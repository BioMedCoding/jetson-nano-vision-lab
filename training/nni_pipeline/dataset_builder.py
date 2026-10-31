"""Dataset builder cacheato per la pipeline NNI.

Costruisce un dataset YOLO a partire da una config dichiarativa (quali
sorgenti pubbliche usare, quante immagini, se includere il dataset locale
catturato) e lo cachea per hash della config, cosi' Stage A -- che richiama
build_dataset() decine di volte con lo stesso dataset_source -- non
riscarica/ricostruisce ogni volta.

Le due sorgenti pubbliche (Open Images V7, LVIS) passano entrambe per lo
stesso percorso generico: fetch via FiftyOne -> export in formato YOLOv5 ->
remap dei class-id sulla lista di classi canoniche (class_maps/
target_classes.yaml) -> merge in un'unica cartella cache con nomi file
prefissati per sorgente per evitare collisioni.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import random
import shutil
import uuid
from pathlib import Path

import yaml

from common.paths import DATASET_CACHE_DIR, LOCAL_CAPTURE_DIR, load_source_class_map
from common.validate_dataset import validate

logger = logging.getLogger("dataset_builder")

VALID_SOURCES = {"oiv7", "samacoco"}
ZOO_DATASET_NAME = {"oiv7": "open-images-v7", "samacoco": "sama-coco"}
# NOTA: la sorgente originariamente pianificata come "lvis" (riannotazione
# densa delle immagini COCO) e' stata sostituita con "samacoco" (Sama-COCO):
# verificato in sessione che fiftyone==1.19.0 non ha ne' un dataset zoo
# "lvis" ne' un modulo fiftyone.utils.lvis -- supporto LVIS rimosso a monte.
# Sama-COCO fornisce lo stesso value proposition (riannotazione di qualita'
# piu' densa sulle stesse immagini COCO) tramite lo zoo nativo, zero lavoro
# extra di integrazione.


def dataset_config_hash(config: dict) -> str:
    canon = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def build_dataset(config: dict, cache_root: Path = DATASET_CACHE_DIR) -> Path:
    """config keys:
        source: "oiv7" | "lvis" | "oiv7+lvis"
        classes: list[str] nomi canonici ordinati per id (vedi
            common.paths.load_canonical_classes())
        max_samples_per_class: int -- cap approssimativo passato a FiftyOne
            (FiftyOne applica il cap sul totale campioni che matchano
            almeno una delle classi richieste, non un cap rigido per
            singola classe: qui moltiplichiamo per il numero di classi
            come stima, non come garanzia esatta)
        split_seed: int
        val_fraction: float
        include_local_capture: bool -- se True richiede local_capture_dataset/
            gia' presente e strutturato in formato YOLO; se False non tocca
            mai quel path (costo zero, nessun crash).

    Ritorna il path a <cache_dir>/data.yaml. Idempotente: se
    <cache_dir>/BUILD_OK esiste gia', riusa senza ricostruire.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    h = dataset_config_hash(config)
    cache_dir = cache_root / h
    ok_marker = cache_dir / "BUILD_OK"
    if ok_marker.exists():
        logger.info("dataset cache hit: %s (source=%s)", h, config["source"])
        return cache_dir / "data.yaml"

    lock_path = cache_root / f"{h}.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if ok_marker.exists():  # ricontrolla dentro il lock (altro processo puo' aver finito)
                return cache_dir / "data.yaml"
            logger.info("dataset cache miss: %s (source=%s) -- costruisco", h, config["source"])
            tmp_dir = cache_root / f"{h}.building-{uuid.uuid4().hex[:8]}"
            try:
                _build_into(config, tmp_dir)
                stats = validate(tmp_dir)
            except Exception:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
            (tmp_dir / "build_config.json").write_text(
                json.dumps({"config": config, "stats": stats}, indent=2)
            )
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            tmp_dir.replace(cache_dir)
            # data.yaml e' stato scritto dentro _build_into() con il path
            # assoluto di tmp_dir (necessario perche' validate() lo richiede
            # gia' pronto prima del rename) -- va riscritto con il path
            # definitivo di cache_dir, altrimenti punta a una directory
            # .building-<uuid> che non esiste piu' dopo il rename.
            _write_data_yaml(cache_dir, config["classes"])
            (cache_dir / "BUILD_OK").touch()
            logger.info("dataset cache build ok: %s (source=%s) stats=%s", h, config["source"], stats)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return cache_dir / "data.yaml"


def _build_into(config: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    classes = config["classes"]
    sources = config["source"].split("+")
    for src in sources:
        if src not in VALID_SOURCES:
            raise ValueError(f"sorgente dataset sconosciuta: {src!r} (valide: {VALID_SOURCES})")
        _fetch_and_merge_source(src, config, out_dir)

    if config.get("include_local_capture"):
        _merge_local_capture(config, out_dir)

    _write_data_yaml(out_dir, classes)


def _fetch_and_merge_source(source: str, config: dict, out_dir: Path) -> None:
    import fiftyone as fo
    import fiftyone.zoo as foz

    name_to_canonical = load_source_class_map(source)
    wanted_source_names = list(name_to_canonical.keys())
    if not wanted_source_names:
        logger.warning("nessuna classe mappata per la sorgente %s, salto", source)
        return

    max_samples = config["max_samples_per_class"] * len(wanted_source_names)
    fo_name = f"cvtrain_{source}_{uuid.uuid4().hex[:8]}"
    scratch = out_dir / f"_scratch_{source}"

    logger.info(
        "fetch %s: classi=%s max_samples~%d", source, wanted_source_names, max_samples
    )
    fo_dataset = foz.load_zoo_dataset(
        ZOO_DATASET_NAME[source],
        split="train",
        label_types=["detections"],
        classes=wanted_source_names,
        max_samples=max_samples,
        dataset_name=fo_name,
    )
    try:
        # split="train" e' forzato esplicitamente: senza, l'exporter YOLOv5
        # di FiftyOne decide da solo la sottocartella (osservato: finisce in
        # images/val/ anche per campioni caricati con split="train" dallo
        # zoo) -- fissarlo rende il layout di scratch/ prevedibile per
        # _remap_and_split(), che gestisce comunque il proprio split
        # train/val a valle in base a config["val_fraction"].
        fo_dataset.export(
            export_dir=str(scratch),
            dataset_type=fo.types.YOLOv5Dataset,
            classes=wanted_source_names,
            split="train",
        )
    finally:
        fo.delete_dataset(fo_name)

    # L'exporter FiftyOne assegna id locali 0..N-1 nell'ordine di
    # wanted_source_names passato a export() -- ricostruiamo qui la stessa
    # corrispondenza per remappare sugli id canonici.
    id_to_canonical = {i: name_to_canonical[name] for i, name in enumerate(wanted_source_names)}
    _remap_and_split(scratch, source, id_to_canonical, config, out_dir)
    shutil.rmtree(scratch, ignore_errors=True)


def _remap_and_split(
    scratch: Path, source: str, id_to_canonical: dict, config: dict, out_dir: Path
) -> None:
    # scratch/images e scratch/labels hanno un solo livello "train/" sotto,
    # perche' _fetch_and_merge_source forza split="train" in export().
    images_dir = scratch / "images" / "train"
    labels_dir = scratch / "labels" / "train"
    if not images_dir.is_dir():
        logger.warning("export FiftyOne per %s non ha prodotto immagini, salto", source)
        return

    images = sorted(p for p in images_dir.glob("*") if p.is_file())
    rng = random.Random(config["split_seed"])
    rng.shuffle(images)
    n_val = max(1, int(len(images) * config["val_fraction"])) if images else 0
    val_set = set(images[:n_val])

    for img_path in images:
        label_path = labels_dir / (img_path.stem + ".txt")
        split = "val" if img_path in val_set else "train"
        new_stem = f"{source}_{img_path.stem}"
        dst_img = out_dir / "images" / split / (new_stem + img_path.suffix)
        dst_label = out_dir / "labels" / split / (new_stem + ".txt")
        shutil.copy2(img_path, dst_img)
        _write_remapped_label(label_path, dst_label, id_to_canonical)


def _write_remapped_label(src_label: Path, dst_label: Path, id_to_canonical: dict) -> None:
    if not src_label.exists():
        dst_label.write_text("")
        return
    lines_out = []
    with open(src_label) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            local_id = int(parts[0])
            canonical_id = id_to_canonical.get(local_id)
            if canonical_id is None:
                continue  # classe non nella mappa canonica, scartata
            parts[0] = str(canonical_id)
            lines_out.append(" ".join(parts))
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def _merge_local_capture(config: dict, out_dir: Path) -> None:
    if not LOCAL_CAPTURE_DIR.is_dir():
        raise RuntimeError(
            f"include_local_capture=True ma {LOCAL_CAPTURE_DIR} non esiste. "
            "Vedi nni_pipeline/README.md sezione 'Abilitare include_local_capture' "
            "prima di attivare questo flag nel search space."
        )
    local_yaml = LOCAL_CAPTURE_DIR / "data.yaml"
    if not local_yaml.is_file():
        raise RuntimeError(
            f"include_local_capture=True ma manca {local_yaml} -- "
            "local_capture_dataset/ non e' strutturato in formato YOLO valido."
        )
    with open(local_yaml) as f:
        local_cfg = yaml.safe_load(f)
    local_names = local_cfg.get("names")
    if isinstance(local_names, dict):
        local_names = [local_names[i] for i in sorted(local_names)]
    if not local_names:
        raise RuntimeError(f"'names' assente o vuoto in {local_yaml}")

    name_to_canonical = load_source_class_map("local_capture")
    id_to_canonical = {i: name_to_canonical.get(name) for i, name in enumerate(local_names)}

    for split in ("train", "val"):
        img_dir = LOCAL_CAPTURE_DIR / "images" / split
        label_dir = LOCAL_CAPTURE_DIR / "labels" / split
        if not img_dir.is_dir():
            raise RuntimeError(f"manca {img_dir} in local_capture_dataset/")
        for img_path in sorted(p for p in img_dir.glob("*") if p.is_file()):
            label_path = label_dir / (img_path.stem + ".txt")
            new_stem = f"local_{img_path.stem}"
            dst_img = out_dir / "images" / split / (new_stem + img_path.suffix)
            dst_label = out_dir / "labels" / split / (new_stem + ".txt")
            shutil.copy2(img_path, dst_img)
            _write_remapped_label(label_path, dst_label, id_to_canonical)


def _write_data_yaml(out_dir: Path, classes: list[str]) -> None:
    # train/val assoluti (niente chiave "path" + relativi): yolov7's
    # check_dataset() risolve train/val cosi' come sono senza unirli a
    # "path" (convenzione solo yolov5/ultralytics) -- verificato in sessione
    # che con path relativi risolve rispetto alla cwd invece che a out_dir.
    # Assoluti funzionano correttamente su tutti e tre gli adapter.
    data = {
        "train": str(out_dir / "images" / "train"),
        "val": str(out_dir / "images" / "val"),
        "nc": len(classes),
        "names": {i: name for i, name in enumerate(classes)},
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
