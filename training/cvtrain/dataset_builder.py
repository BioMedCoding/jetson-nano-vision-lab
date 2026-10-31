"""Cached dataset builder for the NNI pipeline.

Builds a YOLO dataset from a declarative config (which public sources, how
many images, whether to include your own labelled captures) and caches it by
config hash, so Stage A, which calls build_dataset() dozens of times with the
same dataset_source, does not download and rebuild every time.

Both public sources (Open Images V7, Sama-COCO) go through the same path:
fetch via the FiftyOne zoo -> export in YOLOv5 format -> remap class ids to
the canonical class list (nni/class_maps/target_classes.yaml) -> merge into
one cache folder, file names prefixed by source to avoid collisions.
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

from cvtrain.paths import DATASET_CACHE_DIR, LOCAL_CAPTURE_DIR, load_source_class_map
from cvtrain.validate_dataset import validate

logger = logging.getLogger("dataset_builder")

VALID_SOURCES = {"oiv7", "samacoco"}
ZOO_DATASET_NAME = {"oiv7": "open-images-v7", "samacoco": "sama-coco"}
# The source first planned as "lvis" (dense re-annotation of the COCO
# images) was replaced by "samacoco": fiftyone==1.19.0 has neither an "lvis"
# zoo dataset nor a fiftyone.utils.lvis module (removed upstream). Sama-COCO
# offers the same benefit (denser, higher-quality re-annotation of the same
# COCO images) through the native zoo.


def dataset_config_hash(config: dict) -> str:
    canon = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def build_dataset(config: dict, cache_root: Path = DATASET_CACHE_DIR) -> Path:
    """config keys:
        source: "oiv7" | "samacoco" | "oiv7+samacoco"
        classes: list[str], canonical names ordered by id
            (cvtrain.paths.load_canonical_classes())
        max_samples_per_class: int -- approximate cap handed to FiftyOne
            (FiftyOne caps the total number of samples matching ANY of the
            requested classes, not each class: we multiply by the number of
            classes as an estimate, not an exact guarantee)
        split_seed: int
        val_fraction: float
        include_local_capture: bool -- if True, datasets/local_capture/ must
            exist in YOLO format; if False that path is never touched.

    Returns <cache_dir>/data.yaml. Idempotent: if <cache_dir>/BUILD_OK
    exists, the cache is reused without rebuilding.
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
            if ok_marker.exists():  # re-check under the lock: another process may have finished
                return cache_dir / "data.yaml"
            logger.info("dataset cache miss: %s (source=%s), building", h, config["source"])
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
            # _build_into() wrote data.yaml with tmp_dir's absolute path
            # (validate() needs it before the rename): rewrite it with the
            # final cache_dir path, or it would point at a .building-<uuid>
            # folder that no longer exists.
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
            raise ValueError(f"unknown dataset source: {src!r} (valid: {VALID_SOURCES})")
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
        logger.warning("no classes mapped for source %s, skipping", source)
        return

    max_samples = config["max_samples_per_class"] * len(wanted_source_names)
    fo_name = f"cvtrain_{source}_{uuid.uuid4().hex[:8]}"
    scratch = out_dir / f"_scratch_{source}"

    logger.info(
        "fetch %s: classes=%s max_samples~%d", source, wanted_source_names, max_samples
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
        # split="train" is forced: otherwise FiftyOne's YOLOv5 exporter
        # picks the subfolder itself (observed: images/val/ even for samples
        # loaded with split="train"). Fixing it makes scratch/ predictable
        # for _remap_and_split(), which does its own train/val split from
        # config["val_fraction"] anyway.
        fo_dataset.export(
            export_dir=str(scratch),
            dataset_type=fo.types.YOLOv5Dataset,
            classes=wanted_source_names,
            split="train",
        )
    finally:
        fo.delete_dataset(fo_name)

    # FiftyOne's exporter assigns local ids 0..N-1 in the order of
    # wanted_source_names given to export(): rebuild that mapping here to
    # remap onto the canonical ids.
    id_to_canonical = {i: name_to_canonical[name] for i, name in enumerate(wanted_source_names)}
    _remap_and_split(scratch, source, id_to_canonical, config, out_dir)
    shutil.rmtree(scratch, ignore_errors=True)


def _remap_and_split(
    scratch: Path, source: str, id_to_canonical: dict, config: dict, out_dir: Path
) -> None:
    # scratch/images and scratch/labels have a single "train/" level below,
    # because _fetch_and_merge_source forces split="train" in export().
    images_dir = scratch / "images" / "train"
    labels_dir = scratch / "labels" / "train"
    if not images_dir.is_dir():
        logger.warning("FiftyOne export for %s produced no images, skipping", source)
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
                continue  # class not in the canonical map: dropped
            parts[0] = str(canonical_id)
            lines_out.append(" ".join(parts))
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def _merge_local_capture(config: dict, out_dir: Path) -> None:
    if not LOCAL_CAPTURE_DIR.is_dir():
        raise RuntimeError(
            f"include_local_capture=True but {LOCAL_CAPTURE_DIR} does not exist. "
            "See training/nni/README.md, 'Using your own captures', "
            "before enabling this flag in the search space."
        )
    local_yaml = LOCAL_CAPTURE_DIR / "data.yaml"
    if not local_yaml.is_file():
        raise RuntimeError(
            f"include_local_capture=True but {local_yaml} is missing: "
            "datasets/local_capture/ is not a valid YOLO dataset."
        )
    with open(local_yaml) as f:
        local_cfg = yaml.safe_load(f)
    local_names = local_cfg.get("names")
    if isinstance(local_names, dict):
        local_names = [local_names[i] for i in sorted(local_names)]
    if not local_names:
        raise RuntimeError(f"'names' missing or empty in {local_yaml}")

    name_to_canonical = load_source_class_map("local_capture")
    id_to_canonical = {i: name_to_canonical.get(name) for i, name in enumerate(local_names)}

    for split in ("train", "val"):
        img_dir = LOCAL_CAPTURE_DIR / "images" / split
        label_dir = LOCAL_CAPTURE_DIR / "labels" / split
        if not img_dir.is_dir():
            raise RuntimeError(f"missing {img_dir} in datasets/local_capture/")
        for img_path in sorted(p for p in img_dir.glob("*") if p.is_file()):
            label_path = label_dir / (img_path.stem + ".txt")
            new_stem = f"local_{img_path.stem}"
            dst_img = out_dir / "images" / split / (new_stem + img_path.suffix)
            dst_label = out_dir / "labels" / split / (new_stem + ".txt")
            shutil.copy2(img_path, dst_img)
            _write_remapped_label(label_path, dst_label, id_to_canonical)


def _write_data_yaml(out_dir: Path, classes: list[str]) -> None:
    # Absolute train/val (no "path" key + relative paths): yolov7's
    # check_dataset() takes train/val as-is without joining them to "path"
    # (a yolov5/ultralytics-only convention), so relative paths would
    # resolve against the cwd. Absolute paths work for all three trainers.
    data = {
        "train": str(out_dir / "images" / "train"),
        "val": str(out_dir / "images" / "val"),
        "nc": len(classes),
        "names": {i: name for i, name in enumerate(classes)},
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
