"""Path costanti condivise da tutta la pipeline nni_pipeline."""
from pathlib import Path

from common import compat  # noqa: F401 -- side effect: shim filelock/ultralytics, vedi compat.py

PIPELINE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_DIR.parent

YOLOV5_DIR = REPO_ROOT / "yolov5"
YOLOV7_DIR = REPO_ROOT / "yolov7"
LOCAL_CAPTURE_DIR = REPO_ROOT / "local_capture_dataset"

CLASS_MAP_FILE = PIPELINE_DIR / "class_maps" / "target_classes.yaml"

DATASET_CACHE_DIR = PIPELINE_DIR / "dataset_cache"
RUNS_DIR = PIPELINE_DIR / "runs"

DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_canonical_classes() -> list[str]:
    """Lista ordinata dei nomi di classe canonici (indice = class id)."""
    import yaml

    with open(CLASS_MAP_FILE) as f:
        cfg = yaml.safe_load(f)
    canonical = cfg["canonical_classes"]
    return [canonical[i] for i in sorted(canonical)]


def clear_label_cache(data_yaml: Path) -> None:
    """Rimuove labels/{train,val}.cache prima di ogni training.

    yolov5, yolov7 e yolov8 usano tutti lo stesso nome file
    (labels/train.cache, labels/val.cache) per la propria cache di
    label parsate, ma con formati pickle reciprocamente incompatibili.
    Dato che il dataset_source resta fisso mentre l'architettura varia tra i
    trial di Stage A, trial di architetture diverse puntano alla STESSA
    cartella dataset cacheata: senza questa pulizia, un trial yolov5 lascia
    una train.cache che il trial yolov7/yolov8 successivo tenta di
    deserializzare col proprio formato, fallendo con un errore di unpickling
    poco chiaro (verificato in sessione). Costo del rebuild: pochi secondi
    (scansione file), trascurabile rispetto al training vero e proprio.
    """
    data_yaml = Path(data_yaml)
    labels_dir = data_yaml.parent / "labels"
    for split in ("train", "val"):
        for cache in (labels_dir / f"{split}.cache", labels_dir / f"{split}.cache.npy"):
            cache.unlink(missing_ok=True)


def load_source_class_map(source: str) -> dict[str, int | None]:
    """Mappa nome-classe-sorgente -> id canonico (None = classe da scartare)."""
    import yaml

    with open(CLASS_MAP_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["sources"][source]
