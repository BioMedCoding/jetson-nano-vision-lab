"""Paths shared by the whole training side, plus small dataset helpers."""
from pathlib import Path

from cvtrain import compat  # noqa: F401  (side effect: filelock shim, see compat.py)

TRAINING_DIR = Path(__file__).resolve().parent.parent
VENDOR_DIR = TRAINING_DIR / "vendor"
YOLOV5_DIR = VENDOR_DIR / "yolov5"
YOLOV7_DIR = VENDOR_DIR / "yolov7"
WEIGHTS_DIR = TRAINING_DIR / "weights"

NNI_DIR = TRAINING_DIR / "nni"
CLASS_MAP_FILE = NNI_DIR / "class_maps" / "target_classes.yaml"

DATASETS_DIR = TRAINING_DIR / "datasets"
DATASET_CACHE_DIR = DATASETS_DIR / "cache"            # built by dataset_builder.py
LOCAL_CAPTURE_DIR = DATASETS_DIR / "local_capture"   # your labelled Nano captures

RUNS_DIR = TRAINING_DIR / "runs"
NNI_RUNS_DIR = RUNS_DIR / "nni"
QUICKSTART_RUNS_DIR = RUNS_DIR / "quickstart"
EXPORT_DIR = TRAINING_DIR / "export"                  # packaged model folders


def load_canonical_classes():
    """Ordered canonical class names for the NNI pipeline (index = class id)."""
    import yaml
    with open(CLASS_MAP_FILE) as f:
        canonical = yaml.safe_load(f)["canonical_classes"]
    return [canonical[i] for i in sorted(canonical)]


def load_source_class_map(source):
    """Source class name -> canonical id (None = drop that class)."""
    import yaml
    with open(CLASS_MAP_FILE) as f:
        return yaml.safe_load(f)["sources"][source]


def read_names(data_yaml):
    """Class names from a YOLO data.yaml, as a list ordered by id."""
    import yaml
    with open(data_yaml) as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def clear_label_cache(data_yaml):
    """Delete labels/{train,val}.cache before every training run.

    yolov5, yolov7 and ultralytics (yolov8) all cache parsed labels under the
    same file names (labels/train.cache, labels/val.cache) but in mutually
    incompatible pickle formats. When trials of different architectures
    share one dataset folder, a cache left by one family makes the next one
    fail with an obscure unpickling error. Rebuilding costs a few seconds.
    """
    from cvtrain.validate_dataset import split_dirs
    for split in ("train", "val"):
        # The cache sits next to the split's label folder and is named after
        # it (labels/train2017.cache for coco128), not after the split.
        labels_split_dir = split_dirs(data_yaml, split)[1]
        for cache in (labels_split_dir.with_suffix(".cache"),
                      labels_split_dir.with_suffix(".cache.npy")):
            cache.unlink(missing_ok=True)


def resolved_data_yaml(data_yaml, out_path):
    """Write a copy of a YOLO data.yaml with absolute train/val paths and no
    "path" key: the one form all three trainers (notably yolov7) accept."""
    import yaml
    from cvtrain.validate_dataset import split_dirs
    data_yaml = Path(data_yaml).resolve()
    out = {split: str(split_dirs(data_yaml, split)[0]) for split in ("train", "val")}
    names = read_names(data_yaml)
    out["nc"] = len(names)
    out["names"] = {i: n for i, n in enumerate(names)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    return Path(out_path)
