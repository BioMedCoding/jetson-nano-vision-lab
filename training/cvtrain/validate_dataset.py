"""Check a YOLO-format dataset before spending hours training on it.
Used by quickstart.py and by the NNI dataset builder before a cache folder
is marked valid."""
from __future__ import annotations

import glob
from pathlib import Path

import yaml


class DatasetValidationError(ValueError):
    pass


# Used when data.yaml has no train/val keys (the standard YOLO layout).
DEFAULT_SPLIT_DIRS = {"train": "images/train", "val": "images/val"}


def split_dirs(data_yaml: Path, split: str):
    """(images_dir, labels_dir) of a split, following data.yaml like the
    trainers do: train/val are absolute, or relative to "path" (itself
    relative to the yaml's folder), and labels live where the last "images"
    path component is replaced by "labels"."""
    data_yaml = Path(data_yaml).resolve()
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    images = Path(cfg.get(split, DEFAULT_SPLIT_DIRS[split]))
    images = images if images.is_absolute() else root / images
    parts = list(images.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
    return images, Path(*parts)


def validate(dataset: Path) -> dict:
    """Validate a dataset (its folder, or its data.yaml): train/val image and
    label folders exist, 'names' is non-empty, and every class id in the
    labels is below the number of declared classes (the most common, and
    most silent, mistake). Flags (without failing) a suspicious share of
    empty label files. Returns statistics, or raises DatasetValidationError."""
    dataset = Path(dataset)
    data_yaml = dataset if dataset.suffix in (".yaml", ".yml") else dataset / "data.yaml"
    if not data_yaml.is_file():
        raise DatasetValidationError(f"missing {data_yaml}")
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    errors: list[str] = []
    stats: dict = {"n_classes": 0, "n_labels": 0, "n_empty": 0, "images": {}}
    names = cfg.get("names")
    n_names = len(names) if isinstance(names, (list, dict)) else 0
    stats["n_classes"] = n_names
    if not n_names:
        errors.append("'names' missing or empty in data.yaml")

    max_seen, seen_label_dirs = -1, set()
    for split in ("train", "val"):
        images_dir, labels_dir = split_dirs(data_yaml, split)
        for d in (images_dir, labels_dir):
            if not d.is_dir():
                errors.append(f"missing {d}")
        stats["images"][split] = len(glob.glob(str(images_dir / "*")))
        if labels_dir in seen_label_dirs:   # train and val may be the same folder
            continue
        seen_label_dirs.add(labels_dir)
        for txt in glob.glob(str(labels_dir / "*.txt")):
            stats["n_labels"] += 1
            with open(txt) as lf:
                lines = [l.strip() for l in lf if l.strip()]
            if not lines:
                stats["n_empty"] += 1
            for line in lines:
                max_seen = max(max_seen, int(line.split()[0]))

    if n_names and max_seen >= n_names:
        errors.append(f"highest class id in the labels ({max_seen}) >= number of "
                      f"classes in data.yaml ({n_names})")
    if stats["n_labels"] and stats["n_empty"] / stats["n_labels"] > 0.5:
        stats["empty_ratio_warning"] = True
    if errors:
        raise DatasetValidationError("; ".join(errors))
    return stats
