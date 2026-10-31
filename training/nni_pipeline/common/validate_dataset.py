"""Validazione struttura dataset YOLO -- port del validatore heredoc gia'
presente in run_pipeline.sh (stessi controlli, riusati anche dal builder
NNI prima di marcare una cartella cache come valida)."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import yaml


class DatasetValidationError(ValueError):
    pass


def validate(dataset_dir: Path) -> dict:
    """Convalida <dataset_dir>: sottocartelle images/labels {train,val},
    data.yaml con 'names' non vuoto, class id massimo nelle label < numero
    di classi dichiarate, e segnala (senza fallire) un rapporto sospetto di
    label vuote. Ritorna un dict di statistiche in caso di successo, solleva
    DatasetValidationError altrimenti."""
    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    errors: list[str] = []

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        if not (dataset_dir / sub).is_dir():
            errors.append(f"manca {dataset_dir / sub}")

    stats: dict = {"n_classes": 0, "n_labels": 0, "n_empty": 0, "images": {}}

    if not data_yaml.is_file():
        errors.append(f"manca {data_yaml}")
    else:
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names")
        n_names = len(names) if isinstance(names, (list, dict)) else 0
        stats["n_classes"] = n_names
        if not n_names:
            errors.append("'names' assente o vuoto in data.yaml")
        else:
            max_seen = -1
            n_labels, n_empty = 0, 0
            for split in ("train", "val"):
                for txt in glob.glob(str(dataset_dir / "labels" / split / "*.txt")):
                    n_labels += 1
                    with open(txt) as lf:
                        lines = [l.strip() for l in lf if l.strip()]
                    if not lines:
                        n_empty += 1
                    for line in lines:
                        max_seen = max(max_seen, int(line.split()[0]))
            stats["n_labels"] = n_labels
            stats["n_empty"] = n_empty
            if max_seen >= n_names:
                errors.append(
                    f"class id massimo nelle label ({max_seen}) >= classi in "
                    f"data.yaml ({n_names})"
                )
            if n_labels and n_empty / n_labels > 0.5:
                stats["empty_ratio_warning"] = True

        for split in ("train", "val"):
            imgs = glob.glob(str(dataset_dir / "images" / split / "*"))
            stats["images"][split] = len(imgs)

    if errors:
        raise DatasetValidationError("; ".join(errors))
    return stats
