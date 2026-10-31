"""The interface every trainer module (yolov5.py, yolov7.py, yolov8.py)
implements: a module-level `train(...)` with this signature (no classes:
simpler to dispatch through the TRAINERS dict in __init__.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol


class TrainResult(dict):
    """dict with keys: map50_95, map50, precision, recall (all from the best
    epoch), best_weights_path, epochs_run."""


class TrainerAdapter(Protocol):
    def train(
        self,
        *,
        data_yaml: Path,
        weights: str,
        img_size: int,
        epochs: int,
        patience: int,
        batch: int,
        hyp: dict,
        freeze_layers: int,
        run_dir: Path,
        on_epoch: Callable[[int, dict], None] | None = None,
        max_hours: float | None = None,
    ) -> TrainResult: ...
