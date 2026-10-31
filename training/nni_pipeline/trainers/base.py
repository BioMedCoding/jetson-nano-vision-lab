"""Protocollo comune implementato da ogni trainers/train_<arch>.py.

Ogni modulo espone una funzione `train(...)` a livello di modulo con questa
firma (nessuna classe: piu' semplice da dispatchare via dict in trial.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol


class TrainResult(dict):
    """dict con chiavi: map50_95, map50, precision, recall,
    best_weights_path, epochs_run."""


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
