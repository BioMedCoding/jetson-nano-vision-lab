"""Adapter YOLOv8 -- pacchetto pip `ultralytics` (testa anchor-free, gia'
verificata sulla Nano via postprocess_v8()). In-process (nessun subprocess):
callback on_fit_epoch_end riporta le metriche epoca-per-epoca senza polling
di file. Verificato in sessione: trainer.metrics e' un dict con chiavi
"metrics/mAP50-95(B)" ecc., popolato ad ogni validazione
(ultralytics/engine/trainer.py); il callback riceve l'istanza trainer come
unico argomento."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ultralytics import YOLO

from common.paths import clear_label_cache

METRIC_KEYS = {
    "map50_95": "metrics/mAP50-95(B)",
    "map50": "metrics/mAP50(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


def train(
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
) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    clear_label_cache(data_yaml)

    model = YOLO(weights)
    last_metrics: dict = {}

    def _on_fit_epoch_end(trainer) -> None:
        m = trainer.metrics or {}
        snapshot = {k: float(m.get(v, 0.0)) for k, v in METRIC_KEYS.items()}
        last_metrics.update(snapshot)
        if on_epoch is not None:
            on_epoch(trainer.epoch, snapshot)

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    model.train(
        data=str(data_yaml),
        imgsz=img_size,
        epochs=epochs,
        patience=patience,
        batch=batch,
        freeze=freeze_layers if freeze_layers else None,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        device=0,
        cache="ram",
        time=max_hours,  # ultralytics supporta nativamente un cap wall-clock
        **hyp,
    )

    best_weights = run_dir / "train" / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"best.pt non trovato: {best_weights}")
    if not last_metrics:
        raise RuntimeError("Nessuna metrica di validazione catturata durante il training yolov8")

    return {
        **last_metrics,
        "best_weights_path": str(best_weights),
        "epochs_run": epochs,
    }
