"""YOLOv8 trainer: the `ultralytics` pip package (anchor-free head -> decoder
"yolo_v8" on the Nano), run in-process. An on_fit_epoch_end callback reports
per-epoch metrics without polling files: trainer.metrics is a dict with keys
such as "metrics/mAP50-95(B)", filled at every validation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from cvtrain.device import train_device
from cvtrain.paths import clear_label_cache  # imports the filelock shim first

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
    from ultralytics import YOLO

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    clear_label_cache(data_yaml)

    model = YOLO(weights)
    best: dict = {}
    epochs_seen = [0]

    def _on_fit_epoch_end(trainer) -> None:
        m = trainer.metrics or {}
        snapshot = {k: float(m.get(v, 0.0)) for k, v in METRIC_KEYS.items()}
        epochs_seen[0] = trainer.epoch + 1
        # Report the best epoch, like the yolov5/yolov7 trainers do (best.pt
        # is the best epoch too), not simply the last one.
        if not best or snapshot["map50_95"] > best["map50_95"]:
            best.update(snapshot)
        if on_epoch is not None:
            on_epoch(trainer.epoch, snapshot)

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    device = train_device()
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
        device=int(device) if device.isdigit() else device,
        cache="ram",
        time=max_hours,  # ultralytics has a native wall-clock cap
        **hyp,
    )

    best_weights = run_dir / "train" / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"best.pt not found: {best_weights}")
    if not best:
        raise RuntimeError("no validation metrics were captured during yolov8 training")
    return {**best, "best_weights_path": str(best_weights), "epochs_run": epochs_seen[0]}
