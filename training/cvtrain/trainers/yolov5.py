"""YOLOv5 trainer: the classic vendored repo (ultralytics/yolov5, anchor-
based head), for yolov5n and yolov5s. Never use the `ultralytics` pip
package for these two: its yolov5 "u" variants are anchor-free and would
silently need the other decoder on the Nano."""
from __future__ import annotations

import csv
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import yaml

from cvtrain.device import train_device
from cvtrain.paths import YOLOV5_DIR, clear_label_cache

RESULT_KEYS = {
    "map50_95": "metrics/mAP_0.5:0.95",
    "map50": "metrics/mAP_0.5",
    "precision": "metrics/precision",
    "recall": "metrics/recall",
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
    hyp_path = run_dir / "hyp_resolved.yaml"
    with open(hyp_path, "w") as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

    train_name = "train"
    train_out_dir = run_dir / train_name
    results_csv = train_out_dir / "results.csv"

    cmd = [
        sys.executable, "train.py",
        "--data", str(data_yaml),
        "--weights", str(weights),
        "--imgsz", str(img_size),
        "--batch-size", str(batch),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--hyp", str(hyp_path),
        "--freeze", str(freeze_layers),
        "--project", str(run_dir),
        "--name", train_name,
        "--exist-ok",
        "--device", train_device(),
        "--cache", "ram",
    ]

    proc = subprocess.Popen(
        cmd, cwd=str(YOLOV5_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    stop_flag = threading.Event()
    watcher = None
    if on_epoch is not None:
        watcher = threading.Thread(
            target=_tail_results_csv, args=(results_csv, on_epoch, stop_flag), daemon=True
        )
        watcher.start()

    timed_out = threading.Event()
    timer = None
    if max_hours is not None:
        timer = threading.Timer(max_hours * 3600, _terminate, args=(proc, timed_out))
        timer.daemon = True
        timer.start()

    log_lines = []
    for line in proc.stdout:
        log_lines.append(line)
    proc.wait()
    stop_flag.set()
    if watcher is not None:
        watcher.join(timeout=15)
    if timer is not None:
        timer.cancel()

    if proc.returncode != 0 and not timed_out.is_set():
        tail = "".join(log_lines[-60:])
        raise RuntimeError(f"yolov5 train.py failed (exit {proc.returncode}):\n{tail}")

    return _final_metrics(results_csv, train_out_dir)


def _terminate(proc: subprocess.Popen, timed_out: threading.Event) -> None:
    timed_out.set()
    proc.terminate()


def _tail_results_csv(results_csv: Path, on_epoch: Callable, stop_flag: threading.Event) -> None:
    seen = 0
    while not stop_flag.is_set():
        seen = _emit_new_rows(results_csv, on_epoch, seen)
        time.sleep(2)
    _emit_new_rows(results_csv, on_epoch, seen)  # pick up the final rows


def _emit_new_rows(results_csv: Path, on_epoch: Callable, seen: int) -> int:
    if not results_csv.exists():
        return seen
    rows = _read_results_csv(results_csv)
    for row in rows[seen:]:
        on_epoch(row["epoch"], row)
    return len(rows)


def _read_results_csv(results_csv: Path) -> list[dict]:
    rows = []
    with open(results_csv) as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            return rows
        for raw in reader:
            raw = [v for v in raw if v.strip() != ""]
            if not raw:
                continue
            values = dict(zip(header, (float(v) for v in raw)))
            rows.append(
                {
                    "epoch": int(values["epoch"]),
                    "map50_95": values[RESULT_KEYS["map50_95"]],
                    "map50": values[RESULT_KEYS["map50"]],
                    "precision": values[RESULT_KEYS["precision"]],
                    "recall": values[RESULT_KEYS["recall"]],
                }
            )
    return rows


def _final_metrics(results_csv: Path, train_out_dir: Path) -> dict:
    rows = _read_results_csv(results_csv)
    if not rows:
        raise RuntimeError(f"results.csv empty or missing: {results_csv}")
    best = max(rows, key=lambda r: r["map50_95"])
    best_weights = train_out_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"best.pt not found: {best_weights}")
    return {
        "map50_95": best["map50_95"],
        "map50": best["map50"],
        "precision": best["precision"],
        "recall": best["recall"],
        "best_weights_path": str(best_weights),
        "epochs_run": rows[-1]["epoch"] + 1,
    }
