"""Adapter YOLOv7 -- repo classica vendored (WongKinYiu/yolov7, testa
anchor-based, stesso formato output di yolov5 -> decoder "v5" sulla Nano).

Note su incompatibilita' scoperte in sessione e patchate direttamente nel
clone vendorizzato (yolov7/train.py, yolov7/export.py):
- PyTorch >=2.6 richiede weights_only=False per i checkpoint yolov7 (pickle
  di modello completo, repo non piu' mantenuto da WongKinYiu dal 2022).
- NumPy >=2.0 rimuove np.trapz/np.int usati da utils/metrics.py e
  utils/datasets.py.
- train.py NON supporta --patience nativamente: l'early stopping e'
  implementato qui monitorando results.txt e terminando il subprocess.
- data.yaml deve avere train/val come path assoluti: check_dataset() di
  questo repo non supporta la chiave "path" + relativi di yolov5/ultralytics
  (vedi dataset_builder.py::_write_data_yaml).
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import yaml

from common.paths import YOLOV7_DIR, clear_label_cache

# ordine dei 15 token whitespace-separated di ogni riga di results.txt
# (verificato in sessione con una run reale, non dal solo codice sorgente)
RESULT_FIELDS = [
    "epoch_frac", "gpu_mem", "train_box", "train_obj", "train_cls", "train_total",
    "targets", "img_size", "precision", "recall", "map50", "map50_95",
    "val_box", "val_obj", "val_cls",
]


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
    results_txt = train_out_dir / "results.txt"

    cmd = [
        "python", "train.py",
        "--data", str(data_yaml),
        "--weights", str(weights),
        "--cfg", "cfg/training/yolov7-tiny.yaml",
        "--img-size", str(img_size), str(img_size),
        "--batch-size", str(batch),
        "--epochs", str(epochs),
        "--hyp", str(hyp_path),
        "--freeze", str(freeze_layers),
        "--project", str(run_dir),
        "--name", train_name,
        "--exist-ok",
        "--device", "0",
        "--cache-images",
    ]

    proc = subprocess.Popen(
        cmd, cwd=str(YOLOV7_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    stop_flag = threading.Event()
    early_stopped = threading.Event()
    watcher = threading.Thread(
        target=_watch_and_early_stop,
        args=(results_txt, on_epoch, stop_flag, patience, proc, early_stopped),
        daemon=True,
    )
    watcher.start()

    timer = None
    if max_hours is not None:
        timer = threading.Timer(max_hours * 3600, _terminate, args=(proc, early_stopped))
        timer.daemon = True
        timer.start()

    log_lines = []
    for line in proc.stdout:
        log_lines.append(line)
    proc.wait()
    stop_flag.set()
    watcher.join(timeout=15)
    if timer is not None:
        timer.cancel()

    if proc.returncode != 0 and not early_stopped.is_set():
        tail = "".join(log_lines[-60:])
        raise RuntimeError(f"yolov7 train.py fallito (exit {proc.returncode}):\n{tail}")

    return _final_metrics(results_txt, train_out_dir)


def _terminate(proc: subprocess.Popen, flag: threading.Event) -> None:
    flag.set()
    proc.terminate()


def _watch_and_early_stop(
    results_txt: Path,
    on_epoch: Callable | None,
    stop_flag: threading.Event,
    patience: int,
    proc: subprocess.Popen,
    early_stopped: threading.Event,
) -> None:
    seen = 0
    best_map = -1.0
    best_epoch_idx = -1
    while not stop_flag.is_set():
        seen, best_map, best_epoch_idx, last_epoch_idx = _emit_new_rows(
            results_txt, on_epoch, seen, best_map, best_epoch_idx
        )
        if patience and last_epoch_idx is not None and (last_epoch_idx - best_epoch_idx) >= patience:
            early_stopped.set()
            proc.terminate()
            return
        time.sleep(2)
    _emit_new_rows(results_txt, on_epoch, seen, best_map, best_epoch_idx)


def _emit_new_rows(
    results_txt: Path, on_epoch: Callable | None, seen: int, best_map: float, best_epoch_idx: int
) -> tuple[int, float, int, int | None]:
    if not results_txt.exists():
        return seen, best_map, best_epoch_idx, None
    rows = _read_results_txt(results_txt)
    last_epoch_idx = rows[-1]["epoch"] if rows else None
    for row in rows[seen:]:
        if on_epoch is not None:
            on_epoch(row["epoch"], row)
        if row["map50_95"] > best_map:
            best_map = row["map50_95"]
            best_epoch_idx = row["epoch"]
    return len(rows), best_map, best_epoch_idx, last_epoch_idx


def _read_results_txt(results_txt: Path) -> list[dict]:
    rows = []
    with open(results_txt) as f:
        for line in f:
            tokens = line.split()
            if len(tokens) != len(RESULT_FIELDS):
                continue
            epoch_str = tokens[0].split("/")[0]
            rows.append(
                {
                    "epoch": int(epoch_str),
                    "map50_95": float(tokens[RESULT_FIELDS.index("map50_95")]),
                    "map50": float(tokens[RESULT_FIELDS.index("map50")]),
                    "precision": float(tokens[RESULT_FIELDS.index("precision")]),
                    "recall": float(tokens[RESULT_FIELDS.index("recall")]),
                }
            )
    return rows


def _final_metrics(results_txt: Path, train_out_dir: Path) -> dict:
    rows = _read_results_txt(results_txt)
    if not rows:
        raise RuntimeError(f"results.txt vuoto o assente: {results_txt}")
    best = max(rows, key=lambda r: r["map50_95"])
    best_weights = train_out_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"best.pt non trovato: {best_weights}")
    return {
        "map50_95": best["map50_95"],
        "map50": best["map50"],
        "precision": best["precision"],
        "recall": best["recall"],
        "best_weights_path": str(best_weights),
        "epochs_run": rows[-1]["epoch"] + 1,
    }
