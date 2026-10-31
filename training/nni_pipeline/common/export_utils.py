"""Export ONNX per architettura + scrittura classes.txt, per il percorso di
promozione Stage B (promote.py). Ogni funzione esporta alla stessa
risoluzione del training (regola gia' seguita da run_pipeline.sh) e
restituisce il path del file .onnx prodotto."""
from __future__ import annotations

import subprocess
from pathlib import Path

from common.paths import YOLOV5_DIR, YOLOV7_DIR

OPSET = 12  # massimo digeribile da TensorRT 8.2 su JetPack 4.6.1 (vedi README_TRAINING.md)


def export_onnx(arch: str, weights_path: Path, img_size: int) -> Path:
    weights_path = Path(weights_path)
    if arch in ("yolov5n", "yolov5s"):
        return _export_yolov5(weights_path, img_size)
    if arch == "yolov7-tiny":
        return _export_yolov7(weights_path, img_size)
    if arch == "yolov8n":
        return _export_yolov8(weights_path, img_size)
    raise ValueError(f"architettura sconosciuta: {arch!r}")


def _export_yolov5(weights_path: Path, img_size: int) -> Path:
    cmd = [
        "python", "export.py",
        "--weights", str(weights_path),
        "--include", "onnx",
        "--imgsz", str(img_size), str(img_size),
        "--opset", str(OPSET),
        "--simplify",
    ]
    _run(cmd, cwd=YOLOV5_DIR)
    onnx_path = weights_path.with_suffix(".onnx")
    _check_produced(onnx_path)
    return onnx_path


def _export_yolov7(weights_path: Path, img_size: int) -> Path:
    # --grid e' obbligatorio: senza, il layer Detect esporta le feature map
    # grezze per-scala invece del tensore concatenato (1,N,5+nc) che
    # postprocess() sulla Nano si aspetta (verificato in sessione ispezionando
    # lo shape ONNX prodotto con/senza il flag).
    cmd = [
        "python", "export.py",
        "--weights", str(weights_path),
        "--img-size", str(img_size), str(img_size),
        "--batch-size", "1",
        "--grid",
        "--simplify",
        "--device", "cpu",
    ]
    _run(cmd, cwd=YOLOV7_DIR)
    onnx_path = weights_path.with_suffix(".onnx")
    _check_produced(onnx_path)
    return onnx_path


def _export_yolov8(weights_path: Path, img_size: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    produced = model.export(format="onnx", imgsz=img_size, opset=OPSET, simplify=True)
    onnx_path = Path(produced)
    _check_produced(onnx_path)
    return onnx_path


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"export ONNX fallito (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )


def _check_produced(onnx_path: Path) -> None:
    if not onnx_path.exists():
        raise RuntimeError(f"export ONNX non ha prodotto il file atteso: {onnx_path}")


def write_classes_txt(classes: list[str], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.write_text("\n".join(classes) + "\n")
