"""ONNX export per architecture family, at the training resolution (export
at another size and the Nano runs the model at a scale it never saw).

OPSET 12 is the highest TensorRT 8.2 on JetPack 4.6.1 handles reliably.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cvtrain.paths import YOLOV5_DIR, YOLOV7_DIR

OPSET = 12


def export_onnx(arch: str, weights_path: Path, img_size: int) -> Path:
    weights_path = Path(weights_path)
    if arch in ("yolov5n", "yolov5s"):
        return _export_yolov5(weights_path, img_size)
    if arch == "yolov7-tiny":
        return _export_yolov7(weights_path, img_size)
    if arch == "yolov8n":
        return _export_yolov8(weights_path, img_size)
    raise ValueError(f"unknown architecture: {arch!r}")


def _export_yolov5(weights_path: Path, img_size: int) -> Path:
    _run([sys.executable, "export.py", "--weights", str(weights_path),
          "--include", "onnx", "--imgsz", str(img_size), str(img_size),
          "--opset", str(OPSET), "--simplify"], cwd=YOLOV5_DIR)
    return _check_produced(weights_path.with_suffix(".onnx"))


def _export_yolov7(weights_path: Path, img_size: int) -> Path:
    # --grid is mandatory: without it the Detect layer exports the raw
    # per-scale feature maps instead of the concatenated (1, N, 5+nc) tensor
    # the Nano's yolo_v5 decoder expects (checked on the exported shapes).
    _run([sys.executable, "export.py", "--weights", str(weights_path),
          "--img-size", str(img_size), str(img_size), "--batch-size", "1",
          "--grid", "--simplify", "--device", "cpu"], cwd=YOLOV7_DIR)
    return _check_produced(weights_path.with_suffix(".onnx"))


def _export_yolov8(weights_path: Path, img_size: int) -> Path:
    from ultralytics import YOLO
    produced = YOLO(str(weights_path)).export(format="onnx", imgsz=img_size,
                                              opset=OPSET, simplify=True)
    return _check_produced(Path(produced))


def _run(cmd: list, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ONNX export failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def _check_produced(onnx_path: Path) -> Path:
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX export did not produce the expected file: {onnx_path}")
    return onnx_path


def onnx_signature(onnx_path: Path):
    """(input_shape, [output_shapes]) of an ONNX file."""
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=False)
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    return dims(m.graph.input[0]), [dims(o) for o in m.graph.output]


def decoder_from_shapes(outputs):
    """Same rule as nano/apps/build_engine.py: the main (largest 3-D) output
    is (1, N, 5+nc) for yolo_v5 or (1, 4+nc, N) for yolo_v8.
    Returns (decoder, n_classes)."""
    three_d = [s for s in outputs if len(s) == 3]
    if not three_d:
        raise ValueError(f"no 3-D output in {outputs}")
    _, a, b = max(three_d, key=lambda s: s[1] * s[2])
    return ("yolo_v5", b - 5) if a > b else ("yolo_v8", a - 4)
