# Vendored: WongKinYiu/yolov7 (training side)

- Upstream: https://github.com/WongKinYiu/yolov7 (unmaintained since 2023)
- Commit: `a207844` (2023-11-03)
- License: GPL-3.0 (see LICENSE.md)
- Local changes: `LOCAL_PATCHES.diff`:
  - train.py, export.py: compatibility with PyTorch >= 2.6
    (`torch.load(weights_only=False)`) and NumPy >= 2.0 (`np.trapz`, `np.int`).
  - utils/google_utils.py: `attempt_download()` lowercased the whole weights
    path, so weights in any directory with a capital letter were treated as
    missing (and a download attempted). Upstream's demo images (`inference/`, `figure/`)
  are not included.

Used by `training/` for YOLOv7-tiny. Export with `--grid` so the ONNX output
is the concatenated `(1, N, 5 + classes)` tensor the Nano's `yolo_v5`
decoder reads.
