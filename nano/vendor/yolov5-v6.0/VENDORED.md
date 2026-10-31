# Vendored: ultralytics/yolov5 v6.0 (Jetson side)

- Upstream: https://github.com/ultralytics/yolov5, tag `v6.0` (`956be8e`, 2021-10-11)
- License: GPL-3.0 (see LICENSE)
- Local change (`LOCAL_PATCHES.diff`): `utils/plots.py` imports seaborn defensively (the Ubuntu
  18.04 seaborn/scipy pair is incompatible with the venv's numpy 1.19).

The last yolov5 release that runs on the Nano's torch 1.10 / Python 3.6.
Used by `nano/apps/live_detect_torch.py`, `nano/apps/benchmark.py` and by
`nano/apps/build_engine.py` to export `.pt` weights to ONNX on the Nano.
