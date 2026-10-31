# Vendored: ultralytics/yolov5 (training side)

- Upstream: https://github.com/ultralytics/yolov5
- Commit: `0f56a767` (2026-07-17, "Use shared setup-uv action everywhere")
- License: AGPL-3.0 (see LICENSE)
- Local changes: `LOCAL_PATCHES.diff` (train.py, export.py: a small
  `filelock` compatibility shim). Nothing else differs from upstream except
  `utils/loggers/comet/optimizer_config.json`, left out by upstream's own
  `.gitignore`.

Used by `training/` for YOLOv5 n/s fine-tuning and ONNX export. This is
NOT the copy that runs on the Nano: that one is yolov5 v6.0, in
`nano/vendor/yolov5-v6.0` (the Nano's torch 1.10 / Python 3.6 cannot run
this version, and it cannot load checkpoints saved by modern torch: always
go through ONNX).
