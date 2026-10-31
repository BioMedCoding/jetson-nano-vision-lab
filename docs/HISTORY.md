# Project history

This repository starts with a fresh git history (2026-10-31). Before that,
the work lived in two separate repositories and some untracked copies; their
commit log is summarized here so the reasoning is not lost.

## jetson-nano-ov5647 (camera + Jetson userspace)

**2026-07-17: OV5647 support for the Jetson Nano (L4T R32.7.1, kernel 4.9.253).**
Full patched L4T source tree (kernel-4.9 + nvidia + nvgpu + hardware DTS) plus
interactive build/deploy scripts, so it can be rebuilt without NVIDIA's
archives staying online. OV5647 driver ported to the 4.9 v4l2_of API, board
DTS/DTSI wiring it on CAM0 (I2C 0x36, CLK_OUT_3 MCLK, GPIO reset held high),
selectable from the extlinux boot menu next to the stock IMX219 kernel. Fixed
`scripts/Kbuild.include`'s "the-space" trick, which corrupted the dtb list for
ALL Tegra DTBs. GCC 15 compatibility: real bugs fixed (dead address-of-NULL
checks, a missing macro brace, a header-guard typo, an
`ether_addr_equal_64bits` over-read) and new warning classes downgraded; no
driver or Kconfig option disabled.

**2026-07-18: camera capture working end to end + YOLOv5 pipeline.** Fixed the
silent "frame start syncpt timeout" that blocked every capture: five stacked
problems (a `to_camera_common_data()` OOPS on non-camera_common drvdata, missing
pad get/set_fmt so S_FMT was swallowed as -ENOTTY, MIPI PHY calibration
silently skipped without s_data, a MIPI clock-lane mode mismatch, the actual
frame killer, found by reading CIL/CILX_STATUS at timeout time through dynamic
debug, and manual AEC/AGC with nothing driving it, giving black frames).
Prebuilt Image + DTB + module verified at 62.5 fps. Userspace: raw-V4L2 camera
library (GRBG demosaic, bypass_mode=0), detection scripts, yolov5 v6.0
snapshot, torch/torchvision cp36 wheels, a setup script covering every
environment trap met. TensorRT FP16 detection at 30.6 fps live.

**2026-07-18: AGC gain ceiling raised to 15.5x** (0x3a18/0x3a19 = 0x00f8):
indoor scenes went from ~5-7% to ~15% mean brightness, converging in about a
second.

**2026-07-19: web dashboard, tracker / ROI features, more models.** The MJPEG
demo became a test platform: jetson-stats system panel, inference history,
image filters, live thresholds, hot model switching, a centroid tracker with
unique-ID counts, a draggable ROI with enter/exit counting, trajectories.
yolov7-tiny and yolov8n added (new anchor-free decoder) next to yolov5s/m/l,
all checked on the reference images; systemd service.

**2026-07-19: dashboard UI restructured.** Video and ROI always visible,
ROI event feedback, inference stride and JPEG quality as documented fps
levers, class filter, detection on/off, power mode and fan control via jtop,
and control of the separate automagic-fan service found fighting jtop.

## cv_training (training on the PC)

**2026-07-19: NNI multi-architecture training pipeline.** Search over
architecture (yolov5n/s, yolov7-tiny, yolov8n) x dataset source (Open Images
V7, Sama-COCO, both) x hyperparameters; Stage A explores, Stage B replays a
promising trial through export and Nano deploy. Environment fixes: a filelock
version that deadlocked nnictl, a missing mongod for fiftyone-db on a recent
Ubuntu, yolov7 vs current PyTorch/NumPy, a label-cache collision between the
three families.

**2026-07-19: text menu** wrapping setup, smoke test, Stage A/B and deploy;
port validation (a port below 1024 made nnictl fail with an opaque EACCES).

## 2026-10-31: unification (this repository)

The two repositories, copies on the Jetson and on the PC were inventoried
(checksums), merged and restructured into `camera/`, `nano/`, `training/`,
`host/`, translated to English, and verified again on the hardware.
Highlights:

- Offline, fully pinned install on the Jetson (vendored wheelhouse, including
  pre-built pycuda); large files kept as verified chunks (`tools/largefiles.py`).
- The Jetson code became the `jetson_cv` package with a model registry and the
  [model contract](model-contract.md); fixed: YOLOv7 metadata no decoder
  handled, `trt_detect` decoding YOLOv8 with the YOLOv5 decoder, stretched
  16:9 dataset captures from a 4:3 sensor, an ignored dashboard setting,
  journal spam, a lost ONNX export.
- Training: one `cvtrain` package behind both the quickstart and NNI, portable
  setup (PyTorch build chosen for the local GPU), every result packaged and
  checked against the Nano's decoder before deploy; fixed: yolov8 reporting
  its last instead of best epoch, yolov7 failing under any path with a capital
  letter, CUDA-only device selection.
