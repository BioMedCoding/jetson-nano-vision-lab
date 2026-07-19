# cv/ — YOLOv5 inference pipeline for the OV5647 camera

Everything user-space: capture, live detection, TensorRT, benchmarks, dataset
collection. Self-contained on the same principle as `../source/`: the torch /
torchvision cp36 wheels and the yolov5 v6.0 snapshot live **in this repo**,
because every external source for them (NVIDIA forum links, font CDNs, old
release tags) has already proven link-rot-prone.

Verified end to end on 2026-07-18: camera 62.5 fps, YOLOv5n TensorRT FP16
(416) at **30.6 fps live** on a Jetson Nano 4GB.

## Layout

- `cv_projects/` — goes to `~/cv_projects` on the Jetson.
  - `common/camera.py` — **V4L2-raw capture** (direct ioctls, mmap, GRBG
    demosaic). Exposes `open_csi_camera()` with a `cv2.VideoCapture`-like
    object. There is deliberately **no** GStreamer/`nvarguscamerasrc` path:
    Argus has a hardcoded sensor whitelist and will never accept this driver.
  - `01_camera_test/` … `05_dataset_capture/` — runnable scripts, all
    `--headless`-capable (MJPEG stream on `http://<jetson>:8090/`).
  - `models/` — yolov5n/s `.pt` weights + exported `.onnx` + the TensorRT
    engine built and benchmarked on the Nano (engines are
    hardware/TRT-version specific: rebuild with
    `03_tensorrt/build_engine.sh` if JetPack differs).
  - `setup/setup_jetson.sh` — **the** installer (see below).
  - `setup/wheels/` — torchvision wheel + torch wheel split into <100 MB
    chunks (`torch-split/`, GitHub file limit); `setup_jetson.sh` reassembles
    and sha256-verifies it automatically.
- `yolov5/` — yolov5 v6.0 snapshot, goes to `~/yolov5` on the Jetson. One
  local fix: `utils/plots.py` imports seaborn defensively (the system
  seaborn/scipy pair on JetPack 4 is incompatible with venv numpy 1.19).
- `avvia_camera.sh` — run **from the host**: starts capture/detection on the
  Jetson over SSH and opens the MJPEG stream in your browser.

## Install on the Jetson

```bash
# from this repo, on the host:
scp -r cv/cv_projects matteo@<jetson>:~/cv_projects
scp -r cv/yolov5      matteo@<jetson>:~/yolov5

# on the Jetson:
cd ~/cv_projects/setup
./setup_jetson.sh --trt      # ~20 min (pycuda compiles); --no-swap to skip swap
```

The setup script handles every trap we actually hit: venv **must** be Python
3.6 `--system-site-packages` (system cv2/tensorrt/CUDA-torch only exist for
3.6 on JetPack 4), `OPENBLAS_CORETYPE=ARMV8` (numpy "Illegal instruction"),
`LD_PRELOAD=libgomp` (cv2-after-torch "static TLS block" error), pycuda with
`--no-build-isolation` (pip would try to compile numpy 1.12 → xlocale.h
failure), and a local `Arial.ttf` so yolov5 never hits its dead download URL.

## Run

```bash
./avvia_camera.sh          # TensorRT detection, ~30 fps
./avvia_camera.sh test     # camera only, ~62 fps
./avvia_camera.sh torch    # PyTorch detection, ~18 fps (for comparison)
./avvia_camera.sh demo     # experimental dashboard (system/stats/live controls/tracker+zones)
```

Or on the Jetson directly (venv already exports all needed env vars):

```bash
source ~/cv_projects/venv/bin/activate
python3 ~/cv_projects/03_tensorrt/trt_detect.py \
    --engine ~/cv_projects/models/yolov5n_416_fp16.engine --headless
```

## Experimental dashboard (`demo_server.py`)

Same capture→TRT→box pipeline as `trt_detect.py`, evolved into a full test
platform at `http://<jetson>:8090/`: tabbed page with live system stats (CPU/
GPU/RAM/temps/fan via **jetson-stats**, already installed on this system —
no new dependency), inference-time history (mean/p50/p95), adjustable
brightness/contrast/filters applied to both display and inference, live
conf/IoU thresholds, hot model switching from a dropdown, and a lightweight
centroid-tracker (no scipy/Kalman) driving per-class unique-ID counts, a
draggable ROI zone with enter/exit counting, and trajectory drawing.
Autostart at boot: `cv_projects/setup/install_demo_service.sh` (systemd,
run by the user — needs sudo). `trt_detect.py` and `common/viewer.py` are
untouched by this: the dashboard lives in new modules
(`common/state.py`, `filters.py`, `tracker.py`, `jtop_client.py`,
`dashboard_server.py`, `static/`) so the four other scripts keep working
exactly as before.

## Camera facts that userspace must respect

- Sensor mode is fixed: 640x480 RAW10 @ ~60 fps. Other sizes = software
  resize in `camera.py`.
- Bayer order on the wire is **GRBG** → `cv2.COLOR_BayerGR2BGR` (labels in
  driver/DT say BGGR; raw data passes through unchanged, the label is
  cosmetic).
- `TEGRA_CAMERA_CID_VI_BYPASS_MODE` must be 0 (camera.py sets it).
- AEC/AGC run inside the sensor and take ~2-3 s of continuous streaming to
  converge; at 60 fps max integration is ~16 ms, so dim scenes stay dim
  (raise the gain ceiling via sensor regs 0x3a18/0x3a19 if that ever bites).

Benchmarks measured on this exact setup (`04_benchmark/benchmark.py`):

| Backend | size | ms/frame | FPS |
|---|---|---|---|
| PyTorch FP16 | 416 | 43.9 | 22.8 |
| TensorRT FP16 | 416 | 23.6 | 42.3 |

(Model-only time; the live pipeline lands at ~30 fps TRT / ~18 fps torch.)

## Models tested beyond yolov5n (TensorRT FP16, live pipeline fps)

Widened the model set within this Nano's fixed constraints (TensorRT 8.2.1,
JetPack 4.6.1, Python 3.6 aarch64 — export always happens on the host, only
`trtexec` needs to run on the Jetson itself):

| Model | Size | Live fps | Decoder |
|---|---|---|---|
| yolov5n | 416 | 30.6 | v5 (unchanged) |
| yolov7-tiny | 416 | 23.4 | v5-compatible output shape, no new code |
| yolov5s | 416 | 19.9 | v5 |
| yolov8n | 416 | ~21-26 | v8 (new `postprocess_v8`, anchor-free/no objectness) |
| yolov5s | 640 | 10.0 | v5 |
| yolov5m | 416 | 10.3 | v5 |
| yolov5l | 416 | 6.0 | v5 (not vendored: engine alone is ~183MB) |

All accuracy-checked against the standard `bus.jpg`/`zidane.jpg` yolov5
sample images (not just "doesn't crash") before being considered working.
YOLOv8's ONNX output is `(1, 84, N)` — 4 box coords + 80 class scores, no
separate objectness column, transposed vs. v5/v7's `(1, N, 85)` — hence the
dedicated decoder. NanoDet/YOLOX were considered and dropped: no verified
deploy path on this exact TensorRT/JetPack combination was found.
