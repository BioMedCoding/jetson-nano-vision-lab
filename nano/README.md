# nano/: the computer-vision playground on the Jetson

Everything that runs on the Jetson Nano once the OV5647 camera works
(`../camera/`): raw V4L2 capture, TensorRT inference for any model that
follows the [model contract](../docs/model-contract.md), a web dashboard,
benchmarks and dataset capture.

Measured on a Jetson Nano 4 GB, JetPack 4.6.1, MAXN: camera 62.5 fps, YOLOv5n
TensorRT FP16 at 416 about **30 fps live** (full numbers in
[docs/benchmarks.md](../docs/benchmarks.md)).

## Install

```bash
# from your PC (or git clone the repo on the Jetson)
host/sync_to_nano.sh
# on the Jetson
~/jetson-nano-ov5647-vision/nano/setup/setup_jetson.sh
source ~/jetson-nano-ov5647-vision/nano/.venv/bin/activate
cd ~/jetson-nano-ov5647-vision/nano
```

Offline except for a few apt packages: every Python wheel is in
`vendor/wheels/`. Details, the reasons behind the Python 3.6 venv, and
troubleshooting: [setup/SETUP_GUIDE.md](setup/SETUP_GUIDE.md).

## Apps

Every app works with a monitor attached (OpenCV window) or headless (MJPEG
stream at `http://<jetson>:8090/`, or always with `--headless`).

| App | What it does |
|---|---|
| `apps/camera_test.py` | camera only, no AI: real fps, snapshots. Run it first |
| `apps/trt_detect.py --model <id>` | the shortest complete pipeline: camera -> TensorRT -> boxes |
| `apps/dashboard.py` | web dashboard: live video, system stats, inference statistics, image filters, thresholds, hot model switching, tracker with ROI enter/exit counting |
| `apps/live_detect_torch.py` | the same with PyTorch instead of TensorRT, to see the difference (~16 vs ~30 fps) |
| `apps/benchmark.py` | model time only, PyTorch FP32/FP16 vs TensorRT |
| `apps/capture_dataset.py` | save frames (native 640x480) to build a training set |
| `apps/build_engine.py` | add a model (ONNX or yolov5 .pt) and build its TensorRT engine |
| `tests/test_models.py` | check every model on reference images, no camera needed |

From the PC, `host/run_demo.sh [test|trt|torch|dashboard]` starts an app on
the Jetson over SSH and opens the stream in your browser.

## Models

One folder per model in `models/`, each described by a `model.json`:

| Model | Input | Live fps | Decoder |
|---|---|---|---|
| `yolov5n-416` | 416 | 30 | `yolo_v5` (default) |
| `yolov7-tiny-416` | 416 | 23 | `yolo_v5` |
| `yolov8n-416` | 416 | 22 | `yolo_v8` |
| `yolov5s-416` | 416 | 20 | `yolo_v5` |
| `yolov5m-416` | 416 | 10 | `yolo_v5` |
| `yolov5s-640` | 640 | 10 | `yolo_v5` |
| `yolov5l-416` | 416 | 6 | `yolo_v5` |

All are COCO-pretrained (80 classes). The `.engine` files were built on this
exact setup (TensorRT 8.2.1, JetPack 4.6.1); on another JetPack rebuild them
with `python3 apps/build_engine.py <id>`.

Adding your own model:

```bash
# trained on your PC with training/ -> one command does copy + engine build + test
host/deploy_model.sh training/export/<name>
# or, on the Jetson, from any YOLO ONNX (decoder and input size read from the ONNX)
python3 apps/build_engine.py ~/best.onnx --id my-model --labels classes.txt
```

## The dashboard

`python3 apps/dashboard.py` (or install it as a service:
`setup/install_service.sh`). The page at `http://<jetson>:8090/` keeps the
video and the ROI zone always visible on the left, with tabs on the right:

- **System**: CPU/GPU/RAM/temperatures/fan/power mode from jetson-stats.
- **Statistics**: inference time and fps history (mean/median/p95/min/max/std),
  per-class counts (current frame, and unique tracked IDs), mean confidence,
  ROI totals and peak occupancy.
- **Image**: brightness, contrast and filters (grayscale, CLAHE, sharpen,
  blur, edges), applied to what you see AND to what the model sees, with a
  note on how each affects detection: an educational test bench.
- **Detection**: confidence/IoU thresholds, hot model switching, infer every
  N frames (the one real fps lever on this hardware), stream quality, class
  filter.
- **Power**: power mode (nvpmodel), fan profile and speed, the optional
  automagic-fan service. Disabled unless started with
  `--allow-system-control`.

The dashboard has **no authentication**: anyone who can reach port 8090 can
watch the camera. Run it on a network you trust, or `--bind 127.0.0.1` and
use an SSH tunnel.

## Code layout

```
jetson_cv/            the library the apps are built from
  camera.py           raw V4L2 capture of the OV5647 (+ reconnecting wrapper)
  display.py          window or MJPEG stream, shared frame store
  models.py           model registry over models/*/model.json
  inference/          preprocess (letterbox), TensorRT runner, decoders, Detector
  dashboard/          shared state, HTTP/JSON server, static web page
  system/             jetson-stats client, automagic-fan service control
  runtime.py          JetPack workarounds applied before numpy/cv2/torch load
  filters.py, tracker.py, drawing.py, labels.py
apps/                 the programs above
models/               one folder per model (model.json, model.onnx, model.fp16.engine)
setup/                setup_jetson.sh, requirements.lock, systemd unit, service installers
vendor/               wheels, yolov5 v6.0 (PyTorch path, .pt export), jetson-fan-ctl
tests/                test_models.py
```

**Adding a new kind of model** (another detector head, or another task):
write a decoder in `jetson_cv/inference/decoders.py` and register it, then
give the model a `model.json` naming it. Nothing else changes. For a task
other than detection (`"task"` in `model.json`), add a counterpart of
`inference/detector.py` and a drawing function.

## Camera facts that userspace must respect

- The sensor mode is fixed: 640x480 RAW10 at about 60 fps. Other sizes are a
  software resize.
- Bayer order on the wire is **GRBG** (`cv2.COLOR_BayerGR2BGR`), whatever the
  driver/device-tree label says (BGGR).
- `TEGRA_CAMERA_CID_VI_BYPASS_MODE` must be 0 (camera.py sets it).
- AEC/AGC run inside the sensor and need 1-3 s of streaming to converge.
- GStreamer / `nvarguscamerasrc` will **never** work with this sensor: Argus
  has a closed, fixed sensor whitelist.
