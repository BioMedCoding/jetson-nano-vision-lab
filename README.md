# Jetson Nano Vision Lab

Run real-time object detection on an **NVIDIA Jetson Nano** with the cheap
**OV5647 camera (Raspberry Pi Camera v1 and its clones)**, a sensor NVIDIA
never supported on this board, and train your own models for it on a PC.

- **camera/**: a kernel driver and device tree that make the OV5647 work on the
  Nano (JetPack 4.6.1), with prebuilt binaries and the full debugging story.
  640x480 RAW10 at **62.5 fps**.
- **nano/**: a Python playground on the Jetson: raw V4L2 capture, TensorRT
  inference, a live web dashboard, benchmarks, 7 ready-to-run YOLO models.
  YOLOv5n at **~30 fps live** on a $99 board from 2019.
- **training/**: fine-tune YOLOv5 / YOLOv7-tiny / YOLOv8 on your own dataset on
  a PC with a GPU, and deploy the result to the Nano with one command.

```
 ┌──────────── PC (any NVIDIA GPU) ─────────────┐        ┌──────────── Jetson Nano ─────────────┐
 │ your dataset                                 │        │ OV5647 ─► camera/ kernel driver       │
 │   └─► training/quickstart.py                 │        │             └─► /dev/video0           │
 │         train ─► export ONNX ─► check ─►     │  ssh   │                    │                  │
 │         export/<model>/  model.onnx ─────────┼───────►│ nano/models/<model>/                  │
 │                          model.json          │ deploy │   build TensorRT engine (on the Nano) │
 │                          labels.txt          │        │   └─► nano/apps: detect, dashboard,   │
 └──────────────────────────────────────────────┘        │       benchmark, capture dataset      │
                                                         └───────────────────────────────────────┘
```

Everything the board needs is **in this repository**: the patched kernel
source, the CUDA PyTorch wheel for Python 3.6, every other Python wheel,
TensorRT engines, pretrained weights. Download links for a 2019 board's
software stack keep disappearing; this repo should still install the day they
are all gone (`tools/largefiles.py` keeps files over GitHub's 100 MB limit as
verified chunks).

## What you need

- Jetson Nano Developer Kit 4 GB (B01/B02 carrier) with **JetPack 4.6.1**
  (L4T R32.7.1) on the SD card.
- An OV5647 camera module with a 15-pin CSI cable, on the **CAM0** connector.
- For training: a Linux PC with an NVIDIA GPU (tested: RTX 5070) and
  [uv](https://docs.astral.sh/uv/). Optional if you only want to run the
  included models.

## Quick start

1. **Make the camera work** (once): copy the three files in
   `camera/prebuilt/` to the Jetson and add a boot entry
   ([camera/prebuilt/README.md](camera/prebuilt/README.md)), or run
   `camera/scripts/deploy.sh` on the SD card. Reboot into "OV5647".
2. **Connect PC and Jetson** ([docs/networking.md](docs/networking.md)), then
   from the PC: `host/sync_to_nano.sh`.
3. **Install the playground** on the Jetson (offline):
   `~/jetson-nano-vision-lab/nano/setup/setup_jetson.sh`
4. **See it run**: `host/run_demo.sh dashboard` opens the live dashboard in
   your browser.
5. **Train your own model** (optional): `training/setup/setup_host.sh`, then
   `python training/quickstart.py --data your/data.yaml --deploy`.

## Where to read next

| If you want to... | Read |
|---|---|
| bring up an unsupported camera sensor on a Jetson | [camera/docs/CAMERA_DEBUG.md](camera/docs/CAMERA_DEBUG.md): five stacked bugs behind one silent timeout, and how CSI status registers found them without rebuilding |
| understand the Jetson setup and its traps | [nano/setup/SETUP_GUIDE.md](nano/setup/SETUP_GUIDE.md) |
| use the apps and the dashboard | [nano/README.md](nano/README.md) |
| train and deploy your own model | [training/README.md](training/README.md), [training/docs/TRAINING_GUIDE.md](training/docs/TRAINING_GUIDE.md) |
| add another model family or task | [docs/model-contract.md](docs/model-contract.md) |
| see the numbers | [docs/benchmarks.md](docs/benchmarks.md) |
| contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Repository layout

```
camera/      OV5647 kernel support: patched L4T kernel source, patches, prebuilt binaries, docs
nano/        runs on the Jetson: jetson_cv library, apps, models, offline setup
training/    runs on a PC: quickstart, NNI search, trainers, vendored yolov5/yolov7, weights
host/        PC-side helpers: sync to the Jetson, deploy a model, start a demo
tools/       largefiles.py (chunked storage of files over 100 MB)
docs/        model contract, benchmarks, networking, history
```

## License

GPL-3.0 (see `LICENSE`) for this repository's own code and docs, with these
exceptions, each carrying its own license file:

- `camera/` is GPL-2.0: it is the Linux kernel and NVIDIA's L4T kernel source.
- Vendored third-party code keeps its license: `nano/vendor/yolov5-v6.0`
  (GPL-3.0), `training/vendor/yolov5` (AGPL-3.0), `training/vendor/yolov7`
  (GPL-3.0), `nano/vendor/jetson-fan-ctl` (MIT). Wheels in `nano/vendor/wheels`
  are redistributed under their own licenses.
- Model weights derive from the YOLOv5 / YOLOv7 / YOLOv8 releases (trained on
  COCO) and follow those projects' licenses.

OV5647 is a trademark of OmniVision; Jetson, TensorRT and JetPack are
trademarks of NVIDIA. This project is not affiliated with either.
