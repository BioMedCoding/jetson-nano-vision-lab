# Benchmarks

Measured on 2026-10-31 on one Jetson Nano Developer Kit 4 GB, JetPack 4.6.1
(TensorRT 8.2.1), power mode MAXN, with the code in this repository.

## Camera

| Pipeline | fps |
|---|---|
| raw V4L2 capture + GRBG demosaic, 640x480 (`apps/camera_test.py`) | **62.5** |

## Live detection (camera -> inference -> boxes -> MJPEG stream)

Measured in the dashboard (`apps/dashboard.py`, one model after another via
its API), 25 s per model, a few objects in view:

| Model | Input | Live fps |
|---|---|---|
| yolov5n-416 | 416 | **30.1** |
| yolov7-tiny-416 | 416 | 23.2 |
| yolov8n-416 | 416 | 22.3 |
| yolov5s-416 | 416 | 19.7 |
| yolov5m-416 | 416 | 10.1 |
| yolov5s-640 | 640 | 9.7 |
| yolov5l-416 | 416 | 5.9 |

`apps/trt_detect.py` (minimal pipeline): yolov5n 30.6 fps, yolov8n 22.2 fps.
`apps/live_detect_torch.py` (PyTorch FP16 instead of TensorRT): yolov5n at 416,
16.4 fps.

## Model time only (`apps/benchmark.py --models all --sizes 416 --iters 50`)

Synthetic 640x480 frame, no camera, nothing else running on the GPU.

| Backend | Model | Input | ms / frame | fps |
|---|---|---|---|---|
| TensorRT FP16 | yolov5n-416 | 416 | 24.3 | 41.1 |
| TensorRT FP16 | yolov8n-416 | 416 | 27.6 | 36.2 |
| TensorRT FP16 | yolov7-tiny-416 | 416 | 33.9 | 29.5 |
| TensorRT FP16 | yolov5s-416 | 416 | 42.0 | 23.8 |
| TensorRT FP16 | yolov5m-416 | 416 | 89.1 | 11.2 |
| TensorRT FP16 | yolov5s-640 | 640 | 90.7 | 11.0 |
| TensorRT FP16 | yolov5l-416 | 416 | 160.5 | 6.2 |
| PyTorch FP16 | yolov5n | 416 | 48.3 | 20.7 |
| PyTorch FP32 | yolov5n | 416 | 59.8 | 16.7 |
| PyTorch FP16 | yolov5s | 416 | 87.0 | 11.5 |
| PyTorch FP32 | yolov5s | 416 | 93.0 | 10.8 |
| PyTorch FP16 | yolov5m | 416 | 193.4 | 5.2 |
| PyTorch FP32 | yolov5m | 416 | 209.8 | 4.8 |
| PyTorch FP16 | yolov5l | 416 | 354.5 | 2.8 |
| PyTorch FP32 | yolov5l | 416 | 387.8 | 2.6 |

Models fine-tuned on a PC with `training/` and deployed with
`host/deploy_model.sh` run at the speed of their architecture (TensorRT FP16,
416: yolov5n 24.0 ms, yolov8n 27.9 ms, yolov7-tiny 32.2 ms).

## Reading the numbers

- TensorRT FP16 is about 2x faster than PyTorch FP16 for the same model.
- The live pipeline costs about 10 ms per frame on top of the model (capture,
  demosaic, letterbox, decoding, drawing, JPEG encoding), which is why a
  24 ms model runs at ~30 fps live, not 41.
- Beyond yolov5s at 416, live fps drops quickly. For a smooth demo, stay with
  yolov5n / yolov8n / yolov7-tiny at 416, or raise "infer every N frames" in
  the dashboard.
