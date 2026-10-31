#!/usr/bin/env python3
"""Live YOLOv5 detection with PyTorch (no TensorRT), for comparison.

Uses the vendored yolov5 v6.0 code and the model folder's .pt weights,
FP16 on the GPU by default. Roughly 18 fps for yolov5n at 416 vs ~30 fps
with TensorRT (apps/trt_detect.py): this script exists to show that gap.

Only YOLOv5 v6.0 checkpoints load here: the Nano's torch 1.10 / Python 3.6
cannot read checkpoints saved by modern torch. Custom models trained on a PC
go through ONNX -> TensorRT instead.

Usage:
    python3 apps/live_detect_torch.py                    # yolov5n, 416, FP16
    python3 apps/live_detect_torch.py --size 320         # faster, less accurate
    python3 apps/live_detect_torch.py --model yolov5s-416 --size 640
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jetson_cv import runtime  # noqa: E402
runtime.ensure()

import argparse  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from jetson_cv import YOLOV5_DIR  # noqa: E402
from jetson_cv.camera import SENSOR_H, SENSOR_W, open_camera  # noqa: E402
from jetson_cv.display import Viewer  # noqa: E402
from jetson_cv.drawing import draw_fps  # noqa: E402
from jetson_cv.models import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov5n-416",
                    help="model id whose folder holds a yolov5 v6.0 .pt")
    ap.add_argument("--size", type=int, default=416,
                    help="inference size (320/416/640)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--no-half", action="store_true", help="use FP32 instead of FP16")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = never), handy for tests")
    args = ap.parse_args()

    info = resolve(args.model)
    weights = info.weights_path
    if not weights or not os.path.exists(weights):
        sys.exit("Model {} has no .pt weights in its folder.".format(info.id))

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (very slow).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = (not args.no_half) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    print("Loading YOLOv5 from {} (weights {})...".format(YOLOV5_DIR, weights))
    model = torch.hub.load(YOLOV5_DIR, "custom", path=weights, source="local").to(device)
    if half:
        model = model.half()
    model.conf = args.conf

    print("Warm-up (the first inferences compile cuDNN kernels)...")
    for _ in range(2):
        model(np.zeros((SENSOR_H, SENSOR_W, 3), dtype=np.uint8), size=args.size)

    cap = open_camera(flip=args.flip)
    viewer = Viewer("Jetson Nano - YOLOv5 PyTorch", headless=args.headless or None)
    print("Running (size={}, {}).".format(args.size, "FP16" if half else "FP32"))

    fps_ema, n = 0.0, 0
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("No frame received: capture stopped.")
                break
            results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size=args.size)
            annotated = cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR)
            dt = time.time() - t0
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
            infer_ms = sum(results.t) if hasattr(results, "t") else 0
            draw_fps(annotated, fps_ema, "inf: {:.0f} ms".format(infer_ms))

            n += 1
            if viewer.headless and n % 30 == 0:
                print("FPS: {:.1f}  objects: {}".format(fps_ema, len(results.xyxy[0])))
            if not viewer.show(annotated):
                break
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viewer.close()
        print("Stopped. Frames: {}, average fps: {:.1f}".format(n, fps_ema))


if __name__ == "__main__":
    main()
