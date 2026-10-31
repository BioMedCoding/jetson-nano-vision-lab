#!/usr/bin/env python3
"""Minimal live detection with a TensorRT engine: the fastest path on the Nano.

The shortest complete example of the pipeline (camera -> Detector -> boxes ->
window or browser stream). No torch at runtime. Works with any model folder
under nano/models, including custom models trained on a PC (ONNX -> engine).

Usage:
    python3 apps/trt_detect.py                       # yolov5n-416
    python3 apps/trt_detect.py --model yolov8n-416 --headless
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jetson_cv import runtime  # noqa: E402
runtime.ensure()

import argparse  # noqa: E402
import time  # noqa: E402

from jetson_cv.camera import open_camera  # noqa: E402
from jetson_cv.display import Viewer  # noqa: E402
from jetson_cv.drawing import draw_detections, draw_fps  # noqa: E402
from jetson_cv.inference.detector import Detector  # noqa: E402
from jetson_cv.models import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov5n-416",
                    help="model id under nano/models, or a model folder")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    info = resolve(args.model)
    print("Loading {} (decoder {})...".format(info.id, info.decoder))
    detector = Detector(info)
    print("Ready (input {0}x{0}).".format(detector.input_size))

    cap = open_camera(flip=args.flip)
    viewer = Viewer("Jetson Nano - {}".format(info.label), headless=args.headless or None)

    fps_ema, n = 0.0, 0
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("No frame received: capture stopped.")
                break
            dets, infer_ms = detector.detect(frame, args.conf, args.iou)
            draw_detections(frame, dets, detector.names)
            dt = time.time() - t0
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
            draw_fps(frame, fps_ema, "inf: {:.0f} ms".format(infer_ms))

            n += 1
            if viewer.headless and n % 30 == 0:
                print("FPS: {:.1f}  objects: {}".format(fps_ema, len(dets)))
            if not viewer.show(frame):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viewer.close()
        print("Stopped. Average fps: {:.1f}".format(fps_ema))


if __name__ == "__main__":
    main()
