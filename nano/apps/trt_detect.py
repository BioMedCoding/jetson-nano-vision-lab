#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detection live con engine TensorRT FP16 - la via piu' veloce sulla Nano.

Rispetto a live_detect.py (PyTorch) e' tipicamente 3-4x piu' veloce e non
dipende da torch a runtime: serve solo per costruire l'engine (build_engine.sh).
Funziona anche con modelli custom addestrati sull'host (export ONNX -> engine).

Uso:
    python trt_detect.py --engine ../models/yolov5n_416_fp16.engine
"""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2  # noqa: E402

from common.camera import open_csi_camera                        # noqa: E402
from common.viewer import Viewer                                 # noqa: E402
from common.trt_runner import TrtRunner                          # noqa: E402
from common.yolo_utils import (COCO_NAMES, draw_detections,      # noqa: E402
                               draw_fps, postprocess, preprocess)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, help="file .engine da build_engine.sh")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--names", default=None,
                    help="file con un nome classe per riga (default: COCO)")
    args = ap.parse_args()

    names = COCO_NAMES
    if args.names:
        with open(args.names) as f:
            names = [l.strip() for l in f if l.strip()]

    print("Carico engine {}...".format(args.engine))
    runner = TrtRunner(args.engine)
    size = runner.size
    print("Engine pronto (input {}x{}).".format(size, size))

    cap = open_csi_camera(args.width, args.height, args.fps, args.flip)
    viewer = Viewer("Jetson Nano - YOLOv5 TensorRT", headless=args.headless or None)

    fps_ema, n = 0.0, 0
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Frame non ricevuto: pipeline interrotta.")
                break

            img, r, pad = preprocess(frame, size)
            t1 = time.time()
            pred = runner.infer(img)
            infer_ms = (time.time() - t1) * 1000
            dets = postprocess(pred, r, pad, frame.shape,
                               conf_thres=args.conf, iou_thres=args.iou)

            draw_detections(frame, dets, names)
            dt = time.time() - t0
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
            draw_fps(frame, fps_ema, "inf: {:.0f} ms".format(infer_ms))

            n += 1
            if viewer.headless and n % 30 == 0:
                print("FPS: {:.1f}  oggetti: {}".format(fps_ema, len(dets)))
            if not viewer.show(frame):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viewer.close()
        print("Chiuso. FPS medi: {:.1f}".format(fps_ema))


if __name__ == "__main__":
    main()
