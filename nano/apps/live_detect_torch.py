#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detection live YOLOv5 (PyTorch) sulla camera CSI OV5647 - Jetson Nano.

Versione corretta e ottimizzata dello script originale:
  - gira nel venv Python 3.6 creato da setup/setup_jetson.sh (torch CUDA cp36)
  - half precision (FP16) di default sulla GPU
  - inferenza a risoluzione ridotta (--size, default 416) sul frame 640x480
  - appsink senza buffer: latenza costante anche se l'inferenza e' lenta
  - finestra X11 se c'e' un display, altrimenti stream MJPEG sul browser

Uso:
    python live_detect.py                          # yolov5n, 416, FP16
    python live_detect.py --size 320               # piu' veloce, meno preciso
    python live_detect.py --weights ../models/yolov5s.pt --size 640
Suggerimento: prima di misurare le prestazioni:
    sudo nvpmodel -m 0 && sudo jetson_clocks
"""
import os
# Workaround noto su Jetson Nano: senza questa variabile numpy/OpenBLAS
# puo' morire con "Illegal instruction (core dumped)".
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2                      # noqa: E402
import torch                    # noqa: E402

from common.camera import open_csi_camera   # noqa: E402
from common.viewer import Viewer            # noqa: E402
from common.yolo_utils import draw_fps      # noqa: E402


def load_model(yolo_dir, weights, device, half, conf):
    print("Carico YOLOv5 da {} (pesi: {})...".format(yolo_dir, weights))
    model = torch.hub.load(str(yolo_dir), "custom", path=str(weights),
                           source="local")
    model = model.to(device)
    if half and device.type == "cuda":
        model = model.half()
    model.conf = conf
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(BASE / "models" / "yolov5n.pt"))
    ap.add_argument("--yolo-dir", default=str(Path.home() / "yolov5"),
                    help="clone locale del repo ultralytics/yolov5 (v6.0)")
    ap.add_argument("--size", type=int, default=416,
                    help="lato dell'immagine di inferenza (320/416/640)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--no-half", action="store_true",
                    help="disabilita FP16 (usa FP32)")
    ap.add_argument("--headless", action="store_true",
                    help="forza lo stream MJPEG anche con display")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="esci dopo N frame (0 = infinito), utile per i test")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ATTENZIONE: CUDA non disponibile, uso la CPU (molto lento).")
        print("Se sei nel venv giusto qualcosa non va: vedi setup/SETUP_GUIDE.md")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = (not args.no_half) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    model = load_model(args.yolo_dir, args.weights, device, half, args.conf)

    # warmup: la prima inferenza compila i kernel cudnn ed e' lentissima
    print("Warmup...")
    import numpy as np
    for _ in range(2):
        model(np.zeros((args.height, args.width, 3), dtype=np.uint8),
              size=args.size)

    cap = open_csi_camera(args.width, args.height, args.fps, args.flip)
    viewer = Viewer("Jetson Nano - YOLOv5 live", headless=args.headless or None)
    print("Detection avviata ({}p, size={}, {}).".format(
        args.height, args.size, "FP16" if half else "FP32"))

    fps_ema, n = 0.0, 0
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Frame non ricevuto: pipeline interrotta.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb, size=args.size)
            annotated = cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR)

            dt = time.time() - t0
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
            infer_ms = sum(results.t) if hasattr(results, "t") else 0
            draw_fps(annotated, fps_ema, "inf: {:.0f} ms".format(infer_ms))

            n += 1
            if viewer.headless and n % 30 == 0:
                print("FPS: {:.1f}  oggetti: {}".format(
                    fps_ema, len(results.xyxy[0])))
            if not viewer.show(annotated):
                break
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viewer.close()
        print("Chiuso. Frame processati: {}, FPS medi: {:.1f}".format(n, fps_ema))


if __name__ == "__main__":
    main()
