#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cattura immagini dalla camera per costruire un dataset di training.

Le immagini raccolte qui si etichettano poi sull'host (labelImg, Roboflow,
CVAT...) e si usano per il fine-tuning di YOLOv5 sul PC con GPU
(vedi cv_training/README_TRAINING.md sull'host).

Uso:
    python capture_images.py --interval 2            # un frame ogni 2 s
    python capture_images.py --count 200             # si ferma a 200 immagini
Nella finestra: 'c' cattura manuale, 'q'/ESC esce.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.camera import open_csi_camera  # noqa: E402
from common.viewer import Viewer           # noqa: E402

import cv2  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="dataset/images")
    ap.add_argument("--prefix", default="img")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="secondi tra le catture automatiche (0 = solo manuale)")
    ap.add_argument("--count", type=int, default=0,
                    help="numero massimo di immagini (0 = illimitato)")
    ap.add_argument("--width", type=int, default=1280,
                    help="risoluzione di cattura (piu' alta = dataset migliore)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cap = open_csi_camera(args.width, args.height, args.fps, args.flip)
    viewer = Viewer("Cattura dataset", headless=args.headless or None)
    if viewer.headless and args.interval <= 0:
        args.interval = 2.0
        print("Headless: cattura automatica ogni 2 s.")

    saved, last = 0, 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            grab = args.interval > 0 and (time.time() - last) >= args.interval

            preview = frame.copy()
            cv2.putText(preview, "salvate: {}".format(saved), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if not viewer.headless:
                cv2.imshow(viewer.title, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    grab = True
            else:
                viewer.show(preview)

            if grab:
                name = "{}_{}_{:04d}.jpg".format(
                    args.prefix, time.strftime("%Y%m%d_%H%M%S"), saved)
                cv2.imwrite(os.path.join(args.outdir, name), frame)
                saved += 1
                last = time.time()
                print("[{}] {}".format(saved, name))
                if args.count and saved >= args.count:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viewer.close()

    print("\nSalvate {} immagini in {}/".format(saved, args.outdir))
    print("Prossimi passi: copia la cartella sull'host, etichetta le immagini")
    print("e segui cv_training/README_TRAINING.md per il fine-tuning.")


if __name__ == "__main__":
    main()
