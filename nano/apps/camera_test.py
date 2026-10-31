#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test della camera CSI OV5647: verifica pipeline, misura FPS reali, snapshot.

Non richiede torch: e' il primo script da lanciare per validare camera+driver.

Uso:
    python camera_test.py                     # finestra o stream MJPEG (auto)
    python camera_test.py --headless          # forza lo stream MJPEG
    python camera_test.py --snapshot-every 5  # salva un jpg ogni 5 s
Nella finestra: 's' salva uno snapshot, 'q'/ESC esce.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.camera import open_csi_camera  # noqa: E402
from common.viewer import Viewer           # noqa: E402
from common.yolo_utils import draw_fps     # noqa: E402

import cv2  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0,
                    help="flip-method di nvvidconv (2 = capovolgi 180 gradi)")
    ap.add_argument("--headless", action="store_true",
                    help="forza lo stream MJPEG anche con display")
    ap.add_argument("--snapshot-every", type=float, default=0,
                    help="salva uno snapshot ogni N secondi (0 = disattivato)")
    ap.add_argument("--outdir", default="snapshots")
    args = ap.parse_args()

    cap = open_csi_camera(args.width, args.height, args.fps, args.flip)
    viewer = Viewer("OV5647 camera test", headless=args.headless or None)
    os.makedirs(args.outdir, exist_ok=True)

    n, t0 = 0, time.time()
    fps, last_snap = 0.0, time.time()
    print("Camera aperta ({}x{}@{}). 's' = snapshot, 'q' = esci.".format(
        args.width, args.height, args.fps))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame non ricevuto: pipeline interrotta.")
                break
            n += 1
            if n % 30 == 0:
                now = time.time()
                fps = 30.0 / (now - t0)
                t0 = now
                if viewer.headless:
                    print("FPS reali: {:.1f}".format(fps))

            snap = False
            if args.snapshot_every and time.time() - last_snap > args.snapshot_every:
                snap = True
            draw_fps(frame, fps)
            if not viewer.headless:
                cv2.imshow(viewer.title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    snap = True
            else:
                viewer.show(frame)

            if snap:
                path = os.path.join(
                    args.outdir, "snap_{}.jpg".format(time.strftime("%Y%m%d_%H%M%S")))
                cv2.imwrite(path, frame)
                last_snap = time.time()
                print("Snapshot salvato: {}".format(path))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viewer.close()


if __name__ == "__main__":
    main()
