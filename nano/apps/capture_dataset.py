#!/usr/bin/env python3
"""Capture camera frames to build a training dataset.

Images are saved at the sensor's native 640x480 (4:3), never upscaled or
stretched, so what you label is exactly what the model will see live. Label
them on a PC (labelImg, CVAT, Roboflow, ...) and fine-tune there, see
training/README.md.

Usage:
    python3 apps/capture_dataset.py --interval 2      # one frame every 2 s
    python3 apps/capture_dataset.py --count 200       # stop after 200 images
In the window: 'c' captures manually, 'q'/ESC quits.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jetson_cv import runtime  # noqa: E402
runtime.ensure()

import argparse  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402

from jetson_cv.camera import open_camera  # noqa: E402
from jetson_cv.display import Viewer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="dataset/images")
    ap.add_argument("--prefix", default="img")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between automatic captures (0 = manual only)")
    ap.add_argument("--count", type=int, default=0,
                    help="maximum number of images (0 = unlimited)")
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cap = open_camera(flip=args.flip)
    viewer = Viewer("Dataset capture", headless=args.headless or None)
    if viewer.headless and args.interval <= 0:
        args.interval = 2.0
        print("Headless: automatic capture every 2 s.")

    saved, last = 0, 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            grab = args.interval > 0 and (time.time() - last) >= args.interval
            preview = frame.copy()
            cv2.putText(preview, "saved: {}".format(saved), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            key = viewer.poll_key(preview)
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                grab = True
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
    print("\nSaved {} images in {}/".format(saved, args.outdir))
    print("Next: copy the folder to your PC, label the images, then follow")
    print("training/README.md to fine-tune a model.")


if __name__ == "__main__":
    main()
