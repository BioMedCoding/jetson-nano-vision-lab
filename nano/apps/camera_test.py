#!/usr/bin/env python3
"""OV5647 camera test: checks capture, measures real fps, saves snapshots.

No torch, no TensorRT: the first thing to run after booting the OV5647
kernel. Expect ~60 fps at the sensor's native 640x480.

Usage:
    python3 apps/camera_test.py                      # window, or stream if no display
    python3 apps/camera_test.py --headless           # force the browser stream
    python3 apps/camera_test.py --snapshot-every 5   # save a jpg every 5 s
In the window: 's' saves a snapshot, 'q'/ESC quits.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jetson_cv import runtime  # noqa: E402
runtime.ensure()

import argparse  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402

from jetson_cv.camera import SENSOR_H, SENSOR_W, open_camera  # noqa: E402
from jetson_cv.display import Viewer  # noqa: E402
from jetson_cv.drawing import draw_fps  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flip", type=int, default=0,
                    help="0 none, 1 90 ccw, 2 180, 3 90 cw")
    ap.add_argument("--headless", action="store_true",
                    help="force the MJPEG stream even with a display")
    ap.add_argument("--snapshot-every", type=float, default=0,
                    help="save a snapshot every N seconds (0 = off)")
    ap.add_argument("--outdir", default="snapshots")
    args = ap.parse_args()

    cap = open_camera(flip=args.flip)
    viewer = Viewer("OV5647 camera test", headless=args.headless or None)
    os.makedirs(args.outdir, exist_ok=True)

    n, t0 = 0, time.time()
    fps, last_snap = 0.0, time.time()
    print("Camera open ({}x{}). 's' = snapshot, 'q' = quit.".format(SENSOR_W, SENSOR_H))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received: capture stopped.")
                break
            n += 1
            if n % 30 == 0:
                now = time.time()
                fps = 30.0 / (now - t0)
                t0 = now
                if viewer.headless:
                    print("Real fps: {:.1f}".format(fps))

            snap = bool(args.snapshot_every) and time.time() - last_snap > args.snapshot_every
            raw = frame.copy()  # snapshots without the fps overlay
            key = viewer.poll_key(draw_fps(frame, fps))
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                snap = True
            if snap:
                path = os.path.join(args.outdir, "snap_{}.jpg".format(
                    time.strftime("%Y%m%d_%H%M%S")))
                cv2.imwrite(path, raw)
                last_snap = time.time()
                print("Snapshot saved: {}".format(path))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viewer.close()


if __name__ == "__main__":
    main()
