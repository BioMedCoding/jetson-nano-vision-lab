#!/usr/bin/env python3
"""Smoke test for every model in nano/models: load its engine with the
decoder named in model.json and check the detections on yolov5's reference
images. Runs ON the Nano, no camera needed (the dashboard can stay running).

    python3 tests/test_models.py            # every model with an engine
    python3 tests/test_models.py yolov8n-416

A COCO model must find the bus and at least 2 people in bus.jpg and at least
2 people in zidane.jpg. Models with other labels are only checked to load
and run.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jetson_cv import runtime  # noqa: E402
runtime.ensure()

import cv2  # noqa: E402

from jetson_cv import YOLOV5_DIR  # noqa: E402
from jetson_cv.inference.detector import Detector  # noqa: E402
from jetson_cv.labels import class_name  # noqa: E402
from jetson_cv.models import list_models, resolve  # noqa: E402

IMAGES = os.path.join(YOLOV5_DIR, "data", "images")
EXPECT = {"bus": {"bus": 1, "person": 2}, "zidane": {"person": 2}}


def counts(dets, names):
    c = {}
    for d in dets:
        n = class_name(names, d[5])
        c[n] = c.get(n, 0) + 1
    return c


def main():
    infos = [resolve(m) for m in sys.argv[1:]] or [m for m in list_models() if m.has_engine]
    images = {k: cv2.imread(os.path.join(IMAGES, k + ".jpg")) for k in EXPECT}
    failures = 0
    for info in infos:
        det = Detector(info)
        coco = info.labels_spec == "coco80"
        line, ok = [], True
        for img_name, img in images.items():
            dets, ms = det.detect(img, conf=0.4, iou=0.45)
            c = counts(dets, det.names)
            line.append("{}={} ({:.0f} ms)".format(img_name, dict(sorted(c.items())), ms))
            if coco:
                ok &= all(c.get(k, 0) >= v for k, v in EXPECT[img_name].items())
        failures += not ok
        print("{} {:<16} {:<8} {}".format("PASS" if ok else "FAIL", info.id,
                                           info.decoder, "  ".join(line)), flush=True)
        del det
    print("\n{} model(s), {} failure(s)".format(len(infos), failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
