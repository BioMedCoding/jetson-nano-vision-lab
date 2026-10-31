"""jetson_cv: camera capture, TensorRT inference and a web dashboard for a
Jetson Nano with an OV5647 (Raspberry Pi Camera v1) sensor.

Targets JetPack 4.6.1 / Python 3.6: no f-string-only syntax beyond 3.6, no
dataclasses, no walrus. Import order matters on this platform, so apps call
`jetson_cv.runtime.ensure()` before importing numpy/cv2/torch.
"""
import os

NANO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(NANO_DIR, "models")
VENDOR_DIR = os.path.join(NANO_DIR, "vendor")
YOLOV5_DIR = os.path.join(VENDOR_DIR, "yolov5-v6.0")

__version__ = "1.0.0"
