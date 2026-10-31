"""Which device to train on.

Default: the first CUDA GPU if there is one, otherwise the CPU (works, just
slowly). Override with the CVTRAIN_DEVICE environment variable using the
trainers' own syntax: "0", "1", "0,1", "cpu" (and "mps" for yolov8 on Apple
silicon; yolov5/yolov7 here do not support mps).
"""
import os


def train_device():
    forced = os.environ.get("CVTRAIN_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"
