"""One trainer per architecture family, all with the signature in base.py."""
from cvtrain.trainers import yolov5, yolov7, yolov8

TRAINERS = {
    "yolov5n": yolov5,
    "yolov5s": yolov5,
    "yolov7-tiny": yolov7,
    "yolov8n": yolov8,
}
