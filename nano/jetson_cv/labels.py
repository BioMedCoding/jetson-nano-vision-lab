"""Class-name lists. A model's model.json names its labels either as a
built-in set ("coco80") or as a file next to it (one class name per line)."""
import os

COCO80 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

BUILTIN = {"coco80": COCO80}


def load_names(spec, base_dir=None):
    """spec: a built-in name ("coco80") or a path to a one-name-per-line file
    (relative paths resolve against base_dir, usually the model folder)."""
    if spec in BUILTIN:
        return list(BUILTIN[spec])
    path = spec if base_dir is None or os.path.isabs(spec) else os.path.join(base_dir, spec)
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def class_name(names, cls_id):
    return names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
