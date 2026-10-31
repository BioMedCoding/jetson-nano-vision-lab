"""Map the generic NNI search space onto each trainer's native arguments,
handling the differences between the three architecture families (all
verified with real runs):

- yolov5 / yolov7 take augmentation and loss hyperparameters only through
  a --hyp file; yolov8 (the ultralytics package) takes them as keyword
  arguments of model.train().
- yolov8 has no objectness (anchor-free head): no obj / obj_pw / iou_t /
  anchor_t / fl_gamma. Its `cls_pw` exists with a DIFFERENT meaning
  (class-imbalance weighting, not the BCE positive weight), so it is
  excluded explicitly; passing it by name coincidence would silently change
  training.
- yolov8's default optimizer="auto" IGNORES explicit lr0 / momentum, so
  optimizer="SGD" is forced to make the search over them meaningful.
- box / cls loss gains have different numeric scales (yolov5/yolov7 about
  0.05 / 0.5, yolov8 about 7.5 / 0.5): the search space looks for a
  multiplier relative to each architecture's default, not an absolute value.
- freeze_backbone is a boolean in the search space, mapped to an
  architecture-specific layer depth ("10" does not mean the same thing in
  all three).
"""
import os

import yaml

from cvtrain.paths import WEIGHTS_DIR, YOLOV5_DIR, YOLOV7_DIR

ARCHS = ("yolov5n", "yolov5s", "yolov7-tiny", "yolov8n")

# Pretrained COCO starting points, vendored in training/weights/.
WEIGHTS = {arch: WEIGHTS_DIR / (arch + ".pt") for arch in ARCHS}

# Output decoder family on the Nano (nano/jetson_cv/inference/decoders.py).
DECODER = {"yolov5n": "yolo_v5", "yolov5s": "yolo_v5",
           "yolov7-tiny": "yolo_v5", "yolov8n": "yolo_v8"}

# Default loss gains (from yolov5 data/hyps/hyp.scratch-low.yaml, yolov7
# data/hyp.scratch.tiny.yaml, and ultralytics' printed defaults).
ARCH_DEFAULT_BOX_GAIN = {"yolov5n": 0.05, "yolov5s": 0.05, "yolov7-tiny": 0.05, "yolov8n": 7.5}
ARCH_DEFAULT_CLS_GAIN = {"yolov5n": 0.5, "yolov5s": 0.5, "yolov7-tiny": 0.5, "yolov8n": 0.5}

# Backbone depth frozen when freeze_backbone=True. yolov5n/s: 10 is the
# documented convention; yolov8n: 10 is ultralytics' convention for 'n'
# models. yolov7-tiny: 25 is an ESTIMATE (upstream's "backbone of
# yolov7=50" refers to full yolov7, not tiny) -- tune it from results.
ARCH_FREEZE_BACKBONE = {"yolov5n": 10, "yolov5s": 10, "yolov7-tiny": 25, "yolov8n": 10}

# Batch size per (architecture, image size), tuned for a 12 GB GPU and
# deliberately conservative. On a smaller GPU set CVTRAIN_BATCH (e.g. 16).
BATCH_SIZE_TABLE = {
    ("yolov5n", 416): 64, ("yolov5n", 512): 48, ("yolov5n", 640): 32,
    ("yolov5s", 416): 48, ("yolov5s", 512): 32, ("yolov5s", 640): 24,
    ("yolov7-tiny", 416): 48, ("yolov7-tiny", 512): 32, ("yolov7-tiny", 640): 24,
    ("yolov8n", 416): 64, ("yolov8n", 512): 48, ("yolov8n", 640): 32,
}

# Searched keys that yolov8 does not accept, or accepts with another meaning.
_YOLOV8_EXCLUDED_HYP_KEYS = {"obj", "obj_pw", "cls_pw", "iou_t", "anchor_t", "fl_gamma"}

_SHARED_SEARCHED_KEYS = (
    "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
)


def resolve_batch_size(arch, img_size):
    forced = os.environ.get("CVTRAIN_BATCH")
    if forced:
        return int(forced)
    return BATCH_SIZE_TABLE.get((arch, img_size), 16)


def default_hyp(arch):
    """Each family's own default hyperparameters (used by the quickstart)."""
    if arch == "yolov8n":
        return {"optimizer": "SGD"}
    return _load_base_hyp(arch)


def to_trainer_kwargs(*, arch, params, data_yaml, run_dir, epochs, patience):
    """Keyword arguments for cvtrain.trainers.<family>.train(...)."""
    freeze_layers = ARCH_FREEZE_BACKBONE[arch] if params["freeze_backbone"] else 0
    box_gain = ARCH_DEFAULT_BOX_GAIN[arch] * params["box_gain_mult"]
    cls_gain = ARCH_DEFAULT_CLS_GAIN[arch] * params["cls_gain_mult"]
    if arch == "yolov8n":
        hyp = _build_yolov8_hyp(params, box_gain, cls_gain)
    else:
        hyp = _build_yolov5v7_hyp(arch, params, box_gain, cls_gain)
    return {
        "data_yaml": data_yaml,
        "weights": str(WEIGHTS[arch]),
        "img_size": params["img_size"],
        "epochs": epochs,
        "patience": patience,
        "batch": resolve_batch_size(arch, params["img_size"]),
        "hyp": hyp,
        "freeze_layers": freeze_layers,
        "run_dir": run_dir,
    }


def _load_base_hyp(arch):
    base_path = (YOLOV5_DIR / "data/hyps/hyp.scratch-low.yaml"
                 if arch in ("yolov5n", "yolov5s")
                 else YOLOV7_DIR / "data/hyp.scratch.tiny.yaml")
    with open(base_path) as f:
        return yaml.safe_load(f)


def _build_yolov5v7_hyp(arch, params, box_gain, cls_gain):
    hyp = _load_base_hyp(arch)
    for key in _SHARED_SEARCHED_KEYS:
        hyp[key] = params[key]
    hyp["box"] = box_gain
    hyp["cls"] = cls_gain
    return hyp


def _build_yolov8_hyp(params, box_gain, cls_gain):
    hyp = {key: params[key] for key in _SHARED_SEARCHED_KEYS}
    hyp["box"] = box_gain
    hyp["cls"] = cls_gain
    hyp["perspective"] = 0.0  # not searched, same as the yolov5/v7 default
    hyp["optimizer"] = "SGD"  # "auto" would ignore lr0/momentum, see above
    assert not (_YOLOV8_EXCLUDED_HYP_KEYS & hyp.keys())
    return hyp
