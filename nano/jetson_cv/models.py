"""Model registry: one folder per model under nano/models/, described by a
model.json (the "model contract", see docs/model-contract.md):

    models/yolov5n-416/
        model.json          what the files are and how to decode the output
        model.onnx          network exported for this input size (portable)
        model.fp16.engine   TensorRT engine built from model.onnx ON the Nano
        yolov5n.pt          optional: original weights (for the PyTorch path)
        labels.txt          optional: class names when not a built-in set

model.json fields:
    label        human-readable name shown in the dashboard
    task         "detect" (the only task implemented so far)
    decoder      output decoder family, see inference/decoders.py
    input_size   square network input side, in pixels
    labels       "coco80" or a file name in the folder
    onnx, engine, weights   file names in the folder (engine/weights optional)
    source       free text: where the weights come from, how it was exported
"""
import json
import os

from . import MODELS_DIR
from .inference.decoders import canonical_name
from .labels import load_names

MODEL_FILE = "model.json"


class ModelInfo(object):
    def __init__(self, model_dir, meta):
        self.dir = os.path.abspath(model_dir)
        self.id = os.path.basename(self.dir.rstrip("/"))
        self.meta = meta
        self.label = meta.get("label", self.id)
        self.task = meta.get("task", "detect")
        self.decoder = canonical_name(meta["decoder"])
        self.input_size = int(meta["input_size"])
        self.labels_spec = meta.get("labels", "coco80")

    @classmethod
    def from_dir(cls, model_dir):
        with open(os.path.join(model_dir, MODEL_FILE)) as f:
            return cls(model_dir, json.load(f))

    def path(self, key, default=None):
        name = self.meta.get(key, default)
        return os.path.join(self.dir, name) if name else None

    @property
    def engine_path(self):
        return self.path("engine", "model.fp16.engine")

    @property
    def onnx_path(self):
        return self.path("onnx", "model.onnx")

    @property
    def weights_path(self):
        return self.path("weights")

    @property
    def has_engine(self):
        return os.path.isfile(self.engine_path)

    def names(self):
        return load_names(self.labels_spec, base_dir=self.dir)

    def to_json(self, current_id=None):
        return {"id": self.id, "label": self.label, "task": self.task,
                "decoder": self.decoder, "input_size": self.input_size,
                "has_engine": self.has_engine, "current": self.id == current_id}


def list_models(models_dir=MODELS_DIR):
    models = []
    for name in sorted(os.listdir(models_dir)):
        d = os.path.join(models_dir, name)
        if os.path.isfile(os.path.join(d, MODEL_FILE)):
            try:
                models.append(ModelInfo.from_dir(d))
            except (ValueError, KeyError) as e:
                print("[models] skipping {}: invalid {} ({})".format(name, MODEL_FILE, e))
    return models


def resolve(spec, models_dir=MODELS_DIR):
    """spec: a model id ("yolov5n-416"), a model folder, or a path to a file
    inside one (e.g. the .engine). Returns a ModelInfo."""
    candidates = [spec, os.path.join(models_dir, spec)]
    if os.path.isfile(spec):
        candidates.insert(0, os.path.dirname(os.path.abspath(spec)))
    for c in candidates:
        if os.path.isfile(os.path.join(c, MODEL_FILE)):
            return ModelInfo.from_dir(c)
    known = ", ".join(m.id for m in list_models(models_dir))
    raise ValueError("model '{}' not found (known: {})".format(spec, known))
