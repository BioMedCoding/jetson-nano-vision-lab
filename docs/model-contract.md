# The model contract

The one interface between `training/` (a PC) and `nano/` (the Jetson): a
**model folder**. Anything that produces a folder like this can run on the
Nano; anything on the Nano that runs a model reads only this.

```
<model-id>/                  e.g. yolov5n-416, fox-v1
├── model.json               required: how to run and decode the model
├── model.onnx               required to build the engine: the network
├── model.fp16.engine        built ON the Nano from model.onnx (never copied between devices)
├── labels.txt               optional: class names, one per line, in class-id order
└── <weights>.pt             optional: original PyTorch weights (yolov5 v6.0 only, for the PyTorch path)
```

## model.json

```json
{
  "label": "YOLOv5n 416",
  "task": "detect",
  "decoder": "yolo_v5",
  "input_size": 416,
  "labels": "coco80",
  "onnx": "model.onnx",
  "engine": "model.fp16.engine",
  "weights": "yolov5n.pt",
  "source": "ultralytics/yolov5 v6.0 release weights (COCO 2017). ONNX exported on ..."
}
```

| Field | Meaning |
|---|---|
| `label` | name shown in the dashboard |
| `task` | `"detect"`: the only task implemented so far |
| `decoder` | output decoder family, a key of `DECODERS` in `nano/jetson_cv/inference/decoders.py` |
| `input_size` | square network input side in pixels (must match the ONNX) |
| `labels` | `"coco80"` (built in) or a file name in the folder, e.g. `"labels.txt"` |
| `onnx`, `engine`, `weights` | file names in the folder (`engine` and `weights` optional) |
| `source` | free text: where the weights come from, how it was exported, metrics |

The folder name is the model id (`--model <id>` in every app).

## Decoder families

| `decoder` | ONNX output | Produced by |
|---|---|---|
| `yolo_v5` | `(1, N, 5 + classes)`: cx, cy, w, h, objectness, class scores | YOLOv5 (anchor-based, not the "u" variants), YOLOv7 exported with `--grid` |
| `yolo_v8` | `(1, 4 + classes, N)`: cx, cy, w, h, class scores, no objectness | YOLOv8 and later ultralytics heads, YOLOv5 "u" variants |

Both `training/cvtrain/export.py` and `nano/apps/build_engine.py` infer the
family from the ONNX output shape (the larger dimension is N), and
`training/cvtrain/package.py` refuses to package a model whose ONNX does not
match the decoder its architecture should have. That check exists because
the original version of this project shipped YOLOv7 metadata with a decoder
name nothing handled.

Older names `v5`, `v7` and `v8` are accepted as aliases.

## Constraints from the Jetson (JetPack 4.6.1, TensorRT 8.2.1)

- **ONNX opset <= 12**: the highest TensorRT 8.2 handles reliably.
- **Static input shape**: `(1, 3, S, S)`, export without dynamic axes.
- **Export at the training size**: running a model at a size it was not
  trained at costs accuracy.
- **No PyTorch checkpoints across the boundary**: the Nano's torch 1.10 /
  Python 3.6 cannot load checkpoints saved by modern PyTorch. Only ONNX
  travels. (The vendored `.pt` files are yolov5 v6.0 releases, which it can
  load.)
- **Engines are device-specific**: built on the Nano by
  `nano/apps/build_engine.py` (`trtexec --fp16`), never copied from another
  device or JetPack version.

## Adding a new model family or task

1. Write `decode_<family>(pred, ratio, pad, orig_shape, conf_thres, iou_thres)`
   in `nano/jetson_cv/inference/decoders.py`, returning
   `[(x1, y1, x2, y2, score, class_id), ...]`, and register it in `DECODERS`.
2. Teach the shape rule in `build_engine.py` / `cvtrain/export.py` to
   recognize its output (or pass `--decoder` explicitly).
3. For a new **task** (segmentation, pose, classification...), add a
   counterpart of `inference/detector.py` and a drawing function, and give
   the model `"task": "<name>"`. `Detector` refuses tasks other than
   `"detect"`, so a mismatch fails loudly instead of drawing garbage.
