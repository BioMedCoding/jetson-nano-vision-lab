#!/usr/bin/env python3
"""Add a model to nano/models and build its TensorRT engine. Run ON the Nano:
an engine is tied to the GPU and TensorRT version that built it.

Three ways in:
    # 1. rebuild the engine of an existing model folder (from its model.onnx)
    python3 apps/build_engine.py yolov5n-416

    # 2. register an ONNX exported elsewhere (e.g. trained on your PC) and build
    python3 apps/build_engine.py ~/best.onnx --id fox-v1 --labels ~/classes.txt

    # 3. export a YOLOv5 v6.0 .pt to ONNX first (vendored yolov5), then build
    python3 apps/build_engine.py yolov5s.pt --id yolov5s-416 --size 416

For ONNX inputs the input size and the decoder family are read from the
ONNX itself: (1, N, 5+nc) -> yolo_v5 (YOLOv5, YOLOv7 --grid),
(1, 4+nc, N) -> yolo_v8 (YOLOv8 and later). --decoder overrides that.
Building takes ~5-20 min; --onnx-only stops after the model folder is ready.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jetson_cv import runtime  # noqa: E402
runtime.ensure()

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

from jetson_cv import MODELS_DIR, YOLOV5_DIR  # noqa: E402
from jetson_cv.inference.decoders import DECODERS, canonical_name  # noqa: E402
from jetson_cv.models import MODEL_FILE, ModelInfo  # noqa: E402

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
OPSET = 12  # the highest opset TensorRT 8.2 (JetPack 4.6.1) handles reliably


def onnx_io(onnx_path):
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    return dims(m.graph.input[0]), [dims(o) for o in m.graph.output]


def guess_decoder(outputs):
    """Pick the decoder family from the main (largest) output's shape."""
    main = max(outputs, key=lambda s: (len(s) == 3, abs(int(_prod(s)))))
    if len(main) != 3:
        raise ValueError("unexpected output shapes {}: pass --decoder".format(outputs))
    _, a, b = main
    return ("yolo_v5", b - 5) if a > b else ("yolo_v8", a - 4)


def _prod(shape):
    p = 1
    for d in shape:
        p *= max(d, 1)
    return p


def export_yolov5_pt(weights, size, out_dir):
    print("== Exporting {} to ONNX at {}x{} (vendored yolov5 v6.0, opset {}) =="
          .format(weights, size, size, OPSET))
    subprocess.check_call([sys.executable, os.path.join(YOLOV5_DIR, "export.py"),
                           "--weights", weights, "--include", "onnx",
                           "--imgsz", str(size), str(size), "--opset", str(OPSET),
                           "--device", "cpu"])
    produced = os.path.splitext(weights)[0] + ".onnx"
    shutil.move(produced, os.path.join(out_dir, "model.onnx"))
    dst = os.path.join(out_dir, os.path.basename(weights))
    if os.path.abspath(weights) != os.path.abspath(dst):
        shutil.copy2(weights, dst)
    return os.path.basename(weights)


def register(args):
    """Create or update models/<id>/ with model.onnx and model.json."""
    src = os.path.abspath(args.source)
    model_id = args.id or os.path.splitext(os.path.basename(src))[0]
    out_dir = os.path.join(args.models_dir, model_id)
    os.makedirs(out_dir, exist_ok=True)

    weights_name = None
    if src.endswith(".pt"):
        if not args.size:
            sys.exit("--size is required when starting from a .pt")
        weights_name = export_yolov5_pt(src, args.size, out_dir)
    elif src.endswith(".onnx"):
        dst = os.path.join(out_dir, "model.onnx")
        if os.path.abspath(src) != dst:
            shutil.copy2(src, dst)
    else:
        sys.exit("expected a model folder/id, a .onnx or a .pt: {}".format(args.source))

    in_shape, outputs = onnx_io(os.path.join(out_dir, "model.onnx"))
    size = in_shape[-1]
    decoder, n_classes = guess_decoder(outputs)
    if args.decoder:
        decoder = canonical_name(args.decoder)
    print("ONNX: input {}, outputs {} -> decoder {}, {} classes".format(
        in_shape, outputs, decoder, n_classes))

    labels = "coco80"
    if args.labels:
        shutil.copy2(args.labels, os.path.join(out_dir, "labels.txt"))
        labels = "labels.txt"
        with open(args.labels) as f:
            n_names = len([l for l in f if l.strip()])
        if n_names != n_classes:
            print("WARNING: {} has {} names but the model outputs {} classes"
                  .format(args.labels, n_names, n_classes))
    elif n_classes != 80:
        print("WARNING: the model has {} classes, not COCO's 80: pass --labels "
              "classes.txt or the dashboard will show class numbers".format(n_classes))

    # Re-registering an existing model keeps its hand-written fields
    # (label, source, labels file) unless new values are given.
    meta_path = os.path.join(out_dir, MODEL_FILE)
    old = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
    if not args.labels and old.get("labels"):
        labels = old["labels"]
    meta = {"label": args.label or old.get("label") or "{} {}".format(model_id, size),
            "task": "detect", "decoder": decoder, "input_size": size, "labels": labels,
            "onnx": "model.onnx", "engine": "model.fp16.engine"}
    if weights_name or old.get("weights"):
        meta["weights"] = weights_name or old["weights"]
    if args.source_note or old.get("source"):
        meta["source"] = args.source_note or old["source"]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return out_dir


def build_engine(info):
    if not os.path.exists(TRTEXEC):
        sys.exit("trtexec not found at {}: run this on the Jetson".format(TRTEXEC))
    log_path = os.path.join(info.dir, "build.log")
    print("== Building the TensorRT FP16 engine for {} (5-20 min, log: {}) =="
          .format(info.id, log_path))
    with open(log_path, "w") as log:
        rc = subprocess.call([TRTEXEC, "--onnx=" + info.onnx_path,
                              "--saveEngine=" + info.engine_path,
                              "--fp16", "--workspace=1024"],
                             stdout=log, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(info.engine_path):
        sys.exit("trtexec failed (exit {}), see {}".format(rc, log_path))
    print("Engine ready: {}".format(info.engine_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="model id/folder, .onnx file, or yolov5 v6.0 .pt")
    ap.add_argument("--id", help="model folder name (default: from the file name)")
    ap.add_argument("--size", type=int, help="input size, only for a .pt export")
    ap.add_argument("--decoder", choices=sorted(DECODERS),
                    help="override the decoder family read from the ONNX")
    ap.add_argument("--labels", help="class names file (one per line), if not COCO-80")
    ap.add_argument("--label", help="name shown in the dashboard")
    ap.add_argument("--source-note", help="free text stored as model.json 'source'")
    ap.add_argument("--onnx-only", action="store_true",
                    help="prepare the model folder but do not build the engine")
    ap.add_argument("--models-dir", default=MODELS_DIR)
    args = ap.parse_args()

    candidate = (args.source if os.path.isdir(args.source)
                 else os.path.join(args.models_dir, args.source))
    if os.path.isfile(os.path.join(candidate, MODEL_FILE)):
        model_dir = candidate                     # way 1: existing folder
    else:
        model_dir = register(args)                # ways 2 and 3

    info = ModelInfo.from_dir(model_dir)
    if not os.path.exists(info.onnx_path):
        sys.exit("{} has no {} to build from".format(info.id, info.onnx_path))
    if args.onnx_only:
        print("Model folder ready: {} (engine not built)".format(model_dir))
        return
    build_engine(info)
    print("Try it:  python3 apps/trt_detect.py --model {}".format(info.id))
    print("     or  python3 apps/dashboard.py --model {}".format(info.id))


if __name__ == "__main__":
    main()
