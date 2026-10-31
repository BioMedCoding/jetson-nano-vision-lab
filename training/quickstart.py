#!/usr/bin/env python3
"""Fine-tune one model on your dataset and package it for the Nano.

    validate dataset -> train -> export ONNX (at the training size) -> check
    the ONNX against the Nano's decoder -> export/<name>/ model folder

Usage (inside the training venv, see setup/setup_host.sh):
    python quickstart.py --data datasets/mydata/data.yaml
    python quickstart.py --data mydata/data.yaml --arch yolov8n --epochs 150 --name fox-v1
    python quickstart.py --data mydata/data.yaml --deploy      # also deploy to the Nano

The dataset is standard YOLO format (images/{train,val}, labels/{train,val},
data.yaml with "names"). Starting weights are the COCO-pretrained ones in
weights/. Environment: CVTRAIN_DEVICE (e.g. "cpu", "1"), CVTRAIN_BATCH.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvtrain.device import train_device  # noqa: E402
from cvtrain.export import export_onnx  # noqa: E402
from cvtrain.hparams import ARCHS, WEIGHTS, default_hyp, resolve_batch_size  # noqa: E402
from cvtrain.package import package_model  # noqa: E402
from cvtrain.paths import QUICKSTART_RUNS_DIR, TRAINING_DIR, read_names, resolved_data_yaml  # noqa: E402
from cvtrain.trainers import TRAINERS  # noqa: E402
from cvtrain.validate_dataset import DatasetValidationError, validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to the dataset's data.yaml")
    ap.add_argument("--arch", choices=ARCHS, default="yolov5n",
                    help="yolov5n is the fastest on the Nano (~30 fps at 416)")
    ap.add_argument("--img", type=int, default=416,
                    help="training AND inference size on the Nano (416 recommended)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30, help="early-stopping patience")
    ap.add_argument("--batch", type=int, default=None, help="default: table in cvtrain/hparams.py")
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze the first N layers (10 = yolov5/yolov8 backbone)")
    ap.add_argument("--name", default=None, help="run / model name (default: <dataset>-<arch>)")
    ap.add_argument("--deploy", action="store_true",
                    help="run host/deploy_model.sh on the result")
    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    dataset_dir = data_yaml.parent
    name = args.name or "{}-{}".format(dataset_dir.name, args.arch)
    run_dir = QUICKSTART_RUNS_DIR / name

    print("==== [1/4] Validating the dataset ====")
    try:
        stats = validate(data_yaml)
    except DatasetValidationError as e:
        sys.exit("Dataset problem: {}".format(e))
    resolved = resolved_data_yaml(data_yaml, run_dir / "data.yaml")
    names = read_names(resolved)
    print("classes: {} | images: {} | label files: {} ({} empty)".format(
        len(names), stats["images"], stats["n_labels"], stats["n_empty"]))
    if stats.get("empty_ratio_warning"):
        print("WARNING: more than half of the label files are empty")

    print("==== [2/4] Training {} for up to {} epochs at {} on device {} ====".format(
        args.arch, args.epochs, args.img, train_device()))

    def on_epoch(epoch, m):
        print("  epoch {:>3}: mAP50-95 {:.4f}  mAP50 {:.4f}".format(epoch, m["map50_95"], m["map50"]),
              flush=True)

    result = TRAINERS[args.arch].train(
        data_yaml=resolved, weights=str(WEIGHTS[args.arch]), img_size=args.img,
        epochs=args.epochs, patience=args.patience,
        batch=args.batch or resolve_batch_size(args.arch, args.img),
        hyp=default_hyp(args.arch), freeze_layers=args.freeze, run_dir=run_dir,
        on_epoch=on_epoch)
    (run_dir / "final_metrics.json").write_text(json.dumps(result, indent=2))
    print("best epoch: mAP50-95 {:.4f}, mAP50 {:.4f} ({} epochs run)".format(
        result["map50_95"], result["map50"], result["epochs_run"]))

    print("==== [3/4] Exporting ONNX at {0}x{0} ====".format(args.img))
    onnx_path = export_onnx(args.arch, Path(result["best_weights_path"]), args.img)

    print("==== [4/4] Packaging the Nano model folder ====")
    out_dir = package_model(onnx_path=onnx_path, arch=args.arch, img_size=args.img,
                            class_names=names, model_id=name, metrics=result,
                            source_note="Dataset: {}.".format(dataset_dir.name))
    print("Model folder ready: {}".format(out_dir))

    deploy = TRAINING_DIR.parent / "host" / "deploy_model.sh"
    if args.deploy:
        subprocess.check_call([str(deploy), str(out_dir)])
    else:
        print("Deploy it to the Nano with:\n  {} {}".format(deploy, out_dir))


if __name__ == "__main__":
    main()
