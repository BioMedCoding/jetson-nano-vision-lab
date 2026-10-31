#!/usr/bin/env python
"""End-to-end check of every architecture family on a tiny dataset:
train -> export ONNX -> package for the Nano (which verifies the ONNX output
matches the decoder the Nano will use, the input size and the class count).

    python tests/test_trainers.py                 # all 4 architectures, 3 epochs
    python tests/test_trainers.py yolov8n --epochs 1

Downloads coco128 (128 COCO images, ~7 MB) into datasets/ on first use.
The resulting model folders land in export/test-<arch>/ and can be deployed
with host/deploy_model.sh to check them on the Jetson too.
"""
import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvtrain.export import export_onnx  # noqa: E402
from cvtrain.hparams import ARCHS, WEIGHTS, default_hyp, resolve_batch_size  # noqa: E402
from cvtrain.package import package_model  # noqa: E402
from cvtrain.paths import DATASETS_DIR, RUNS_DIR, read_names, resolved_data_yaml  # noqa: E402
from cvtrain.trainers import TRAINERS  # noqa: E402

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def coco128():
    """Return a data.yaml for coco128 (train = val, it's a smoke test)."""
    root = DATASETS_DIR / "coco128"
    if not (root / "images" / "train2017").is_dir():
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        zpath = DATASETS_DIR / "coco128.zip"
        print("Downloading", COCO128_URL)
        urllib.request.urlretrieve(COCO128_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATASETS_DIR)
        zpath.unlink()
    from cvtrain.paths import load_canonical_classes  # the 80 COCO names
    data = root / "data.yaml"
    names = load_canonical_classes()
    data.write_text("path: {}\ntrain: images/train2017\nval: images/train2017\n"
                    "names:\n{}".format(root, "".join(
                        "  {}: {}\n".format(i, n) for i, n in enumerate(names))))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archs", nargs="*", default=list(ARCHS), help=", ".join(ARCHS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--img", type=int, default=416)
    args = ap.parse_args()

    data = coco128()
    results = []
    for arch in args.archs:
        t0 = time.time()
        run_dir = RUNS_DIR / "tests" / arch
        resolved = resolved_data_yaml(data, run_dir / "data.yaml")
        try:
            res = TRAINERS[arch].train(
                data_yaml=resolved, weights=str(WEIGHTS[arch]), img_size=args.img,
                epochs=args.epochs, patience=0, batch=min(16, resolve_batch_size(arch, args.img)),
                hyp=default_hyp(arch), freeze_layers=0, run_dir=run_dir)
            onnx = export_onnx(arch, Path(res["best_weights_path"]), args.img)
            out = package_model(onnx_path=onnx, arch=arch, img_size=args.img,
                                class_names=read_names(resolved), model_id="test-" + arch,
                                metrics=res, source_note="tests/test_trainers.py on coco128.")
            results.append((arch, "PASS", "mAP50 {:.3f} -> {} ({:.0f} s)".format(
                res["map50"], out, time.time() - t0)))
        except Exception as e:  # report every architecture, then fail at the end
            results.append((arch, "FAIL", "{}: {}".format(type(e).__name__, (str(e).strip().splitlines() or [""])[-1][:300])))
        print("{} {}: {}".format(results[-1][1], arch, results[-1][2]), flush=True)

    print("\n" + "\n".join("{} {:<12} {}".format(s, a, d) for a, s, d in results))
    sys.exit(0 if all(s == "PASS" for _, s, _ in results) else 1)


if __name__ == "__main__":
    main()
