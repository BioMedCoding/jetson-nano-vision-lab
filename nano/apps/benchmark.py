#!/usr/bin/env python3
"""Inference benchmark on the Nano: PyTorch FP32/FP16 vs TensorRT FP16.

Times the model alone on a synthetic image (no camera), so the numbers
isolate model cost. For reproducible results first run:
    sudo nvpmodel -m 0 && sudo jetson_clocks

Usage:
    python3 apps/benchmark.py                          # yolov5n: torch + TensorRT
    python3 apps/benchmark.py --models all --no-torch  # every engine in nano/models
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jetson_cv import runtime  # noqa: E402
runtime.ensure()

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from jetson_cv import YOLOV5_DIR  # noqa: E402
from jetson_cv.models import list_models, resolve  # noqa: E402


def bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1000.0  # ms


def bench_torch(weights, sizes, frame, iters, rows):
    import torch
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, a CPU benchmark is not meaningful.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Loading PyTorch model {}...".format(os.path.basename(weights)))
    model = torch.hub.load(YOLOV5_DIR, "custom", path=weights, source="local").to(device)
    for half in (False, True):
        if half and device.type != "cuda":
            continue
        m = model.half() if half else model.float()
        for size in sizes:
            ms = bench(lambda: m(frame, size=size), iters=iters)
            rows.append(("PyTorch " + ("FP16" if half else "FP32"),
                         os.path.basename(weights), size, ms))
            print("  {} size={}: {:.1f} ms".format(rows[-1][0], size, ms))


def bench_trt(info, frame, iters, rows):
    from jetson_cv.inference.preprocess import preprocess
    from jetson_cv.inference.trt_runner import TrtRunner
    print("Loading TensorRT engine {}...".format(info.id))
    runner = TrtRunner(info.engine_path)
    img, _, _ = preprocess(frame, runner.size)
    ms = bench(lambda: runner.infer(img), iters=iters)
    rows.append(("TensorRT FP16", info.id, runner.size, ms))
    print("  {}: {:.1f} ms".format(info.id, ms))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolov5n-416"],
                    help="model ids, or 'all' for every model with an engine")
    ap.add_argument("--no-torch", action="store_true", help="skip the PyTorch runs")
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 416, 640],
                    help="input sizes for the PyTorch runs")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if args.models == ["all"]:
        infos = [m for m in list_models() if m.has_engine]
    else:
        infos = [resolve(m) for m in args.models]

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    rows = []
    if not args.no_torch:
        for info in infos:
            weights = info.weights_path
            if weights and os.path.exists(weights) and info.decoder == "yolo_v5":
                bench_torch(weights, args.sizes, frame, args.iters, rows)
    for info in infos:
        if info.has_engine:
            bench_trt(info, frame, args.iters, rows)

    print("\n{:<14} {:<18} {:>5} {:>9} {:>7}".format("Backend", "Model", "size", "ms/frame", "FPS"))
    print("-" * 57)
    for backend, name, size, ms in rows:
        print("{:<14} {:<18} {:>5} {:>9.1f} {:>7.1f}".format(backend, name, size, ms, 1000.0 / ms))
    print("\nModel time only: the live pipeline (camera, conversions, drawing,"
          " streaming) adds overhead.")


if __name__ == "__main__":
    main()
