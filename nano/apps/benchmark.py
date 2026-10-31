#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark di inferenza YOLOv5 sulla Jetson Nano: PyTorch FP32/FP16 e TensorRT.

Misura i tempi su un'immagine sintetica (niente camera: isola il costo del
modello). Per risultati riproducibili prima esegui:
    sudo nvpmodel -m 0 && sudo jetson_clocks

Uso:
    python benchmark.py                                   # solo PyTorch
    python benchmark.py --engine ../models/yolov5n_416_fp16.engine
"""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402


def bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1000  # ms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(BASE / "models" / "yolov5n.pt"))
    ap.add_argument("--yolo-dir", default=str(Path.home() / "yolov5"))
    ap.add_argument("--engine", default=None, help="engine TensorRT opzionale")
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 416, 640])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    rows = []

    import torch
    if not torch.cuda.is_available():
        print("ATTENZIONE: CUDA non disponibile, benchmark su CPU poco utile.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("Carico il modello PyTorch...")
    model = torch.hub.load(args.yolo_dir, "custom", path=args.weights,
                           source="local").to(device)

    for half in (False, True):
        if half and device.type != "cuda":
            continue
        m = model.half() if half else model.float()
        for size in args.sizes:
            ms = bench(lambda: m(frame, size=size), iters=args.iters)
            rows.append(("PyTorch " + ("FP16" if half else "FP32"), size, ms))
            print("  {} size={}: {:.1f} ms".format(rows[-1][0], size, ms))

    if args.engine:
        from common.trt_runner import TrtRunner
        from common.yolo_utils import preprocess
        print("Carico engine TensorRT...")
        runner = TrtRunner(args.engine)
        img, _, _ = preprocess(frame, runner.size)
        ms = bench(lambda: runner.infer(img), iters=args.iters)
        rows.append(("TensorRT FP16", runner.size, ms))
        print("  TensorRT size={}: {:.1f} ms".format(runner.size, ms))

    print("\n{:<16} {:>6} {:>10} {:>8}".format("Backend", "size", "ms/frame", "FPS"))
    print("-" * 44)
    for name, size, ms in rows:
        print("{:<16} {:>6} {:>10.1f} {:>8.1f}".format(name, size, ms, 1000.0 / ms))
    print("\nNota: e' il solo tempo modello; la pipeline completa (camera +"
          " conversioni + display) aggiunge overhead.")


if __name__ == "__main__":
    main()
