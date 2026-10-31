#!/usr/bin/env python
"""Standalone smoke test for cvtrain/dataset_builder.py (no NNI, no GPU).

Builds tiny datasets (max_samples_per_class=20) for the 3 supported
dataset_source variants, validates each, and checks that
include_local_capture=True without datasets/local_capture/ raises the
documented error instead of a generic crash.

Usage (from training/nni/): python smoke_test/make_tiny_sample.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cvtrain.dataset_builder import build_dataset  # noqa: E402
from cvtrain.paths import DATASET_CACHE_DIR, load_canonical_classes  # noqa: E402
from cvtrain.validate_dataset import validate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

SMOKE_CACHE_ROOT = DATASET_CACHE_DIR.parent / "cache_smoke"


def base_config(source: str) -> dict:
    return {
        "source": source,
        "classes": load_canonical_classes(),
        "max_samples_per_class": 20,
        "split_seed": 0,
        "val_fraction": 0.2,
        "include_local_capture": False,
    }


def run_source_variant(source: str) -> None:
    print(f"\n=== build_dataset(source={source!r}) ===")
    config = base_config(source)
    data_yaml = build_dataset(config, cache_root=SMOKE_CACHE_ROOT)
    stats = validate(data_yaml.parent)
    print(f"PASS: {source} -> {data_yaml} stats={stats}")


def run_negative_local_capture_test() -> None:
    print("\n=== include_local_capture=True without datasets/local_capture/ ===")
    config = base_config("oiv7")
    config["include_local_capture"] = True
    config["split_seed"] = 999  # different hash: does not reuse run_source_variant's cache
    try:
        build_dataset(config, cache_root=SMOKE_CACHE_ROOT)
    except RuntimeError as e:
        print(f"PASS: RuntimeError raised as expected: {e}")
        return
    raise AssertionError(
        "FAIL: build_dataset did not raise RuntimeError with "
        "include_local_capture=True and no datasets/local_capture/"
    )


def main() -> None:
    for source in ("oiv7", "samacoco", "oiv7+samacoco"):
        run_source_variant(source)
    run_negative_local_capture_test()
    print("\nAll dataset builder smoke tests passed.")


if __name__ == "__main__":
    main()
