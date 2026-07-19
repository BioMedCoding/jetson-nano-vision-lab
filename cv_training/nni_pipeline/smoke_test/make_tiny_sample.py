#!/usr/bin/env python
"""Smoke test standalone (no NNI, no GPU) per dataset_builder.py.

Costruisce dataset minuscoli (max_samples_per_class=20) per le 3 varianti
di dataset_source supportate, valida ciascuna, e verifica che
include_local_capture=True senza local_capture_dataset/ presente sollevi
l'errore documentato invece di un crash generico.

Uso: python smoke_test/make_tiny_sample.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import DATASET_CACHE_DIR, load_canonical_classes
from common.validate_dataset import validate
from dataset_builder import build_dataset

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

SMOKE_CACHE_ROOT = DATASET_CACHE_DIR.parent / "dataset_cache_smoke"


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
    print("\n=== include_local_capture=True senza local_capture_dataset/ ===")
    config = base_config("oiv7")
    config["include_local_capture"] = True
    config["split_seed"] = 999  # hash diverso, non riusa cache di run_source_variant
    try:
        build_dataset(config, cache_root=SMOKE_CACHE_ROOT)
    except RuntimeError as e:
        print(f"PASS: RuntimeError sollevato come atteso: {e}")
        return
    raise AssertionError(
        "FAIL: build_dataset non ha sollevato RuntimeError con "
        "include_local_capture=True e local_capture_dataset/ assente"
    )


def main() -> None:
    for source in ("oiv7", "samacoco", "oiv7+samacoco"):
        run_source_variant(source)
    run_negative_local_capture_test()
    print("\nTutti i test smoke del dataset builder sono passati.")


if __name__ == "__main__":
    main()
