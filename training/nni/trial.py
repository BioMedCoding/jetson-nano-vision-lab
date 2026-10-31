#!/usr/bin/env python
"""NNI entry point for Stage A ("explore"). Receives the parameters NNI
sampled, builds or reuses the dataset (cache-aware), trains the chosen
architecture and reports mAP50-95 to NNI per epoch and at the end.

CVTRAIN_SMOKE=1 shrinks epochs, patience and samples for the end-to-end
smoke test (nni_config_smoke.yml) while running exactly the same code.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nni  # noqa: E402

from cvtrain.dataset_builder import build_dataset  # noqa: E402
from cvtrain.hparams import to_trainer_kwargs  # noqa: E402
from cvtrain.paths import DATASET_CACHE_DIR, NNI_RUNS_DIR, load_canonical_classes  # noqa: E402
from cvtrain.trainers import TRAINERS  # noqa: E402

SMOKE = bool(os.environ.get("CVTRAIN_SMOKE"))
EXPLORE_EPOCHS = 2 if SMOKE else 25
EXPLORE_PATIENCE = 1 if SMOKE else 8
# Model quality drives the budget here, not wall-clock time. 500 per class is
# the top of the 150-500 range docs/TRAINING_GUIDE.md recommends for hard
# classes, applied to all 80 canonical classes. Download cost is paid once
# per dataset_source (cached), not per trial.
MAX_SAMPLES_PER_CLASS = 20 if SMOKE else 500
SPLIT_SEED = 42
VAL_FRACTION = 0.15


def main() -> None:
    params = nni.get_next_parameter()
    trial_id = nni.get_trial_id()
    run_dir = NNI_RUNS_DIR / trial_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ds_config = {
        "source": params["dataset_source"],
        "classes": load_canonical_classes(),
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "include_local_capture": params["include_local_capture"],
    }
    data_yaml = build_dataset(ds_config, cache_root=DATASET_CACHE_DIR)

    kwargs = to_trainer_kwargs(arch=params["architecture"], params=params,
                               data_yaml=data_yaml, run_dir=run_dir,
                               epochs=EXPLORE_EPOCHS, patience=EXPLORE_PATIENCE)

    # Written BEFORE training: if it fails halfway, the resolved config is
    # still there to see what was attempted. promote.py (Stage B) reads only
    # this file, never NNI's internal storage (not a stable API across
    # versions), to replay a trial in full.
    (run_dir / "resolved_config.json").write_text(json.dumps({
        "schema_version": 1,
        "trial_id": trial_id,
        "stage": "explore",
        "params": params,
        "dataset_config": ds_config,
        "data_yaml": str(data_yaml),
    }, indent=2, default=str))

    def on_epoch(epoch, metrics):
        nni.report_intermediate_result(metrics["map50_95"])

    final = TRAINERS[params["architecture"]].train(**kwargs, on_epoch=on_epoch)
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
    nni.report_final_result(final["map50_95"])


if __name__ == "__main__":
    main()
