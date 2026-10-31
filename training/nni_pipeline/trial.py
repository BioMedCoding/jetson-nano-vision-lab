#!/usr/bin/env python
"""Entry point NNI per Stage A ('explore'). Riceve i parametri campionati da
NNI, costruisce/risolve il dataset (cache-aware), allena l'architettura
scelta, riporta mAP@0.5:0.95 per epoca e finale a NNI.

Variabile d'ambiente CVTRAIN_SMOKE=1: riduce epoche/pazienza/campioni per il
test end-to-end (smoke_test/, nni_config_smoke.yml) riusando esattamente lo
stesso codice del run reale.
"""
from __future__ import annotations

import json
import os

import nni

from common.hparam_mapping import to_trainer_kwargs
from common.paths import DATASET_CACHE_DIR, RUNS_DIR, load_canonical_classes
from dataset_builder import build_dataset
from trainers import train_yolov5, train_yolov7, train_yolov8

TRAINER_MAP = {
    "yolov5n": train_yolov5,
    "yolov5s": train_yolov5,
    "yolov7-tiny": train_yolov7,
    "yolov8n": train_yolov8,
}

SMOKE = bool(os.environ.get("CVTRAIN_SMOKE"))
EXPLORE_EPOCHS = 2 if SMOKE else 25
EXPLORE_PATIENCE = 1 if SMOKE else 8
# La qualita' guida le scelte di budget in questo progetto, non il tempo
# (indicazione esplicita dell'utente: va bene superare il budget orario di
# Stage A pur di garantire qualita'). 500/classe e' il valore alto della
# forchetta 150-500/classe raccomandata in GUIDA_TRAINING_DATASET.md per
# classi "difficili" -- qui applicato a tutte le 80 classi canoniche
# (COCO-80, vedi class_maps/target_classes.yaml). Costo di download: una
# tantum per dataset_source (cacheato), non per trial.
MAX_SAMPLES_PER_CLASS = 20 if SMOKE else 500
SPLIT_SEED = 42
VAL_FRACTION = 0.15


def main() -> None:
    params = nni.get_next_parameter()
    trial_id = nni.get_trial_id()
    run_dir = RUNS_DIR / trial_id
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

    kwargs = to_trainer_kwargs(
        arch=params["architecture"],
        params=params,
        data_yaml=data_yaml,
        run_dir=run_dir,
        epochs=EXPLORE_EPOCHS,
        patience=EXPLORE_PATIENCE,
    )

    # Dump PRIMA di allenare: se il training fallisce a meta', resta comunque
    # la config risolta per capire cosa e' stato tentato. promote.py (Stage
    # B) legge SOLO questo file (non lo storage interno di NNI, che non e'
    # un'API stabile tra versioni) per rilanciare un trial per intero.
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial_id": trial_id,
                "stage": "explore",
                "params": params,
                "dataset_config": ds_config,
                "data_yaml": str(data_yaml),
            },
            indent=2,
            default=str,
        )
    )

    trainer = TRAINER_MAP[params["architecture"]]

    def on_epoch(epoch, metrics):
        nni.report_intermediate_result(metrics["map50_95"])

    final = trainer.train(**kwargs, on_epoch=on_epoch)
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
    nni.report_final_result(final["map50_95"])


if __name__ == "__main__":
    main()
