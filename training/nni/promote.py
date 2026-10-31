#!/usr/bin/env python
"""Stage B ("promote"): replay a promising Stage A trial in full, with more
epochs and an explicit wall-clock safety cap, then export and package it
for the Nano (and optionally deploy it).

Reads ONLY runs/nni/<trial_id>/resolved_config.json, never NNI's internal
storage (nnictl experiment export / db), which is not a stable API across
versions. Listing the best trials scans runs/nni/*/final_metrics.json on
disk: same information, no dependency on NNI's export format.

Usage:
    python promote.py                          # list the best trials in runs/nni/
    python promote.py --trial-id <id>          # promote one, package it in export/
    python promote.py --trial-id <id> --deploy # ... and deploy it to the Nano
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvtrain.dataset_builder import build_dataset  # noqa: E402
from cvtrain.export import export_onnx  # noqa: E402
from cvtrain.hparams import to_trainer_kwargs  # noqa: E402
from cvtrain.package import package_model  # noqa: E402
from cvtrain.paths import (DATASET_CACHE_DIR, NNI_RUNS_DIR, TRAINING_DIR,  # noqa: E402
                           load_canonical_classes)
from cvtrain.trainers import TRAINERS  # noqa: E402


def list_promising(top_k: int) -> list:
    """Top-k (trial_id, mAP50-95, architecture), best first. Skips
    *_promoted folders (already promoted)."""
    scored = []
    if not NNI_RUNS_DIR.is_dir():
        return scored
    for run_dir in NNI_RUNS_DIR.iterdir():
        if not run_dir.is_dir() or run_dir.name.endswith("_promoted"):
            continue
        metrics_path = run_dir / "final_metrics.json"
        config_path = run_dir / "resolved_config.json"
        if not (metrics_path.is_file() and config_path.is_file()):
            continue
        metrics = json.loads(metrics_path.read_text())
        config = json.loads(config_path.read_text())
        scored.append((run_dir.name, metrics["map50_95"], config["params"]["architecture"]))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]


def promote(trial_id, epochs, patience, max_hours, deploy):
    resolved_path = NNI_RUNS_DIR / trial_id / "resolved_config.json"
    if not resolved_path.is_file():
        raise SystemExit(f"no resolved_config.json for trial {trial_id}: {resolved_path}")
    resolved = json.loads(resolved_path.read_text())
    params = resolved["params"]
    ds_config = resolved["dataset_config"]
    arch = params["architecture"]
    print(f"Promoting trial {trial_id}: architecture={arch}, dataset_source={ds_config['source']}, "
          f"epochs={epochs}, patience={patience}, cap={max_hours}h")

    data_yaml = build_dataset(ds_config, cache_root=DATASET_CACHE_DIR)
    promoted_dir = NNI_RUNS_DIR / f"{trial_id}_promoted"
    kwargs = to_trainer_kwargs(arch=arch, params=params, data_yaml=data_yaml,
                               run_dir=promoted_dir, epochs=epochs, patience=patience)
    promoted_dir.mkdir(parents=True, exist_ok=True)
    (promoted_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2))

    def on_epoch(epoch, metrics):
        print(f"  epoch {epoch}: mAP50-95={metrics['map50_95']:.4f}", flush=True)

    final = TRAINERS[arch].train(**kwargs, on_epoch=on_epoch, max_hours=max_hours)
    (promoted_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
    print(f"Training done: mAP50-95={final['map50_95']:.4f} ({final['epochs_run']} epochs)")

    onnx_path = export_onnx(arch, Path(final["best_weights_path"]), params["img_size"])
    out_dir = package_model(onnx_path=onnx_path, arch=arch, img_size=params["img_size"],
                            class_names=load_canonical_classes(),
                            model_id=f"{arch}-{params['img_size']}-{trial_id}",
                            metrics=final,
                            source_note=f"NNI trial {trial_id}, dataset {ds_config['source']}.")
    print(f"Model folder ready: {out_dir}")
    deploy_script = TRAINING_DIR.parent / "host" / "deploy_model.sh"
    if deploy:
        subprocess.run([str(deploy_script), str(out_dir)], check=True)
    else:
        print(f"Deploy it with:  {deploy_script} {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial-id", help="trial id from runs/nni/ (without it: list the best)")
    ap.add_argument("--top-k", type=int, default=5, help="how many trials to list")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--max-hours", type=float, default=48.0,
                    help="safety net against an unattended run, not a target: patience "
                         "already stops a run that stopped improving; raise it freely")
    ap.add_argument("--deploy", action="store_true", help="run host/deploy_model.sh afterwards")
    args = ap.parse_args()

    if args.trial_id:
        promote(args.trial_id, args.epochs, args.patience, args.max_hours, args.deploy)
        return
    ranked = list_promising(args.top_k)
    if not ranked:
        print("No trials in runs/nni/ yet (run Stage A first: "
              "nnictl create --config nni_config_explore.yml)")
        return
    print(f"Top {len(ranked)} trials by mAP50-95:")
    for trial_id, score, arch in ranked:
        print(f"  {trial_id}  map50_95={score:.4f}  arch={arch}")
    print("\nTo promote: python promote.py --trial-id <id> [--epochs N] [--deploy]")


if __name__ == "__main__":
    main()
