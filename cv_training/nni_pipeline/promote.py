#!/usr/bin/env python
"""Stage B ('promote'): rilancia per intero un trial promettente di Stage A
con epoche estese e un tetto wall-clock esplicito, poi esporta ed
eventualmente deploya sulla Nano.

Legge SOLO runs/<trial_id>/resolved_config.json -- mai lo storage interno di
NNI (nnictl experiment export/db), che non e' un'API stabile tra versioni e
in questa sessione ha gia' mostrato piu' di una sorpresa (vedi il deadlock di
nnictl documentato in nni_pipeline/README.md). La modalita' "quali trial sono
promettenti" scansiona runs/*/final_metrics.json direttamente sul disco
locale, stessa informazione, zero dipendenza dal formato di export di NNI.

Uso:
    python promote.py                                    # elenca i migliori trial trovati in runs/
    python promote.py --trial-id <id>                    # promuove un trial specifico
    python promote.py --trial-id <id> --deploy user@host # promuove e fa il deploy sulla Nano
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from common.export_utils import export_onnx, write_classes_txt
from common.hparam_mapping import to_trainer_kwargs
from common.paths import DATASET_CACHE_DIR, PIPELINE_DIR, RUNS_DIR, load_canonical_classes
from dataset_builder import build_dataset
from trainers import train_yolov5, train_yolov7, train_yolov8

TRAINER_MAP = {
    "yolov5n": train_yolov5,
    "yolov5s": train_yolov5,
    "yolov7-tiny": train_yolov7,
    "yolov8n": train_yolov8,
}
POSTPROCESS = {"yolov5n": "v5", "yolov5s": "v5", "yolov7-tiny": "v7", "yolov8n": "v8"}


def list_promising(top_k: int) -> list[tuple[str, float, str]]:
    """Scansiona runs/*/final_metrics.json, ritorna i top_k trial_id per
    mAP@0.5:0.95 con la relativa architettura, dal migliore al peggiore.
    Ignora le cartelle *_promoted (gia' promosse da un run precedente)."""
    scored = []
    if not RUNS_DIR.is_dir():
        return scored
    for run_dir in RUNS_DIR.iterdir():
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


def promote(
    trial_id: str, epochs: int, patience: int, max_hours: float | None, deploy_host: str | None
) -> None:
    resolved_path = RUNS_DIR / trial_id / "resolved_config.json"
    if not resolved_path.is_file():
        raise SystemExit(f"resolved_config.json non trovato per il trial {trial_id}: {resolved_path}")
    resolved = json.loads(resolved_path.read_text())
    params = resolved["params"]
    ds_config = resolved["dataset_config"]
    arch = params["architecture"]

    print(f"Promuovo trial {trial_id}: architettura={arch}, dataset_source={ds_config['source']}, "
          f"epoche={epochs}, patience={patience}, tetto={max_hours}h")

    data_yaml = build_dataset(ds_config, cache_root=DATASET_CACHE_DIR)

    promoted_run_dir = RUNS_DIR / f"{trial_id}_promoted"
    kwargs = to_trainer_kwargs(
        arch=arch, params=params, data_yaml=data_yaml, run_dir=promoted_run_dir,
        epochs=epochs, patience=patience,
    )
    (promoted_run_dir).mkdir(parents=True, exist_ok=True)
    (promoted_run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2))

    def on_epoch(epoch, metrics):
        print(f"  epoca {epoch}: mAP@0.5:0.95={metrics['map50_95']:.4f}")

    trainer = TRAINER_MAP[arch]
    final = trainer.train(**kwargs, on_epoch=on_epoch, max_hours=max_hours)
    (promoted_run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
    print(f"Training completo: mAP@0.5:0.95={final['map50_95']:.4f} ({final['epochs_run']} epoche)")

    onnx_path = export_onnx(arch, Path(final["best_weights_path"]), params["img_size"])
    classes_txt = onnx_path.parent / "classes.txt"
    write_classes_txt(load_canonical_classes(), classes_txt)
    print(f"Export ONNX: {onnx_path}")

    # Punto critico: il postprocess va calcolato esplicitamente
    # dall'architettura, mai lasciato al fallback per nome file di
    # build_engine.sh -- con 4 architetture in gioco quel fallback (solo
    # "yolov8*"->v8, altrimenti v5) non e' affidabile.
    postprocess = POSTPROCESS[arch]
    deploy_cmd = [
        str(PIPELINE_DIR / "deploy.sh"), str(onnx_path), str(classes_txt),
        str(params["img_size"]), postprocess,
    ]
    if deploy_host:
        deploy_cmd.append(deploy_host)
    subprocess.run(deploy_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial-id", help="trial_id da runs/ (senza: elenca i migliori trovati)")
    parser.add_argument("--top-k", type=int, default=5, help="quanti trial elencare senza --trial-id")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument(
        "--max-hours", type=float, default=48.0,
        help="rete di sicurezza contro un run incustodito, non un obiettivo: la qualita' "
             "guida il budget in questo progetto, non il tempo. patience si occupa gia' di "
             "fermare un training che ha smesso di migliorare; alza pure questo valore se "
             "serve piu' margine.",
    )
    parser.add_argument("--deploy", metavar="user@host")
    args = parser.parse_args()

    if args.trial_id:
        promote(args.trial_id, args.epochs, args.patience, args.max_hours, args.deploy)
        return

    ranked = list_promising(args.top_k)
    if not ranked:
        print("Nessun trial trovato in runs/ (esegui prima Stage A: nnictl create --config nni_config_explore.yml)")
        return
    print(f"Top {len(ranked)} trial per mAP@0.5:0.95:")
    for trial_id, score, arch in ranked:
        print(f"  {trial_id}  map50_95={score:.4f}  arch={arch}")
    print("\nPer promuovere: python promote.py --trial-id <id> [--epochs N] [--deploy user@host]")


if __name__ == "__main__":
    main()
