"""Mappa il search space generico di NNI sui kwargs nativi di ciascun
trainer (trainers/train_<arch>.py), gestendo le differenze scoperte in
sessione tra le tre famiglie di architettura:

- yolov5/yolov7 accettano gli iperparametri di augmentation/loss solo via
  file --hyp; yolov8 (pacchetto ultralytics) li accetta come kwargs diretti
  di model.train().
- yolov8 non ha objectness (testa anchor-free): niente obj/obj_pw/iou_t/
  anchor_t/fl_gamma. Il suo `cls_pw` esiste ma con significato DIVERSO
  (class-imbalance weighting, non BCE positive weight) -- va escluso, non
  solo "lasciato al default": passarlo per errore cambierebbe silenziosamente
  il training.
- yolov8 usa optimizer="auto" di default, che IGNORA lr0/momentum passati
  esplicitamente (verificato in sessione) -- va sempre forzato
  optimizer="SGD" per rendere la ricerca NNI su lr0/momentum efficace.
- box/cls loss gain hanno scale numeriche diverse tra le famiglie (yolov5/
  yolov7 ~0.05/0.5, yolov8 ~7.5/0.5, verificato in sessione): il search
  space non cerca un valore assoluto ma un moltiplicatore relativo al
  default di ciascuna architettura.
- freeze_layers e' un booleano nel search space (freeze_backbone) mappato
  su una profondita' di layer specifica per architettura, perche' "10" non
  significa la stessa cosa in tutte e tre (backbone yolov5n/s ~10 layer,
  yolov7-tiny e yolov8n hanno profondita' diverse).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from common.paths import PIPELINE_DIR, YOLOV5_DIR, YOLOV7_DIR

WEIGHTS = {
    "yolov5n": YOLOV5_DIR / "yolov5n.pt",
    "yolov5s": YOLOV5_DIR / "yolov5s.pt",
    "yolov7-tiny": YOLOV7_DIR / "yolov7-tiny.pt",
    "yolov8n": PIPELINE_DIR / "weights" / "yolov8n.pt",
}

# gain di default per architettura (verificati in sessione leggendo
# data/hyps/hyp.scratch-low.yaml, data/hyp.scratch.tiny.yaml, e i log di un
# run reale di ultralytics che stampa i propri default box=7.5/cls=0.5)
ARCH_DEFAULT_BOX_GAIN = {"yolov5n": 0.05, "yolov5s": 0.05, "yolov7-tiny": 0.05, "yolov8n": 7.5}
ARCH_DEFAULT_CLS_GAIN = {"yolov5n": 0.5, "yolov5s": 0.5, "yolov7-tiny": 0.5, "yolov8n": 0.5}

# Profondita' di freeze del backbone quando freeze_backbone=True nel search
# space. yolov5n/s: 10 e' la convenzione documentata nella GUIDA del
# progetto. yolov8n: 10 e' la convenzione documentata da ultralytics per i
# modelli 'n'. yolov7-tiny: STIMA non verificata puntualmente (il commento
# di --freeze nel train.py originale, "backbone of yolov7=50", si riferisce
# al yolov7 pieno, non alla variante tiny che ha meno layer) -- 25 e' una
# stima approssimativa basata sulla profondita' totale osservata (~208-263
# layer secondo il contesto), da verificare/tarare guardando i risultati.
ARCH_FREEZE_BACKBONE = {"yolov5n": 10, "yolov5s": 10, "yolov7-tiny": 25, "yolov8n": 10}

# batch size per architettura/risoluzione (12GB VRAM RTX 5070): valori
# conservativi, c'e' ampio margine osservato nei run di prova in sessione
# (yolov5n/yolov8n/yolov7-tiny a batch=4/img=416 usavano <0.4GB VRAM ciascuno)
BATCH_SIZE_TABLE = {
    ("yolov5n", 416): 64, ("yolov5n", 512): 48, ("yolov5n", 640): 32,
    ("yolov5s", 416): 48, ("yolov5s", 512): 32, ("yolov5s", 640): 24,
    ("yolov7-tiny", 416): 48, ("yolov7-tiny", 512): 32, ("yolov7-tiny", 640): 24,
    ("yolov8n", 416): 64, ("yolov8n", 512): 48, ("yolov8n", 640): 32,
}

# chiavi searched dal search space che yolov8 (ultralytics) NON accetta o
# accetta con significato diverso -- escluse esplicitamente, mai lasciate
# passare per "coincidenza di nome"
_YOLOV8_EXCLUDED_HYP_KEYS = {"obj", "obj_pw", "cls_pw", "iou_t", "anchor_t", "fl_gamma"}


def resolve_batch_size(arch: str, img_size: int) -> int:
    return BATCH_SIZE_TABLE[(arch, img_size)]


def to_trainer_kwargs(
    *, arch: str, params: dict, data_yaml: Path, run_dir: Path, epochs: int, patience: int
) -> dict:
    """Ritorna i kwargs pronti per trainers.train_<arch>.train(...)."""
    freeze_layers = ARCH_FREEZE_BACKBONE[arch] if params["freeze_backbone"] else 0
    box_gain = ARCH_DEFAULT_BOX_GAIN[arch] * params["box_gain_mult"]
    cls_gain = ARCH_DEFAULT_CLS_GAIN[arch] * params["cls_gain_mult"]

    if arch == "yolov8n":
        hyp = _build_yolov8_hyp(params, box_gain, cls_gain)
    else:
        hyp = _build_yolov5v7_hyp(arch, params, box_gain, cls_gain)

    return {
        "data_yaml": data_yaml,
        "weights": str(WEIGHTS[arch]),
        "img_size": params["img_size"],
        "epochs": epochs,
        "patience": patience,
        "batch": resolve_batch_size(arch, params["img_size"]),
        "hyp": hyp,
        "freeze_layers": freeze_layers,
        "run_dir": run_dir,
    }


_SHARED_SEARCHED_KEYS = (
    "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
)


def _build_yolov5v7_hyp(arch: str, params: dict, box_gain: float, cls_gain: float) -> dict:
    base_path = (
        YOLOV5_DIR / "data/hyps/hyp.scratch-low.yaml"
        if arch in ("yolov5n", "yolov5s")
        else YOLOV7_DIR / "data/hyp.scratch.tiny.yaml"
    )
    with open(base_path) as f:
        hyp = yaml.safe_load(f)
    for key in _SHARED_SEARCHED_KEYS:
        hyp[key] = params[key]
    hyp["box"] = box_gain
    hyp["cls"] = cls_gain
    return hyp


def _build_yolov8_hyp(params: dict, box_gain: float, cls_gain: float) -> dict:
    hyp = {key: params[key] for key in _SHARED_SEARCHED_KEYS}
    hyp["box"] = box_gain
    hyp["cls"] = cls_gain
    hyp["perspective"] = 0.0  # non cercato, coerente col default yolov5/v7
    # optimizer="auto" (default ultralytics) ignora lr0/momentum passati
    # esplicitamente -- verificato in sessione con un run reale.
    hyp["optimizer"] = "SGD"
    assert not (_YOLOV8_EXCLUDED_HYP_KEYS & hyp.keys())
    return hyp
