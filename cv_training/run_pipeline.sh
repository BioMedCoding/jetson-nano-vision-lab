#!/bin/bash
# Pipeline di fine-tuning YOLOv5 sull'host (RTX 5070), end-to-end: valida il
# dataset, allena, esporta ONNX alla stessa risoluzione del training, prepara
# (o esegue) il deploy sulla Jetson. Wrapper di convenienza attorno ai passi
# gia' documentati in README_TRAINING.md, che resta la spiegazione di
# dettaglio per debug manuale passo-passo.
#
# Uso:
#   ./run_pipeline.sh dataset/ yolov5n.pt
#   ./run_pipeline.sh dataset/ yolov5s.pt --img 416 --epochs 150 --batch 32 --name mio_run
#   ./run_pipeline.sh dataset/ yolov5n.pt --deploy matteo@10.42.0.24
#
# Richiede l'ambiente gia' pronto (una tantum): ./setup_host.sh
# NOTA: --name riutilizzato su run successivi sovrascrive in-place la stessa
# cartella runs/train/<name> (--exist-ok), cosi' il percorso di best.pt resta
# prevedibile; usa nomi diversi per tenere run distinti.
set -e

BASE="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 2 ]; then
    echo "Uso: ./run_pipeline.sh <dataset_dir> <weights.pt> [opzioni]"
    exit 1
fi
DATASET="$1"; WEIGHTS="$2"; shift 2

IMG=416
EPOCHS=100
BATCH=64
NAME="run"
DEPLOY_HOST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --img) IMG="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --deploy) DEPLOY_HOST="$2"; shift 2 ;;
        *) echo "Argomento sconosciuto: $1"; exit 1 ;;
    esac
done

[ -d "$DATASET" ] || { echo "ERRORE: dataset non trovato: $DATASET"; exit 1; }
DATASET="$(cd "$DATASET" && pwd)"
[ -f "$BASE/.venv/bin/activate" ] || { echo "ERRORE: venv mancante. Esegui prima ./setup_host.sh"; exit 1; }
[ -d "$BASE/yolov5" ] || { echo "ERRORE: yolov5/ mancante. Esegui prima ./setup_host.sh"; exit 1; }
source "$BASE/.venv/bin/activate"

echo "==== [1/4] Valido il dataset ===="
python3 - "$DATASET" << 'PYEOF'
import glob
import os
import sys

import yaml

dataset_dir = sys.argv[1]
data_yaml = os.path.join(dataset_dir, "data.yaml")
ok = True

for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    p = os.path.join(dataset_dir, sub)
    if not os.path.isdir(p):
        print("ERRORE: manca {}".format(p))
        ok = False

if not os.path.isfile(data_yaml):
    print("ERRORE: manca {}".format(data_yaml))
    ok = False
else:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names")
    n_names = len(names) if isinstance(names, (list, dict)) else 0
    if not n_names:
        print("ERRORE: 'names' assente o vuoto in data.yaml")
        ok = False
    else:
        print("Classi in data.yaml: {}".format(n_names))
        max_seen = -1
        n_labels, n_empty = 0, 0
        for split in ("train", "val"):
            for txt in glob.glob(os.path.join(dataset_dir, "labels", split, "*.txt")):
                n_labels += 1
                with open(txt) as lf:
                    lines = [l.strip() for l in lf if l.strip()]
                if not lines:
                    n_empty += 1
                for line in lines:
                    max_seen = max(max_seen, int(line.split()[0]))
        print("File label trovati: {} ({} vuoti)".format(n_labels, n_empty))
        if max_seen >= n_names:
            print("ERRORE: class id massimo nelle label ({}) >= classi in "
                 "data.yaml ({})".format(max_seen, n_names))
            ok = False
        if n_labels and n_empty / n_labels > 0.5:
            print("ATTENZIONE: oltre meta' dei file label e' vuota "
                 "({}/{})".format(n_empty, n_labels))

    for split in ("train", "val"):
        imgs = glob.glob(os.path.join(dataset_dir, "images", split, "*"))
        print("images/{}: {} file".format(split, len(imgs)))

sys.exit(0 if ok else 1)
PYEOF

echo "==== [2/4] Training (yolov5/train.py, ${EPOCHS} epoche, img ${IMG}) ===="
cd "$BASE/yolov5"
python train.py --img "$IMG" --batch "$BATCH" --epochs "$EPOCHS" \
    --data "$DATASET/data.yaml" --weights "$WEIGHTS" --name "$NAME" --exist-ok

BEST="$BASE/yolov5/runs/train/$NAME/weights/best.pt"
[ -f "$BEST" ] || { echo "ERRORE: best.pt non trovato in runs/train/$NAME/weights/"; exit 1; }

echo "==== [3/4] Export ONNX (stessa risoluzione del training: ${IMG}x${IMG}) ===="
python export.py --weights "$BEST" --include onnx --imgsz "$IMG" "$IMG" \
    --opset 12 --simplify
ONNX="${BEST%.pt}.onnx"
[ -f "$ONNX" ] || { echo "ERRORE: export ONNX fallito"; exit 1; }

CLASSES_TXT="$(dirname "$ONNX")/classes.txt"
python3 - "$DATASET/data.yaml" "$CLASSES_TXT" << 'PYEOF'
import sys

import yaml

with open(sys.argv[1]) as f:
    names = yaml.safe_load(f)["names"]
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
with open(sys.argv[2], "w") as f:
    f.write("\n".join(names) + "\n")
PYEOF

echo "==== [4/4] Deploy ===="
if [ -n "$DEPLOY_HOST" ]; then
    KEY="${JETSON_KEY:-$HOME/.ssh/jetson_session/jetson_key}"
    echo "Copio su $DEPLOY_HOST:~/cv_projects/models/ ..."
    scp -i "$KEY" "$ONNX" "$DEPLOY_HOST:~/cv_projects/models/custom.onnx"
    scp -i "$KEY" "$CLASSES_TXT" "$DEPLOY_HOST:~/cv_projects/models/classes.txt"
    echo
    echo "Fatto. Sulla Jetson ora:"
    echo "  cd ~/cv_projects/03_tensorrt && ./build_engine.sh ../models/custom.onnx $IMG"
    echo "  python demo_server.py --engine ../models/custom_${IMG}_fp16.engine --names ../models/classes.txt --headless"
else
    echo "Fatto."
    echo "  ONNX:    $ONNX"
    echo "  Classi:  $CLASSES_TXT"
    echo "Per portarlo sulla Jetson (sostituisci l'host):"
    echo "  scp -i ~/.ssh/jetson_session/jetson_key $ONNX matteo@<jetson>:~/cv_projects/models/custom.onnx"
    echo "  scp -i ~/.ssh/jetson_session/jetson_key $CLASSES_TXT matteo@<jetson>:~/cv_projects/models/classes.txt"
    echo "  # poi sulla Jetson: ./build_engine.sh ../models/custom.onnx $IMG"
fi
