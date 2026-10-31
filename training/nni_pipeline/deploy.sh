#!/bin/bash
# Tail di deploy riutilizzabile da promote.py (Stage B), estratta a parte
# invece di modificare run_pipeline.sh (che resta invariato, quick-path
# solo-yolov5) o di shellare sull'intero script (che trascinerebbe anche il
# training). A differenza di run_pipeline.sh, qui il decoder da usare sulla
# Nano (--postprocess) e' SEMPRE esplicito: con 4 architetture diverse in
# gioco il fallback per nome file di build_engine.sh (yolov8*->v8, else v5)
# non e' affidabile.
#
# Uso:
#   ./deploy.sh <onnx> <classes.txt> <img_size> <v5|v7|v8> [user@host]
set -e

if [ $# -lt 4 ]; then
    echo "Uso: ./deploy.sh <onnx> <classes.txt> <img_size> <v5|v7|v8> [user@host]"
    exit 1
fi
ONNX="$1"; CLASSES_TXT="$2"; IMG="$3"; POSTPROCESS="$4"; DEPLOY_HOST="${5:-}"

[ -f "$ONNX" ] || { echo "ERRORE: onnx non trovato: $ONNX"; exit 1; }
[ -f "$CLASSES_TXT" ] || { echo "ERRORE: classes.txt non trovato: $CLASSES_TXT"; exit 1; }
case "$POSTPROCESS" in v5|v7|v8) ;; *) echo "ERRORE: postprocess deve essere v5, v7 o v8: $POSTPROCESS"; exit 1 ;; esac

if [ -n "$DEPLOY_HOST" ]; then
    KEY="${JETSON_KEY:-$HOME/.ssh/jetson_session/jetson_key}"
    echo "Copio su $DEPLOY_HOST:~/cv_projects/models/ ..."
    scp -i "$KEY" "$ONNX" "$DEPLOY_HOST:~/cv_projects/models/custom.onnx"
    scp -i "$KEY" "$CLASSES_TXT" "$DEPLOY_HOST:~/cv_projects/models/classes.txt"
    echo
    echo "Fatto. Sulla Jetson ora:"
    echo "  cd ~/cv_projects/03_tensorrt && ./build_engine.sh ../models/custom.onnx $IMG --postprocess $POSTPROCESS"
    echo "  python demo_server.py --engine ../models/custom_${IMG}_fp16.engine --names ../models/classes.txt --headless"
else
    echo "Fatto."
    echo "  ONNX:    $ONNX"
    echo "  Classi:  $CLASSES_TXT"
    echo "Per portarlo sulla Jetson (sostituisci l'host):"
    echo "  scp -i ~/.ssh/jetson_session/jetson_key $ONNX matteo@<jetson>:~/cv_projects/models/custom.onnx"
    echo "  scp -i ~/.ssh/jetson_session/jetson_key $CLASSES_TXT matteo@<jetson>:~/cv_projects/models/classes.txt"
    echo "  # poi sulla Jetson: ./build_engine.sh ../models/custom.onnx $IMG --postprocess $POSTPROCESS"
fi
