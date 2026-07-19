#!/bin/bash
# Costruisce un engine TensorRT FP16 da pesi YOLOv5 (.pt) o da un ONNX gia' pronto.
# Da eseguire SULLA JETSON (l'engine e' legato all'hardware su cui viene creato).
#
# Uso:
#   ./build_engine.sh                          # yolov5n.pt, 416
#   ./build_engine.sh ../models/yolov5s.pt 640
#   ./build_engine.sh ../models/custom.onnx 416   # ONNX esportato sull'host
#   ./build_engine.sh ../models/yolov7tiny.onnx 416 --postprocess v7
#
# --postprocess {v5,v7,v8}: forza il decoder da usare nella dashboard (scritto
# nel sidecar .json accanto all'engine). Se omesso, viene dedotto dal prefisso
# del nome file (yolov8* -> v8, altrimenti v5): usare il flag per i modelli
# custom il cui nome non lo rende ovvio.
#
# Primo utilizzo: installa onnx/pycuda nel venv (pycuda compila, ~15 min).
set -e

POSTPROCESS_OVERRIDE=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --postprocess) POSTPROCESS_OVERRIDE="$2"; shift 2 ;;
        --postprocess=*) POSTPROCESS_OVERRIDE="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]}"

BASE="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS="${1:-$BASE/models/yolov5n.pt}"
SIZE="${2:-416}"
YOLO_DIR="${YOLO_DIR:-$HOME/yolov5}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERRORE: questo script va eseguito sulla Jetson, non sull'host."; exit 1
fi
[ -f "$BASE/venv/bin/activate" ] || { echo "ERRORE: venv mancante, esegui prima setup/setup_jetson.sh"; exit 1; }
source "$BASE/venv/bin/activate"
export OPENBLAS_CORETYPE=ARMV8
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1  # evita "cannot allocate memory in static TLS block" (cv2 dopo torch su aarch64)

# dipendenze per l'export/inferenza TRT (no-op se gia' installate)
python -c "import onnx" 2>/dev/null || pip install -r "$BASE/setup/requirements_trt.txt"

case "$WEIGHTS" in
  *.onnx)
    ONNX="$WEIGHTS"
    ;;
  *.pt)
    echo "== Export ONNX (opset 12, ${SIZE}x${SIZE}) =="
    python "$YOLO_DIR/export.py" --weights "$WEIGHTS" --include onnx \
        --imgsz "$SIZE" "$SIZE" --opset 12 --device cpu
    ONNX="${WEIGHTS%.pt}.onnx"
    ;;
  *)
    echo "ERRORE: atteso un file .pt o .onnx"; exit 1
    ;;
esac
[ -f "$ONNX" ] || { echo "ERRORE: ONNX non trovato: $ONNX"; exit 1; }

ENGINE="${ONNX%.onnx}_${SIZE}_fp16.engine"
echo "== Build engine TensorRT FP16 (puo' richiedere 10-20 minuti) =="
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 --workspace=1024

# Sidecar di metadata per la dashboard (demo_server.py / dashboard_server.py):
# indica quale decoder di post-processing usare per questo engine.
BASENAME=$(basename "$ENGINE")
if [ -n "$POSTPROCESS_OVERRIDE" ]; then
    POSTPROCESS="$POSTPROCESS_OVERRIDE"
else
    case "$BASENAME" in
        yolov8*) POSTPROCESS="v8" ;;
        *)       POSTPROCESS="v5" ;;
    esac
fi
SIDECAR="${ENGINE%.engine}.json"
cat > "$SIDECAR" << JSONEOF
{"postprocess": "$POSTPROCESS", "input_size": $SIZE, "label": "${BASENAME%.engine}"}
JSONEOF

echo
echo "Engine creato: $ENGINE"
echo "Metadata: $SIDECAR (postprocess=$POSTPROCESS)"
echo "Prova con:  python $BASE/03_tensorrt/trt_detect.py --engine $ENGINE"
echo "Oppure con la dashboard: python $BASE/03_tensorrt/demo_server.py --engine $ENGINE --headless"
