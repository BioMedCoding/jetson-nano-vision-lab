#!/bin/bash
# Deploy a model folder (model.onnx + model.json [+ labels.txt]) to the Jetson,
# build its TensorRT engine there, and smoke-test it.
#
# Usage:  host/deploy_model.sh training/export/<model-id>
# Config: host/jetson.env (see jetson.env.example)
#
# Then on the Jetson:  python3 apps/dashboard.py --model <model-id>
# (or pick it from the dashboard's model list, it shows up after a reload).
set -e
. "$(dirname "$0")/common.sh"

SRC="${1:?usage: host/deploy_model.sh <model-folder>}"
SRC="$(cd "$SRC" && pwd)"
ID="$(basename "$SRC")"
[ -f "$SRC/model.json" ] && [ -f "$SRC/model.onnx" ] || {
    echo "ERROR: $SRC needs model.json and model.onnx (see docs/model-contract.md)"; exit 1; }

echo "== Copying $ID to $JETSON_HOST:$JETSON_DIR/nano/models/$ID"
ssh "$JETSON_HOST" "mkdir -p $JETSON_DIR/nano/models/$ID"
rsync -a --info=stats0 "$SRC/" "$JETSON_HOST:$JETSON_DIR/nano/models/$ID/"

echo "== Building the TensorRT engine on the Jetson (5-20 min)"
ssh "$JETSON_HOST" "cd $JETSON_DIR/nano && .venv/bin/python -u apps/build_engine.py $ID"

echo "== Smoke test"
ssh "$JETSON_HOST" "cd $JETSON_DIR/nano && .venv/bin/python -u tests/test_models.py $ID" \
    | grep -v "TRT\] \[W\]" || true

echo
echo "Done. On the Jetson:  python3 apps/trt_detect.py --model $ID"
echo "                 or:  python3 apps/dashboard.py --model $ID"
