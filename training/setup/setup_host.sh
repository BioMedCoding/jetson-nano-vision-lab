#!/bin/bash
# Set up the training environment on a PC: training/.venv with Python 3.12,
# PyTorch for your hardware, and every other package pinned by
# setup/requirements.lock (yolov5/yolov7 deps, ultralytics, NNI, FiftyOne,
# ONNX tooling), with the one deliberate override in setup/overrides.txt.
#
# Usage:  training/setup/setup_host.sh [--torch-backend auto|cu128|cu126|cu118|cpu|rocm6.3|...]
#
# Tested on: Ubuntu, x86_64, NVIDIA RTX 5070 (Blackwell, needs CUDA 12.8
# wheels, i.e. cu128). --torch-backend auto (default) lets uv pick the
# PyTorch build matching your NVIDIA driver, so older GPUs get a build that
# supports them; with no GPU you get the CPU build (training works, slowly;
# the NNI search needs a GPU). If auto picks a build that misbehaves, force
# the combination verified for this repo: --torch-backend cu128.
#
# Uses uv (https://docs.astral.sh/uv/) to manage Python itself: it does not
# depend on, or touch, the system Python.
set -e

TRAINING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND=auto
while [ $# -gt 0 ]; do
    case "$1" in
        --torch-backend) BACKEND="$2"; shift 2 ;;
        --torch-backend=*) BACKEND="${1#*=}"; shift ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
cd "$TRAINING_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it (one static binary) with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "or see https://docs.astral.sh/uv/getting-started/installation/ , then re-run this."
    exit 1
fi

echo "==== [1/4] Python 3.12 virtual environment (training/.venv) ===="
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
PY=.venv/bin/python

echo "==== [2/4] PyTorch 2.11 for backend '$BACKEND' + everything in setup/requirements.lock ===="
# One resolution for everything, so torch and the packages that depend on it
# can never disagree about which build is installed.
uv pip install --python "$PY" --torch-backend "$BACKEND" \
    torch==2.11.0 torchvision==0.26.0 \
    -r setup/requirements.lock --override setup/overrides.txt

echo "==== [3/4] MongoDB for FiftyOne (only used by the NNI dataset builder) ===="
# fiftyone-db ships a mongod only for the Linux releases it knows; on newer
# ones (e.g. Ubuntu 26.04) FiftyOne fails with "MongoDB could not be installed
# on your system". Then install MongoDB's own 8.0.17 build (the one verified
# here), checking its sha256. Not vendored: ~100 MB, under MongoDB's SSPL.
MONGO_BIN_DIR="$("$PY" -c 'import fiftyone, os; print(os.path.join(os.path.dirname(fiftyone.__file__), "db", "bin"))')"
if [ -x "$MONGO_BIN_DIR/mongod" ]; then
    echo "mongod present: $MONGO_BIN_DIR/mongod"
elif [ "$(uname -m)" = "x86_64" ]; then
    MONGO_TGZ=mongodb-linux-x86_64-ubuntu2404-8.0.17.tgz
    MONGO_SHA=fe7bccea2ac1eed16867e9ae5a60481455ee30796f285fe67d0d02a6c2abbdda
    TMP=$(mktemp -d)
    echo "Downloading $MONGO_TGZ (~100 MB)..."
    if curl -fsSL -o "$TMP/$MONGO_TGZ" "https://fastdl.mongodb.org/linux/$MONGO_TGZ" \
       && echo "$MONGO_SHA  $TMP/$MONGO_TGZ" | sha256sum -c --quiet; then
        tar -xzf "$TMP/$MONGO_TGZ" -C "$TMP" --wildcards '*/bin/mongod'
        mkdir -p "$MONGO_BIN_DIR"
        install -m 755 "$TMP"/mongodb-*/bin/mongod "$MONGO_BIN_DIR/mongod"
        echo "Installed $MONGO_BIN_DIR/mongod"
    else
        echo "WARNING: could not get MongoDB; the NNI dataset builder will not work"
        echo "         (quickstart.py does not need it). See nni/README.md."
    fi
    rm -rf "$TMP"
else
    echo "WARNING: no mongod for this platform; see nni/README.md (only NNI needs it)."
fi

echo "==== [4/4] Check ===="
"$PY" - <<'EOF'
import torch, ultralytics, onnx, filelock
cuda = torch.cuda.is_available()
print("torch      :", torch.__version__, "| CUDA:", cuda,
      "| GPU:", torch.cuda.get_device_name(0) if cuda else "-")
print("ultralytics:", ultralytics.__version__, "| onnx:", onnx.__version__,
      "| filelock:", filelock.__version__, "(must be < 3.12 for NNI)")
if cuda:
    x = torch.randn(256, 256, device="cuda")
    assert float((x @ x).abs().sum()) > 0
    print("GPU compute: OK")
else:
    print("No CUDA GPU: training will run on the CPU (slow). Set CVTRAIN_DEVICE to override.")
EOF

cat <<EOF

Done. Activate with:   source $TRAINING_DIR/.venv/bin/activate
Then:                  python quickstart.py --help      (one model on your dataset)
                       ./menu.sh                         (everything, including the NNI search)
EOF
