#!/bin/bash
# One-shot setup of the Python environment on the Jetson Nano.
# Run ON the Jetson, from anywhere:   nano/setup/setup_jetson.sh
#
# Target: JetPack 4.6.1 (L4T R32.7.1, Ubuntu 18.04, Python 3.6), the last
# JetPack for the original Nano. Everything Python comes from this repo
# (nano/vendor/wheels, pinned by nano/setup/requirements.lock): no PyPI, no
# NVIDIA download links, no compilation.
#
# What it does:
#   1. apt: the few system libraries the torch wheel links against (skip with --no-apt)
#   2. reassembles the large vendored files (torch wheel) from their chunks
#   3. creates nano/.venv: Python 3.6 with --system-site-packages, because
#      OpenCV (cv2) and TensorRT exist only as JetPack system packages
#   4. installs every pip package offline from nano/vendor/wheels
#   5. installs jetson-stats (jtop) system-wide if missing (dashboard System tab)
#   6. verifies the result
#
# Options:
#   --no-apt    skip step 1 (offline machine that already has the packages)
#   --swap      also create a 4 GB /swapfile (only useful for heavy builds)
set -e

NANO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$(cd "$NANO_DIR/.." && pwd)"
VENV="$NANO_DIR/.venv"
WHEELS="$NANO_DIR/vendor/wheels"
WANT_APT=1; WANT_SWAP=0
for a in "$@"; do
    case "$a" in
        --no-apt) WANT_APT=0 ;;
        --swap) WANT_SWAP=1 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "Unknown option: $a"; exit 1 ;;
    esac
done

if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERROR: run this on the Jetson Nano, not on your PC."; exit 1
fi
if ! grep -q "R32 (release), REVISION: 7" /etc/nv_tegra_release; then
    echo "WARNING: this was built and tested on L4T R32.7.x (JetPack 4.6.x). Found:"
    head -1 /etc/nv_tegra_release
    echo "The vendored wheels and TensorRT engines may not work. Continuing in 5 s..."
    sleep 5
fi

echo "==== [1/6] System packages (apt) ===="
if [ "$WANT_APT" = 1 ]; then
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-pip \
        libopenblas-base libopenmpi-dev libomp-dev
else
    echo "Skipped (--no-apt)."
fi
if [ "$WANT_SWAP" = 1 ] && [ ! -f /swapfile ]; then
    echo "Creating a 4 GB /swapfile..."
    sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
    sudo mkswap /swapfile && sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "==== [2/6] Reassembling large vendored files ===="
python3 "$REPO_DIR/tools/largefiles.py" join nano/

echo "==== [3/6] Virtual environment (Python 3.6, system site packages) ===="
if [ ! -f "$VENV/bin/activate" ]; then
    python3.6 -m venv --system-site-packages "$VENV"
fi
# Workarounds that must be in the environment before Python starts
# (see nano/jetson_cv/runtime.py for the why):
add_env() { grep -q "$1" "$VENV/bin/activate" || echo "export $1=$2" >> "$VENV/bin/activate"; }
add_env OPENBLAS_CORETYPE ARMV8
add_env LD_PRELOAD /usr/lib/aarch64-linux-gnu/libgomp.so.1
# Ignore packages in ~/.local: a stray `pip install --user torch` would
# otherwise silently shadow the venv's copy.
add_env PYTHONNOUSERSITE 1
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==== [4/6] Python packages (offline, from nano/vendor/wheels) ===="
PIP="pip install --no-index --find-links $WHEELS"
$PIP "pip==21.3.1" "setuptools==59.6.0" "wheel==0.37.1"
$PIP -r "$NANO_DIR/setup/requirements.lock"
$PIP "$WHEELS"/torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl \
     "$WHEELS"/torchvision-0.11.0a0+fa347eb-cp36-cp36m-linux_aarch64.whl

# yolov5 v6.0 tries to download Arial.ttf from a URL that no longer exists:
# put a local TTF where it looks, so it never touches the network.
mkdir -p ~/.config/Ultralytics
[ -f ~/.config/Ultralytics/Arial.ttf ] || \
    cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ~/.config/Ultralytics/Arial.ttf

echo "==== [5/6] jetson-stats (jtop) ===="
if python3 -c "import jtop" 2>/dev/null; then
    echo "jtop already installed."
else
    echo "Installing jetson-stats 4.3.2 system-wide (needs sudo)..."
    sudo -H pip3 install --no-index --find-links "$WHEELS" \
        jetson-stats==4.3.2 distro==1.9.0 smbus2==0.6.1
    sudo systemctl restart jtop.service || true
    sudo usermod -aG jtop "$USER"
    echo "NOTE: log out and back in once, so your user joins the 'jtop' group."
fi
if ! id -nG "$USER" | grep -qw video; then
    sudo usermod -aG video "$USER"
    echo "NOTE: added $USER to the 'video' group (camera access): log out and back in."
fi

echo "==== [6/6] Verification ===="
python - <<'EOF'
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
import numpy, torch, torchvision, cv2, tensorrt, pycuda.driver, onnx
ok = torch.cuda.is_available() and "/.venv/" in torch.__file__
print("numpy       :", numpy.__version__)
print("torch       :", torch.__version__, "| CUDA:", torch.cuda.is_available(), "|", torch.__file__)
print("torchvision :", torchvision.__version__)
print("cv2         :", cv2.__version__)
print("tensorrt    :", tensorrt.__version__)
print("onnx        :", onnx.__version__)
print("\nALL GOOD" if ok else "\nCHECK FAILED: torch must come from nano/.venv and see CUDA")
raise SystemExit(0 if ok else 1)
EOF

cat <<EOF

Setup complete. In each new shell:
    source $VENV/bin/activate

Then, from $NANO_DIR:
    python3 apps/camera_test.py --headless      # camera only, ~60 fps
    python3 apps/trt_detect.py --headless       # TensorRT detection, ~30 fps
    python3 apps/dashboard.py                   # web dashboard on :8090

Start the dashboard at boot:  nano/setup/install_service.sh
Max performance:              sudo nvpmodel -m 0 && sudo jetson_clocks
EOF
