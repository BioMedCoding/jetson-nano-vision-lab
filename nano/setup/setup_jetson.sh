#!/bin/bash
# Setup completo dell'ambiente YOLOv5 sulla Jetson Nano (JetPack 4.6.1, Ubuntu 18.04).
# Da eseguire SULLA JETSON:   cd ~/cv_projects/setup && ./setup_jetson.sh
#
# Cosa fa:
#   1. installa i pacchetti apt necessari (incluse le librerie python "pesanti"
#      che il venv erediterà: matplotlib/pandas/seaborn, mai da compilare)
#   2. crea swap da 4 GB se assente (serve per eventuali compilazioni)
#   3. crea il venv Python 3.6 con --system-site-packages in ~/cv_projects/venv
#   4. installa numpy -> torch (wheel NVIDIA) -> torchvision (wheel Qengineering)
#   5. installa i requirements pip e verifica tutto
#
# Opzioni:
#   --trt       installa anche onnx+pycuda per la pipeline TensorRT (pycuda
#               compila da sorgente: ~15 minuti in piu')
#   --no-swap   non creare lo swapfile
set -e

BASE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$BASE/venv"
WHEELS="$BASE/setup/wheels"
WANT_TRT=0; WANT_SWAP=1
for a in "$@"; do
    case "$a" in
        --trt) WANT_TRT=1 ;;
        --no-swap) WANT_SWAP=0 ;;
    esac
done

if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERRORE: questo script va eseguito sulla Jetson Nano, non sull'host."
    exit 1
fi

echo "==== [1/7] Pacchetti di sistema (apt) ===="
sudo apt-get update
sudo apt-get install -y \
    python3-venv python3-dev python3-pip \
    libopenblas-base libopenmpi-dev libomp-dev \
    libjpeg-dev zlib1g-dev \
    python3-matplotlib python3-pandas python3-seaborn python3-pil

if [ "$WANT_SWAP" = 1 ]; then
    echo "==== [2/7] Swap ===="
    SWAP_KB=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
    if [ "$SWAP_KB" -lt 3800000 ]; then
        if [ ! -f /swapfile ]; then
            echo "Creo /swapfile da 4 GB..."
            sudo fallocate -l 4G /swapfile
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            grep -q '^/swapfile' /etc/fstab || \
                echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        else
            sudo swapon /swapfile 2>/dev/null || true
        fi
    else
        echo "Swap gia' sufficiente."
    fi
else
    echo "==== [2/7] Swap: saltato (--no-swap) ===="
fi

echo "==== [3/7] Virtualenv Python 3.6 ===="
# --system-site-packages e' FONDAMENTALE: fa ereditare al venv l'OpenCV di
# sistema (compilato con GStreamer per la camera CSI), tensorrt e le librerie
# apt. Il vecchio venv 'jetson_yolo' falliva perche' era Python 3.8: su
# JetPack 4 torch CUDA e cv2 esistono solo per Python 3.6.
if [ ! -f "$VENV/bin/activate" ]; then
    python3.6 -m venv --system-site-packages "$VENV"
fi
# workaround "Illegal instruction" di numpy/OpenBLAS su Nano: sempre attivo nel venv
grep -q OPENBLAS_CORETYPE "$VENV/bin/activate" || \
    echo 'export OPENBLAS_CORETYPE=ARMV8' >> "$VENV/bin/activate"
# workaround "cannot allocate memory in static TLS block" quando cv2 viene
# importato dopo torch (libgomp aarch64): preload sempre attivo nel venv
grep -q 'LD_PRELOAD=.*libgomp' "$VENV/bin/activate" || \
    echo 'export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1' >> "$VENV/bin/activate"
source "$VENV/bin/activate"
export OPENBLAS_CORETYPE=ARMV8

echo "==== [4/7] pip + numpy + torch + torchvision ===="
# ultime versioni dei tool compatibili con Python 3.6
pip install --upgrade "pip==21.3.1" "setuptools==59.6.0" "wheel==0.37.1"
pip install "numpy==1.19.5"

# la wheel torch (233 MB) vive nella repo splittata in chunk <100 MB
# (limite file di GitHub): se manca intera, ricomponila e verifica lo sha256
if ! ls "$WHEELS"/torch-1.10*.whl >/dev/null 2>&1 && \
   ls "$WHEELS"/torch-split/*.part-00 >/dev/null 2>&1; then
    echo "Ricompongo la wheel torch dai chunk..."
    P0=$(ls "$WHEELS"/torch-split/*.part-00)
    cat "${P0%.part-00}".part-* > "$WHEELS/$(basename "${P0%.part-00}")"
    (cd "$WHEELS" && sha256sum -c torch-split/SHA256SUM)
fi
TORCH_WHL=$(ls "$WHEELS"/torch-1.10*.whl 2>/dev/null | head -1)
TV_WHL=$(ls "$WHEELS"/torchvision-0.11*.whl 2>/dev/null | head -1)
[ -n "$TORCH_WHL" ] || { echo "ERRORE: wheel torch mancante in $WHEELS"; exit 1; }
python -c "import torch" 2>/dev/null || pip install "$TORCH_WHL"

if ! python -c "import torchvision" 2>/dev/null; then
    if [ -n "$TV_WHL" ] && pip install "$TV_WHL"; then
        echo "torchvision installato dalla wheel precompilata."
    else
        echo "Wheel torchvision non disponibile/installabile: compilo da sorgente (~60-90 min)..."
        sudo apt-get install -y git
        rm -rf /tmp/torchvision-src
        git clone --branch v0.11.1 --depth 1 https://github.com/pytorch/vision /tmp/torchvision-src
        cd /tmp/torchvision-src
        export BUILD_VERSION=0.11.1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="5.3"
        export PATH=/usr/local/cuda/bin:$PATH
        python setup.py install
        cd - >/dev/null
    fi
fi

echo "==== [5/7] Requirements pip ===="
pip install -r "$BASE/setup/requirements_jetson.txt"
if [ "$WANT_TRT" = 1 ]; then
    echo "-- extra TensorRT (pycuda compila: pazienza...) --"
    export PATH=/usr/local/cuda/bin:$PATH
    pip install onnx==1.10.2 protobuf==3.19.6
    # NO build isolation: altrimenti pip compila numpy 1.12 come build-dep
    # (vincolo python 3.6) e fallisce su xlocale.h con glibc recenti
    pip install pycuda==2021.1 --no-build-isolation
fi

# yolov5 all'import prova a scaricare Arial.ttf da un URL ormai morto (308):
# mettiamo un TTF locale al posto giusto cosi' non tenta mai la rete
mkdir -p ~/.config/Ultralytics
[ -f ~/.config/Ultralytics/Arial.ttf ] || \
    cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ~/.config/Ultralytics/Arial.ttf

echo "==== [6/7] Verifica ===="
python - <<'EOF'
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
ok = True
import numpy, torch
print("numpy       :", numpy.__version__)
print("torch       :", torch.__version__, "| CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    ok = False
    print("  !! CUDA NON disponibile: qualcosa non va con la wheel torch")
import torchvision
print("torchvision :", torchvision.__version__)
import torchvision.ops  # verifica gli op compilati (NMS)
import cv2
print("cv2         :", cv2.__version__, "| GStreamer:",
      "YES" in cv2.getBuildInformation().split("GStreamer")[1][:40])
try:
    import tensorrt
    print("tensorrt    :", tensorrt.__version__)
except ImportError:
    print("tensorrt    : non visibile (ok se non usi la pipeline TRT)")
print("\nTUTTO OK" if ok else "\nVERIFICA FALLITA: vedi messaggi sopra")
EOF

echo "==== [7/7] jetson-stats (jtop, per il pannello Sistema della dashboard) ===="
if python -c "import jtop" 2>/dev/null; then
    echo "jtop OK (gia' installato e importabile nel venv)."
else
    echo "jtop non importabile in questo venv: su un sistema NUOVO installalo con:"
    echo "  sudo -H pip3 install -U jetson-stats"
    echo "  sudo systemctl restart jtop.service"
    echo "  sudo usermod -aG jtop \$USER"
    echo "  (poi disconnettiti/riconnettiti: serve una nuova sessione per il gruppo)"
    echo "Questi comandi richiedono sudo + un nuovo login: eseguili tu stesso, non li lancia questo script."
fi

echo
echo "Setup completato. Per usare l'ambiente:"
echo "  source ~/cv_projects/venv/bin/activate"
echo "  cd ~/cv_projects/01_camera_test && python camera_test.py"
echo
echo "Dashboard sperimentale (sistema/statistiche/controlli/tracker):"
echo "  cd ~/cv_projects/03_tensorrt && python demo_server.py --engine ../models/yolov5n_416_fp16.engine --headless"
echo "Per farla partire da sola al boot: setup/install_demo_service.sh"
echo
echo "Prestazioni massime:  sudo nvpmodel -m 0 && sudo jetson_clocks"
