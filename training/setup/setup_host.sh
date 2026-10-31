#!/bin/bash
# Setup ambiente di training sull'HOST (RTX 5070, CUDA 12.8+): yolov5,
# yolov7-tiny, yolov8n (via ultralytics) e la pipeline NNI in nni_pipeline/.
# Uso:  cd ~/jetson_work/cv_training && ./setup_host.sh
#
# Usa uv (gestore python standalone) perche' il python di sistema e' un 3.14
# senza venv/pip e PyTorch supporta al massimo il 3.13.
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installo uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==== venv Python 3.12 ===="
uv venv --python 3.12 .venv
source .venv/bin/activate

echo "==== PyTorch cu128 (richiesto dalla RTX 5070 / sm_120) ===="
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "==== YOLOv5 (master: mantenuto e compatibile con torch recente) ===="
[ -d yolov5 ] || git clone https://github.com/ultralytics/yolov5.git
uv pip install -r yolov5/requirements.txt
uv pip install onnx onnxslim   # per l'export verso la Jetson

echo "==== YOLOv7 (WongKinYiu, testa anchor-based come v5 -- necessaria per"
echo "     yolov7-tiny, che sulla Nano usa lo stesso decoder postprocess() di v5) ===="
[ -d yolov7 ] || git clone https://github.com/WongKinYiu/yolov7.git
# yolov7/requirements.txt pin numpy<1.24 / protobuf<4.21.3 (pre-Python-3.12,
# la build di numpy vecchio fallisce senza distutils): override per tenere
# le versioni moderne gia' installate da torch/yolov5/onnx.
uv pip install -r yolov7/requirements.txt --override requirements-overrides-yolov7.txt
[ -f yolov7/yolov7-tiny.pt ] || \
    curl -L -o yolov7/yolov7-tiny.pt \
    https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
# richiesto dal nuovo torch.onnx.export basato su dynamo (torch >=2.x);
# senza, l'export ONNX di yolov7/export.py fallisce con "No module named
# 'onnxscript'" (yolov5's export.py invece funziona gia' senza, verificare
# se cambia in futuro).
uv pip install onnxscript onnxsim

echo "==== ultralytics (pacchetto pip -- SOLO per yolov8n, testa anchor-free"
echo "     gia' verificata sulla Nano via postprocess_v8(). NON usare questo"
echo "     pacchetto per yolov5/yolov7: le sue varianti 'u' (es. yolov5nu) sono"
echo "     anchor-free e romperebbero silenziosamente postprocess() sulla Nano ===="
uv pip install ultralytics
echo "---- verifica che ultralytics non abbia declassato il build cu128 di torch ----"
python - <<'EOF'
import torch
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "ultralytics potrebbe aver declassato torch a un build CPU-only: "
    "reinstalla torch/torchvision con --index-url cu128 come sopra"
)
EOF

echo "==== NNI (Microsoft Neural Network Intelligence -- orchestrazione HPO/AutoML) ===="
uv pip install nni
# nni fissa filelock<3.12 (pin datato) E il suo nnictl si blocca in un
# deadlock silenzioso se un filelock piu' recente e' installato (verificato
# in sessione: SimplePreemptiveLock in nni/tools/nnictl/common_utils.py
# eredita da filelock.SoftFileLock, il cui comportamento interno e' cambiato
# in modo incompatibile in filelock >=3.12). NON alzare filelock oltre
# questo pin per "risolvere" l'import di ultralytics (che richiede
# AsyncFileLock, aggiunto dopo 3.12): quello si sistema con lo shim in
# common/compat.py + le patch in yolov5/train.py e yolov5/export.py, non
# alzando la versione di filelock.
uv pip install "filelock<3.12"

echo "==== FiftyOne (fetch Open Images V7 / Sama-COCO filtrati per classe) ===="
uv pip install fiftyone

echo "==== Verifica ===="
python - <<'EOF'
import torch
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU  :", torch.cuda.get_device_name(0))
EOF

echo
echo "Fatto. Attiva con:  source ~/jetson_work/cv_training/.venv/bin/activate"
echo "Quick-path manuale (solo yolov5): segui README_TRAINING.md"
echo "Pipeline NNI (tutte e 4 le architetture): segui nni_pipeline/README.md"
