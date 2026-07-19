# Guida al setup — YOLOv5 su Jetson Nano (JetPack 4.6.1)

Guida per creare da zero l'ambiente che fa girare la detection sulla camera
OV5647. Vale per: **Jetson Nano, L4T R32.7.1 / JetPack 4.6.1, Ubuntu 18.04**.

## Perché Python 3.6 (e perché prima non funzionava)

Su JetPack 4.x le uniche versioni con accelerazione hardware sono legate a
Python 3.6:

| Componente | Da dove arriva | Python |
|---|---|---|
| PyTorch 1.10 con CUDA | wheel NVIDIA (`setup/wheels/torch-1.10.0a0*.whl`) | **solo 3.6** |
| torchvision 0.11 con CUDA | wheel Qengineering (`setup/wheels/torchvision-0.11*.whl`) | **solo 3.6** |
| OpenCV 4.1.1 con GStreamer | pacchetto di sistema JetPack | **solo 3.6** |
| TensorRT 8.2 | pacchetto di sistema JetPack | **solo 3.6** |

Il vecchio venv `jetson_yolo` era stato creato con **Python 3.8**: dentro non
poteva esistere né torch con CUDA né `cv2` con GStreamer, da lì tutti gli
errori di dipendenze. Un `pip install opencv-python` non risolve: la build
PyPI **non ha GStreamer** e quindi non può aprire `nvarguscamerasrc`.

La soluzione è un venv **3.6** creato con `--system-site-packages`, che
eredita dal sistema le librerie compilate da NVIDIA e prende da pip solo il
resto (pinnato alle ultime versioni compatibili con 3.6).

## Setup automatico (consigliato)

Sulla Jetson:

```bash
cd ~/cv_projects/setup
chmod +x setup_jetson.sh          # solo la prima volta
./setup_jetson.sh                 # ambiente base (PyTorch)
./setup_jetson.sh --trt           # base + extra TensorRT (pycuda, ~15 min in piu')
```

Serve la connessione a internet solo per apt e per le poche wheel pip
(le due wheel "difficili", torch e torchvision, sono già in `setup/wheels/`).

Al termine lo script esegue una verifica e deve stampare `TUTTO OK`.

## Setup manuale (cosa fa lo script, passo per passo)

```bash
# 1. Dipendenze di sistema. Le librerie python "pesanti" (matplotlib, pandas,
#    seaborn) arrivano da apt: compilarle con pip su Python 3.6/aarch64 non è
#    più praticabile nel 2026.
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-pip \
    libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev \
    python3-matplotlib python3-pandas python3-seaborn python3-pil

# 2. Venv 3.6 che vede i pacchetti di sistema
python3.6 -m venv --system-site-packages ~/cv_projects/venv
source ~/cv_projects/venv/bin/activate
export OPENBLAS_CORETYPE=ARMV8    # vedi Troubleshooting

# 3. Tool pip: ultime versioni che supportano Python 3.6
pip install --upgrade "pip==21.3.1" "setuptools==59.6.0" "wheel==0.37.1"

# 4. Nell'ordine: numpy, poi torch, poi torchvision
pip install numpy==1.19.5
pip install ~/cv_projects/setup/wheels/torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl
pip install ~/cv_projects/setup/wheels/torchvision-0.11.0a0+fa347eb-cp36-cp36m-linux_aarch64.whl

# 5. Il resto
pip install -r ~/cv_projects/setup/requirements_jetson.txt
```

## Verifica rapida

```bash
source ~/cv_projects/venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# atteso: 1.10.0a0+git36449ea True
python -c "import cv2; print('GStreamer' in cv2.getBuildInformation())"
# atteso: True
python -c "import torchvision.ops"      # nessun errore = op CUDA ok
```

Poi il primo test reale (senza torch, solo camera):

```bash
cd ~/cv_projects/01_camera_test && python camera_test.py
```

## Troubleshooting

**`Illegal instruction (core dumped)` appena importi numpy/torch**
Bug noto di OpenBLAS sul Cortex-A57 della Nano. Soluzione:
`export OPENBLAS_CORETYPE=ARMV8` (lo script lo aggiunge in automatico
all'`activate` del venv; anche tutti gli script del progetto lo impostano).

**La camera non si apre / si apre nera dopo un crash**
Il daemon argus resta appeso: `sudo systemctl restart nvargus-daemon`.
Verifica anche che il device esista (`ls /dev/video0`) e che il kernel in
boot sia quello con il driver OV5647 (`uname -r` → `4.9.253`).

**`cv2` non si importa nel venv**
Sei in un venv creato senza `--system-site-packages`, oppure con il Python
sbagliato. Controlla: `cat ~/cv_projects/venv/pyvenv.cfg` → deve dire
`version = 3.6.9` e `include-system-site-packages = true`. Se non è così,
cancella il venv e rilancia il setup. Non installare mai `opencv-python`
da pip in questo ambiente.

**Prima inferenza lentissima (30-60 s)**
Normale: cudnn fa l'autotuning e il modello viene "fuso" alla prima passata.
Gli script fanno warmup, ma il primo avvio del processo resta lento.

**Out of memory / il desktop si blocca durante l'inferenza**
La Nano ha 4 GB condivisi CPU+GPU. Assicurati che lo swap sia attivo
(`free -h`), chiudi Chromium, e considera la modalità headless (`--headless`,
senza X11 si liberano ~500 MB): lo stream si guarda dal browser del PC.

**Prestazioni basse**
`sudo nvpmodel -m 0 && sudo jetson_clocks` (modalità 10 W + clock fissi).
Con yolov5n a 416: PyTorch FP16 ~4-6 FPS, TensorRT FP16 ~12-20 FPS.

**Pannello Sistema della dashboard vuoto/`jtop_ok: false`**
Il setup verifica `jtop` all'ultimo step ([7/7]): se manca (sistema nuovo),
`sudo -H pip3 install -U jetson-stats && sudo systemctl restart jtop.service
&& sudo usermod -aG jtop $USER`, poi disconnettersi/riconnettersi (serve una
nuova sessione perché il gruppo venga effettivo). Su questo sistema è già
installato e attivo, nessuna azione necessaria.

**Avvio automatico della dashboard al boot**
`cv_projects/setup/install_demo_service.sh` installa un servizio systemd
(`cv-demo.service`, restart automatico su crash). Va eseguito sulla Jetson
con il proprio utente (usa sudo internamente).

**pip dà errori SSL o "requires a different Python"**
Stai usando il pip di sistema o una versione troppo nuova/vecchia. Nel venv
il pip giusto è il 21.3.1 (ultimo per Python 3.6), installato dal setup.

## Vecchi file superati da questo setup

- `~/jetson_yolo/` — venv Python 3.8, causa originale del problema: si può cancellare
- `~/test.py` — workaround "bridge" 3.8→3.6, non più necessario
- `~/live_detect_v5.py` — sostituito da `02_live_detection/live_detect.py`
- `~/Miniconda3-latest-Linux-aarch64.sh` — non serve (e conda su aarch64/py3.6 non aiuta)
- `~/torch-1.10.0a0*.whl` — copiata in `setup/wheels/`, l'originale in home è ridondante
