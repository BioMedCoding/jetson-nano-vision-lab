# Training YOLOv5 sull'host → deploy su Jetson Nano

Fine-tuning di YOLOv5 sulla RTX 5070 di questo PC e deploy del modello sulla
Nano. Il percorso di deploy consigliato è **ONNX → TensorRT**: il PyTorch
moderno dell'host e il torch 1.10 della Nano non sono compatibili a livello di
checkpoint, ma l'ONNX li disaccoppia completamente.

Questo file è il quick-start (setup → `run_pipeline.sh` → deploy). Per una
guida più a fondo — come allenare bene (hyperparametri, freeze, lettura dei
risultati, overfitting) e dove trovare dataset gratuiti più grandi/migliori
di COCO per le classi che mancano — vedi
[`GUIDA_TRAINING_DATASET.md`](GUIDA_TRAINING_DATASET.md).

## 0. Setup (una tantum)

```bash
cd ~/jetson_work/cv_training && chmod +x setup_host.sh && ./setup_host.sh
source .venv/bin/activate
```

## Percorso rapido: run_pipeline.sh

I passi §1-4 sotto restano la spiegazione di dettaglio (utile per debug
manuale), ma una volta che il dataset è pronto (§1) l'intera pipeline
training→export→deploy si lancia con un solo comando:

```bash
./run_pipeline.sh dataset/ yolov5n.pt                                 # stampa i comandi scp
./run_pipeline.sh dataset/ yolov5s.pt --img 416 --epochs 150 --name volpe
./run_pipeline.sh dataset/ yolov5n.pt --deploy matteo@10.42.0.24       # fa anche lo scp
```

Valida la struttura del dataset (cartelle, `data.yaml`, coerenza tra il numero
di classi dichiarate e il class-id massimo usato nelle label — l'errore più
comune e più silenzioso se non controllato) **prima** di lanciare ore di
training, poi allena, esporta ONNX alla stessa risoluzione del training
(evita il classico mismatch train/inferenza) e genera `classes.txt` da
`data.yaml`. Con `--deploy` fa anche lo scp sulla Jetson; senza, stampa i
comandi pronti da eseguire.

## 1. Dataset

Raccogli le immagini sulla Nano con `~/cv_projects/05_dataset_capture/` e
copiale qui (scp dalla Jetson o direttamente dalla microSD). Struttura YOLO:

```
dataset/
├── images/train/   *.jpg
├── images/val/     *.jpg  (10-20% delle immagini)
├── labels/train/   *.txt  (una riga per box: class x_c y_c w h, normalizzati)
└── labels/val/     *.txt
```

Per etichettare: [labelImg](https://github.com/HumanSignal/labelImg) in
formato YOLO, oppure Roboflow/CVAT (esportano già così).

Crea `dataset/data.yaml`:

```yaml
path: /home/matteo/jetson_work/cv_training/dataset
train: images/train
val: images/val
names:
  0: classe_a
  1: classe_b
```

## 2. Training

Parti sempre da `yolov5n` (o `yolov5s`): sono gli unici sensati per la Nano.

```bash
cd yolov5
python train.py --img 416 --batch 64 --epochs 100 \
    --data ../dataset/data.yaml --weights yolov5n.pt --name ov5647_run
# risultati in runs/train/ov5647_run/  (best.pt, curve, matrici di confusione)
```

Consigli per la 5070 (12 GB): `--img 416 --batch 64` per yolov5n; se la VRAM
avanza, alza il batch. `--img 416` combacia con la risoluzione di inferenza
sulla Nano: allena e inferisci alla stessa scala.

## 3. Export per la Jetson

```bash
python export.py --weights runs/train/ov5647_run/weights/best.pt \
    --include onnx --imgsz 416 416 --opset 12 --simplify
```

`--opset 12` è il massimo digeribile da TensorRT 8.2 su JetPack 4.6.
L'output `best.onnx` ha lo stesso formato (1, N, 5+num_classi) che gli script
della Nano si aspettano.

## 4. Deploy sulla Nano

```bash
# dall'host (Jetson accesa in rete) oppure copiando sulla microSD:
scp runs/train/ov5647_run/weights/best.onnx matteo@<ip-jetson>:~/cv_projects/models/custom.onnx

# sulla Jetson:
cd ~/cv_projects/03_tensorrt
./build_engine.sh ../models/custom.onnx 416
python trt_detect.py --engine ../models/custom_416_fp16.engine --names ../models/classes.txt
```

`classes.txt`: un nome di classe per riga, stesso ordine di `data.yaml`.

## Pesi di partenza e dataset più aggiornati

I pesi vendorizzati (`yolov5n.pt`/`yolov5s.pt`) sono la release **v6.0**
(2021, pre-addestrata su COCO 2017): è la coppia peso/architettura con cui
`postprocess()` nella Nano è stata verificata, percorso a costo-decoder zero.

- **YOLOv5 v7.0 "u" (ultralytics, anchor-free)**: pesi pretrained più
  accurati a parità di dimensione. **Attenzione**: la testa "u" è imparentata
  con quella di YOLOv8 (vedi `cv/README.md` del progetto principale) — non è
  garantito che l'export ONNX resti nel formato `(1, N, 85)` che
  `postprocess()` si aspetta. Usarla è un compito di verifica-decoder (shape
  reale dell'ONNX, eventuale `postprocess_v8`), non un drop-in gratuito. Se
  vuoi restare a costo zero, allena da v6.0.
- **Dataset più recenti di COCO 2017**: [Roboflow
  Universe](https://universe.roboflow.com/) ha migliaia di dataset
  YOLO-ready recenti, spesso specifici per scenario (angolazione, distanza,
  condizioni di luce) — questo è il vantaggio reale, non il semplice "essere
  più nuovi": un mirror Roboflow di puro COCO non aggiunge nulla ai pesi COCO
  2017 già in uso. Cerca dataset con inquadratura simile a questa camera CSI
  (fissa, dall'alto/frontale), esporta in formato "YOLOv5 PyTorch" (produce
  già la struttura `images/labels/data.yaml` di §1) e unisci alle immagini
  catturate con `05_dataset_capture` per adattare il modello alla scena reale
  — il fine-tuning locale conta più della fonte del dataset base.

## Note

- **Non** copiare il `.pt` addestrato qui dentro il flusso PyTorch della Nano
  (`live_detect.py`): il torch 1.10/py3.6 della Jetson non carica checkpoint
  salvati da torch moderni. Usa sempre il percorso ONNX → TensorRT.
- Se ti serve provare il modello sull'host con la webcam/immagini:
  `python detect.py --weights runs/train/.../best.pt --source 0` (nel venv host).
- La cartella `dataset/` e `runs/` possono crescere molto: sono escluse dal
  deploy, sulla Nano viaggia solo l'ONNX.
