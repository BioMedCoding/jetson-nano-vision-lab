# cv_projects — Computer vision su Jetson Nano + OV5647

Base per i progetti di visione su questa Jetson Nano (JetPack 4.6.1,
kernel 4.9.253 custom con driver OV5647). Detection YOLOv5 accelerata GPU,
con percorso PyTorch e percorso TensorRT.

## Partenza rapida

```bash
# 1. una tantum: crea l'ambiente (vedi setup/SETUP_GUIDE.md per i dettagli)
cd ~/cv_projects/setup && chmod +x setup_jetson.sh && ./setup_jetson.sh

# 2. in ogni sessione:
source ~/cv_projects/venv/bin/activate

# 3. test camera, poi detection
cd ~/cv_projects/01_camera_test  && python camera_test.py
cd ~/cv_projects/02_live_detection && python live_detect.py
```

Senza monitor collegato gli script passano da soli in modalità stream:
apri `http://<ip-della-jetson>:8090/` dal browser del PC.

**Dashboard sperimentale** (sistema/statistiche/controlli live/tracker+zone,
vedi §"Dashboard" più sotto):

```bash
cd ~/cv_projects/03_tensorrt
python demo_server.py --engine ../models/yolov5n_416_fp16.engine --headless
```

## Struttura

| Cartella | Contenuto |
|---|---|
| `setup/` | script di setup, guida, requirements, wheel torch/torchvision cp36, servizio systemd |
| `models/` | pesi/engine YOLOv5(n/s)/v7-tiny/v8n + sidecar `.json` di metadata |
| `common/` | camera CSI, viewer (finestra/MJPEG semplice), utils YOLO (decoder v5+v8), runner TRT, più `state.py`/`filters.py`/`tracker.py`/`jtop_client.py`/`dashboard_server.py`/`static/` per la dashboard |
| `01_camera_test/` | verifica camera e FPS reali, snapshot — nessuna dipendenza da torch |
| `02_live_detection/` | detection live PyTorch FP16 (per confronto) |
| `03_tensorrt/` | build engine FP16, detection TensorRT minimale (`trt_detect.py`) e dashboard sperimentale (`demo_server.py`) |
| `04_benchmark/` | confronto ms/FPS tra FP32, FP16 e TensorRT |
| `05_dataset_capture/` | raccolta immagini per addestrare modelli custom |
| `venv/` | (creato dal setup) ambiente Python 3.6 |

Il repo `~/yolov5` (clone v6.0) resta dov'è: gli script lo usano via
`torch.hub.load(..., source='local')`.

## Dashboard sperimentale

`03_tensorrt/demo_server.py` è l'evoluzione di `trt_detect.py` in una vera
piattaforma di test: stessa pipeline camera→TRT→box, più pagina web
(`http://<jetson>:8090/`) a due colonne — **video e zona ROI sempre
visibili a sinistra** (con barra di stato che lampeggia sui singoli
ingressi/uscite dalla zona) mentre a destra si naviga tra le schede dei
parametri, così si vede sempre l'effetto live di quello che si cambia:

- **Sistema** — CPU/GPU/RAM/temperature/ventola/power-mode da jetson-stats
  (jtop), già installato di serie su questo sistema (sola lettura).
- **Statistiche** — storico tempo di inferenza e fps (media/mediana/p95/
  min/max/dev. standard, grafico con selettore serie), conteggi per classe
  (frame corrente + ID unici cumulativi via tracker leggero), confidenza
  media, picco massimo di presenze simultanee nella zona ROI.
- **Immagine** — luminosità/contrasto/filtri (grigio/CLAHE/nitidezza/
  sfocatura/contorni), applicati sia a schermo sia all'inferenza; ogni
  controllo ha una breve spiegazione di quando/perché usarlo e dell'impatto
  sulla detection.
- **Rilevamento** — soglie conf/IoU regolabili live, cambio modello a caldo
  da un menu, parametri extra (traiettorie, conteggio per classe), più i
  controlli di performance/leggibilità: salto inferenza ogni N frame
  (l'unica vera leva di fps su questo hardware), qualità JPEG dello stream,
  filtro classi da mostrare.
- **Alimentazione** — power mode (nvpmodel) e profilo ventola
  automatico/manuale con slider velocità, più un tasto "Prestazioni
  massime".

Il vecchio testo bruciato sull'immagine (FPS/tempo inferenza) è stato
rimosso: quei dati vivono nell'header e nel tab Statistiche, l'immagine
resta pulita per mostrare solo detection e zona ROI.

Autostart al boot: `setup/install_demo_service.sh` (richiede sudo, va
eseguito dall'utente sulla Jetson).

## Modelli disponibili e prestazioni (416×416 salvo indicazione, stanza con
luce normale, `nvpmodel -m 0` + `jetson_clocks`)

| Modello | FPS live | Note |
|---|---|---|
| yolov5n | 30.6 | default, il più leggero |
| yolov7-tiny | 23.4 | miglior compromesso: più accurato di n, quasi stessa velocità |
| yolov5s | 19.9 | più accurato di n, circa 2/3 dell'fps |
| yolov8n | ~21-26 | richiede decoder `postprocess_v8` (incluso), testa anchor-free |
| yolov5s @640 | 10.0 | risoluzione più alta, stesso modello di yolov5s |
| yolov5m | 10.3 | oltre questo punto la fluidità live scende molto |
| yolov5l | 6.0 | funzionante ma impraticabile per una demo fluida; non vendorizzato nel repo (motivi di dimensione, vedi sotto) |

Tutti gli engine tranne yolov5l sono già in `models/` con il sidecar di
metadata che dice alla dashboard quale decoder usare (`postprocess`: `v5` per
YOLOv5/v7, `v8` per YOLOv8 — vedi `build_engine.sh --postprocess`). yolov5l
è riproducibile ma non incluso: il solo file `.engine` supera 180MB (limite
soft del repo <95MB per file) per un guadagno di fluidità marginale.

## Modelli custom

1. Raccogli immagini con `05_dataset_capture/capture_images.py`
2. Sul PC host (RTX 5070): etichetta e addestra — `~/jetson_work/cv_training/
   run_pipeline.sh dataset/ yolov5n.pt --deploy matteo@<jetson>` fa da solo
   validazione dataset→training→export→scp (dettagli in
   `README_TRAINING.md` sull'host)
3. Sulla Jetson: `03_tensorrt/build_engine.sh models/custom.onnx 416`
   (aggiungi `--postprocess v8` se il modello ha una testa stile YOLOv8/senza
   objectness esplicito — per YOLOv5/v7 non serve, viene dedotto)
4. `python 03_tensorrt/demo_server.py --engine models/custom_416_fp16.engine --names classes.txt --headless`
   (o `trt_detect.py` per la demo minimale)

Il percorso ONNX→TensorRT è quello consigliato per i modelli custom: evita
ogni incompatibilità tra il PyTorch moderno dell'host e l'1.10 della Nano.
