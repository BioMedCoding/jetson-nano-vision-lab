# Guida completa: training YOLOv5 da host + dataset gratuiti più grandi

Approfondimento di `README_TRAINING.md` (che resta il quick-start:
setup → `run_pipeline.sh` → deploy). Questa guida copre due cose che lì sono
solo accennate: **come allenare bene** (non solo "far girare `train.py`") e
**dove trovare dataset gratuiti più grandi/migliori di COCO 2017** per
migliorare le classi che interessano davvero a questo progetto (camera fissa
CSI, scena giardino/esterno, target tipici: persone, veicoli, animali).

Indice:
1. [Perché host e non Jetson](#1-perché-host-e-non-jetson)
2. [Dataset: come costruirlo bene](#2-dataset-come-costruirlo-bene)
3. [Dove trovare dataset gratuiti più grandi](#3-dove-trovare-dataset-gratuiti-più-grandi)
4. [Scaricare e convertire in formato YOLO](#4-scaricare-e-convertire-in-formato-yolo)
5. [Unire dataset diversi senza rompere gli ID classe](#5-unire-dataset-diversi-senza-rompere-gli-id-classe)
6. [Training: come farlo bene, non solo farlo girare](#6-training-come-farlo-bene-non-solo-farlo-girare)
7. [Leggere i risultati](#7-leggere-i-risultati)
8. [Etichettatura semi-automatica (active learning)](#8-etichettatura-semi-automatica-active-learning)
9. [Export e deploy](#9-export-e-deploy)
10. [Checklist end-to-end](#10-checklist-end-to-end)

---

## 1. Perché host e non Jetson

La RTX 5070 (12 GB VRAM) allena un'epoca di yolov5n a 416px su qualche
migliaio di immagini in secondi; la Nano non ha senso usarla per training
(niente CUDA compute capability sufficiente per PyTorch moderno, e il
torch 1.10/py3.6 della Nano è comunque incompatibile con i checkpoint di un
torch recente). Il disaccoppiamento è l'ONNX: si allena sull'host con
qualunque versione di PyTorch, si esporta ONNX, e quello lo si porta sulla
Nano per `build_engine.sh` → TensorRT. Vedi `README_TRAINING.md` §3-4 per i
comandi.

## 2. Dataset: come costruirlo bene

### Quante immagini servono

Regola pratica per fine-tuning (non training da zero, si parte sempre da
pesi COCO pretrained): **150-500 immagini per classe** con oggetti già
visibili e box corretti danno risultati utilizzabili; sotto le 100 il
modello impara rumore invece che il pattern. Per una classe "difficile"
(oggetto piccolo, tante pose, sfondo variabile — es. un animale che si
muove in giardino con luce diversa giorno/notte) conviene stare sul lato
alto del range, o compensare con più epoche di data augmentation.

### Split e bilanciamento

- **train/val**: 80/20 o 85/15. Non serve un test set separato per uso
  amatoriale: `val` durante il training è già la stima onesta di mAP.
- **Bilanciamento tra classi**: se una classe ha 500 immagini e un'altra 20,
  il modello tende a ignorare la seconda. Non serve un bilanciamento
  perfetto, ma un rapporto oltre 10:1 tra classe più numerosa e meno
  numerosa è un campanello d'allarme — o si raccolgono più immagini della
  classe rara, o (§5) si "importa" quella classe da un dataset pubblico più
  grande.
- **Diversità delle condizioni**: per una camera fissa, la variabile che
  conta di più non è il numero di immagini ma **quanto coprono le
  condizioni reali** — luce diurna/notturna/IR, pioggia, oggetto vicino e
  lontano dal sensore, angolazioni diverse se la camera si è mai spostata.
  100 immagini che coprono bene queste variazioni battono 1000 immagini
  tutte prese nello stesso pomeriggio.

### Strumenti di annotazione (bounding box in formato YOLO)

| Strumento | Tipo | Note |
|---|---|---|
| [labelImg](https://github.com/HumanSignal/labelImg) | locale, desktop | già citato in README_TRAINING.md, semplice, esporta YOLO nativo |
| [CVAT](https://www.cvat.ai/) | self-hosted (docker) o cloud free tier | supporto "smart annotation" con SAM per disegnare box più veloce, buono se le immagini sono tante |
| [Label Studio](https://labelstud.io/) | self-hosted | più generico (non solo detection), utile se in futuro si vuole anche classificare |
| [Roboflow](https://roboflow.com/annotate) | cloud, free tier limitato | annotazione + hosting + augmentation + export diretto in formato "YOLOv5 PyTorch" (struttura già pronta per `run_pipeline.sh`) |

Per questo progetto, dato il volume atteso (centinaia, non decine di
migliaia, di immagini), **labelImg locale o CVAT self-hosted** sono
sufficienti e non dipendono da un account cloud.

## 3. Dove trovare dataset gratuiti più grandi

COCO 2017 (già vendorizzato, pesi v6.0 pretrained) è il punto di partenza
ma è del 2017, 80 classi generiche, e non specializzato per una camera
fissa da giardino/esterno. Le fonti sotto **non sostituiscono** le immagini
catturate con `05_dataset_capture` — quelle restano le più preziose perché
sono la scena reale — ma sono ottime per: (a) classi rare nella scena reale
di cui servono più esempi, (b) pretraining/fine-tuning intermedio prima di
specializzare sulla scena vera, (c) recuperare classi che a COCO mancano
del tutto (es. specie animali specifiche).

### Dataset generalisti, più grandi di COCO

| Dataset | Dimensione | Perché usarlo | Licenza |
|---|---|---|---|
| **[Open Images V7](https://storage.googleapis.com/openimages/web/index.html)** (Google) | ~9M immagini, 600 classi, ~16M box | Il più utile in assoluto qui: enormemente più classi e immagini di COCO, e si può scaricare **solo le classi che servono** (es. `Fox`, `Cat`, `Dog`, `Bird`, `Person`, `Car`) invece del dataset intero — vedi §4 | CC-BY 4.0 (per immagine, verificare singolarmente se si redistribuisce) |
| **[Objects365](https://www.objects365.org/)** | 2M immagini, 365 classi, ~30M box | Già presente come `yolov5/data/Objects365.yaml` con autodownload di ultralytics. Più classi/box di COCO, ma il train set completo è **centinaia di GB** — scaricare tutto non è consigliato per questo progetto | Solo uso non commerciale/ricerca |
| **[LVIS](https://www.lila.science/) / [lvisdataset.org](https://www.lvisdataset.org/)** | riusa le immagini COCO, 1203 classi, coda lunga | Non richiede scaricare nuove immagini se COCO è già presente: solo nuove annotazioni, molto più dettagliate (oggetti piccoli/rari annotati che COCO ignora) | CC-BY 4.0, come COCO |
| **[Roboflow Universe](https://universe.roboflow.com/)** | migliaia di dataset comunitari | Cercare per scenario specifico (es. "backyard camera", "security camera person detection", "car detection driveway") invece che per classe generica — spesso già in formato "YOLOv5 PyTorch" pronto | **Variabile per dataset**, controllare la licenza di ognuno prima di redistribuire (molti CC BY 4.0, alcuni più restrittivi) |

### Dataset specifici per fauna selvatica / camera trap

Rilevanti se tra le classi di interesse ci sono animali (il progetto ha già
un run di esempio chiamato "volpe" in `run_pipeline.sh`): COCO ha solo
poche classi animali generiche (`bird`, `cat`, `dog`, `horse`, ecc.), niente
di specifico per fauna selvatica europea.

| Dataset | Note |
|---|---|
| **[LILA BC](https://lila.science/)** ("Labeled Information Library of Alexandria: Biology and Conservation") | Aggregatore centrale di dataset da camera trap **con licenza aperta** (CC0/CDLA), pensati esattamente per questo scenario: camera fissa, giorno/notte/IR, animali selvatici. Include Snapshot Serengeti, Caltech Camera Traps, WCS Camera Traps, NACTI (North American Camera Trap Images), iWildCam. Non tutte le specie europee sono coperte, ma la somiglianza di setup (camera fissa esterna) le rende utili anche solo per il pretraining su "sagoma animale in scena naturale" |
| **[iNaturalist](https://www.inaturalist.org/pages/developers)** | Enorme, ma prevalentemente **classificazione** (etichetta immagine intera, non bounding box) — utile se in futuro si vuole distinguere la specie dopo che un detector generico ha già isolato "animale", meno diretto per il training di detection attuale |
| **Open Images V7, classi animali** | Ha box per molte specie (`Fox`, `Squirrel`, `Hedgehog`, `Raccoon`, ecc. a seconda della copertura) — spesso la fonte più pratica perché già in formato detection e filtrabile per classe (§4) |

### Dataset per scenari specifici già pronti nel repo yolov5 vendorizzato

`cv/yolov5/data/*.yaml` include già gli script di autodownload di
ultralytics per alcuni dataset — utili solo se lo scenario combacia, non
in generale per questo progetto:

- `VisDrone.yaml` — vista aerea da drone, non rilevante per una camera
  fissa a terra.
- `Argoverse.yaml`, `xView.yaml` — guida autonoma / satellitare, non
  rilevanti.
- `SKU-110K.yaml`, `GlobalWheat2020.yaml` — scenari verticali (retail,
  agricoltura), non rilevanti salvo interesse specifico.
- `VOC.yaml` — piccolo (20 classi), utile solo come sanity-check rapido
  della pipeline di training, non per migliorare le performance reali.

### Ricerca generica

- **[Kaggle Datasets](https://www.kaggle.com/datasets)** — filtro per
  "object detection", licenza indicata per ogni dataset, richiede API key
  gratuita per il download via CLI.
- **[Google Dataset Search](https://datasetsearch.research.google.com/)** —
  motore di ricerca meta su dataset pubblici di ogni tipo, utile per
  scenari molto specifici (es. "driveway security camera dataset").

## 4. Scaricare e convertire in formato YOLO

Il modo più pratico per Open Images V7 e LVIS (evita di scaricare dataset
interi da terabyte quando servono solo poche classi) è
**[FiftyOne](https://docs.voxel51.com/)** di Voxel51, che ha export diretto
al formato YOLOv5 che `run_pipeline.sh` si aspetta già:

```bash
# nel venv di ~/jetson_work/cv_training (source .venv/bin/activate)
uv pip install fiftyone

python - <<'EOF'
import fiftyone as fo
import fiftyone.zoo as foz

# scarica SOLO le classi che servono, non l'intero Open Images V7
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Fox", "Cat", "Dog", "Bird", "Person"],
    max_samples=2000,       # limite per classe/query, alza se serve di più
)

dataset.export(
    export_dir="oiv7_export",
    dataset_type=fo.types.YOLOv5Dataset,
    classes=["Fox", "Cat", "Dog", "Bird", "Person"],
)
EOF
```

Questo produce già una cartella `oiv7_export/` con la struttura
`images/`, `labels/`, `dataset.yaml` compatibile con quella richiesta in
§1 di `README_TRAINING.md` (potrebbe servire solo rinominare/spostare per
farla combaciare esattamente con `images/train`, `images/val`).

Per **LVIS** stesso procedimento, `foz.load_zoo_dataset("lvis", split="train", classes=[...])`
— utile soprattutto se COCO è già scaricato, perché LVIS riusa le stesse
immagini e aggiunge solo annotazioni più dense.

Per **Roboflow Universe**, ogni dataset ha un pulsante "Download" →
formato "YOLOv5 PyTorch" che produce direttamente `images/labels/data.yaml`
già nella struttura giusta, niente conversione manuale.

Per **Objects365**, dato il volume, meglio scaricare da
`yolov5/data/Objects365.yaml` solo se serve davvero l'intero dataset (es.
per un pretraining generico), sapendo che serve molto spazio disco e tempo
di download — non è la scelta di default per aggiungere una manciata di
classi mancanti.

## 5. Unire dataset diversi senza rompere gli ID classe

Questo è **l'errore più comune e più silenzioso** (lo stesso che
`run_pipeline.sh` già valida per un singolo dataset — qui si applica anche
mescolando fonti): se il dataset catturato in giardino ha `0=persona,
1=volpe, 2=gatto` e quello scaricato da Open Images ha `0=Fox, 1=Cat`, non
si possono semplicemente copiare le cartelle `labels/` insieme — gli ID
numerici si sovrappongono e le classi vengono mischiate senza errori
visibili finché non si guardano i risultati (mAP basso, box con classe
sbagliata).

Procedura corretta:
1. Deciso l'elenco finale delle classi (es. `["persona", "volpe", "gatto",
   "auto"]`), scrivere un piccolo script che **rimappa** ogni file
   `labels/*.txt` di ogni dataset sorgente sostituendo il primo numero di
   ogni riga con il nuovo indice corrispondente (dizionario
   `{vecchio_nome: nuovo_id}` per ogni dataset, dato che nomi e ID variano
   per fonte).
2. Copiare le immagini rimappate in un'unica cartella `dataset/images/{train,val}`
   e le label corrispondenti in `dataset/labels/{train,val}`, **senza
   collisioni di nome file** (prefissare col nome del dataset sorgente se
   serve, es. `oiv7_00042.jpg`).
3. Scrivere un unico `data.yaml` finale con la lista di classi concordata.
4. Far girare comunque la validazione già in `run_pipeline.sh` (§1) prima
   di allenare: controlla proprio questo tipo di disallineamento
   (`class id massimo nelle label >= classi in data.yaml`).

Non serve un tool esterno per questo: 20-30 righe di Python (glob + rewrite
riga per riga) bastano; non vale la pena costruire un'astrazione più
generale per un progetto di questa scala.

## 6. Training: come farlo bene, non solo farlo girare

`run_pipeline.sh` già allena con dei default ragionevoli
(`--img 416 --epochs 100 --batch 64`, coerenti con la risoluzione di
inferenza sulla Nano). Quello che segue è per quando i risultati di default
non bastano.

### Hyperparametri (`--hyp`)

`cv/yolov5/data/hyps/` contiene più profili:

- `hyp.scratch-low.yaml` — augmentation leggera, adatta a **fine-tuning**
  da pesi pretrained con **pochi dati** (il caso tipico di questo
  progetto: poche centinaia di immagini reali + eventuale dataset
  pubblico aggiunto). **Questo è il default consigliato** quando si parte
  da `yolov5n.pt`/`yolov5s.pt` già pretrained COCO.
- `hyp.scratch-high.yaml` — augmentation più aggressiva, pensata per
  dataset grandi o training più lungo da zero; con poche centinaia di
  immagini rischia di introdurre più rumore che segnale.
- `hyp.finetune.yaml` — learning rate più basso, pensato esplicitamente
  per fine-tuning breve su pesi già buoni; da provare se `scratch-low`
  overfitta rapidamente (val loss che risale dopo poche epoche).
- `hyp.finetune_objects365.yaml` — specifico per fine-tuning da pesi
  Objects365, non rilevante qui.

```bash
python train.py --img 416 --batch 64 --epochs 100 \
    --data ../dataset/data.yaml --weights yolov5n.pt --name ov5647_run \
    --hyp data/hyps/hyp.scratch-low.yaml
```

### Freeze del backbone (`--freeze`)

Con pochi dati, allenare *tutta* la rete (backbone incluso) rischia di
"dimenticare" le feature generiche imparate su COCO (catastrophic
forgetting) e overfittare sulle poche immagini nuove. Congelare il
backbone e allenare solo la testa converge più in fretta ed è più stabile
con dataset piccoli:

```bash
python train.py ... --freeze 10   # congela il backbone (primi 10 layer)
```

Da provare quando: dataset < ~300 immagini totali, o quando si nota che il
training "esplode" (loss che oscilla molto) con `--freeze 0`. Da evitare
quando: si sta allenando su un dataset grande e diverso da COCO (es. dopo
aver unito molte immagini Open Images con classi nuove) — lì conviene
allenare tutta la rete per farla adattare meglio.

### Batch size e risoluzione (12 GB VRAM)

| Modello | `--img` | `--batch` indicativo |
|---|---|---|
| yolov5n | 416 | 64-96 |
| yolov5n | 640 | 32-48 |
| yolov5s | 416 | 32-48 |
| yolov5s | 640 | 16-24 |

`--img` deve **combaciare con la risoluzione di inferenza sulla Nano**
(già impostata a 416 in `build_engine.sh`): allenare a 640 e inferire a
416 introduce un mismatch scala che degrada l'accuratezza — motivo per
cui `run_pipeline.sh` esporta sempre alla stessa risoluzione del training.
Se serve più accuratezza su oggetti piccoli, si alza la risoluzione **sia**
in training che nel deploy (aumenta anche il costo di inferenza sulla
Nano, verificare l'fps risultante).

### Altri flag utili

- `--patience 30` — ferma il training se il val mAP non migliora per 30
  epoche invece di aspettare tutte le epoche richieste (evita di sprecare
  tempo GPU su un training che ha già smesso di migliorare).
- `--multi-scale` — varia la risoluzione ±50% durante il training, rende
  il modello più robusto a oggetti di dimensioni diverse (utile per una
  camera dove la distanza dell'oggetto varia molto); costa più tempo per
  epoca.
- `--cache ram` — se il dataset sta in RAM, velocizza molto le epoche
  successive alla prima (default già `ram` se non specificato).

### Monitoraggio

TensorBoard è già nei requirements (`tensorboard>=2.4.1`):

```bash
tensorboard --logdir runs/train
```

apri `http://localhost:6006` per seguire loss/mAP/precision/recall epoca
per epoca mentre allena. `wandb` è commentato nei requirements — non
necessario per un progetto di questa scala, TensorBoard locale basta.

## 7. Leggere i risultati

In `runs/train/<name>/`:

- `results.png` — curve di loss (box/obj/cls) e metriche (precision,
  recall, mAP@0.5, mAP@0.5:0.95) per epoca. **Segnale di overfitting**: le
  loss di training continuano a scendere ma quelle di validation
  risalgono, o mAP smette di migliorare mentre la training loss continua a
  scendere — rimedi: più dati (anche da dataset pubblici, §3), più
  augmentation (`hyp.scratch-high` invece di `-low`), `--freeze` del
  backbone, o semplicemente fermarsi all'epoca migliore (`best.pt` è già
  salvato automaticamente, non serve l'ultima epoca).
- `confusion_matrix.png` — quali classi vengono confuse tra loro; utile
  se due classi visivamente simili (es. gatto/volpe di notte, sagoma
  simile) si scambiano spesso — segnale che serve più contrasto nei dati
  (più esempi che le distinguano chiaramente) più che più epoche.
- `PR_curve.png`, `P_curve.png`, `R_curve.png` — per scegliere la soglia
  di confidenza da usare in deploy (`conf` regolabile live nella dashboard,
  tab Rilevamento): un mAP alto con precision bassa a soglie basse
  suggerisce di alzare la soglia default per ridurre i falsi positivi.
- `weights/best.pt` — checkpoint alla epoca con miglior mAP su val, quello
  da esportare (già quello che usa `run_pipeline.sh`).

## 8. Etichettatura semi-automatica (active learning)

Una volta che esiste un modello anche solo discreto (es. il primo
fine-tuning con poche centinaia di immagini), usarlo per **pre-etichettare**
le nuove immagini catturate è molto più veloce che annotare da zero:

```bash
python detect.py --weights runs/train/ov5647_run/weights/best.pt \
    --source nuove_immagini/ --save-txt --save-conf --conf-thres 0.25
```

Questo produce già file `labels/*.txt` in formato YOLO nelle predizioni del
modello corrente. Il passo umano si riduce a **correggere** gli errori
(box mancanti, box sbagliati, classe sbagliata) in labelImg/CVAT invece di
disegnare tutto da zero — utile soprattutto man mano che si accumulano
nuove catture da `05_dataset_capture` nel tempo, e converge verso un
modello via via più specializzato sulla scena reale.

## 9. Export e deploy

Vedi `README_TRAINING.md` §3-4 (`export.py --include onnx --opset 12`, poi
`build_engine.sh` sulla Nano) — invariato rispetto a questa guida, che si
concentra sul percorso *prima* dell'export.

## 10. Checklist end-to-end

1. Definire la lista di classi finale (quella che si vuole nella
   dashboard).
2. Raccogliere immagini reali con `05_dataset_capture` (anche poche decine
   per iniziare).
3. Se qualche classe è scarsa nella scena reale, integrare da Open Images
   V7 via FiftyOne (§3-4), filtrando solo quelle classi.
4. Annotare/correggere con labelImg o CVAT.
5. Se si uniscono fonti diverse, rimappare gli ID classe (§5) e validare
   con lo script già in `run_pipeline.sh`.
6. Allenare partendo da `yolov5n.pt`/`yolov5s.pt`, `--hyp
   hyp.scratch-low.yaml`, eventualmente `--freeze 10` se i dati sono pochi
   (§6).
7. Leggere `results.png`/`confusion_matrix.png`, decidere se servono più
   dati o più epoche (§7).
8. Esportare ONNX alla stessa risoluzione del training, deploy sulla Nano
   (`run_pipeline.sh ... --deploy matteo@<ip-jetson>`).
9. Verificare a video (dashboard), raccogliere nuovi casi che sbagliano,
   pre-etichettarli col modello corrente (§8), tornare al punto 4 per un
   nuovo giro.
