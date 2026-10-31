# Pipeline di training NNI-driven

Estende il quick-path manuale di `../README_TRAINING.md` (solo yolov5) con
una ricerca NNI su architettura × sorgente dataset × iperparametri, su tutte
e 4 le architetture gia' deployate sulla Nano (yolov5n, yolov5s,
yolov7-tiny, yolov8n). Il piano completo/contesto e' in
`~/.claude/plans/ho-cambiato-i-permessi-groovy-marshmallow.md`.

## Modo piu' semplice: il menu

```bash
cd ~/jetson_work/cv_training && ./menu.sh
```

Menu testuale che copre tutto il resto di questo file (setup, modifica
classi, smoke test, avvio/stato/stop di Stage A, promozione Stage B,
deploy manuale, stato GPU/disco) senza dover ricordare i comandi. Le
sezioni sotto restano la spiegazione di dettaglio, utile per debug manuale
o per capire cosa fa ciascuna voce del menu.

## Filosofia di budget: qualita', non tempo

In questo progetto la qualita' del modello guida le scelte di budget
(immagini per classe, epoche, durata dei run), non il tempo. I tetti orari
di Stage A/B (`maxExperimentDuration`, `--max-hours`) sono reti di
sicurezza contro un run lasciato incustodito, non obiettivi — vanno alzati
senza esitare se serve piu' margine per la qualita'.

## Classi target

`class_maps/target_classes.yaml` usa le 80 classi COCO-2017 ufficiali
(stesso schema dei pesi pretrained gia' in uso, cosi' il fine-tuning
specializza invece di ripartire da zero), copertura interni domestici +
esterni giardino/strada, senza fauna selvatica. Sama-COCO mappa tutte le 80
classi 1:1 (verificato); Open Images V7 ne copre 78/80 (mancano
equivalenti diretti affidabili per "frisbee" e "cup", che restano coperte
solo da Sama-COCO). Vedi i commenti nel file per la motivazione e come
aggiungere/togliere classi in futuro.

## Setup (una tantum)

```bash
cd ~/jetson_work/cv_training && ./setup_host.sh   # estende il venv esistente (idempotente)
source .venv/bin/activate
```

## Come funziona: due stadi

- **Stage A "explore"**: un esperimento NNI ampio (architettura × dataset ×
  iperparametri), trial brevi (25 epoche, patience 8) con early-stopping
  (Medianstop) per esplorare tanti punti a basso costo — qui il costo
  contenuto per trial e' intenzionale (serve a classificare le
  combinazioni promettenti, non a produrre un modello finale), rete di
  sicurezza sulla durata a 48h. Il modello finale lo produce Stage B.
- **Stage B "promote"** (`promote.py`): dato un trial_id promettente di
  Stage A, rilancia solo quella configurazione con epoche estese (default
  250, patience 40, rete di sicurezza 48h) fino a export ONNX + deploy
  sulla Nano — qui la qualita' del modello finale e' l'obiettivo, alza pure
  epoche/tetto se serve.

## Smoke test (fare SEMPRE prima di un run reale)

```bash
python smoke_test/make_tiny_sample.py          # dataset builder standalone, <2 min, no GPU
nnictl create --config nni_config_smoke.yml --port 8081
# aspetta che entrambi i trial siano SUCCEEDED (http://localhost:8081, o:
# curl -s http://localhost:8081/api/v1/nni/trial-jobs)
nnictl stop --all
python promote.py --trial-id <uno-dei-due-trial-id> --epochs 2 --patience 1
```

## Stage A (run reale, rete di sicurezza 48h, non un obiettivo)

```bash
nnictl create --config nni_config_explore.yml --port 8080
# segui su http://localhost:8080, o nnictl experiment status
```

## Stage B: promuovere un trial

```bash
python promote.py                                          # elenca i migliori trial trovati in runs/
python promote.py --trial-id <id>                           # promuove, stampa i comandi scp
python promote.py --trial-id <id> --deploy matteo@10.42.0.24  # promuove e fa il deploy
```

## Abilitare `include_local_capture`

Il dataset catturato con `05_dataset_capture` sulla Nano non e' ancora
etichettato, quindi e' escluso di default (`include_local_capture: [false]`
in `search_space.json`). Per abilitarlo quando sara' pronto:

1. Etichetta le immagini (labelImg/CVAT, formato YOLO) e struttura la
   cartella in `../local_capture_dataset/` con `images/{train,val}`,
   `labels/{train,val}`, `data.yaml` (stesso formato di
   `README_TRAINING.md` §1).
2. Compila `class_maps/target_classes.yaml` sezione `local_capture` coi
   nomi classe reali usati durante l'etichettatura.
3. In `search_space.json`, cambia `"include_local_capture": {"_type":
   "choice", "_value": [false]}` in `[false, true]`.

Nessun'altra modifica: `dataset_builder.py` gestisce gia' il merge.

## Problemi noti dell'ambiente (gia' risolti in questa sessione, per riferimento)

Se rifai `setup_host.sh` da zero o su un'altra macchina, questi sono gia'
gestiti dallo script/dal codice, ma utile saperli se qualcosa si rompe di
nuovo dopo un aggiornamento pacchetti:

- **`filelock` deve restare `<3.12`**: `nnictl` (nni 3.0) va in un deadlock
  silenzioso — non un crash, un blocco permanente su "Starting web
  server..." — con `filelock>=3.12` (il suo `SimplePreemptiveLock` interno
  eredita da `filelock.SoftFileLock` in un modo che si rompe tra versioni).
  `ultralytics.data.converter` pero' richiede `filelock.AsyncFileLock`
  (aggiunto dopo 3.12) solo per l'*import*: risolto con uno shim
  (`common/compat.py`, e le stesse righe patchate in `yolov5/train.py` e
  `yolov5/export.py`) che inietta un `AsyncFileLock` fittizio invece di
  alzare la versione di filelock. Se compare di nuovo un blocco su
  "Starting web server...", controlla `pip show filelock` prima di tutto.
- **yolov7 (repo non mantenuta dal 2022)**: patchata direttamente in
  `../yolov7/train.py` ed `../yolov7/export.py` per `torch.load(...,
  weights_only=False)` (default cambiato in PyTorch >=2.6) e per
  `np.trapz`/`np.int` (rimossi in NumPy >=2.0).
- **`fiftyone-db` non riconosce Ubuntu molto recenti**: se `mongod` non si
  trova (`fiftyone.core.service.ServiceExecutableNotFound`), il pip
  package `fiftyone-db` non ha un binario per la tua versione di Ubuntu nel
  suo mapping interno (`fiftyone_db/setup.py`). Scarica manualmente il
  tarball generico piu' recente da `fastdl.mongodb.org/linux/` ed estrai
  `mongod` in `.venv/lib/python3.*/site-packages/fiftyone/db/bin/mongod`
  (eseguibile).
- **`labels/train.cache` e' condiviso e incompatibile tra repo**: yolov5,
  yolov7 e yolov8 usano tutti lo stesso nome file per la cache label ma con
  formati diversi. `common/paths.clear_label_cache()` lo cancella prima di
  ogni training — se lo disattivi per qualche motivo, aspettati errori di
  unpickling quando l'architettura cambia sullo stesso dataset cacheato.

## File

Vedi il piano per la struttura completa. In sintesi: `dataset_builder.py`
(cache by config-hash), `trainers/train_{yolov5,yolov7,yolov8}.py`
(adapter per famiglia di architettura), `trial.py` (entry point NNI, Stage
A), `promote.py` (Stage B), `common/hparam_mapping.py` (search space
generico -> kwargs nativi, con le differenze tra architetture: scale di
box/cls gain, chiavi escluse per yolov8, optimizer forzato SGD).
