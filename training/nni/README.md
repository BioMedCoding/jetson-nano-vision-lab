# NNI hyperparameter search

Extends the single-run quickstart (`../quickstart.py`) with a
[Microsoft NNI](https://github.com/microsoft/nni) search over architecture x
dataset source x hyperparameters, across the four architectures the Nano
runs (yolov5n, yolov5s, yolov7-tiny, yolov8n).

Easiest way in: `../menu.sh`, a text menu that covers everything below
(setup, class list, smoke test, start/status/stop Stage A, promote, deploy,
GPU/disk status).

## Budget philosophy: quality, not time

Model quality drives the budget here (images per class, epochs, run length),
not wall-clock time. The hour caps of Stage A and B
(`maxExperimentDuration`, `--max-hours`) are safety nets against a run left
unattended, not targets: raise them without hesitation if quality needs it.

Be aware of what that means: a real Stage A downloads tens of thousands of
images (12+ GB per dataset source) and runs for one to two days on a single
RTX 5070.

## Target classes

`class_maps/target_classes.yaml` uses the 80 official COCO-2017 classes (the
same scheme as the pretrained weights, so fine-tuning specializes rather than
starting over). Sama-COCO maps all 80 classes 1:1; Open Images V7 covers
78/80 (frisbee and cup only come from Sama-COCO). The file explains how to
add or remove classes.

## How it works: two stages

- **Stage A "explore"** (`trial.py`, `nni_config_explore.yml`): a broad NNI
  experiment (architecture x dataset x hyperparameters) with short trials
  (25 epochs, patience 8) and Medianstop early stopping, to rank many
  combinations cheaply. Its purpose is ranking, not producing the final
  model.
- **Stage B "promote"** (`promote.py`): replays one promising Stage A trial
  in full (default 250 epochs, patience 40), exports ONNX and packages a
  Nano model folder in `../export/`, optionally deploying it.

Each trial writes `../runs/nni/<trial_id>/resolved_config.json` before it
trains; `promote.py` reads only that file, never NNI's internal storage.

## Smoke test (ALWAYS run it before a real run)

From `training/nni/`, with the venv active:

```bash
python smoke_test/make_tiny_sample.py          # dataset builder alone, no GPU
nnictl create --config nni_config_smoke.yml --port 8081
# wait until both trials are SUCCEEDED (http://localhost:8081, or
#   curl -s http://localhost:8081/api/v1/nni/trial-jobs)
nnictl stop --all
python promote.py --trial-id <one-of-the-two-trial-ids> --epochs 2 --patience 1
```

## Stage A (the real run)

```bash
nnictl create --config nni_config_explore.yml --port 8080
# follow it at http://localhost:8080, or: nnictl experiment status
```

## Stage B: promote a trial

```bash
python promote.py                              # list the best trials found
python promote.py --trial-id <id>              # promote: train, export, package
python promote.py --trial-id <id> --deploy     # ... and deploy to the Nano
```

## Using your own captures

Captures from `nano/apps/capture_dataset.py` are not labelled, so they are
off by default (`include_local_capture: [false]` in `search_space.json`).
To use them:

1. Label the images (labelImg / CVAT, YOLO format) and arrange them in
   `training/datasets/local_capture/` with `images/{train,val}`,
   `labels/{train,val}` and a `data.yaml`.
2. Fill in the `local_capture` section of `class_maps/target_classes.yaml`
   with the class names you used while labelling.
3. In `search_space.json` change `"include_local_capture": {"_type":
   "choice", "_value": [false]}` to `[false, true]`.

Nothing else: `cvtrain/dataset_builder.py` already handles the merge.

## Known environment traps (already handled, for reference)

All of these are handled by `setup/setup_host.sh` and the code, but worth
knowing if something breaks after a package update:

- **filelock must stay below 3.12.** nnictl (NNI 3.0) deadlocks silently
  (not a crash: it hangs forever at "Starting web server...") with
  filelock >= 3.12. ultralytics needs `filelock.AsyncFileLock` (newer) only
  to import, which a shim provides (`cvtrain/compat.py`, and the same lines
  patched into `vendor/yolov5/train.py` and `export.py`). If nnictl hangs at
  "Starting web server...", check `pip show filelock` first.
- **yolov7 is unmaintained since 2023:** patched for PyTorch >= 2.6 and
  NumPy >= 2.0 (`vendor/yolov7/LOCAL_PATCHES.diff`).
- **fiftyone-db may not recognize very recent Ubuntu releases.** If `mongod`
  is not found (`fiftyone.core.service.ServiceExecutableNotFound`), the
  `fiftyone-db` package has no binary mapped for your Ubuntu version.
  Download the most recent generic tarball from `fastdl.mongodb.org/linux/`
  and extract `mongod` into
  `.venv/lib/python3.*/site-packages/fiftyone/db/bin/mongod` (executable).
- **Label caches clash between families.** yolov5, yolov7 and ultralytics use
  the same `labels/*.cache` names with incompatible formats;
  `cvtrain.paths.clear_label_cache()` deletes them before every training.
- **Ports below 1024 need root**; nnictl then fails with an unclear EACCES.
