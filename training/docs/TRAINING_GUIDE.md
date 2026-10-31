# Training guide: training well, and free datasets bigger than COCO

[`../README.md`](../README.md) is the quick start (setup ->
`quickstart.py` -> deploy). This guide covers two things that page only
touches on: **how to train well** (not just how to get a training run going)
and **where to find free datasets bigger or better than COCO 2017**, to
improve the classes that matter for a fixed camera (people, vehicles,
animals in a home, garden or street scene).

Contents:
1. [Why a PC and not the Jetson](#1-why-a-pc-and-not-the-jetson)
2. [Building a good dataset](#2-building-a-good-dataset)
3. [Where to find bigger free datasets](#3-where-to-find-bigger-free-datasets)
4. [Downloading and converting to YOLO format](#4-downloading-and-converting-to-yolo-format)
5. [Merging datasets without breaking class ids](#5-merging-datasets-without-breaking-class-ids)
6. [Training well, not just training](#6-training-well-not-just-training)
7. [Reading the results](#7-reading-the-results)
8. [Semi-automatic labelling (active learning)](#8-semi-automatic-labelling-active-learning)
9. [End-to-end checklist](#9-end-to-end-checklist)

---

## 1. Why a PC and not the Jetson

A desktop GPU (an RTX 5070 here, 12 GB) trains an epoch of yolov5n at 416 px
on a few thousand images in seconds. Training on the Nano is impractical:
its GPU and 4 GB of shared RAM would take orders of magnitude longer, and
its torch 1.10 / Python 3.6 cannot even read checkpoints written by modern
PyTorch. ONNX decouples the two: train on the PC with any PyTorch version,
export ONNX, and the Nano turns it into a TensorRT engine
(`host/deploy_model.sh` does both steps).

## 2. Building a good dataset

### How many images

Rule of thumb for fine-tuning (we always start from COCO-pretrained weights,
never from scratch): **150-500 images per class**, with clearly visible
objects and correct boxes, give usable results; below 100 the model learns
noise rather than the pattern. For a "hard" class (small object, many poses,
changing background, e.g. an animal moving through a garden in day and night
light) stay at the high end of the range, or compensate with more epochs of
augmentation.

### Split and balance

- **train/val**: 80/20 or 85/15. A separate test set is not needed for
  hobby use: `val` during training is already an honest mAP estimate.
- **Balance between classes**: if one class has 500 images and another 20,
  the model tends to ignore the second. Perfect balance is not needed, but a
  ratio above 10:1 between the most and least frequent class is a warning
  sign: collect more images of the rare class, or import it from a bigger
  public dataset (section 5).
- **Variety of conditions**: for a fixed camera, what matters most is not
  the number of images but **how well they cover real conditions**: day,
  night or IR light, rain, objects near and far from the sensor, other angles
  if the camera ever moved. 100 images covering these variations beat 1000
  taken on the same afternoon.

### Annotation tools (YOLO-format bounding boxes)

| Tool | Type | Notes |
|---|---|---|
| [labelImg](https://github.com/HumanSignal/labelImg) | local desktop app | simple, exports YOLO natively |
| [CVAT](https://www.cvat.ai/) | self-hosted (Docker) or cloud free tier | "smart annotation" with SAM draws boxes faster; good for many images |
| [Label Studio](https://labelstud.io/) | self-hosted | more general (not only detection) |
| [Roboflow](https://roboflow.com/annotate) | cloud, limited free tier | annotation + hosting + augmentation + direct "YOLOv5 PyTorch" export |

For hundreds (not tens of thousands) of images, **labelImg or a self-hosted
CVAT** are enough and need no cloud account.

## 3. Where to find bigger free datasets

COCO 2017 (what the pretrained weights know) is the starting point, but it
is from 2017, has 80 generic classes, and is not specialized for a fixed
outdoor camera. The sources below **do not replace** frames captured with
`nano/apps/capture_dataset.py` (those are the most valuable: they are the
real scene), but they are great for (a) classes rare in your scene that need
more examples, (b) an intermediate fine-tuning step before specializing on
the real scene, (c) classes COCO lacks altogether (e.g. specific animal
species).

### General-purpose datasets bigger than COCO

| Dataset | Size | Why | License |
|---|---|---|---|
| **[Open Images V7](https://storage.googleapis.com/openimages/web/index.html)** (Google) | ~9M images, 600 classes, ~16M boxes | The most useful here: far more classes and images than COCO, and you can download **only the classes you need** (e.g. `Fox`, `Cat`, `Dog`, `Bird`, `Person`, `Car`), see section 4 | annotations CC BY 4.0; images under their own licenses (check before redistributing) |
| **[Objects365](https://www.objects365.org/)** | 2M images, 365 classes, ~30M boxes | Has an autodownload in `vendor/yolov5/data/Objects365.yaml`. More classes and boxes than COCO, but the full train set is **hundreds of GB**: not recommended here | non-commercial / research only |
| **[Sama-COCO](https://www.sama.com/sama-coco-dataset)** | the COCO images, re-annotated | Denser, higher-quality boxes on the same images (in the FiftyOne zoo as `sama-coco`; used by the NNI pipeline) | CC BY 4.0 |
| **[LVIS](https://www.lvisdataset.org/)** | the COCO images, 1203 classes, long tail | New, much more detailed annotations (small and rare objects COCO ignores) without new images. Note: FiftyOne 1.19 no longer has an `lvis` zoo dataset, so download it from the LVIS site | CC BY 4.0 |
| **[Roboflow Universe](https://universe.roboflow.com/)** | thousands of community datasets | Search by scenario (e.g. "backyard camera", "security camera person detection", "car detection driveway") rather than by class; usually ready in "YOLOv5 PyTorch" format | **varies per dataset**: check each license |

### Wildlife / camera-trap datasets

Relevant if animals are among your classes: COCO only has a few generic
ones (`bird`, `cat`, `dog`, `horse`, ...), nothing specific for wildlife.

| Dataset | Notes |
|---|---|
| **[LILA BC](https://lila.science/)** (Labeled Information Library of Alexandria: Biology and Conservation) | Hub of **openly licensed** camera-trap datasets (CC0 / CDLA), made for exactly this setup: fixed camera, day/night/IR, wild animals. Includes Snapshot Serengeti, Caltech Camera Traps, WCS Camera Traps, NACTI, iWildCam. Not every species is covered, but the similar setup makes them useful even just for "animal shape in a natural scene" pre-training |
| **[iNaturalist](https://www.inaturalist.org/pages/developers)** | Huge, but mostly **classification** (whole-image label, no boxes): useful to tell species apart after a detector has found "animal", less direct for detection training |
| **Open Images V7, animal classes** | Boxes for many species (`Fox`, `Squirrel`, `Hedgehog`, `Raccoon`, ...): often the most practical source, already detection-format and filterable by class (section 4) |

### Scenario datasets with autodownload in the vendored yolov5

`vendor/yolov5/data/*.yaml` includes ultralytics' autodownload scripts for
some datasets; useful only if the scenario matches:

- `VisDrone.yaml`: aerial drone views, not a ground-level fixed camera.
- `Argoverse.yaml`, `xView.yaml`: autonomous driving / satellite.
- `SKU-110K.yaml`, `GlobalWheat2020.yaml`: retail, agriculture.
- `VOC.yaml`: small (20 classes), useful only as a quick pipeline sanity check.

### General search

- **[Kaggle Datasets](https://www.kaggle.com/datasets)**: filter by "object
  detection"; license shown per dataset; free API key for CLI downloads.
- **[Google Dataset Search](https://datasetsearch.research.google.com/)**:
  meta search engine, useful for very specific scenarios.

## 4. Downloading and converting to YOLO format

For Open Images V7 (and Sama-COCO), the practical way, which avoids
downloading terabytes when you need a few classes, is
**[FiftyOne](https://docs.voxel51.com/)** (installed by
`setup/setup_host.sh`), which exports straight to the YOLOv5 format:

```bash
source training/.venv/bin/activate
python - <<'EOF'
import fiftyone as fo
import fiftyone.zoo as foz

classes = ["Fox", "Cat", "Dog", "Bird", "Person"]
# download ONLY these classes, not the whole of Open Images V7
dataset = foz.load_zoo_dataset("open-images-v7", split="train",
                               label_types=["detections"], classes=classes,
                               max_samples=2000)
dataset.export(export_dir="oiv7_export", dataset_type=fo.types.YOLOv5Dataset,
               classes=classes)
EOF
```

This produces `oiv7_export/` with `images/`, `labels/` and a `dataset.yaml`
in the expected layout (you may only need to arrange `images/train` and
`images/val`). The NNI pipeline does exactly this, with caching, in
`cvtrain/dataset_builder.py`.

For **Roboflow Universe**, each dataset has "Download" -> "YOLOv5 PyTorch",
which gives `images/`, `labels/` and `data.yaml` in the right structure.

For **Objects365**, given its size, use `vendor/yolov5/data/Objects365.yaml`
only if you really need the whole dataset (e.g. generic pre-training).

## 5. Merging datasets without breaking class ids

This is **the most common and most silent mistake**: if your garden dataset
has `0=person, 1=fox, 2=cat` and the Open Images export has `0=Fox, 1=Cat`,
you cannot just copy the `labels/` folders together. The numeric ids overlap
and classes get mixed with no visible error until you look at the results
(low mAP, boxes with the wrong class).

The right procedure:
1. Decide the final class list (e.g. `["person", "fox", "cat", "car"]`) and
   **remap** every `labels/*.txt` of every source, replacing the first
   number of each line with the new id (a `{source_name: new_id}` dict per
   dataset, since names and ids differ per source).
2. Copy the images into one `dataset/images/{train,val}` and the remapped
   labels into `dataset/labels/{train,val}`, **without file-name
   collisions** (prefix with the source name, e.g. `oiv7_00042.jpg`).
3. Write one final `data.yaml` with the agreed class list.
4. Let `quickstart.py` validate it before training: its first step checks
   exactly this mismatch (highest class id in the labels >= number of
   classes in `data.yaml`).

20-30 lines of Python (glob + line-by-line rewrite) are enough; see
`_write_remapped_label()` in `cvtrain/dataset_builder.py` for a worked
example.

## 6. Training well, not just training

`quickstart.py` trains with reasonable defaults (416 px, matching the
Nano's inference size; up to 100 epochs with early stopping; the family's
default hyperparameters). What follows is for when the defaults are not
enough.

### Hyperparameters

For yolov5, the profiles live in `vendor/yolov5/data/hyps/`:

- `hyp.scratch-low.yaml`: light augmentation, suited to **fine-tuning**
  pretrained weights on **little data** (the typical case here: a few
  hundred real images, maybe plus a public dataset). **The recommended
  default**, and what `quickstart.py` uses for yolov5.
- `hyp.scratch-med.yaml` / `hyp.scratch-high.yaml`: stronger augmentation for
  big datasets or long training; with a few hundred images it tends to add
  more noise than signal.
- `hyp.no-augmentation.yaml`: no augmentation at all; useful only to check
  whether augmentation is what hurts.

yolov7-tiny uses `vendor/yolov7/data/hyp.scratch.tiny.yaml`; yolov8 uses
ultralytics' defaults (with `optimizer="SGD"` forced, see
`cvtrain/hparams.py`). The NNI pipeline searches these values for you.

### Freezing the backbone (`--freeze`)

With little data, training the *whole* network risks "forgetting" the
generic features learned on COCO (catastrophic forgetting) and overfitting
the few new images. Freezing the backbone and training only the head
converges faster and is more stable on small datasets:

```bash
python quickstart.py --data mydata/data.yaml --freeze 10   # yolov5 / yolov8 backbone
```

Try it when the dataset has fewer than ~300 images, or when training
"explodes" (loss oscillating wildly) without it. Avoid it on a big dataset
that differs from COCO (e.g. after merging many Open Images classes): there
the whole network should adapt.

### Batch size and resolution

`cvtrain/hparams.py` holds a batch-size table tuned for 12 GB of VRAM (e.g.
yolov5n at 416: 64; yolov5s at 640: 24). On a smaller GPU set
`CVTRAIN_BATCH=16` (or pass `--batch`).

The image size must **match the inference size on the Nano**: training at
640 and running at 416 introduces a scale mismatch that costs accuracy,
which is why the export always uses the training size. If you need more
accuracy on small objects, raise the size in **both** (`--img 640`), and
check the resulting fps on the Nano (yolov5s drops from ~20 to ~10 fps).

### Other useful knobs

- `--patience 30` (quickstart default): stop when val mAP has not improved
  for 30 epochs instead of running every requested epoch.
- yolov5's `--multi-scale` (varies the size by +-50% during training) makes
  the model more robust to objects at different distances, at the cost of
  time per epoch: run `vendor/yolov5/train.py` directly to use it.
- The trainers cache images in RAM (`--cache ram`): the epochs after the
  first are much faster, as long as the dataset fits in memory.

### Monitoring

TensorBoard is installed:

```bash
tensorboard --logdir training/runs
```

and open `http://localhost:6006` to follow loss / mAP / precision / recall
per epoch while it trains.

## 7. Reading the results

In `runs/quickstart/<name>/train/`:

- `results.png`: loss curves (box / obj / cls) and metrics (precision,
  recall, mAP50, mAP50-95) per epoch. **Overfitting sign**: training losses
  keep falling while validation losses rise again, or mAP stops improving
  while the training loss keeps falling. Remedies: more data (section 3),
  more augmentation, freezing the backbone, or simply keeping the best epoch
  (`best.pt` is saved automatically and is what gets exported).
- `confusion_matrix.png`: which classes get confused with each other. If two
  visually similar classes swap often (e.g. cat / fox at night), the data
  needs more examples that tell them apart, more than it needs more epochs.
- `PR_curve.png`, `P_curve.png`, `R_curve.png`: help choose the confidence
  threshold on the Nano (adjustable live in the dashboard, Detection tab): a
  high mAP with low precision at low thresholds suggests raising the
  default threshold to cut false positives.
- `weights/best.pt`: the checkpoint of the best validation epoch, the one
  that gets exported.

## 8. Semi-automatic labelling (active learning)

Once you have even a decent model (e.g. the first fine-tune on a few hundred
images), using it to **pre-label** new captures is much faster than
labelling from scratch:

```bash
cd training/vendor/yolov5
python detect.py --weights ../../runs/quickstart/<name>/train/weights/best.pt \
    --source new_images/ --save-txt --save-conf --conf-thres 0.25
```

This writes YOLO-format `labels/*.txt` with the current model's predictions.
The human step shrinks to **correcting** mistakes (missing boxes, wrong
boxes, wrong class) in labelImg / CVAT, and each round makes the model more
specialized on the real scene.

## 9. End-to-end checklist

1. Decide the final class list (what you want to see in the dashboard).
2. Collect real frames with `nano/apps/capture_dataset.py` (a few dozen are
   enough to start).
3. If a class is rare in the real scene, add it from Open Images V7 via
   FiftyOne (sections 3-4), filtering only those classes.
4. Label / correct with labelImg or CVAT.
5. When merging sources, remap class ids (section 5).
6. Train: `python quickstart.py --data ... --arch yolov5n` (add
   `--freeze 10` if data is scarce, section 6).
7. Read `results.png` / `confusion_matrix.png` and decide whether you need
   more data or more epochs (section 7).
8. Deploy: `host/deploy_model.sh training/export/<name>` (or `--deploy`).
9. Check it live in the dashboard, collect the cases it gets wrong,
   pre-label them with the current model (section 8), and go back to step 4.
