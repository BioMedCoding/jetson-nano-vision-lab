# training/: fine-tune models on a PC, deploy them to the Nano

The Nano can run models but cannot train them in any reasonable time. This
part runs on a PC with a GPU: it fine-tunes YOLO detectors on your data and
packages each result as a **model folder** that the Nano picks up as is
(the [model contract](../docs/model-contract.md)).

```
your dataset ──► quickstart.py ──► export/<name>/ ──► host/deploy_model.sh ──► Nano
                 (or the NNI search)  model.onnx       copies it, builds the
                                      model.json       TensorRT engine ON the
                                      labels.txt       Nano, smoke-tests it
```

ONNX is the only thing that travels: the Nano's torch 1.10 / Python 3.6
cannot load checkpoints saved by modern PyTorch, and a TensorRT engine can
only be built on the device that will run it.

## Setup (once)

Needs Linux x86_64 with [uv](https://docs.astral.sh/uv/) installed. A CUDA
GPU is strongly recommended; tested on an RTX 5070.

```bash
training/setup/setup_host.sh          # creates training/.venv
source training/.venv/bin/activate
```

`setup_host.sh --torch-backend auto` (the default) lets uv pick the PyTorch
build that matches your NVIDIA driver; `--torch-backend cu128` forces the
exact combination this repo was verified with; `cpu` works without a GPU
(slowly). Everything else is pinned in `setup/requirements.lock`.

## Quickstart: one model on your dataset

```bash
cd training
python quickstart.py --data /path/to/dataset/data.yaml                  # yolov5n, 416
python quickstart.py --data /path/to/data.yaml --arch yolov8n --name fox-v1 --deploy
```

It validates the dataset (the classic silent mistake: a class id in the
labels higher than the number of classes in `data.yaml`), trains with early
stopping, exports ONNX **at the training size**, checks the ONNX against the
decoder the Nano will use, and writes `export/<name>/`. With `--deploy` it
also runs `host/deploy_model.sh`.

| `--arch` | Nano fps (live, 416) | Notes |
|---|---|---|
| `yolov5n` | ~30 | fastest; the default |
| `yolov7-tiny` | ~23 | more accurate than v5n, nearly as fast |
| `yolov8n` | ~22 | anchor-free head (`yolo_v8` decoder) |
| `yolov5s` | ~20 | more accurate, 2/3 of the fps |

Train at the size you will run at: `--img 416` (default) matches the Nano
models. Starting weights are the COCO-pretrained ones in `weights/`.

### Dataset format

Standard YOLO layout, e.g. from Roboflow ("YOLOv5 PyTorch" export), CVAT or
labelImg:

```
mydata/
├── data.yaml          names: {0: fox, 1: cat}   (train/val paths optional)
├── images/train/  *.jpg     images/val/  *.jpg  (10-20% of the images)
└── labels/train/  *.txt     labels/val/  *.txt  (class x_c y_c w h, normalized)
```

Collect images with the camera itself (`nano/apps/capture_dataset.py`, native
640x480): a model fine-tuned on frames from the real camera and scene beats
one trained only on internet images. See
[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for how to train well
(dataset size, epochs, freezing, reading the results) and where to find free
datasets.

## The NNI search (optional, heavy)

`nni/` runs a [Microsoft NNI](https://github.com/microsoft/nni)
hyperparameter search over architecture x public dataset (Open Images V7,
Sama-COCO via FiftyOne) x augmentation/optimizer settings, then "promotes"
the best trial to a long training. It downloads tens of thousands of images
and runs for days: see [nni/README.md](nni/README.md). `./menu.sh` drives
everything from a text menu.

## Layout

| Path | What |
|---|---|
| `quickstart.py` | one dataset, one architecture -> model folder |
| `cvtrain/` | shared code: trainers (yolov5 / yolov7 / yolov8), export, packaging, dataset builder |
| `nni/` | NNI search: `trial.py` (Stage A), `promote.py` (Stage B), configs, class map |
| `menu.sh`, `menu.py` | text menu over all of the above |
| `setup/` | `setup_host.sh`, `requirements.lock`, `overrides.txt` |
| `weights/` | COCO-pretrained starting weights (vendored: download links rot) |
| `vendor/` | patched yolov5 and yolov7 (see each `VENDORED.md`) |
| `tests/test_trainers.py` | trains, exports and packages all 4 architectures on coco128 |
| `datasets/`, `runs/`, `export/` | created as you work; git-ignored |

Environment variables: `CVTRAIN_DEVICE` (`0`, `1`, `cpu`, ...; default: first
GPU, else CPU) and `CVTRAIN_BATCH` (batch size, if the default table in
`cvtrain/hparams.py`, tuned for 12 GB, is too much for your GPU).
