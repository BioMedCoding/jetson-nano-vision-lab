# Jetson Nano setup guide (JetPack 4.6.1)

How the Python environment for `nano/` is built, why it is built that way,
and what to do when something goes wrong. Applies to: **Jetson Nano 4 GB,
L4T R32.7.1 / JetPack 4.6.1, Ubuntu 18.04**, with the OV5647 kernel from
`camera/` booted.

## TL;DR

```bash
# on your PC: copy nano/ and tools/ to the Jetson (or git clone the repo there)
host/sync_to_nano.sh
# on the Jetson:
~/jetson-nano-vision-lab/nano/setup/setup_jetson.sh          # ~5 min, offline except apt
source ~/jetson-nano-vision-lab/nano/.venv/bin/activate
```

The script ends with a check that must print `ALL GOOD`.

## Why Python 3.6, and why `--system-site-packages`

On JetPack 4 the only hardware-accelerated builds are tied to Python 3.6:

| Component | Comes from | Python |
|---|---|---|
| PyTorch 1.10 with CUDA | NVIDIA wheel, vendored in `vendor/wheels/` | **3.6 only** |
| torchvision 0.11 with CUDA ops | Qengineering wheel, vendored | **3.6 only** |
| OpenCV 4.1.1 | JetPack system package | **3.6 only** |
| TensorRT 8.2 | JetPack system package | **3.6 only** |

A venv on any other Python (3.8, conda, ...) can have neither CUDA torch nor
the system `cv2`/`tensorrt`, which is where most "dependency hell" on the
Nano comes from. `pip install opencv-python` does not help either. The
answer is a **Python 3.6 venv created with `--system-site-packages`**: it
inherits NVIDIA's compiled libraries from the system and takes everything
else from the vendored wheels.

## Why everything is vendored

NVIDIA's download links for JetPack 4 wheels, forum attachments and old
release assets disappear over time, and building pycuda or torchvision on the
Nano takes 15-90 minutes and often fails on a modern toolchain. So:

- `vendor/wheels/` holds a wheel for every pip package (pycuda and matplotlib
  pre-built for aarch64; torch stored as verified chunks, see
  `tools/largefiles.py`);
- `setup/requirements.lock` pins every package, direct and indirect, to the
  version verified on 2026-10-31;
- `pip install --no-index` means the install never touches PyPI.

Only `apt` needs the network (a few system libraries the torch wheel links
against); `setup_jetson.sh --no-apt` skips it on a machine that already has
them.

## What the script does, step by step

```bash
# 1. system libraries for the torch wheel
sudo apt-get install -y python3-venv python3-pip libopenblas-base libopenmpi-dev libomp-dev
# 2. reassemble the chunked torch wheel and check its sha256
python3 tools/largefiles.py join nano/
# 3. the venv, plus three exports appended to its activate script
python3.6 -m venv --system-site-packages nano/.venv
#    OPENBLAS_CORETYPE=ARMV8, LD_PRELOAD=libgomp, PYTHONNOUSERSITE=1 (see Troubleshooting)
# 4. every pip package, offline
pip install --no-index --find-links nano/vendor/wheels -r nano/setup/requirements.lock
pip install --no-index --find-links nano/vendor/wheels torch-1.10*.whl torchvision-0.11*.whl
# 5. jetson-stats (jtop) system-wide, for the dashboard's System tab
sudo -H pip3 install --no-index --find-links nano/vendor/wheels jetson-stats==4.3.2
```

Then the first real test, camera only:

```bash
cd ~/jetson-nano-vision-lab/nano
python3 apps/camera_test.py --headless      # expect "Real fps: 62.5"
python3 tests/test_models.py                # every model on the reference images
```

## Optional services

- **Desktop shortcut** (nothing at boot): `setup/install_launcher.sh` puts a
  "Vision Dashboard" icon on the desktop and in the applications menu. It
  opens a terminal running the dashboard and the browser on
  `http://localhost:8090/`; closing the terminal stops it.
- **Dashboard at boot**: `setup/install_service.sh` installs `cv-demo.service`
  (restart on crash). Add `--allow-system-control` to let the web page change
  power mode and fan settings. `--uninstall` removes it.
- **Temperature-based fan control** (5 V PWM fan): `setup/install_fan_service.sh`
  installs the vendored automagic-fan service and the narrow sudo rule the
  dashboard needs to start/stop it.

## Troubleshooting

**`Illegal instruction (core dumped)` as soon as numpy/torch is imported**
Known OpenBLAS bug on the Nano's Cortex-A57. Fix: `OPENBLAS_CORETYPE=ARMV8`.
The venv's `activate` exports it, the systemd unit sets it, and every app sets
it at start (`jetson_cv/runtime.py`).

**`cannot allocate memory in static TLS block` when importing cv2 after torch**
Needs `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` *before* Python
starts. Same three places as above; the apps re-launch themselves with it if
it is missing.

**torch imports from `~/.local` instead of the venv**
A `pip install --user torch` done at some point shadows the venv. The venv
sets `PYTHONNOUSERSITE=1` to ignore `~/.local`; check with
`python -c "import torch; print(torch.__file__)"` (must be under `nano/.venv`).

**The camera does not open**
Check that the OV5647 kernel is booted (`ls /dev/video0`,
`lsmod | grep ov5647`) and that nothing else holds the device: the dashboard
service does (`systemctl stop cv-demo`, or `fuser /dev/video0`). Never try
GStreamer / `nvarguscamerasrc` with this sensor: Argus has a fixed whitelist
and will not accept it; capture is raw V4L2 (`jetson_cv/camera.py`). Kernel
side: `camera/docs/HARDWARE.md`.

**`cv2` does not import in the venv**
The venv was created without `--system-site-packages` or with the wrong
Python. `cat nano/.venv/pyvenv.cfg` must say `version = 3.6.9` and
`include-system-site-packages = true`; otherwise delete `nano/.venv` and
re-run the setup. Never `pip install opencv-python` here.

**"Invalid TensorRT engine" / engine fails to load**
Engines are tied to the TensorRT version and GPU that built them; the
vendored ones were built on JetPack 4.6.1. On another JetPack, rebuild:
`python3 apps/build_engine.py <model-id>`. TensorRT 8.2 also prints "Using an
engine plan file across different models of devices is not recommended"
even for engines built on this very board: harmless.

**The first inference is very slow (30-60 s)**
Normal for the PyTorch path: cuDNN autotunes on the first passes. The apps
warm up, but the first start of a process stays slow. The TensorRT path
does not have this problem.

**Out of memory / the desktop freezes**
The Nano has 4 GB shared between CPU and GPU. Make sure swap is on
(`free -h`; JetPack's zram is usually enough, `setup_jetson.sh --swap` adds a
4 GB swap file), close Chromium, and prefer `--headless` (without X11 about
500 MB more are free): watch the stream from your PC's browser.

**Low fps**
`sudo nvpmodel -m 0 && sudo jetson_clocks` (MAXN mode, fixed clocks). With
yolov5n at 416: about 30 fps live with TensorRT, about 16-18 with PyTorch
FP16 (see `docs/benchmarks.md`).

**Dashboard System tab empty / `jtop_ok: false`**
jetson-stats missing, or your user is not yet in the `jtop` group: log out and
back in after the setup (group changes need a new session).

**pip says SSL errors or "requires a different Python"**
You are using the system pip, or a newer pip than Python 3.6 supports. Inside
the venv pip is 21.3.1, the last version for Python 3.6.
