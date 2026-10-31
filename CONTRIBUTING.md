# Contributing: how to change things without breaking the archive

This repo has one unusual property to respect: it is an **archive of
record** for an old board. Sources, binaries and docs must agree with each
other and with what actually runs on a Jetson Nano, and nothing it needs may
depend on a download link. Every workflow below ends with "verify on
hardware, then commit source + binaries + docs together".

## Ground rules

- **Verify on the device.** A change to `camera/` or `nano/` is not done until
  it ran on a Jetson Nano with JetPack 4.6.1.
- **No new download links.** Anything an install needs lives in the repo
  (wheels, weights, engines). Files of 95 MB or more are stored as chunks:
  `tools/largefiles.py split <file>`, which also records them in
  `largefiles.manifest` and git-ignores the original. Never commit a file of
  100 MB or more (GitHub's limit).
- **Vendored code keeps a `VENDORED.md`** (upstream, commit, license) and a
  `LOCAL_PATCHES.diff` against that commit. Patch it only when necessary, and
  regenerate the diff.
- **English** in code, comments, UI text and docs.

## Kernel side (`camera/`)

1. Edit `camera/kernel-source/` (it is the build tree).
2. Build: `camera/scripts/build.sh`, or by hand from
   `camera/kernel-source/kernel/kernel-4.9` with
   `export CROSS_COMPILE=aarch64-linux-gnu- ARCH=arm64`:
   - `make -j$(nproc) Image` (seconds when incremental)
   - `make -j$(nproc) M=drivers/media/i2c modules` (ov5647.ko only)
   - `make -j$(nproc) dtbs` (if the device tree changed)
3. Deploy: `camera/scripts/deploy.sh` (SD card), or copy by hand to
   `/boot/Image-ov5647`, `/boot/dtb/...`, `/lib/modules/4.9.253/...` plus
   `sudo depmod -a`, and **reboot**: the ov5647 module cannot be hot-reloaded
   (the built-in VI driver pins it) and Image/DTB need a boot anyway.
4. Verify on hardware. Minimum bar for camera changes:
   `python3 nano/apps/camera_test.py --headless` (expect ~62.5 fps) and a
   look at an actual frame. For anything CSI-related, enable the zero-cost
   tracing first: `camera/docs/CAMERA_DEBUG.md`, "The technique".
5. Document the change as a patch, before committing the source:
   `git diff -- camera/kernel-source/<files> > camera/patches/00XX-short-name.patch`
   and add a row to `camera/patches/README.md` saying *why*, not just what
   (the table is a debugging diary: keep it one).
6. Refresh `camera/prebuilt/` with the binaries you just verified, never with
   unverified ones.
7. One commit with all of it: source + patch + prebuilt + docs.

Rules learned the hard way (see `camera/docs/`): never change the kernel
LOCALVERSION (the modules directory must equal `uname -r` or the Nano does
not boot userspace); never "fix" a build error by disabling a driver or
Kconfig option; debug prints do not ship (use `dev_dbg` + dynamic debug, not
`pr_info`).

## Jetson userspace (`nano/`)

1. Edit in the repo, `host/sync_to_nano.sh`, test on the Jetson inside
   `nano/.venv`. Every app supports `--headless` (MJPEG on port 8090), so
   everything can be tested over SSH.
2. Run `nano/tests/test_models.py` after touching inference code, and the
   camera apps after touching capture code.
3. Respect the camera facts in `nano/README.md` (fixed 640x480 RAW10 mode,
   GRBG demosaic, bypass_mode=0). Do not reintroduce GStreamer/Argus paths:
   they cannot work with this sensor.
4. Code must run on **Python 3.6** (no dataclasses, no walrus, no
   `from __future__ import annotations`).
5. A new Python dependency must install **offline on Python 3.6 aarch64**:
   download or build its wheel on the Jetson, add it to `nano/vendor/wheels/`,
   pin it (and its dependencies) in `nano/setup/requirements.lock`, and check
   that a fresh `setup_jetson.sh --no-apt` still ends with `ALL GOOD`.
6. A new model: a folder in `nano/models/` per the
   [model contract](docs/model-contract.md), built with
   `nano/apps/build_engine.py`, passing `tests/test_models.py`.

## Training side (`training/`)

1. Work inside `training/.venv` (`training/setup/setup_host.sh`).
2. Run `python tests/test_trainers.py` after touching `cvtrain/`: it trains,
   exports and packages all four architectures on coco128 in a few minutes.
3. A dependency change: update `setup/requirements.lock` from a working
   environment (`uv pip freeze`, minus torch / torchvision / nvidia-* /
   cuda-* / triton, which `setup_host.sh` installs for the local GPU).

## Publishing

`gh repo create <name> --public --source . --push`. No Git LFS is needed as
long as every file stays below 100 MB (check before committing:
`find . -path ./.git -prune -o -size +95M -print` should list only
originals that `largefiles.manifest` tracks).
