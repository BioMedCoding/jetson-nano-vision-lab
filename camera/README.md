# Jetson Nano — OV5647 (Raspberry Pi Camera V1 clone) kernel + YOLOv5 pipeline

Adds OV5647 camera support to a Jetson Nano Developer Kit (B01/B02 carrier)
as a **selectable alternate kernel**, alongside the stock IMX219-only image —
plus the complete YOLOv5 inference pipeline that consumes it.

**Status (2026-07-18): fully working, verified on hardware.** Camera captures
640x480 RAW10 at 62.5 fps via raw V4L2; YOLOv5n TensorRT FP16 runs at
30.6 fps live. Getting there required fixing five stacked problems behind a
single silent `frame start syncpt timeout` — the whole story is in
`docs/CAMERA_DEBUG.md` and it's the most reusable artifact in this repo if
you're bringing up *any* non-NVIDIA-blessed sensor on a Jetson.

This repository is self-contained: it carries the **full NVIDIA L4T R32.7.1
kernel source** (kernel 4.9.253 — `kernel-4.9`, `nvidia`, `nvgpu` and the
`hardware` device-tree source, ~950 MB total) with every fix already applied,
plus the two scripts needed to rebuild and redeploy it. No external download
is required to build — that's a deliberate choice: NVIDIA's source archives
for older L4T releases have a habit of quietly disappearing or moving when a
new major release ships, so the source of truth lives here instead of behind
a link. See `patches/README.md` if you just want to read *what* changed
without checking out the whole tree.

## What's actually in here

- `source/public/` — the full kernel + hardware source tree, patched.
- `prebuilt/` — **verified binaries** (Image, DTB, ov5647.ko): deploy and use
  the camera without building anything. See `prebuilt/README.md`.
- `cv/` — the complete userspace: V4L2 capture library, YOLOv5 live/TRT
  detection scripts, torch/torchvision wheels (vendored, GitHub-size-split),
  yolov5 snapshot, one-shot installer and launcher. See `cv/README.md`.
- `patches/` — the same changes as individual, reviewable unified diffs
  (documentation only — not needed to build, `source/public/` already has
  them applied). Start with `patches/README.md`.
- `docs/CAMERA_DEBUG.md` — how the "frame start syncpt timeout" wall was
  actually diagnosed (CSI status registers via dynamic debug, no rebuilds)
  and the five fixes behind it.
- `docs/OV5647_DEBUG_HANDOFF.it.md` — the full raw debugging log (Italian),
  including every dead end.
- `docs/HARDWARE.md` — wiring, connector, GPIO/I2C-mux notes, and what the
  "Failed to load kernel modules" boot failure actually meant here.
- `docs/GCC15_NOTES.md` — why/how this builds with a modern GCC instead of
  NVIDIA's original Linaro 7.3.1 toolchain.
- `config/nano-ov5647.config` — the exact, known-good `.config` this was
  last built and verified with (camera confirmed working end to end).
- `scripts/build.sh` — interactive build wizard.
- `scripts/deploy.sh` — interactive deploy wizard (SD card / USB / NVMe).

## Quick start — camera + inference, no kernel build

```bash
git clone <this-repo-url> && cd jetson-nano-ov5647

# 1. kernel: copy the three files in prebuilt/ onto the Jetson
#    (destinations + boot entry: prebuilt/README.md, or use scripts/deploy.sh)

# 2. userspace: copy cv/cv_projects -> ~/cv_projects and cv/yolov5 -> ~/yolov5
#    on the Jetson, then run ~/cv_projects/setup/setup_jetson.sh --trt there

# 3. from the host:
./cv/avvia_camera.sh          # live TensorRT detection in your browser
```

## Quick start — rebuild the kernel yourself

```bash
sudo apt install gcc-aarch64-linux-gnu   # if not already installed

./scripts/build.sh
#  -> builds arch/arm64/boot/Image and the camera DTB

sudo ./scripts/deploy.sh
#  -> pick the target SD card/rootfs from a list, copies Image + DTB +
#     modules, rewrites extlinux.conf, asks how eager the boot menu
#     should be about auto-booting the modified kernel
```

Both scripts are interactive: they'll ask before anything destructive, and
`deploy.sh` refuses to touch the disk your own machine is booted from and
requires you to type the target device name back to confirm before writing.

After booting the new kernel:

```bash
ls /dev/video0
dmesg | grep -iE 'ov5647|vi |nvcsi'

# actual capture test (after cv/ setup):
python3 ~/cv_projects/01_camera_test/camera_test.py --headless
#  -> "FPS reali: 62.5", MJPEG stream on http://<jetson>:8090/
```

Note for anyone tempted by GStreamer: `nvarguscamerasrc` will **never** work
with this sensor — Argus is closed-source with a fixed sensor whitelist.
Capture is raw V4L2 (`cv/cv_projects/common/camera.py` wraps it; Bayer order
is GRBG regardless of the BGGR label, see `docs/CAMERA_DEBUG.md`).

See `docs/HARDWARE.md` for the full verification checklist and what to do if
`/dev/video0` doesn't show up.

## Why this exists

The Nano's stock L4T image only has an in-tree driver for the IMX219 (the
official Pi Camera V2). A V1-generation OV5647 clone needs its own sensor
driver plus a device-tree entry describing its I2C address, MCLK, and CSI
lane wiring — none of which exist in NVIDIA's default tree. This repo adds
all of that as an alternate boot entry, so the stock camera path is never
touched or put at risk.

Everything that went wrong along the way (a mistaken IMX219 I2C address left
over from copy-pasting, a missing MCLK, a device-tree include that silently
duplicated the whole camera port graph, a GNU Make whitespace trick that
broke *every* `.dtbo` build — not just this one, a modules-directory name
that didn't match `uname -r`, a truncated boot menu entry) is documented
where the fix lives, in `patches/README.md` and `docs/`. If you hit a
similar symptom on a different Jetson project, that's the first place to
look.

## Board / camera assumptions

- Carrier board: P3449-0000 B01/B02 (DTB suffix `b00`).
- Camera on **CAM0** (the connector nearer the barrel jack), OV5647 at I2C
  address `0x36`, driven by `CLK_OUT_3` at 24 MHz.

If your board or wiring differs, `docs/HARDWARE.md` explains what to change
and why each of these values is what it is.

## Extending this archive

If you want to use this repo as a general "things that would be a pain to
lose" archive for other Jetson Nano projects (other vendor source drops,
firmware blobs, datasheets), keep each one under its own top-level directory
(e.g. `vendor/<name>/`) with a short `README.md` inside stating where it came
from and its license — same pattern as `source/public/` here. Don't dump
things loose at the repo root; it's the first thing that makes an archive
repo unnavigable a year later.

## License

GPL-2.0 (see `LICENSE`) — inherited from the Linux kernel and NVIDIA's L4T
source, which this repository is a derivative of in full.
