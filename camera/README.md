# camera/: OV5647 (Raspberry Pi Camera v1) support for the Jetson Nano

Adds OV5647 camera support to a Jetson Nano Developer Kit (B01/B02 carrier)
as a **selectable alternate kernel**, alongside the stock IMX219-only image.
Nothing here involves machine learning: this part ends at `/dev/video0`
delivering frames. Everything that uses those frames is in `../nano/`.

**Status: working, verified on hardware.** The camera captures 640x480 RAW10
at 62.5 fps via raw V4L2. Getting there meant fixing five stacked problems
behind a single silent `frame start syncpt timeout`. The whole story is in
[docs/CAMERA_DEBUG.md](docs/CAMERA_DEBUG.md), the most reusable part of this
repo if you are bringing up *any* sensor NVIDIA does not support on a Jetson.

This directory is self-contained: it carries the **full NVIDIA L4T R32.7.1
kernel source** (kernel 4.9.253: `kernel-4.9`, `nvidia`, `nvgpu` and the
`hardware` device-tree source, ~950 MB) with every fix applied, plus the
scripts to rebuild and redeploy it. Building needs no download, on purpose:
NVIDIA's source archives for older L4T releases tend to move or disappear
when a new major release ships, so the source of truth lives here instead of
behind a link.

## What's in here

| Path | What |
|---|---|
| `kernel-source/` | the full kernel + hardware device-tree source, patched |
| `prebuilt/` | **verified binaries** (Image, DTB, ov5647.ko): use the camera without building anything, see [prebuilt/README.md](prebuilt/README.md) |
| `patches/` | the same changes as individual, reviewable unified diffs (documentation only: `kernel-source/` already has them), start with [patches/README.md](patches/README.md) |
| `config/nano-ov5647.config` | the exact `.config` last built and verified |
| `scripts/build.sh` | interactive build wizard (cross-compiles on an x86_64 PC) |
| `scripts/deploy.sh` | interactive deploy wizard (SD card / USB / NVMe) |
| `tools/v4l2_raw_capture.py` | standalone raw capture test, no dependencies on the rest of the repo |
| `docs/CAMERA_DEBUG.md` | how "frame start syncpt timeout" was diagnosed (CSI status registers via dynamic debug, no rebuilds) and the five fixes |
| `docs/HARDWARE.md` | wiring, connector, GPIO / I2C-mux notes, and what the "Failed to load kernel modules" boot failure meant here |
| `docs/GCC15_NOTES.md` | why and how this builds with a modern GCC instead of NVIDIA's Linaro 7.3.1 |
| `docs/DEBUG_LOG.md` | the full raw debugging log, every dead end included |

## Quick start: use the prebuilt kernel

Copy the three files in `prebuilt/` to the Jetson and add a boot entry
(destinations and the `extlinux.conf` entry are in
[prebuilt/README.md](prebuilt/README.md)), or let `scripts/deploy.sh` do it
on the SD card. Reboot into the "OV5647" entry, then:

```bash
ls /dev/video0
dmesg | grep -iE 'ov5647|vi |nvcsi'
python3 camera/tools/v4l2_raw_capture.py      # raw capture test, no setup needed
```

and continue with `../nano/` (`nano/apps/camera_test.py --headless` should
report "Real fps: 62.5").

## Quick start: rebuild the kernel yourself

```bash
sudo apt install gcc-aarch64-linux-gnu      # any recent cross GCC (tested: 15.2)
camera/scripts/build.sh                     # builds Image, modules and the camera DTB
sudo camera/scripts/deploy.sh               # pick the target SD card / rootfs from a list
```

Both scripts are interactive and ask before anything destructive;
`deploy.sh` refuses to touch the disk your own machine boots from and asks
you to type the target device name back before writing.

Note for anyone tempted by GStreamer: `nvarguscamerasrc` will **never** work
with this sensor. Argus is closed source with a fixed sensor whitelist.
Capture is raw V4L2 (`nano/jetson_cv/camera.py`), and the Bayer order on the
wire is GRBG whatever the BGGR label says (see docs/CAMERA_DEBUG.md).

## Why this exists

The Nano's stock L4T image only has an in-tree driver for the IMX219 (the
official Pi Camera v2). A v1-generation OV5647 (or clone) needs its own
sensor driver plus a device-tree entry describing its I2C address, MCLK and
CSI lane wiring, none of which exist in NVIDIA's tree. This adds all of that
as an alternate boot entry, so the stock camera path is never touched.

Everything that went wrong along the way (an IMX219 I2C address left over
from copy-pasting, a missing MCLK, a device-tree include that silently
duplicated the whole camera port graph, a GNU Make whitespace trick that
broke *every* `.dtbo` build, a modules-directory name that did not match
`uname -r`, a truncated boot menu entry) is documented where the fix lives,
in `patches/README.md` and `docs/`. If you hit a similar symptom on another
Jetson project, look there first.

## Board / camera assumptions

- Carrier board: P3449-0000 B01/B02 (DTB suffix `b00`).
- Camera on **CAM0** (the connector nearer the barrel jack), OV5647 at I2C
  address `0x36`, driven by `CLK_OUT_3` at 24 MHz.

If your board or wiring differs, `docs/HARDWARE.md` explains what to change
and why each value is what it is.

## License

GPL-2.0 (see `LICENSE`), inherited from the Linux kernel and NVIDIA's L4T
source, of which this directory is a derivative.
