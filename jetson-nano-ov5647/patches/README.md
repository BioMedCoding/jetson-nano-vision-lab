# Patch index

These patches are **documentation only** — the source tree under `source/public/`
already has every one of them applied, so you do **not** need to apply these to
build. They exist so the changes are reviewable file-by-file instead of buried
in a ~950 MB source drop, and so they can be cherry-picked or re-applied to a
different L4T source tree with `patch -p1` or `git apply`, run from
`source/public/`.

All patches are unified diffs against the pristine NVIDIA L4T R32.7.1 source
(kernel 4.9.253), extracted from `kernel_src.tbz2` as shipped by NVIDIA.

## Camera enablement (OV5647 / Raspberry Pi Camera V1 clone)

| Patch | What |
|---|---|
| `0001-ov5647-driver-add.patch` | New V4L2 sensor driver `drivers/media/i2c/ov5647.c`. Ported from a newer mainline version that used the post-4.15 `v4l2_fwnode` API; kernel 4.9 only has `v4l2_of` (`v4l2_of_parse_endpoint()`), so the parsing call was swapped back to the old API. Also made `xclk` optional and 25 MHz-only assumption relaxed to just warn (Tegra drives MCLK at 24 MHz, not 25), and the sensor's MCLK is now enabled around the I2C detect probe — the OV5647 does not answer on I2C at all without a running MCLK, which was the root cause of `probe of 6-0036 failed with error -2` at first boot. |
| `0002-ov5647-kconfig-makefile.patch` | Registers `CONFIG_VIDEO_OV5647` in `drivers/media/i2c/Kconfig` and the `obj-$(CONFIG_VIDEO_OV5647) += ov5647.o` build rule in the Makefile. |
| `0003-ov5647-dts-board-file.patch` | New board DTS `tegra210-p3448-0000-p3449-0000-b00-ov5647.dts` — a clone of the stock B00 board file with the IMX219 camera include swapped for OV5647. The stock IMX219 board file is left untouched; this is a **selectable alternative**, chosen via the `extlinux.conf` boot menu, not a replacement. |
| `0004-ov5647-dts-Makefile.patch` | Registers the new `.dtb` build target in `hardware/nvidia/platform/t210/porg/kernel-dts/Makefile`. |
| `0005-ov5647-dtsi-sensor-node.patch` | New `tegra210-camera-ov5647.dtsi`: the sensor's `vi`/`nvcsi` port graph and the `i2c@546c0000` node `rbpcv2_ov5647_a@36` (address **0x36**, the OV5647's real I2C address — an earlier draft had it at the IMX219 address 0x10 by copy-paste, which produced "failed to create i2c device" at boot). Adds `clocks = <&tegra_car TEGRA210_CLK_CLK_OUT_3>` (24 MHz MCLK) — missing in the first version, which meant the sensor never got a clock and consistently failed I2C probe (`-ENOENT`/`-2`). |
| `0006-ov5647-dtsi-porg-overlay.patch` | Board-specific overlay: wires `reset-gpios` (CAM1_PWDN, GPIO S7) to the correct node, sets that GPIO **high** at boot via a `gpio-hog` (the mainline driver never drives it itself — left low it keeps the camera in reset), and adds a second hog forcing the B02 carrier's CAM_I2C analog mux to the CAM0 connector. |

**Prerequisite fixed for ALL dtb builds, not just this one:** `0007-kbuild-include-the-space-fix.patch` — `scripts/Kbuild.include`'s `the-space` whitespace trick (relying on a trailing space surviving `:=`) silently broke under this toolchain, corrupting `dtb-y`/`dtbo-y` list construction and making every `.dtbo` overlay target collide on the literal name `_ddot_`. Fixed by switching to the standard `the-empty := ` / `the-space := $(the-empty) $(the-empty)` idiom. Confirmed this broke the **stock** `a01.dtb` too, before any OV5647 changes — unrelated to the camera, just never triggered until a full `dtbs` build was attempted.

## GCC 15 toolchain compatibility

This machine only has GCC 15.2.0 installed (no legacy Linaro 7.3.1 the L4T
R32.7.1 docs call for). Building `vmlinux` + all modules with it surfaced
real bugs GCC 15's stricter default diagnostics catch, plus some warning
classes that needed downgrading to keep unrelated vendor drivers compiling
(**no driver or Kconfig option was disabled** — see rationale below).

| Patch | What |
|---|---|
| `0008-top-makefile-gcc15-warnings.patch` | Downgrades ~12 GCC15-only warning classes (`-Waddress`, `-Wmisleading-indentation`, `-Wstringop-overread`, `-Wuse-after-free`, etc.) from error back to warning, via the same `cc-disable-warning` pattern NVIDIA already used in this Makefile for older warning classes. Changes no driver behavior, disables no hardware — see `../docs/GCC15_NOTES.md`. |
| `0009-etherdevice-h-stringop-overread.patch` | `ether_addr_equal_64bits()` always takes the byte-wise-safe path now. The 8-byte-word compare it used before assumed an 8-byte-capable buffer; several callers pass a genuinely 6-byte array (e.g. `struct ieee80211_hdr.addr3`), which GCC15 correctly flags as a potential overread. Fixes the root cause for patches 0010–0012 at once. |
| `0010`–`0012` | Three callers (`ipvlan_core.c`, `ath/main.c`, `at76c50x-usb.c`) switched from `ether_addr_equal_64bits()` to plain `ether_addr_equal()` — same fix pattern, at the call sites GCC flagged directly. |
| `0013-cxgb4-debugfs-macro-braces.patch` | Added explicit braces to a `for` loop inside a multi-line macro (`-Wmisleading-indentation`). No behavior change — the loop body was already only the one statement; this just makes it unambiguous to the compiler and to readers. |
| `0014-stepper-pca-dead-code.patch` | Removed `if (!(&stepper->stepper_dev->chardev)) return -ENODEV;` — address-of a struct field is never NULL, so this was dead code (`-Waddress`). |
| `0015-virtio-pci-common-waddress.patch` | Wrapped a `free_cpumask_var()` loop in a targeted `#pragma GCC diagnostic ignored "-Waddress"` — the array-typed field it tests trips the same always-true-address warning, but silencing this one narrowly (rather than the whole file/build) preserves the check everywhere else it's meaningful. |
| `0016-hid-atv-jarvis-uniq-check.patch` | **Real bug**, not just a warning: `if (hdev->uniq)` tested a pointer that is never NULL (fixed-size buffer), so the check always passed; changed to `hdev->uniq[0]` to actually test whether the string is empty, which was clearly the intent. |
| `0017-mttcan-remove-wrong-init.patch` | `mttcan_hw_init()` was marked `__init` while reachable from `.probe()` after boot (deferred probe / module reload) — real use-after-free-of-discarded-init-memory risk. Removed the `__init` annotation. |
| `0018-r8168-dead-null-check.patch` | Removed a dead `|| (tp->Rx_skbuff == NULL)` check — `Rx_skbuff` is a fixed-size array member, can't be NULL. This is the Nano's real onboard Gigabit NIC; deliberately **not** disabled to fix the warning. |
| `0019-bcmdhd-dbg-macro-braces.patch` | `DHD_ERROR()` macro was missing braces around an `if`, so the following `nv_sprintf` arguments ran unconditionally regardless of the debug level check. Added braces to restore the intended conditional. |
| `0020`, `0021` | `if (&pdev->dev.driver == NULL)` in `vi-notify.c` and `tegra-camera-rtcpu.c` (camera capture pipeline) — address-of a struct field, always false, dead code. Fixed to `pdev->dev.driver == NULL`. |
| `0022-trusty-ote-header-guard-typo.patch` | Header guard was `_TRUSTY_OTE_H_` on the `#ifndef` line but `__TRUSTY_OTE_H_` (extra underscore) on the closing `#define`/`#endif` pairing, meaning the guard never fully worked. Fixed to match. |

See `../docs/GCC15_NOTES.md` for the full reasoning and why disabling
Kconfig options was deliberately avoided as a fix strategy.

## Camera capture fixes ("frame start syncpt timeout" — 2026-07-18)

With patches 0001-0006 the sensor probed and `/dev/video0` existed, but every
capture attempt timed out (`video4linux video0: frame start syncpt timeout!0`
repeated, zero frames, no other errors). The full debugging record — including
every hypothesis that did NOT pan out — lives in
`../docs/CAMERA_DEBUG.md`. These three patches are the resolution; with them
applied the camera delivers real frames at ~62.5 fps (640x480 RAW10).

| Patch | What |
|---|---|
| `0023-camera-common-s-data-self-check.patch` | `to_camera_common_data()` in `camera_common.h` did `container_of()` on the drvdata of *any* device whose DT node has `modeN` children, even when the bound driver never registered a real `camera_common_data` — reinterpreting unrelated memory and crashing the kernel at probe (`tegra_channel_populate_dev_info` OOPS). Now validates the candidate struct via its `->dev` self-reference (always set by `camera_common_initialize()`) and returns NULL for non-camera_common consumers. |
| `0024-ov5647-capture-fixes.patch` | Four changes to `ov5647.c`, found in sequence: **(a)** minimal `camera_common` integration (allocate + `camera_common_initialize()` in probe, register `&s_data->subdev`) — without it `tegra_csi_mipi_calibrate()` silently skipped MIPI PHY calibration for this channel (`chan->s_data == NULL` guard in `csi.c`); **(b)** register table ported from current mainline: RAW10 (`SBGGR10_1X10`, `0x3034=0x18`) instead of the historical 8-bit table whose `0x3034=0x08` isn't even a documented value, plus `pad->get_fmt/set_fmt` ops (kernel 4.9's `v4l2_subdev_pad_ops`) without which `VIDIOC_S_FMT` failed with a silently-swallowed `-ENOTTY`; **(c)** **the actual frame-killer**: MIPI clock-lane mode. `__sensor_init()` used to end with `0x4800=0x04` (clock lane gate OFF = continuous clock) while the DT declares `discontinuous_clk="yes"` — the Tegra CIL waited for LP-state sequences on the clock lane that never came, accumulating `CLK_LANE_CTRL_ERR`/`DATA_LANEx_CTRL_ERR` (CILX_STATUS `0x00040041`) and never locking a single packet. Now stream-on writes `0x4800=0x34` (clock-lane gate + line sync + bus idle, i.e. the gated/non-continuous mode the DT declares — mainline's `clock-noncontinuous` path) and stream-off/init park the lanes in LP-11 (`0x4800=0x25`) so the receiver always sees a clean LP→HS transition; **(d)** `0x3503=0x00` (AEC/AGC auto) — mainline uses manual (`0x03`) but drives exposure/gain via V4L2 controls this driver doesn't implement, so manual meant power-on defaults ≈ black frames; **(e)** AGC gain ceiling raised to 15.5x (`0x3a18/0x3a19=0x00f8`) — the sensor default caps auto-gain so low that AEC plateaus far under its own target in indoor light. |
| `0025-ov5647-dtsi-mode0-real-values.patch` | `mode0` in the sensor DTSI described a fictitious 3264x2464@21fps/182.4MHz mode (copy-paste from IMX219). The driver transmits exactly one fixed mode; `mode0` now carries its real values: 640x480, `bayer_bggr`, `line_length=1852`, `pix_clk_hz=58333000`, `embedded_metadata_height=0`. These feed `camera_common`/CSI settle-time math, so they must match what's actually on the wire. |

Two operational notes that are **not** in any patch but matter for userspace
(both encoded in `cv/cv_projects/common/camera.py`):

- The Tegra-specific V4L2 control `TEGRA_CAMERA_CID_VI_BYPASS_MODE` must be
  **0** for V4L2 capture: with bypass=1 `vi2_channel_start_streaming()` returns
  right after enabling the sensor — no capture kthread, no syncpoint armed, no
  DMA. (Bypass mode exists for the Argus/nvcamera stack, which programs VI
  itself — not usable with this sensor anyway, Argus has a fixed sensor
  whitelist.)
- The stream's real Bayer order is **GRBG** (verified against a known-color
  reference object), even though driver + DT label it BGGR. Raw bytes pass
  through unchanged, so the label is harmless — but demosaic in userspace must
  use GRBG (`cv2.COLOR_BayerGR2BGR`).
