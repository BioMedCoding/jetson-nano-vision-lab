# Hardware notes

## Board

Jetson Nano Developer Kit, carrier board revision **B01/B02** family
(P3449-0000, DTB suffix `b00` — the DTB naming NVIDIA uses for this carrier
generation; confirmed working, do not second-guess it against the physical
silkscreen revision letter).

## Camera

OmniVision **OV5647** — a Raspberry Pi Camera V1 clone, connected to the
**CAM0** connector (the one nearer the barrel power jack; silkscreened
`CAM0` on the carrier board). The stock Nano image supports IMX219 (Pi
Camera V2) out of the box; OV5647 is not in NVIDIA's default device tree, so
this whole project exists to add it as a selectable alternative — see
`../patches/`.

Physical connection: 15-pin flat flex cable, blue tab facing the Ethernet
port side, silver contacts facing the HDMI port side (same orientation as
a stock IMX219 module).

### Why CAM0 and not CAM1

The B01/B02 carrier board routes both CSI connectors through a GPIO-selected
analog I2C mux (`CAM_I2C_MUX`, GPIO `TEGRA_GPIO(I, 0)`). The device tree in
this repo hard-drives that mux **low**, which selects CAM0. If you need the
camera on CAM1 instead, you'd need to either drive that GPIO high instead, or
extend the DT with an `i2c-mux-gpio` node the way NVIDIA's stock dual-IMX219
overlay does it (see `tegra210-porg-camera-rbpcv2-dual-imx219.dtsi` in
`kernel-source/hardware/` for the pattern) — this repo's DT does not do that,
single-camera-on-CAM0 only.

### Sensor wiring the DT expects

- **I2C address**: `0x36` (the *real* OV5647 address — the first attempt
  at this DT had it at IMX219's `0x10` by copy-paste, see patch 0005).
- **MCLK**: driven by Tegra's `CLK_OUT_3` at 24 MHz (not the 25 MHz the
  Raspberry Pi uses — the driver only warns, doesn't fail, on the
  difference).
- **Reset/power-down GPIO**: `TEGRA_GPIO(S, 7)` (CAM1_PWDN net name on the
  schematic — the net name is a NVIDIA carry-over from the reference design
  and does *not* imply "camera 1" here), held **high** from boot via a
  `gpio-hog` so the sensor is out of reset by the time the driver probes it.
  The mainline `ov5647.c` driver does not drive this pin itself.

## What "Failed to load kernel modules" actually meant here

Two unrelated problems produced this, both fixed and both worth knowing about
if you ever see it again after modifying this tree further:

1. **Modules directory name mismatch.** The kernel is built with
   `CONFIG_LOCALVERSION=""`, so `uname -r` is plain `4.9.253` — modules
   **must** live in `/lib/modules/4.9.253` on the target rootfs, not
   `4.9.253-something`. If you rename anything, `depmod -b <rootfs-root>
   4.9.253` needs to be re-run afterwards. `scripts/deploy.sh` in this repo
   handles this automatically.
2. **Truncated `APPEND` line in `extlinux.conf`.** A hand-edited boot entry
   had lost its `console=`/`fbcon=` arguments partway through. Always keep
   the exact same `APPEND` arguments as the stock entry, just point `LINUX`/
   `FDT` at the new files. `scripts/deploy.sh` writes this out for you.

## Verifying the camera after boot

```bash
ls /dev/video0
dmesg | grep -iE 'ov5647|vi |nvcsi|i2c'
v4l2-ctl --list-formats-ext -d /dev/video0
media-ctl -p
```

A working probe log looks like `ov5647 6-0036: OmniVision OV5647 camera
driver probed`. If you instead see `probe of 6-0036 failed with error -2`,
the sensor isn't answering on I2C at all — check the MCLK (`clocks`
property) and the CAM_I2C_MUX GPIO state first; that error is not a "sensor
detected wrong ID" error, it means the sensor is silent on the bus.
