# How "frame start syncpt timeout" was actually debugged and fixed

After patches 0001-0006 the OV5647 probed cleanly, `/dev/video0` existed and
`STREAMON` succeeded — but `DQBUF` never returned a frame. dmesg showed only:

```
video4linux video0: frame start syncpt timeout!0
```

repeated every ~200 ms, with **no other error anywhere**. That single symptom
hid five independent problems, stacked. They are recorded here in the order
they were found, because the *method* matters as much as the fixes: three of
the five were invisible until the previous one was removed.

The full raw debugging log (in Italian, with every dead end and every dmesg
excerpt) is in `DEBUG_LOG.md`. The kernel-side fixes are patches
`0023`-`0025`; the userspace rules live in `../../nano/jetson_cv/camera.py`.

## The five problems

### 1. Kernel OOPS at probe — `to_camera_common_data()` trusted any drvdata
`container_of()` was applied to the drvdata of any device whose DT node has
`modeN` children — but a plain V4L2 subdev driver (like mainline ov5647)
stores its own `v4l2_subdev` there, not a `camera_common_data`. Result:
garbage dereference in `tegra_channel_populate_dev_info`. Fixed with a
self-reference sanity check (patch 0023).

### 2. `VIDIOC_S_FMT` silently did nothing — missing pad ops
The driver had no `.pad->get_fmt/set_fmt`, so Tegra's `tegra_channel_set_format`
got `-ENOIOCTLCMD`, and the ioctl fell through to a default handler that
no-ops. Nothing logged. Fixed by implementing fixed-mode get/set_fmt
(part of patch 0024). Lesson: on this stack a failed S_FMT looks *identical*
to a successful one from userspace unless you check buffer sizes.

### 3. MIPI PHY calibration silently skipped
`tegra_csi_mipi_calibrate()` skips any channel with `chan->s_data == NULL` —
true for every sensor that isn't a camera_common consumer. No log line, no
error: the PHY just stays uncalibrated. Fixed by minimally integrating
camera_common in the driver (allocate `s_data`, `camera_common_initialize()`,
register `&s_data->subdev` — patch 0024). After this, calibration ran and
returned 0. **Still zero frames** — which is what forced the register-level
step below.

### 4. THE killer: MIPI clock-lane mode mismatch
Reading the CSI receiver status registers at the exact moment of the timeout
(see "The technique" below) showed, cycle after cycle:

```
TEGRA_VI_CSI_ERROR_STATUS      0x00000000
TEGRA_CSI_PIXEL_PARSER_STATUS  0x00000000   <- not one packet ever decoded
TEGRA_CSI_CIL_STATUS           0x00000010   <- CTRL_ERR: LP sequence aborted
TEGRA_CSI_CILX_STATUS          0x00040041   <- CTRL_ERR on clock lane (bit 0),
                                              data lane 0 (bit 6), lane 1 (bit 18)
```

Translation: the sensor **was** transmitting (all three lanes show activity —
this ruled out wiring/port/electrical theories in one shot), but every
LP→HS handshake failed. Cause: the DT declares `discontinuous_clk = "yes"`
(gated clock), but `__sensor_init()` ended with `0x4800 = 0x04` — clock-lane
gating OFF, i.e. a **continuous** HS clock, already running before the
receiver was armed. The Tegra CIL waited forever for clock-lane LP sequences
that never came. Fix (patch 0024): stream-on writes `0x4800 = 0x34`
(clock-lane gate + line sync + bus idle — the gated mode the DT declares;
mainline's `clock-noncontinuous`), stream-off and init park all lanes in
LP-11 (`0x4800 = 0x25`) so the receiver always sees a clean transition.
First frames arrived immediately after this change.

Related discovery with opposite sign: the V4L2 control `bypass_mode`
(`TEGRA_CAMERA_CID_VI_BYPASS_MODE`) must be **0** — with 1,
`vi2_channel_start_streaming()` returns right after enabling the sensor:
no capture thread, no syncpoint, not even the timeout message. (Bypass
exists for the Argus stack, which programs VI itself.)

### 5. Black frames — manual AEC with nothing driving it
Mainline's 10-bit table sets `0x3503 = 0x03` (manual AEC/AGC) and drives
exposure/gain through V4L2 controls this driver doesn't implement; power-on
defaults ≈ zero exposure, so frames sat at the black level (~16/1023).
Fix: `0x3503 = 0x00`, sensor-internal auto exposure/gain (patch 0024).
Convergence takes ~2-3 s of continuous streaming.

## The technique: reading CSI status with zero rebuilds

The decisive step cost nothing. The timeout recovery path already dumps every
relevant register — `tegra_channel_capture_error()` → `tegra_csi_status()` —
but via `dev_dbg`, invisible by default. The kernel config here has
`CONFIG_DYNAMIC_DEBUG=y`, so:

```bash
echo 'file csi2_fops.c +p' | sudo tee /sys/kernel/debug/dynamic_debug/control
echo 'file vi2_fops.c +p'  | sudo tee /sys/kernel/debug/dynamic_debug/control
# run any capture attempt, then read dmesg
```

No rebuild, no reboot, survives until reboot. Enabling `csi2_fops.c` also
prints the computed clock/CIL settle times at every stream start — the next
thing you'd tune if the status registers had shown SOT errors instead of
CTRL errors.

Register bit meanings (Tegra X1/TK1 TRM, CIL block):
- `CIL_STATUS` bit 4 = `CILA_CTRL_ERR`: LP-01/LP-10 followed by LP-11 instead
  of entering HS/escape — an aborted handshake.
- `CILX_STATUS` bit 0 = clock-lane control error; bits 5/17 = SOT multi-bit
  error lane 0/1 (what `tegra_csi_error()` masks with `0x00020020`); bits
  6/18 = data-lane 0/1 control error. Lane-1 fields repeat lane-0 fields
  12 bits higher.

How to read the outcome:
- **All zeros** → receiver saw literally nothing: wiring, wrong CSI port in
  DT, sensor not actually streaming. (Not our case.)
- **CTRL errors, parser zero** (our case) → signal present, handshake/timing
  config wrong: clock continuity, settle times, lane count.
- **SOT/CRC/parser errors** → link works, data corrupted: settle-time tuning
  territory (`cil_settletime` in DT, sensor 0x4800/0x4802 timing).

## Bayer order: verify, don't trust

Driver and DT say `SBGGR10`; the pixels are actually **GRBG**. Raw data
passes through VI untouched, so the label is cosmetic — but the demosaic
isn't. Method: put an object of known color (red) in front of the lens,
demosaic the same raw frame all four ways, pick the one where red is red
(`cv2.COLOR_BayerGR2BGR`; the GB variant renders it blue = R/B swapped,
the other two look desaturated). Thirty seconds, zero ambiguity.

## What did NOT fix it (so nobody retries these)

- PLL compensation for the 24 vs 25 MHz MCLK difference (proportional
  multiplier rescale, `0x3036`): no effect; mainline uses the same PLL
  values for other XVCLKs.
- RAW8 → RAW10 migration alone: correct (RAW8's `0x3034=0x08` isn't even a
  documented value) and kept, but not sufficient.
- CSI lane deskew: `deskew_lanes = 0` on this setup, the whole path is
  inert on T210.
- Forcing `bypass_mode = 1`: actively harmful, see problem 4.
