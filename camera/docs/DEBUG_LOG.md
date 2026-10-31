# OV5647 on the Jetson Nano: camera debugging log (as of 2026-07-18)

> **SOLVED (2026-07-18, following session).** The camera captures frames and the
> whole CV pipeline works (TensorRT detection at 30 fps live). See
> [Resolution](#resolution-2026-07-18-following-session) at the end for the causes
> and fixes. The rest of this document is kept as the history of the debugging,
> dead ends included: [CAMERA_DEBUG.md](CAMERA_DEBUG.md) is the clean write-up.
>
> Translated from the original Italian. Paths such as `~/jetson_work/...`,
> `~/cv_projects` and the SSH address are those of the original development
> machine; in this repo the kernel tree is `camera/kernel-source/` and the
> userspace is `nano/`.

This file was written to be the ONLY source needed to resume the work from
scratch, without re-reading the original conversation. It contains: setup,
goal, everything that was done (working and not), the sources found online,
the exact state of the system, and ideas for the next step.

## 1. Setup: how to reconnect and where things are

**Jetson Nano** (target):
```
ssh -i ~/.ssh/jetson_session/jetson_key matteo@10.42.0.24
```
Commands that need `sudo` need an interactive TTY (they cannot run from a
non-interactive tool; run them in the user's real terminal):
```
ssh -t -i ~/.ssh/jetson_session/jetson_key matteo@10.42.0.24
```
Target hostname: `matteo-desktop` (since renamed `matteo-jetson`). Kernel
running: 4.9.253, build `#15` (the number increases with every new Image
booted; check `uname -a` to know which build is running).

**Kernel sources** (host, where it is compiled):
- L4T root: `/home/matteo/jetson_work/Linux_for_Tegra/source/public/`
- "Vanilla" kernel (contains `drivers/media/i2c/ov5647.c`, the sensor driver):
  `.../kernel/kernel-4.9/`
- Tegra VI/CSI/camera_common drivers (NVIDIA's, source available):
  `.../kernel/nvidia/drivers/media/platform/tegra/camera/`
  - `vi/channel.c`: VI channel logic (format negotiation, stream on/off, mode setup)
  - `vi/vi2_fops.c`: VI2 fops (T210/Nano uses VI2+CSI2, NOT VI4/VI5/nvcsi, which
    are for Xavier/Orin: don't get confused reading csi4/csi5/vi4/vi5 fops, they
    are not this platform)
  - `csi/csi.c`: common CSI logic, including `tegra_csi_mipi_calibrate()`
  - `csi/csi2_fops.c`: T210-specific CSI2 fops, including `csi2_mipi_cal()` and
    `csi2_start_streaming()`
  - `camera_common.c`: the camera_common framework (`camera_common_initialize()` etc.)
  - `.../include/media/camera_common.h`: contains `to_camera_common_data()` (see fix #1)
- Sensor device tree:
  `.../hardware/nvidia/platform/t210/porg/kernel-dts/porg-platforms/tegra210-camera-ov5647.dtsi`
- Official working sensor for reference: `drivers/media/i2c/imx219.c` (uses the
  full tegracam framework, which we do not; see section 4)

**Build** (always from `.../kernel/kernel-4.9/`):
```bash
export CROSS_COMPILE=aarch64-linux-gnu- ARCH=arm64
make -j$(nproc) Image                          # kernel Image (~10 s, incremental)
make -j$(nproc) M=drivers/media/i2c modules    # ov5647.ko only (~1-2 s)
make -j$(nproc) dtbs                           # all DTBs, ours included
```
`.config` already exists in the tree, no need to regenerate it.

**Deploy to the target** (via `scp` + interactive sudo commands run by the user):
- Image -> `/boot/Image-ov5647` (referenced by the default entry in
  `/boot/extlinux/extlinux.conf`; the LABEL already exists, no need to touch it)
- Module -> `/lib/modules/4.9.253/kernel/drivers/media/i2c/ov5647.ko`, then `sudo depmod -a`
- DTB -> `/boot/dtb/tegra210-p3448-0000-p3449-0000-b00-ov5647.dtb` (path confirmed
  by reading `FDT` in extlinux.conf)
- A **reboot is always needed** after touching the Image or DTB (not just the
  module). The module alone CANNOT be reloaded live either: `rmmod ov5647` fails
  with "Module is in use", because the (built-in) VI driver keeps a persistent
  reference to the subdev even with the fd closed, and this driver has no clean
  unbind on the VI side. A full reboot is the only reliable way found.

**Deploy pattern used for the whole session** (works, repeatable):
1. Build locally.
2. `scp` to `~/something.{ko,dtb}` or `~/Image-xxx` on the target (never
   overwrite `/boot/...` directly with scp, permission denied: copy to home first).
3. Give the user 2-4 commands to run **in their real terminal** (not a tool):
   `ssh -t ...`, then `sudo cp ...`, then (if needed) `sudo depmod -a`, then
   `sudo reboot`. **Important**: do NOT chain the commands with `&&` on one line
   passed to `ssh -t "..."`: the user's terminal sometimes wraps long commands and
   breaks the line, running the `cp` destination path as a command of its own
   (seen several times: "cp: missing destination file operand" followed by
   "bash: line 1: /path/of/the/file: Permission denied"). Give short commands,
   one per line.
4. Wait for the reboot by polling (the host is reachable ~5 s after the reboot):
   ```bash
   SSH="ssh -i ~/.ssh/jetson_session/jetson_key -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new matteo@10.42.0.24"
   for i in $(seq 1 30); do $SSH "echo up" 2>/dev/null | grep -q up && break; sleep 5; done
   ```

**Capture test script** (on the target, persistent): `~/v4l2_raw_test3.py` (in
this repo: `camera/tools/v4l2_raw_capture.py`). It does raw V4L2 ioctls via
`fcntl.ioctl` (no OpenCV/GStreamer, bypassing Argus completely): opens
`/dev/video0`, sets `VIDIOC_S_CTRL` to force `TEGRA_CAMERA_CID_VI_BYPASS_MODE=1`,
then `S_FMT(640x480, BG10)` -> `REQBUFS` -> `QUERYBUF`+`mmap`+`QBUF` x4 ->
`STREAMON` -> tries 5 `DQBUF` with a 2 s timeout each -> `STREAMOFF`. If it
captures real frames, it saves 4 Bayer demosaic variants (`bggr2bgr`,
`rggb2bgr`, `gbrg2bgr`, `grbg2bgr`) as `/tmp/ov5647_*.jpg` on the target, to
download and inspect visually to find the right Bayer order (this step was
never reached: not a single frame captured yet).
To re-run it: `source ~/cv_projects/venv/bin/activate && python3 ~/v4l2_raw_test3.py`.

**Python environment on the target**: venv at `~/cv_projects/venv` (Python 3.6
`--system-site-packages`, required by the cp36 torch/torchvision wheels used
elsewhere in the project).

## 2. Original goal

The wider project is a working YOLOv5 detector on a Jetson Nano with an OV5647
camera (Raspberry Pi Camera v1, clone). The current bottleneck: **the camera
does not capture a single frame**, whatever the application software
(Argus/`nvarguscamerasrc`, GStreamer, OpenCV, or direct raw V4L2: all fail the
same way at driver level).

Note: Argus/`nvarguscamerasrc` was ruled out for good. It is a closed-source
stack with a fixed sensor whitelist and will never recognize a non-"tegracam"
sensor, whatever is fixed in the V4L2 driver. All further work is on **raw
V4L2**, bypassing Argus.

## 3. Reference hardware / configuration

- Original Jetson Nano devkit (not the 2 GB), L4T R32.7.1, kernel 4.9.253, JetPack 4.6.1.
- Camera: OV5647 (Raspberry Pi Camera v1 clone), 15-pin connector, **connected
  DIRECTLY** to the Jetson Nano's CAM0 connector, **no adapter**. The user
  confirmed this explicitly: it is NOT an adapter / pin-mismatch wiring problem
  (I had raised that hypothesis by confusing the Jetson Nano with the Jetson
  TX1, which has a different CSI connector and really needs a 15-to-22-pin
  adapter according to RidgeRun's TX1 docs; that does NOT apply to the Nano,
  which takes the 15-pin cable directly, a standard, well-documented setup for
  exactly this hardware pairing).
- Starting driver: `drivers/media/i2c/ov5647.c`, the "raw" mainline Linux driver
  (Ramiro Oliveira / Synopsys), NOT integrated with NVIDIA's
  `camera_common`/`tegracam`. This is the heart of the problem: the only sensors
  NVIDIA tests its VI/CSI stack with are integrated with `camera_common`, and
  this driver was not at all at the start of the session.

## 4. What was fixed (confirmed, stable)

### 4.1 Kernel crash at probe (FIXED)
**Symptom**: OOPS ("Internal error: Accessing user space memory outside
uaccess.h routines") in `tegra_channel_populate_dev_info` when the sensor probes.

**Cause**: `to_camera_common_data()` in `camera_common.h` did a `container_of()`
on the `drvdata` of ANY device whose DT node had `modeN` children, blindly
assuming it was a real `camera_common_data`. Our `ov5647.c` (before the fix in
4.3) left its own raw `v4l2_subdev` in drvdata, not a `camera_common_data`:
reinterpreting that memory as a different structure dereferenced garbage
pointers.

**Fix** (file: `kernel/nvidia/include/media/camera_common.h`, function
`to_camera_common_data()`): a self-reference check. After the `container_of()`,
check that `s_data->dev == dev`; if not, return `NULL` instead of trusting it
blindly. `camera_common_initialize()` always sets `->dev` to the sensor's own
device, so this self-check tells a real camera_common consumer from one that
is not.

```c
static inline struct camera_common_data *to_camera_common_data(
	const struct device *dev)
{
	struct camera_common_data *s_data;

	if (!sensor_common_parse_num_modes(dev))
		return NULL;

	s_data = container_of(dev_get_drvdata(dev),
			    struct camera_common_data, subdev);
	if (s_data->dev != dev)
		return NULL;

	return s_data;
}
```
**Status**: confirmed stable for the whole session (dozens of reboots, never
seen again).

### 4.2 Broken V4L2 format negotiation (FIXED)
**Symptom**: `VIDIOC_S_FMT` seemed to succeed but changed nothing; after much
debugging it turned out to fail silently with `-ENOTTY`, masked for hours by a
bug of ours (see 5.1).

**Real cause**: `ov5647.c` did not implement `.pad->get_fmt` / `.pad->set_fmt`
in `v4l2_subdev_pad_ops`, only `.enum_mbus_code`. Without them,
`tegra_channel_try_format` / `set_format` in `channel.c` (which call
`v4l2_subdev_call(sd, pad, set_fmt, ...)`) failed with `-ENOIOCTLCMD` ->
`-ENOTTY`, and the ioctl was routed to the default handler
`tegra_channel_default_ioctl`, which silently does nothing.

**Fix**: added `ov5647_get_fmt` / `ov5647_set_fmt`, which always report the fixed
resolution actually transmitted (first 640x480 SBGGR8, later moved to SBGGR10,
see 4.4), hooked to `.pad->get_fmt` / `.pad->set_fmt`.

**Status**: confirmed working: `REQBUFS`/`QUERYBUF` correctly allocate buffers
of 307200 bytes (640x480x1, 8-bit version) or 614400 bytes (640x480x2, current
10-bit version).

### 4.3 MIPI PHY calibration never run (FOUND and integrated, not decisive yet)
**Symptom**: after fixing 4.1 and 4.2, `STREAMON` succeeds but `DQBUF` always
times out; dmesg repeatedly shows `video4linux video0: frame start syncpt
timeout!0`, with no other error (no I2C error, no explicit CSI PHY error).

**Cause found** (file `kernel/nvidia/drivers/media/platform/tegra/camera/csi/csi.c`,
function `tegra_csi_mipi_calibrate()`):
```c
list_for_each_entry(chan, &csi->csi_chans, list) {
    if (chan->pg_mode) continue;
    if (chan->s_data == NULL) continue;   // <-- THIS
    ret = csi->fops->mipical(chan);
    ...
}
```
`chan->s_data` is filled by `to_camera_common_data(chan->subdev_on_csi->dev)`,
which for our sensor (never a real camera_common consumer) always returned
`NULL` (correctly, given fix 4.1). Result: **the MIPI PHY calibration was
silently skipped** for our channel, for the whole session until this fix. A
never-calibrated receiver can stay armed but unable to lock onto any valid HS
transition: 100% consistent with the "zero frames, zero errors" observed.

Confirmed with a temporary diagnostic `pr_info` (later removed from
`csi2_fops.c`; a second, more permanent one stayed in `csi.c`, see section 6):
before the fix, `s_data=NULL`, `sensor_sd=NULL`, the calibration function was
never called. After the fix, `s_data`/`sensor_sd` are valid (non-NULL
addresses) and `fops->mipical()` explicitly returns `0` (success, also checked
by reading the real implementation in `mipical/mipi_cal.c`: not a stub, it
really runs `select_lanes()` + `tegra_mipi_wait()` with genuine error checks).

**Fix implemented** (file: `kernel/kernel-4.9/drivers/media/i2c/ov5647.c`): a
MINIMAL integration with `camera_common`, not the full `tegracam` framework
(which would need converting to `regmap`, `ctrl_ops` for exposure/gain,
`sensor_ops`: too much work for what is needed), using the low-level APIs
directly:
- `struct ov5647` no longer embeds its own `struct v4l2_subdev sd`; it holds a
  `struct camera_common_data *s_data` instead.
- `to_state()` now gets the driver struct via
  `container_of(sd, struct camera_common_data, subdev)->priv`.
- In `ov5647_probe()`: allocate `s_data`, fill the minimum fields (`dev`,
  `priv`, `frmfmt`/`numfmts` with one 640x480 entry, `def_mode=0`,
  `def_width=640`, `def_height=480`, `def_clk_freq=24000000`), call
  `camera_common_initialize(s_data, "ov5647")` (which runs
  `camera_common_parse_ports()` + `camera_common_parse_general_properties()` +
  `sensor_common_init_sensor_properties()`, reading the existing DT `modeN`
  nodes), then set `s_data->mode_prop_idx = 0`, and register `&s_data->subdev`
  (no longer a separate subdev) via `v4l2_i2c_subdev_init()`.
- `ov5647_remove()` updated accordingly.

**Status**: the integration compiles cleanly, does not crash, and the MIPI
calibration **now really runs and returns success** (checked via debug print,
see above). Despite that, **the frame start still never arrives**: see section 5
for what was ruled out next.

### 4.4 Sensor register table moved to 10-bit (done, not decisive on its own)
See 5.3 for the reasoning. Kept anyway because it is the most correct and
verified version available (a faithful port from the currently maintained
mainline Linux driver, tested by third parties on real hardware). Not reverted
to 8-bit.

### 4.5 Device-tree mode0 data corrected (done, not decisive on its own)
File: `hardware/nvidia/platform/t210/porg/kernel-dts/porg-platforms/tegra210-camera-ov5647.dtsi`,
node `mode0`. It originally described a completely fictitious
3264x2464@21fps / 182.4 MHz mode (never actually transmitted by this driver,
which implements ONE fixed 640x480 mode through its register table). Corrected
with the real values of our stream (taken from the verified upstream driver):
`active_w=640`, `active_h=480`, `pixel_t=bayer_bggr`, `line_length=1852` (= HTS),
`pix_clk_hz=58333000`, `embedded_metadata_height=0`. The other 5 `modeN` nodes
(mode1-mode5) were left unchanged and still fictitious/unused: they affect
nothing because `mode_prop_idx` always points to 0.

## 5. What was tried and did NOT work (and why it looked promising)

All these tests ran WITH fixes 4.1 + 4.2 applied and stable, and (from the point
indicated) with fix 4.3 (camera_common) too. Every single attempt below produced
the IDENTICAL symptom: `STREAMON` ok, then `frame start syncpt timeout!0`
repeated, zero additional log errors, zero frames ever received.

### 5.1 (History, already fixed, mentioned for context) A bug in our Python test
For many hours the real bug (4.2, masked ENOTTY) was hidden by a trivial mistake
of ours: a wrong size for `struct v4l2_format` in the Python test script (204
bytes instead of 208: the format's inner union is rounded to 8 bytes for struct
alignment). With the wrong ioctl code, `__video_do_ioctl` did not recognize
`VIDIOC_S_FMT` and routed it to the default handler, which silently did
nothing, giving the illusion that "S_FMT succeeds but does nothing". Found with
an unconditional `pr_info` at the top of `video_usercopy()`
(kernel-4.9/drivers/media/v4l2-core/v4l2-ioctl.c) printing side by side the
`cmd` received and the `VIDIOC_S_FMT` constant computed by the kernel itself:
the difference (`0xc0cc5605` vs `0xc0d05605`) revealed the size error.
**Lesson**: when doing raw ioctls by hand in Python, always compute the real C
structs' `sizeof()` precisely; never trust a manual count of the fields.

### 5.2 PLL compensation for a 24 MHz clock instead of 25 MHz (did NOT fix it)
**Hypothesis**: the driver explicitly checks `if (xclk_freq != 25000000)
dev_warn("unusual xclk frequency")`, and prints this warning at every boot,
because the Jetson's physical MCLK (fixed at board level, shared by all camera
sensors, see `assigned-clock-rates` for `TEGRA210_CLK_EXTERN3` in the
board-level DT, NOT in the sensor dtsi) is 24 MHz, not the 25 MHz the whole
sensor register table (PLL multiplier/dividers included) was originally tuned
for. Register `0x4802` (D-PHY timing control) defaults to `0x00` = automatic
computation of ALL HS/CLK timing parameters from the real PLL clock, so a PLL
off by 4% would propagate the error to every MIPI timing parameter.

**Attempted fix**: proportional compensation of the PLL multiplier (register
`0x3036`): `new = round(old x 25/24)`. With `old=0x46=70` -> `new=73=0x49`.

**Result**: no change, the exact same failure.

**Why it was (probably) not the cause**: it later turned out (looking at the
register table of the currently maintained mainline driver, see 5.3) that
`0x3036=0x46` is IDENTICAL between their 10-bit version and our original 8-bit
one. The PLL multiplier does not change between the two modes in the real,
working driver, suggesting the PLL itself is not the problem (or at least that
my proportional compensation, though mathematically reasonable, does not
capture the real mechanism, since upstream does nothing like it and presumably
works on other boards with an XVCLK other than 25 MHz without such a fix).

### 5.3 Full port to 10-bit from the actively maintained mainline driver (did NOT fix it on its own, but kept)
**Hypothesis**: the online research (section 7) showed that ALL working
OV5647-on-Jetson implementations found (RidgeRun's commercial ones for Nano AND
TX1, a community project "giraffai" written from scratch with Tegracam
integration) use RAW10 (10-bit), NEVER RAW8. Our starting driver
(`drivers/media/i2c/ov5647.c`, a historical version from the Linux mainline
repo, author Ramiro Oliveira / Synopsys) uses `MEDIA_BUS_FMT_SBGGR8_1X8` (8-bit)
with register `0x3034=0x08`: a value that, checking the official OV5647
datasheet (section 6.1, system control registers table), **does not even match
a documented value** of the `mipi_bit_mode` field (bits [3:0] of `0x3034`):
only `0000` = 8-bit and `0001` = 10-bit are defined, "others: reserved for
future use". `0x08` in binary has bits [3:0] = `1000`, neither of them.

**Attempted fix**: faithfully ported from
`https://raw.githubusercontent.com/torvalds/linux/master/drivers/media/i2c/ov5647.c`
(current mainline version, updated author, uses `struct reg_sequence` +
`regmap`; we translated the register values into our simpler `regval_list`
structure, keeping the existing raw i2c write logic) its
"`ov5647_640x480_10bpp`" mode (original comment: "10-bit VGA full FOV 60fps.
2x2 binned and subsampled down to VGA", `.pixel_rate=58333000`, `.hts=1852`,
`.vts=0x1f8`=504). Changed `MEDIA_BUS_FMT_SBGGR8_1X8` ->
`MEDIA_BUS_FMT_SBGGR10_1X10` in `get_fmt`/`set_fmt`/`enum_mbus_code`/
`ov5647_open`. Checked that the Tegra format table (`vi/vi2_formats.h`) already
has a native entry for `SBGGR10_1X10` -> `RAW10` CSI datatype, so nothing needed
changing on the Tegra side. Test script updated for `fourcc('BG10')` and 2-byte
pixels.

**Result**: `REQBUFS`/`QUERYBUF` now correctly allocate 614400 bytes
(640x480x2, confirming format negotiation holds with the new format too), but
the exact same `frame start syncpt timeout`. No improvement.

**Status**: the code was KEPT (not reverted to 8-bit) because it is the most
correct and verified version available anyway, even though it did not fix the
main problem on its own.

### 5.4 CSI lane deskew (CHECKED, not relevant)
`tegra_channel_set_stream()` in `channel.c` has a second calibration mechanism,
`nvcsi_deskew_apply_check()`, enabled if `!chan->bypass && !chan->pg_mode &&
chan->deskew_ctx->deskew_lanes`. Checked via debug print: `deskew_ctx` is not
NULL but `deskew_lanes=0`, so the branch never runs. Moreover,
`nvcsi_deskew_apply_check` is a stub that always returns 0 on T210 unless
`CONFIG_TEGRA_GRHOST_NVCSI` is enabled (deskew is an nvcsi feature, typically
T194+/Xavier, not T210/Nano). **Ruled out as the cause.**

### 5.5 VI bypass mode forced on explicitly (did NOT fix it)
**Hypothesis**: `chan->bypass` (V4L2 control `TEGRA_CAMERA_CID_VI_BYPASS_MODE`,
default `.def=0` = OFF) decides whether the VI channel writes RAW data straight
to memory (bypass=1, what we want for raw capture) or goes through a more
ISP-aware path (bypass=0). Checked via debug print that during all previous
tests `bypass=0` (never explicitly set by our final test script; an earlier,
abandoned attempt tried to set it but it was never brought back into the final
script). Hypothesis: maybe with bypass=0 the channel expects ISP-side
signals/config that our raw sensor never provides, blocking the frame start.

**Attempted fix**: an explicit `VIDIOC_S_CTRL` call in the test script
(`~/v4l2_raw_test3.py`) setting `TEGRA_CAMERA_CID_VI_BYPASS_MODE=1` (index 1 =
`SWITCH_ON` in `switch_ctrl_qmenu`) BEFORE `S_FMT`/`STREAMON`.

**Result**: `S_CTRL` succeeds (`set VI_BYPASS_MODE=1 (SWITCH_ON) ok`), and a
debug print confirmed `bypass=1` during the following `set_stream(on=1)`, but
the exact same `frame start syncpt timeout`. No change.

## 6. EXACT state of the system at this point

**On the target** (after the last reboot, kernel build `#15`):
- `/boot/Image-ov5647` = build with: fix 4.1 (camera_common.h) + fix 4.2
  (get/set_fmt) + fix 4.3 (camera_common integration in ov5647.c, compiled into
  the Image because `camera_common.c`/`channel.c`/`csi.c`/`csi2_fops.c` are
  `obj-y`, built-in) + **two diagnostic `pr_info` still present**:
  - `kernel/nvidia/drivers/media/platform/tegra/camera/csi/csi.c`, inside
    `tegra_csi_mipi_calibrate()`, prints `CLAUDE_DBG mipi_calibrate: chan=...
    pg_mode=... s_data=... sensor_sd=... numlanes=...` and then
    `CLAUDE_DBG mipi_calibrate: fops->mipical returned %d`.
  - `kernel/nvidia/drivers/media/platform/tegra/camera/vi/channel.c`, inside
    `tegra_channel_set_stream()`, prints `CLAUDE_DBG set_stream: on=...
    bypass=... pg_mode=... deskew_ctx=...` and (if not NULL)
    `CLAUDE_DBG set_stream: deskew_lanes=...`.

  These prints were **deliberately left in** because they are useful to keep
  debugging. Once the problem is solved they must be removed before the work is
  considered done (search for `CLAUDE_DBG` in both files). *(Done since: neither
  the sources in this repo nor `prebuilt/Image-ov5647` contain them.)*
- `/lib/modules/4.9.253/kernel/drivers/media/i2c/ov5647.ko` = version with the
  10-bit register table (5.3) + camera_common integration (4.3). Declared
  format: SBGGR10_1X10, fixed 640x480.
- `/boot/dtb/tegra210-p3448-0000-p3449-0000-b00-ov5647.dtb` = with the corrected
  `mode0` (4.5).
- `~/v4l2_raw_test3.py` = version forcing bypass=1 (5.5) + fourcc BG10 + correct
  FMT_SIZE=208. This is the most up-to-date test script.

**Locally** (host sources, exactly matching the above):
- `kernel/kernel-4.9/drivers/media/i2c/ov5647.c`: full driver as described in 4.3 + 4.4.
- `kernel/nvidia/include/media/camera_common.h`: fix 4.1 present.
- `kernel/nvidia/drivers/media/platform/tegra/camera/csi/csi.c`: debug print present (section 6).
- `kernel/nvidia/drivers/media/platform/tegra/camera/vi/channel.c`: debug print present (section 6).
- `hardware/.../tegra210-camera-ov5647.dtsi`: corrected mode0 (4.5).

**Latest test result** (identical over several attempts):
```
set VI_BYPASS_MODE=1 (SWITCH_ON) ok
S_FMT -> 640 640 ...  (note: our Python script's parsing of this particular print
                        probably has a struct alignment bug in the debug output;
                        REQBUFS/QUERYBUF are correct and reliable, so the REAL
                        format negotiation is verified correct even if this line
                        looks odd)
G_FMT (chan->format) -> 0 640 ...
REQBUFS -> count= 4
QUERYBUF 0..3 -> length= 614400   (= 640*480*2, correct for 10-bit)
STREAMON ok
timeout waiting for frame 0
```
dmesg shows: MIPI calibration run successfully (`fops->mipical returned 0`),
`bypass=1` confirmed, then repeated `frame start syncpt timeout!0`, no other
error.

## 7. Online research: what was found

- **NVIDIA developer forum** (thread "Mipi-driver : frame start syncpt
  timeout!0", 3 pages): the exact same symptom with a custom sensor on a
  Jetson. **Never conclusively solved** in the thread: it drifted into hardware
  checks (pin connection, clock) without a conclusion. Other NVIDIA threads on
  the same error ("Got frame start syncpt timeout!0", "Nano camera capture
  error", etc.): a recurring pattern, no definitive public solution found for a
  case equivalent to ours.
- **`Moore123/ov5647_jetson`** (GitHub): an attempt at an OV5647 driver for
  Jetson with tegracam integration (`tegracam_core.h`), explicitly marked
  **"Not Working Yet"** in its README.
- **RidgeRun** (`developer.ridgerun.com`, a serious embedded consultancy): sells
  commercial OV5647 drivers for Jetson Nano AND Jetson TX1, **both explicitly
  RAW10 / Bayer BGGR 10-bit**, never RAW8. No public price; the patch tarball
  only on direct commercial contact (not freely downloadable). No public
  technical detail on how they solve CSI/PLL timing.
- **"giraffai.com" blog**, 5-part series "Developing OV5647 camera sensor Linux
  device driver for Jetson Nano": the only public account of someone writing an
  OV5647 driver FROM SCRATCH with real Tegracam integration and **getting real
  frames** (command quoted: `v4l2-ctl --set-fmt-video=... pixel=RG10 ...
  --set-ctrl bypass_mode=0`; note: they use `bypass_mode=0`, NOT 1 as in our
  hypothesis 5.5; an inconsistency never cleared up). It does not document the
  exact technical details of how they solved CSI timing / frame sync: the author
  refers to "a demo video" instead of explaining it in writing.
- **`torvalds/linux` mainline `drivers/media/i2c/ov5647.c`** (current version,
  much newer than ours): used as the source for the port in 5.3. Confirms `xclk`
  must be exactly 25 MHz (in recent versions it is now a **fatal error**, not
  just a warning). Uses `regmap`/`cci_*` instead of raw i2c. No dynamic PLL
  computation for XVCLKs other than 25 MHz: "adapting to a different XVCLK" is
  not solved upstream either.
- **Official OV5647 datasheet** (OmniVision, "PRELIMINARY SPECIFICATION",
  version 1.0, November 2009, downloaded and read page by page): confirms "input
  clock frequency: 6~27 MHz" (so 24 MHz is well within the sensor's supported
  range in theory), "PLL... generates a default 96 MHz internal clock from a
  6-27 MHz input" (implying the PLL should normalize the output whatever the
  input, if configured correctly for that input). Table 6-1 (system control
  registers, pages 6-3/6-4) gave the exact bit fields of `0x3034`-`0x3037` (PLL
  charge pump, mipi_bit_mode, system_clk_div, scale_divider_mipi, PLL multiplier
  4-252, pll_root_div, pll_prediv) used to derive the fix in 5.2 (which turned
  out insufficient on its own).

## 8. Untested ideas for the next attempt

In order of estimated cost/information (cheapest and most informative first):

1. **Read the real CSI status registers at the moment of the timeout** (never
   done). `csi/csi2_fops.c` already has the `csi_read()`/`cil_read()` helpers
   used everywhere to write `TEGRA_CSI_PIXEL_PARSER_STATUS`,
   `TEGRA_CSI_CIL_STATUS`, `TEGRA_CSI_CILX_STATUS` (seen written with
   `0xFFFFFFFF` to "clear" the status in `csi2_start_streaming()`, so they are
   readable and meaningful). A `pr_info` that READS these registers right after
   `nvhost_syncpt_wait_timeout_ext()` fails in `vi2_fops.c` (around lines
   443-454, where it already prints "frame start syncpt timeout") would give the
   real hardware state of the D-PHY receiver: whether it saw even a single LP->HS
   transition (probably a framing/CRC problem, closer to a software/fine timing
   bug) or saw absolutely NOTHING (much more indicative of an electrical/physical
   problem or a wrong lane / CSI port configuration). Probably the single best
   next step for information versus risk.

2. **Try Tegra's built-in Test Pattern Generator (TPG)** (never done). The VI2
   driver supports `chan->pg_mode`, a path that generates synthetic data inside
   the SoC, completely independent of the physical sensor and cable. If the TPG
   produces valid frames through the same VI/CSI channel, the problem is
   isolated to the sensor or cable (not Tegra); if the TPG fails with the same
   timeout too, it points to a VI/CSI configuration problem independent of the
   sensor (never considered so far). See `graph.c`/`channel.c` for how pg_mode
   is enabled (probably a dedicated `tpg` DT entry, or debugfs).

3. **Check the physical CSI port/lane assignment** (never checked thoroughly).
   The Jetson Nano has several CSI ports (CAM0/CAM1) mapped to NVCSI ports A/B
   with specific lane routing. If the sensor DT declares the wrong CSI port for
   the physical connector actually used, calibration and reception would happen
   on the wrong physical lanes: consistent with "everything configured right but
   zero signal ever received". Check `csi_port`/`reg` in the DT nodes
   (`vi_port0`, `csi_chan0`, etc. in `tegra210-camera-ov5647.dtsi`) against the
   connector actually used (CAM0 vs CAM1).

4. **Compare line by line with `digitallyamar/ov5647`** (GitHub, found but never
   examined in detail: `https://github.com/digitallyamar/ov5647`). A newer
   repository than the initial research, not yet checked for technical content
   or status.

5. **Ask RidgeRun for a quote** (commercial, not technical): they claim a working
   driver for the Jetson Nano specifically. Budget permitting, it could solve it
   outright, skipping the whole investigation.

6. **Reconsider the hardware**: an IMX219 (Raspberry Pi Camera v2, natively
   supported by the Jetson Nano, zero driver work) remains the most pragmatic
   option if the main goal is "a working camera for YOLOv5" rather than "making
   this specific sensor work". The user said buying one is not possible now: to
   reconsider if points 1-4 do not unblock anything.

## Resolution (2026-07-18, following session)

Step 8.1 (reading the CSI status registers at the timeout) was done WITHOUT a
rebuild: the timeout recovery path already calls `tegra_channel_capture_error()`
-> `tegra_csi_status()`, which prints exactly the registers wanted, only at
`dev_dbg` level; with `CONFIG_DYNAMIC_DEBUG=y` (already on) it is enough to run
`echo 'file csi2_fops.c +p' | sudo tee /sys/kernel/debug/dynamic_debug/control`
(same for `vi2_fops.c`), re-run the test and read dmesg.

**Diagnosis**: `PIXEL_PARSER_STATUS=0x0` always (never decoded a packet), but
`CIL_STATUS=0x10` (CILA_CTRL_ERR: LP sequences aborted in LP11) and
`CILX_STATUS=0x00040041` (CTRL_ERR on the clock lane, bit 0 + data lane 0, bit 6
+ data lane 1, bit 18). So: the sensor was transmitting, but the receiver never
locked LP -> HS.

**Cause 1, MIPI clock**: the DT declares `discontinuous_clk="yes"` but
`__sensor_init()` wrote `0x4800=0x04` (clock lane gate OFF -> CONTINUOUS clock),
overwriting the register table's 0x34; moreover the sensor came out of power-on
already streaming (never an observable LP11 -> HS transition). Fix in
`ov5647.c` (like mainline's `clock-noncontinuous`): stream_on -> `0x4800=0x34`
(gate + line sync + bus idle); stream_off -> `0x4800=0x25` (lanes in LP-11);
`__sensor_init()` ends with `ov5647_stream_off()`.

**Cause 2, bypass inverted (correcting 5.5)**: `TEGRA_CAMERA_CID_VI_BYPASS_MODE`
must be **0**: with bypass=1 `vi2_channel_start_streaming()` returns right after
the sensor's set_stream: no capture kthread, no syncpt armed (which is why those
runs did not even print the timeout message). giraffai was right.

**Cause 3, black frames**: the mainline 10-bit table uses `0x3503=0x03`
(manual AEC/AGC, driven by V4L2 controls this driver does not have) -> exposure
and gain ~0. Fix: `0x3503=0x00` (auto). AEC converges in ~2-3 s of continuous
streaming.

**Real Bayer order: GRBG** (checked with a red object: correct with
`cv2.COLOR_BayerGR2BGR`; driver/DT declare BGGR, a label that is harmless for
the raw pipeline).

**CV pipeline (every script tested and working)**:
- `common/camera.py` rewritten on raw V4L2 ioctls (backup:
  `camera.py.argus_backup`); identical interface (read/isOpened/release), GRBG
  demosaic, ~62.5 fps. (In this repo: `nano/jetson_cv/camera.py`.)
- 01 camera_test: 62.5 fps; 02 live_detect (torch FP16 yolov5n @ 416): 17.8 fps;
  03 trt_detect (FP16 416 engine, built): **30.6 fps live**; 04 benchmark: TRT
  23.6 ms vs PyTorch FP16 43.9 ms @ 416; 05 capture_images ok (use
  `--width 640 --height 480`, other sizes are a software resize).
- Environment fixes made on the target: optional seaborn import in
  `~/yolov5/utils/plots.py` (yolov5 then auto-installed seaborn 0.11.2 in the
  venv); `~/.config/Ultralytics/Arial.ttf` (a copy of DejaVuSans, the download
  URL is dead); pycuda 2021.1 installed with `--no-build-isolation` (build
  isolation compiled numpy 1.12 -> xlocale.h error);
  `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` for cv2-after-torch
  (added to build_engine.sh; also needed to run trt_detect/benchmark).

**Remaining cleanup (not blocking)**: remove the two `CLAUDE_DBG` `pr_info`s
(csi.c, channel.c) from the Image *(done since)*; possibly align the format
label (SGRBG10 in the driver + `pixel_t=bayer_grbg` in the DT).
`~/v4l2_raw_test3.py` now captures 300 frames (bypass=0).

## 9. A note on the reliability of this document

Every fact about code, paths, register values and test results in this
document was verified directly (reading the sources, running on real hardware,
reading dmesg) during the session; none are untested hypotheses, except where
explicitly marked "hypothesis" / "not yet tested". The web research results
(section 7) are summaries of external sources and should be re-checked before
basing new decisions on them (the links were valid at the time of the
research, 2026-07-18).
