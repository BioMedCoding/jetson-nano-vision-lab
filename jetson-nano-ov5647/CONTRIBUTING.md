# Working on this repo — how to add features without breaking the archive

This repo has one unusual property to respect: it is the **archive of record**
— sources, binaries and docs must stay in agreement with each other and with
what actually runs on the Jetson. Every workflow below ends with "verify on
hardware, then commit source + prebuilt + docs together".

## The two-tree situation (on the original dev machine)

On the machine this was developed on, kernel work happens in a separate
build tree (`~/jetson_work/Linux_for_Tegra/source/public/`) and is then
synced here. If you're starting fresh from a clone, you don't have that
split: `source/public/` in this repo **is** your build tree —
`scripts/build.sh` builds it in place. Either way the rule is the same:
whatever tree you build from, the repo must end up containing the exact
sources of the binaries in `prebuilt/`.

## Adding/changing a kernel-side feature (driver, DT)

1. Edit sources (in your build tree or directly in `source/public/`).
2. Build: `./scripts/build.sh`, or by hand from `source/public/kernel/kernel-4.9`:
   `export CROSS_COMPILE=aarch64-linux-gnu- ARCH=arm64`
   - `make -j$(nproc) Image` (~10 s incremental)
   - `make -j$(nproc) M=drivers/media/i2c modules` (ov5647.ko only, seconds)
   - `make -j$(nproc) dtbs` (if DT changed)
3. Deploy to the Jetson: `scp` to the user's home, then on the Jetson
   `sudo cp` into place (`/boot/Image-ov5647`, `/boot/dtb/…`,
   `/lib/modules/4.9.253/…` + `sudo depmod -a`) and **reboot** — the ov5647
   module cannot be hot-reloaded (the built-in VI driver pins it) and Image/DTB
   obviously need a boot. `scripts/deploy.sh` automates the SD-card variant.
4. Verify on hardware. Minimum bar for camera changes:
   `python3 ~/cv_projects/01_camera_test/camera_test.py --headless`
   (expect ~62.5 fps) and a look at an actual frame. For anything CSI-related,
   arm the zero-cost tracing first — see `docs/CAMERA_DEBUG.md`, "The
   technique" (dynamic debug on `csi2_fops.c`/`vi2_fops.c`).
5. Sync back into the repo (if you edited an external tree) and regenerate the
   patch documentation:
   ```bash
   git diff d3f7855bb --relative=source/public -- source/public/<file> \
       > patches/00XX-short-name.patch
   ```
   New logical change = new numbered patch + a row in `patches/README.md`
   (say *why*, not just what — the table is a debugging diary, keep it one).
6. Refresh `prebuilt/` with the binaries you just verified, never with
   unverified ones. If you verified a build, prebuilt gets *that* build.
7. One commit with all of it: source + patch + prebuilt + docs.

Ground rules learned the hard way (see `docs/`): never rename the kernel
LOCALVERSION (modules dir must equal `uname -r` or the Nano won't boot
userspace); never "fix" a build error by disabling a driver or Kconfig option;
debug prints don't ship — use `dev_dbg` + dynamic debug instead of `pr_info`.

## Adding/changing CV features (detection scripts, camera userspace)

1. Edit under `cv/cv_projects/` (repo) and rsync to the Jetson, or edit on the
   Jetson and rsync back — but the repo copy is the one that counts.
2. Test on the Jetson inside the venv (`source ~/cv_projects/venv/bin/activate`
   — it already exports `OPENBLAS_CORETYPE` and `LD_PRELOAD`). All scripts
   support `--headless` (MJPEG on `http://<jetson>:8090/`), so everything is
   testable over SSH.
3. Camera invariants that scripts must respect are listed in `cv/README.md`
   ("Camera facts"): fixed 640x480 RAW10 mode, GRBG demosaic, bypass_mode=0.
   Don't reintroduce GStreamer/Argus paths — they can't work with this sensor.
4. New Python dependency? It must install on **Python 3.6 aarch64, offline
   preferred**: add it to `setup/requirements_jetson.txt` only after testing
   `pip install` on the Nano; if it needs compilation or a vendored wheel, put
   the wheel in `setup/wheels/` (split any file ≥95 MB like torch:
   `split -b 90M`, add the sha256 to a `SHA256SUM`, teach `setup_jetson.sh`
   to reassemble — it already does this for torch, copy that pattern).
5. If you touched `setup_jetson.sh` or the launcher, keep the three copies in
   sync: this repo, the Jetson, and any Desktop backup in use.

## Adding a new vendored archive (new sensor source drop, firmware, datasheets)

Own top-level directory (`vendor/<name>/` pattern), with a README stating
origin, date, license, and why it's vendored. Files ≥95 MB: split + checksum,
as above. That's what keeps this repo useful after the upstream links die —
which is its whole reason to exist.

## Publishing

The repo currently lives locally (no remote). To publish:
`gh auth login`, then
`gh repo create jetson-nano-ov5647 --public --source . --push`.
Largest file is 94 MB, so plain GitHub works — no LFS needed. Keep it that
way: check any new file against the 100 MB limit *before* committing it.
