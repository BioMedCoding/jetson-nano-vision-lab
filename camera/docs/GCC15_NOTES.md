# Building L4T R32.7.1 (kernel 4.9.253) with GCC 15

NVIDIA's documented toolchain for L4T R32.7.1 is Linaro GCC 7.3.1
(`gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu`), 8 major versions
behind. This tree has only ever been built here with the Ubuntu-packaged
`aarch64-linux-gnu-gcc` 15.2.0, and it works end to end (`make dtbs`,
`make Image`, `make modules` — 664 modules, all installed and running on
target). No Docker container or legacy toolchain turned out to be necessary.

## Ground rule followed throughout

**Never disable a driver or Kconfig option to make a build error go away**,
even for hardware that obviously isn't on this board (10G NICs, exotic WiFi
chipsets pulled in by the generic arm64 defconfig). The goal of every fix in
`../patches/` is "everything works exactly like the stock L4T image, plus the
camera" — not a trimmed-down kernel. When GCC15's stricter default
diagnostics broke a build, the fix was always, in this order:

1. Fix the actual bug, if the warning revealed a real logic error (most of
   the `drivers/nvidia/...` fixes below are this — dead `&x == NULL` checks,
   a missing macro brace that skipped a conditional, a mis-typed header
   guard).
2. Otherwise, downgrade only the specific new warning class from error back
   to warning in the top-level `Makefile`, using the same `cc-disable-warning`
   pattern NVIDIA's own Makefile already uses for older warning classes. This
   changes no driver behavior and disables no hardware.
3. Only disable a Kconfig option if you, the person building this, decide you
   don't need that hardware. Nothing here does that by default.

## Why the dtb build failed first (unrelated to the camera)

The very first full build attempt failed on `.dtbo` overlay targets with
`target given more than once` and `cc1: .. No such file or directory` — on
**stock** dtbs, before any camera-specific file existed. Root cause:
`scripts/Kbuild.include`'s `the-space` variable relied on a GNU Make trick
(`the-space := ` then `the-space += ` with a trailing space) that no longer
worked, corrupting `dtb-y`/`dtbo-y` list construction in
`arch/arm64/boot/dts/Makefile`. This was **not** a GNU Make version issue —
reproduced identically with a from-source GNU Make 4.3, ruling out a
"stricter duplicate pattern rule" theory. Fixed with the standard, robust
idiom (`the-empty := ` / `the-space := $(the-empty) $(the-empty)`) — see
`../patches/0007-kbuild-include-the-space-fix.patch`.

## Toolchain state this was built with

```
$ aarch64-linux-gnu-gcc --version
aarch64-linux-gnu-gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0
```
Installed via `apt install gcc-aarch64-linux-gnu` on Ubuntu — no separate
Linaro toolchain download needed. If a future Ubuntu/GCC release breaks
something new, the fix pattern above (real bug first, `cc-disable-warning`
second) is the one to keep following.
