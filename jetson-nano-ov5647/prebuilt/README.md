# Prebuilt kernel artifacts — ready to deploy, no build needed

Built from exactly the sources in `../source/public/` (GCC 15.2.0 cross
aarch64, config `../config/nano-ov5647.config`) and verified on hardware:
OV5647 capturing at 62.5 fps, YOLOv5n TensorRT inference at 30.6 fps live
(2026-07-18).

| File | Destination on the Jetson |
|---|---|
| `Image-ov5647` | `/boot/Image-ov5647` |
| `tegra210-p3448-0000-p3449-0000-b00-ov5647.dtb` | `/boot/dtb/` |
| `ov5647.ko` | `/lib/modules/4.9.253/kernel/drivers/media/i2c/` then `sudo depmod -a` |

The kernel identifies as plain `4.9.253` (no LOCALVERSION) **on purpose**: the
modules directory name must match `uname -r` exactly or the Nano fails to boot
usable userspace ("Failed to load kernel modules"). Don't "fix" that.

Boot entry (`/boot/extlinux/extlinux.conf`) — add alongside the stock one, or
let `../scripts/deploy.sh` do all of this interactively:

```
LABEL ov5647
      MENU LABEL OV5647 kernel
      LINUX /boot/Image-ov5647
      FDT /boot/dtb/tegra210-p3448-0000-p3449-0000-b00-ov5647.dtb
      INITRD /boot/initrd
      APPEND ${cbootargs} quiet root=/dev/mmcblk0p1 rw rootwait rootfstype=ext4 console=ttyS0,115200n8 console=tty0 fbcon=map:0 net.ifnames=0
```

(Copy `APPEND` from the existing `primary` entry of *your* SD image rather
than trusting the line above — cboot args can differ between L4T point
releases.)

A reboot is required after replacing Image or DTB. The ov5647 module alone
can't be hot-reloaded either (the built-in VI driver holds a permanent
reference to the subdev) — reboot after replacing it too.
