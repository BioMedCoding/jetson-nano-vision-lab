#!/usr/bin/env bash
# Interactive deploy wizard: copies a built kernel Image + DTB + modules onto
# a Jetson Nano's rootfs (SD card, USB, or NVMe) and wires up the extlinux
# boot menu. Must be run as root (needs to write to the target's /boot and
# /lib/modules, and 'depmod' needs root to read them).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KDIR="$REPO_ROOT/source/public/kernel/kernel-4.9"
KERNEL_RELEASE="4.9.253"
DTB_NAME="tegra210-p3448-0000-p3449-0000-b00-ov5647.dtb"

BOLD=$'\033[1m'; RESET=$'\033[0m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
step()  { echo "${BOLD}==> $*${RESET}"; }
info()  { echo "    $*"; }
warn()  { echo "${YELLOW}    $*${RESET}"; }
error() { echo "${RED}    $*${RESET}" >&2; }
die()   { error "$*"; exit 1; }

ask() {
  local prompt="$1" default="$2" reply
  read -r -p "$prompt [$default]: " reply || true
  echo "${reply:-$default}"
}

ask_yn() {
  local prompt="$1" default="$2" reply
  read -r -p "$prompt [${default}]: " reply || true
  reply="${reply:-$default}"
  [[ "$reply" =~ ^[Yy] ]]
}

[ "$(id -u)" -eq 0 ] || die "Run this with sudo — it writes to the target's /boot and /lib/modules."

IMAGE_SRC="$KDIR/arch/arm64/boot/Image"
DTB_SRC="$KDIR/arch/arm64/boot/dts/$DTB_NAME"
[ -f "$IMAGE_SRC" ] || die "No built Image at $IMAGE_SRC — run scripts/build.sh first."
[ -f "$DTB_SRC" ]   || die "No built DTB at $DTB_SRC — run scripts/build.sh first."

step "Jetson Nano OV5647 kernel — deploy wizard"
info "Kernel Image: $IMAGE_SRC"
info "DTB:          $DTB_SRC"
echo

step "Finding target devices"
ROOT_DISK="$(lsblk -no PKNAME "$(findmnt -no SOURCE /)" 2>/dev/null || true)"
[ -n "$ROOT_DISK" ] || ROOT_DISK="$(basename "$(findmnt -no SOURCE / | sed -E 's/p?[0-9]+$//')" )"

# lsblk -P (key=value pairs) is used instead of the plain/raw table format
# because MOUNTPOINT is empty for an unmounted card (the common case for a
# freshly-inserted SD card) — a plain-format column would collapse under
# 'read' and misalign every field after it.
CAND_NAMES=(); CAND_SIZES=(); CAND_MNTS=()
while IFS= read -r line; do
  eval "$line"
  [ "$TYPE" = "part" ] || continue
  [ "$FSTYPE" = "ext4" ] || continue
  [ "$PKNAME" = "$ROOT_DISK" ] && continue
  CAND_NAMES+=("$NAME"); CAND_SIZES+=("$SIZE"); CAND_MNTS+=("$MOUNTPOINT")
done < <(lsblk -P -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT,PKNAME)

if [ "${#CAND_NAMES[@]}" -eq 0 ]; then
  die "No ext4 partitions found besides the host's own disk ($ROOT_DISK). Is the SD card / target disk connected and its rootfs partition an ext4 one?"
fi

echo "    #  DEVICE       SIZE     MOUNTPOINT"
for i in "${!CAND_NAMES[@]}"; do
  printf "    %d  /dev/%-9s %-8s %s\n" "$((i+1))" "${CAND_NAMES[$i]}" "${CAND_SIZES[$i]}" "${CAND_MNTS[$i]:-<not mounted>}"
done
echo

CHOICE="$(ask "Pick target partition by number" "1")"
IDX=$((CHOICE-1))
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "${#CAND_NAMES[@]}" ] || die "Invalid choice."
DEV_NAME="${CAND_NAMES[$IDX]}"
DEV_SIZE="${CAND_SIZES[$IDX]}"
DEV_MNT="${CAND_MNTS[$IDX]}"
DEV="/dev/$DEV_NAME"

warn "About to write to ${BOLD}$DEV${RESET}${YELLOW} (${DEV_SIZE}) — double-check this is really the Jetson's rootfs and not something else."
CONFIRM="$(ask "Type the device name to confirm ($DEV_NAME)" "")"
[ "$CONFIRM" = "$DEV_NAME" ] || die "Confirmation did not match — aborting, nothing was touched."
echo

step "Mounting"
if [ -n "$DEV_MNT" ]; then
  TARGET="$DEV_MNT"
  info "Already mounted at $TARGET"
else
  MOUNT_OUT="$(udisksctl mount -b "$DEV" 2>&1)" || die "Mount failed: $MOUNT_OUT"
  TARGET="$(echo "$MOUNT_OUT" | grep -oP "(?<=at ).*" | tr -d '.')"
  info "Mounted at $TARGET"
fi

EXTLINUX="$TARGET/boot/extlinux/extlinux.conf"
[ -f "$EXTLINUX" ] || die "$EXTLINUX not found — this doesn't look like a flashed L4T rootfs. Aborting."
echo

step "Installing kernel Image + DTB"
mkdir -p "$TARGET/boot/dtb"
install -m 644 -o root -g root "$IMAGE_SRC" "$TARGET/boot/Image-ov5647"
install -m 644 -o root -g root "$DTB_SRC" "$TARGET/boot/dtb/$DTB_NAME"
info "Installed /boot/Image-ov5647 and /boot/dtb/$DTB_NAME"
echo

if ask_yn "Install kernel modules too? (needed the first time, or after any driver change)" "y"; then
  step "Installing modules"
  OLD_MOD_DIR="$TARGET/lib/modules/$KERNEL_RELEASE"
  if [ -d "$OLD_MOD_DIR" ]; then
    BACKUP="${OLD_MOD_DIR}.bak-$(date +%Y%m%d%H%M%S)"
    mv "$OLD_MOD_DIR" "$BACKUP"
    info "Backed up existing modules to $BACKUP"
  fi
  make -C "$KDIR" ARCH=arm64 modules_install INSTALL_MOD_PATH="$TARGET" >/dev/null
  depmod -b "$TARGET" "$KERNEL_RELEASE"
  info "Modules installed to $OLD_MOD_DIR and depmod run"
else
  info "Skipped module install."
fi
echo

step "Boot menu"
echo "  1) Dual-boot safety net — modified kernel boots by default, 10s window to interrupt (recommended)"
echo "  2) Boot the modified kernel almost immediately (~1s window to interrupt)"
echo "  3) Always show the menu and wait for a manual choice"
BOOT_CHOICE="$(ask "Choice" "1")"
case "$BOOT_CHOICE" in
  1) TIMEOUT_VAL=100 ;;
  2) TIMEOUT_VAL=10 ;;
  3) TIMEOUT_VAL=0 ;;
  *) die "Unknown choice: $BOOT_CHOICE" ;;
esac

EXISTING_APPEND="$(grep -m1 -E '^\s*APPEND' "$EXTLINUX" | sed -E 's/^\s*APPEND\s*//')"
if [ -z "$EXISTING_APPEND" ]; then
  warn "Could not find an existing APPEND line to copy console/root arguments from."
  EXISTING_APPEND='${cbootargs} quiet root=/dev/mmcblk0p1 rw rootwait rootfstype=ext4 console=ttyS0,115200n8 console=tty0 fbcon=map:0 net.ifnames=0'
  warn "Falling back to a generic SD-card default — verify this matches your setup:"
  warn "$EXISTING_APPEND"
fi

BACKUP_CONF="${EXTLINUX}.bak-$(date +%Y%m%d%H%M%S)"
cp "$EXTLINUX" "$BACKUP_CONF"
info "Backed up existing extlinux.conf to $BACKUP_CONF"

cat > "$EXTLINUX" <<EOF
TIMEOUT $TIMEOUT_VAL
DEFAULT ov5647

MENU TITLE L4T boot options

LABEL primary
      MENU LABEL primary kernel
      LINUX /boot/Image
      INITRD /boot/initrd
      APPEND $EXISTING_APPEND

LABEL ov5647
      MENU LABEL Jetson Nano with OV5647 camera
      LINUX /boot/Image-ov5647
      FDT /boot/dtb/$DTB_NAME
      INITRD /boot/initrd
      APPEND $EXISTING_APPEND
EOF
info "Wrote new $EXTLINUX (TIMEOUT=$TIMEOUT_VAL, DEFAULT=ov5647)"
echo

mkdir -p "$TARGET/etc"
{
  echo "Deployed by jetson-nano-ov5647/scripts/deploy.sh"
  echo "Date: $(date -Is)"
  echo "Kernel release: $KERNEL_RELEASE"
  echo "DTB: $DTB_NAME"
} > "$TARGET/etc/ov5647-kernel-deploy.info"

sync
step "${GREEN}Done${RESET}"
info "extlinux.conf still has the stock kernel as a 'primary' fallback entry —"
info "interrupt the boot menu (any key on a serial console or attached keyboard)"
info "and pick it if the modified kernel doesn't come up."
echo
if ask_yn "Unmount $TARGET now?" "y"; then
  sync
  udisksctl unmount -b "$DEV" && info "Unmounted — safe to remove the card."
fi
