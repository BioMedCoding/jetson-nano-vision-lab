#!/usr/bin/env bash
# Interactive build wizard for the Jetson Nano OV5647 kernel.
# Builds dtbs + Image + modules from camera/kernel-source/kernel/kernel-4.9,
# which already has every patch in ../patches applied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KDIR="$REPO_ROOT/kernel-source/kernel/kernel-4.9"

BOLD=$'\033[1m'; RESET=$'\033[0m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
step()  { echo "${BOLD}==> $*${RESET}"; }
info()  { echo "    $*"; }
warn()  { echo "${YELLOW}    $*${RESET}"; }
error() { echo "${RED}    $*${RESET}" >&2; }

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

if [ ! -d "$KDIR" ]; then
  error "Kernel source not found at $KDIR"
  error "Did you clone the full repository (not a shallow/partial checkout)?"
  exit 1
fi

step "Jetson Nano OV5647 kernel — build wizard"
info "Repo root:    $REPO_ROOT"
info "Kernel tree:  $KDIR"
echo

step "Checking cross toolchain"
CROSS_COMPILE="$(ask "Cross-compile prefix" "aarch64-linux-gnu-")"
if ! command -v "${CROSS_COMPILE}gcc" >/dev/null 2>&1; then
  error "${CROSS_COMPILE}gcc not found on PATH."
  info  "On Ubuntu/Debian: sudo apt install gcc-aarch64-linux-gnu"
  info  "See docs/GCC15_NOTES.md — this tree is verified to build with"
  info  "the distro's GCC (tested with 15.2.0), NVIDIA's older Linaro"
  info  "toolchain is not required."
  exit 1
fi
info "Found: $("${CROSS_COMPILE}gcc" --version | head -1)"
echo

step "Kernel configuration"
CONFIG_REF="$REPO_ROOT/config/nano-ov5647.config"
if [ -f "$KDIR/.config" ]; then
  info "An existing .config is already present in the kernel tree."
  if ask_yn "Overwrite it with the known-good reference config from config/nano-ov5647.config?" "n"; then
    cp "$CONFIG_REF" "$KDIR/.config"
    info "Reference config installed."
  fi
else
  info "No .config present — installing the known-good reference config."
  cp "$CONFIG_REF" "$KDIR/.config"
fi

if ask_yn "Open 'make menuconfig' now to review/adjust config before building?" "n"; then
  make -C "$KDIR" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" menuconfig
else
  make -C "$KDIR" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" olddefconfig >/dev/null
fi
echo

JOBS="$(ask "Parallel build jobs" "$(nproc)")"

step "What do you want to build?"
echo "  1) Everything (dtbs + Image + modules) — recommended for a first build"
echo "  2) Just the device trees (fast, after a DT-only change)"
echo "  3) Just the kernel Image (after a driver-code-only change)"
echo "  4) Just the modules"
TARGET_CHOICE="$(ask "Choice" "1")"

run_make() {
  info "make ARCH=arm64 CROSS_COMPILE=$CROSS_COMPILE -j$JOBS $*"
  make -C "$KDIR" ARCH=arm64 CROSS_COMPILE="$CROSS_COMPILE" -j"$JOBS" "$@"
}

echo
step "Building"
case "$TARGET_CHOICE" in
  1) run_make dtbs; run_make Image; run_make modules ;;
  2) run_make dtbs ;;
  3) run_make Image ;;
  4) run_make modules ;;
  *) error "Unknown choice: $TARGET_CHOICE"; exit 1 ;;
esac

echo
step "${GREEN}Build finished${RESET}"
IMAGE="$KDIR/arch/arm64/boot/Image"
DTB="$KDIR/arch/arm64/boot/dts/tegra210-p3448-0000-p3449-0000-b00-ov5647.dtb"
[ -f "$IMAGE" ] && info "Kernel Image: $IMAGE ($(du -h "$IMAGE" | cut -f1))"
[ -f "$DTB" ]   && info "Camera DTB:   $DTB"
info "Built .ko module objects are still inside the source tree next to their"
info "*.c files; deploy.sh runs 'make modules_install' straight onto the SD"
info "card, there's no separate local staging step."
echo
info "Next: run camera/scripts/deploy.sh to copy this onto the Jetson's SD card."
