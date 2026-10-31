#!/bin/bash
# Install / remove the systemd service that starts the dashboard at boot
# (and restarts it if it crashes). Run ON the Jetson; uses sudo.
#
# Usage:
#   nano/setup/install_service.sh                         # install and start
#   nano/setup/install_service.sh --allow-system-control  # ... with power/fan controls
#   nano/setup/install_service.sh --model yolov8n-416     # ... with another model
#   nano/setup/install_service.sh --uninstall             # stop, disable, remove
#
# Any option other than --uninstall is passed to apps/dashboard.py.
set -e

UNIT_NAME="cv-demo.service"
NANO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$NANO_DIR/setup/systemd/$UNIT_NAME.in"
DEST="/etc/systemd/system/$UNIT_NAME"

if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERROR: run this on the Jetson, not on your PC."; exit 1
fi

if [ "$1" = "--uninstall" ]; then
    sudo systemctl disable --now "$UNIT_NAME" 2>/dev/null || true
    sudo rm -f "$DEST"
    sudo systemctl daemon-reload
    echo "Removed $UNIT_NAME."
    exit 0
fi

[ -x "$NANO_DIR/.venv/bin/python3" ] || { echo "ERROR: run nano/setup/setup_jetson.sh first."; exit 1; }
if ! id -nG "$USER" | grep -qw video; then
    echo "ERROR: $USER is not in the 'video' group (needed for /dev/video0)."
    echo "  sudo usermod -aG video $USER   then log out and back in."
    exit 1
fi

TMP=$(mktemp)
sed -e "s#@USER@#$USER#g" -e "s#@NANO_DIR@#$NANO_DIR#g" -e "s#@EXTRA_ARGS@#$*#" \
    "$TEMPLATE" > "$TMP"
if [ -f "$DEST" ] && ! cmp -s "$TMP" "$DEST"; then
    echo "Replacing the existing $DEST (diff below):"
    diff "$DEST" "$TMP" || true
fi
sudo install -m 644 "$TMP" "$DEST"
rm -f "$TMP"
sudo systemctl daemon-reload
sudo systemctl enable "$UNIT_NAME"
sudo systemctl restart "$UNIT_NAME"

echo
echo "Installed. Check it with:"
echo "  systemctl status $UNIT_NAME"
echo "  journalctl -u $UNIT_NAME -f"
echo "  open http://$(hostname).local:8090/ in a browser on your network"
