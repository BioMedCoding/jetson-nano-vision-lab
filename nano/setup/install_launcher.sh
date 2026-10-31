#!/bin/bash
# Put a "Vision Dashboard" shortcut on the Jetson's desktop and in the
# applications menu. Nothing starts at boot: double-click it when you want the
# dashboard (it runs setup/launch_dashboard.sh in a terminal window).
# Run ON the Jetson, as your desktop user (no sudo).
#
# Usage:
#   nano/setup/install_launcher.sh               # install
#   nano/setup/install_launcher.sh --uninstall   # remove
set -e

NAME="jetson-vision-dashboard.desktop"
NANO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
APPS_DIR="$HOME/.local/share/applications"

if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERROR: run this on the Jetson, not on your PC."; exit 1
fi

if [ "$1" = "--uninstall" ]; then
    rm -f "$DESKTOP_DIR/$NAME" "$APPS_DIR/$NAME"
    echo "Removed the shortcut."
    exit 0
fi

chmod +x "$NANO_DIR/setup/launch_dashboard.sh"
mkdir -p "$DESKTOP_DIR" "$APPS_DIR"
cat > "$APPS_DIR/$NAME" <<EOF
[Desktop Entry]
Type=Application
Name=Vision Dashboard
Comment=Live camera + TensorRT detection dashboard (http://localhost:8090/)
Exec=gnome-terminal --title "Vision Dashboard" -- $NANO_DIR/setup/launch_dashboard.sh
Icon=camera-web
Terminal=false
Categories=Graphics;Video;
EOF
chmod +x "$APPS_DIR/$NAME"
install -m 755 "$APPS_DIR/$NAME" "$DESKTOP_DIR/$NAME"
# GNOME/Unity only runs desktop-folder launchers marked as trusted.
gio set "$DESKTOP_DIR/$NAME" metadata::trusted true 2>/dev/null || true

echo "Installed: $DESKTOP_DIR/$NAME (also in the applications menu)."
echo "Double-click it to start the dashboard; close its window to stop it."
