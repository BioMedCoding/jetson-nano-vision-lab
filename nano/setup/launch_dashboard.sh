#!/bin/bash
# Start the dashboard on the Jetson's own desktop and open it in the browser.
# The desktop shortcut made by setup/install_launcher.sh runs this in a
# terminal window: closing that window (or Ctrl+C) stops the dashboard.
#
# Usage: nano/setup/launch_dashboard.sh [dashboard.py options]
# Default options: --allow-system-control (power mode and fan from the page).
NANO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--allow-system-control)

if [ ! -x "$NANO_DIR/.venv/bin/python3" ]; then
    echo "ERROR: run $NANO_DIR/setup/setup_jetson.sh first."
    read -r -p "Press Enter to close." _; exit 1
fi
if systemctl is-active --quiet cv-demo 2>/dev/null; then
    echo "The cv-demo service is running and holds the camera."
    echo "Stop it with:  sudo systemctl stop cv-demo"
    read -r -p "Press Enter to close." _; exit 1
fi

URL="http://localhost:8090/"
echo "Starting the dashboard; the browser opens at $URL in a few seconds."
echo "Close this window or press Ctrl+C to stop it."
( sleep 8 && xdg-open "$URL" >/dev/null 2>&1 ) &

cd "$NANO_DIR" || exit 1
"$NANO_DIR/.venv/bin/python3" -u apps/dashboard.py --headless "${ARGS[@]}"
echo
read -r -p "The dashboard stopped. Press Enter to close." _
