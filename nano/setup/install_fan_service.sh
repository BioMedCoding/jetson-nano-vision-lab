#!/bin/bash
# Optional: install the automagic-fan temperature-based fan controller
# (vendored from github.com/Pyrestone/jetson-fan-ctl, MIT) and let the
# dashboard control it. Only useful with a 5 V PWM fan on the Nano.
# Run ON the Jetson; uses sudo.
#
# Besides upstream's installer this grants the current user, and nothing
# more:
#   - write access to /etc/automagic-fan/config.json (via the user's group)
#   - passwordless `systemctl start|stop|restart automagic-fan.service`
# which is what the dashboard's Power tab needs (with --allow-system-control).
set -e

NANO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUDOERS=/etc/sudoers.d/automagic-fan-dashboard

(cd "$NANO_DIR/vendor/jetson-fan-ctl" && sudo ./install.sh)

sudo chgrp "$(id -gn)" /etc/automagic-fan/config.json
sudo chmod 664 /etc/automagic-fan/config.json

RULE="$USER ALL=(root) NOPASSWD: /bin/systemctl start automagic-fan.service, /bin/systemctl stop automagic-fan.service, /bin/systemctl restart automagic-fan.service"
TMP=$(mktemp)
echo "$RULE" > "$TMP"
sudo visudo -cf "$TMP"
sudo install -m 440 "$TMP" "$SUDOERS"
rm -f "$TMP"
echo "Done: automagic-fan installed and controllable from the dashboard."
