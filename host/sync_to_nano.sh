#!/bin/bash
# Copy the parts of this repo that the Jetson needs (nano/, tools/, the
# large-file manifest) to $JETSON_HOST:$JETSON_DIR, without the kernel source
# or the training side. Large files are sent reassembled (not as chunks).
#
# Usage:  host/sync_to_nano.sh [--dry-run]
# Config: host/jetson.env (see jetson.env.example)
#
# Alternative without this script: git clone the repo on the Jetson (needs
# ~2 GB free; a sparse checkout of nano/ and tools/ is enough).
set -e
. "$(dirname "$0")/common.sh"

python3 "$REPO_DIR/tools/largefiles.py" join nano/ >/dev/null

ssh "$JETSON_HOST" "mkdir -p $JETSON_DIR"
rsync -az --info=stats1 --delete $1 \
    --exclude '.venv/' --exclude '__pycache__/' --exclude '*.part-[0-9][0-9]' \
    --exclude 'build.log' --exclude 'snapshots/' --exclude 'dataset/' \
    "$REPO_DIR/nano" "$REPO_DIR/tools" "$REPO_DIR/largefiles.manifest" \
    "$JETSON_HOST:$JETSON_DIR/"
echo "Synced to $JETSON_HOST:$JETSON_DIR"
echo "First time on the Jetson:  $JETSON_DIR/nano/setup/setup_jetson.sh"
