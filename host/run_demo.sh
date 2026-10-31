#!/bin/bash
# Start an app on the Jetson over SSH and open its stream in your browser.
#
# Usage:
#   host/run_demo.sh                 # TensorRT detection, ~30 fps (default)
#   host/run_demo.sh test            # camera only, ~60 fps
#   host/run_demo.sh torch           # PyTorch detection, ~16 fps (for comparison)
#   host/run_demo.sh dashboard       # the full web dashboard
#   host/run_demo.sh trt --model yolov8n-416    # extra options go to the app
#
# Config: host/jetson.env (see jetson.env.example). Ctrl+C stops it.
# The stream is at http://<jetson>:8090/
set -e
. "$(dirname "$0")/common.sh"

MODE="${1:-trt}"; shift || true
case "$MODE" in
  test)      APP="apps/camera_test.py" ;;
  torch)     APP="apps/live_detect_torch.py" ;;
  dashboard) APP="apps/dashboard.py" ;;
  trt)       APP="apps/trt_detect.py" ;;
  *) echo "Unknown mode '$MODE' (test | trt | torch | dashboard)"; exit 1 ;;
esac

if [ "$(ssh "$JETSON_HOST" systemctl is-active cv-demo 2>/dev/null)" = "active" ]; then
    echo "The dashboard service (cv-demo) is running and holds the camera."
    echo "It is already at the URL below; to run '$MODE' instead, stop it first:"
    echo "  ssh -t $JETSON_HOST sudo systemctl stop cv-demo"
    MODE=service
fi

# The address the browser should use: what ssh resolves JETSON_HOST to.
ADDR="$(ssh -G "$JETSON_HOST" | awk '/^hostname /{print $2}')"
URL="http://$ADDR:8090/"
echo "Stream in a few seconds at: $URL  (Ctrl+C to stop)"
( sleep 6 && xdg-open "$URL" >/dev/null 2>&1 ) &
[ "$MODE" = service ] && exit 0

ssh -t "$JETSON_HOST" "cd $JETSON_DIR/nano && .venv/bin/python -u $APP --headless $*"
