#!/bin/bash
# Avvia la camera OV5647 sulla Jetson via SSH e apre lo stream nel browser.
#
# Uso:
#   ./avvia_camera.sh          # detection YOLOv5 TensorRT live (~30 fps)
#   ./avvia_camera.sh test     # solo camera, senza inference (~62 fps)
#   ./avvia_camera.sh torch    # detection PyTorch (~18 fps, per confronto)
#   ./avvia_camera.sh demo     # dashboard sperimentale (sistema/statistiche/
#                              # controlli live/tracker+zone), stesso ~30 fps
#
# Configurazione via variabili d'ambiente (default = setup di casa):
#   JETSON_HOST=matteo@10.42.0.24  JETSON_KEY=~/.ssh/jetson_session/jetson_key
#
# Ctrl+C per fermare. Lo stream MJPEG e' su http://<jetson>:8090/
set -e

JETSON="${JETSON_HOST:-matteo@10.42.0.24}"
KEY="${JETSON_KEY:-$HOME/.ssh/jetson_session/jetson_key}"
HOST_IP="${JETSON#*@}"
URL="http://$HOST_IP:8090/"

case "${1:-trt}" in
  test)  CMD="python3 01_camera_test/camera_test.py --headless" ;;
  torch) CMD="python3 02_live_detection/live_detect.py --headless" ;;
  demo)  CMD="python3 03_tensorrt/demo_server.py --engine models/yolov5n_416_fp16.engine --headless" ;;
  trt|*) CMD="python3 03_tensorrt/trt_detect.py --engine models/yolov5n_416_fp16.engine --headless" ;;
esac

echo "Stream tra qualche secondo su: $URL  (Ctrl+C per fermare)"
( sleep 6 && xdg-open "$URL" 2>/dev/null ) &

# il venv esporta gia' OPENBLAS_CORETYPE e LD_PRELOAD (vedi setup_jetson.sh)
ssh -t -i "$KEY" "$JETSON" \
  "cd ~/cv_projects && source venv/bin/activate && $CMD"
