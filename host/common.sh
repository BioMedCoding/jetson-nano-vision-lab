# Sourced by the host/ scripts: loads host/jetson.env and checks the repo.
HOST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HOST_DIR/.." && pwd)"
[ -f "$HOST_DIR/jetson.env" ] && . "$HOST_DIR/jetson.env"
JETSON_HOST="${JETSON_HOST:-jetson}"
JETSON_DIR="${JETSON_DIR:-~/jetson-nano-vision-lab}"
