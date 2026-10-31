"""HTTP server for the dashboard: static page, MJPEG stream, JSON API.

Standard library only (BaseHTTPRequestHandler + ThreadingMixIn).

Threading rule: model switches NEVER happen here (an HTTP thread). The
handler only records the request in AppState; the main loop in
apps/dashboard.py performs it. The pycuda context is bound to the thread
that created it, so building a TensorRT runner from another thread would be
the most dangerous concurrency bug in the whole dashboard.

Security: there is no authentication. Anyone who can reach the port can
watch the camera and change detection settings. Power-mode and fan
endpoints are disabled unless the app is started with
--allow-system-control. Run it on a trusted network only.
"""
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from ..display import ThreadedHTTPServer, local_ip, stream_mjpeg
from ..models import list_models
from ..system import fan_service

STATIC_DIR = Path(__file__).resolve().parent / "static"

_EXT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json",
}

_SYSTEM_ENDPOINTS = ("/api/power", "/api/power/max", "/api/fan",
                     "/api/fan-service", "/api/fan-service/config")


def _make_handler(state, jtop_client, frame_store, models_dir, get_names,
                  allow_system_control):
    static_dir = STATIC_DIR

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # no per-request logging
            pass

        # ---------- response helpers ----------
        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path, content_type):
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def _json_endpoint(self, fn):
            """Run fn(body) and map the usual errors to HTTP status codes."""
            try:
                fn(self._read_json_body())
            except ValueError as e:  # includes json.JSONDecodeError
                self._send_json({"error": str(e)}, status=400)
            except TypeError as e:
                self._send_json({"error": "invalid body: {}".format(e)}, status=400)
            except RuntimeError as e:
                self._send_json({"error": str(e)}, status=500)

        # ---------- GET ----------
        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    self._send_file(static_dir / "dashboard.html", _EXT_TYPES[".html"])
                elif self.path.startswith("/static/"):
                    self._serve_static(self.path[len("/static/"):])
                elif self.path == "/stream":
                    stream_mjpeg(self, frame_store)
                elif self.path == "/api/stats":
                    snapshot = state.get_stats_snapshot()
                    snapshot["system"] = jtop_client.get_snapshot()
                    snapshot["system_control"] = allow_system_control
                    self._send_json(snapshot)
                elif self.path == "/api/models":
                    current = state.get_model_info().get("id")
                    self._send_json([m.to_json(current) for m in list_models(models_dir)])
                elif self.path == "/api/classes":
                    self._send_json(list(get_names()))
                elif self.path == "/api/fan-service":
                    self._send_json(fan_service.get_status())
                else:
                    self._send_json({"error": "not found"}, status=404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _serve_static(self, rel_path):
            candidate = (static_dir / rel_path).resolve()
            if candidate != static_dir and static_dir not in candidate.parents:
                self._send_json({"error": "forbidden"}, status=403)
                return
            if not candidate.is_file():
                self._send_json({"error": "not found"}, status=404)
                return
            content_type = _EXT_TYPES.get(
                candidate.suffix, mimetypes.guess_type(str(candidate))[0]
                or "application/octet-stream")
            self._send_file(candidate, content_type)

        # ---------- POST ----------
        def do_POST(self):
            if self.path in _SYSTEM_ENDPOINTS and not allow_system_control:
                self._send_json({"error": "system control is disabled; start the "
                                 "dashboard with --allow-system-control"}, status=403)
                return
            routes = {
                "/api/params": self._handle_params,
                "/api/model": self._handle_model_switch,
                "/api/power": self._handle_power,
                "/api/power/max": self._handle_max_performance,
                "/api/fan": self._handle_fan,
                "/api/fan-service": self._handle_fan_service,
                "/api/fan-service/config": self._handle_fan_service_config,
            }
            handler = routes.get(self.path)
            if handler is None:
                self._send_json({"error": "not found"}, status=404)
            else:
                self._json_endpoint(handler)

        def _handle_params(self, patch):
            if not isinstance(patch, dict):
                raise ValueError("the body must be a JSON object")
            self._send_json(state.update_params(patch))

        def _handle_model_switch(self, body):
            model_id = body.get("id") if isinstance(body, dict) else None
            known = {m.id: m for m in list_models(models_dir)}
            if model_id not in known:
                raise ValueError("unknown model: {}".format(model_id))
            if not known[model_id].has_engine:
                raise ValueError("model {} has no TensorRT engine yet".format(model_id))
            state.request_model_switch(model_id)
            self._send_json({"status": "switching"}, status=202)

        def _handle_power(self, body):
            mode_id = body.get("mode_id") if isinstance(body, dict) else None
            if not isinstance(mode_id, int):
                raise ValueError("mode_id must be an integer")
            modes = jtop_client.get_snapshot().get("power_modes") or []
            if modes and not (0 <= mode_id < len(modes)):
                raise ValueError("mode_id out of range")
            jtop_client.request_command({"type": "power_mode", "id": mode_id})
            self._send_json({"status": "queued"}, status=202)

        def _handle_max_performance(self, _body):
            jtop_client.request_command({"type": "max_performance"})
            self._send_json({"status": "queued"}, status=202)

        def _handle_fan(self, body):
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            profile, speed = body.get("profile"), body.get("speed")
            if profile is None and speed is None:
                raise ValueError("give at least one of profile, speed")
            if profile is not None and profile not in ("manual", "temp_control"):
                raise ValueError("profile must be 'manual' or 'temp_control'")
            if speed is not None:
                speed = int(speed)
                if not (0 <= speed <= 100):
                    raise ValueError("speed must be in [0,100]")
            jtop_client.request_command({"type": "fan", "profile": profile, "speed": speed})
            self._send_json({"status": "queued"}, status=202)

        def _handle_fan_service(self, body):
            action = body.get("action") if isinstance(body, dict) else None
            if action not in ("start", "stop"):
                raise ValueError("action must be 'start' or 'stop'")
            fan_service.control(action)
            self._send_json(fan_service.get_status())

        def _handle_fan_service_config(self, patch):
            if not isinstance(patch, dict):
                raise ValueError("the body must be a JSON object")
            fan_service.write_config(patch)
            status = fan_service.get_status()
            if status.get("active"):
                # fanctl.py reads its config only at start: restart it, or
                # the change would not apply until the next manual restart.
                fan_service.control("restart")
                status = fan_service.get_status()
            self._send_json(status)

    return Handler


class DashboardServer(object):
    def __init__(self, state, jtop_client, frame_store, models_dir, get_names,
                 port=8090, bind="0.0.0.0", allow_system_control=False):
        self.port = port
        handler = _make_handler(state, jtop_client, frame_store, models_dir,
                                get_names, allow_system_control)
        self._server = ThreadedHTTPServer((bind, port), handler)
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True
        self._thread.start()
        print("[dashboard] http://{}:{}/".format(local_ip(), self.port))

    def shutdown(self):
        self._server.shutdown()
