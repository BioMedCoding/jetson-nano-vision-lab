# -*- coding: utf-8 -*-
"""Server HTTP della dashboard sperimentale.

Clona il pattern di common/viewer.py (BaseHTTPRequestHandler + ThreadingMixIn,
stdlib puro, zero dipendenze esterne) ed estende con route JSON per
statistiche/controlli/modelli e la pagina statica. viewer.py non viene
toccato: e' ancora usato da 01/02/04/05.

Il cambio modello NON avviene mai qui (thread della richiesta HTTP): imposta
solo un flag su AppState, consumato dal loop principale in demo_server.py.
Il contesto CUDA di pycuda e' legato al thread che lo crea (pycuda.autoinit
in trt_runner.py), quindi ricostruire TrtRunner da un thread diverso sarebbe
il bug di concorrenza piu' pericoloso di tutta la dashboard.
"""
import json
import mimetypes
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

from common import fan_service

_EXT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json",
}


def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def guess_postprocess_from_name(filename):
    name = filename.lower()
    if name.startswith("yolov8"):
        return "v8"
    return "v5"  # v5/v7/custom-da-v5: stesso formato output (1,N,85+)


def list_models(models_dir, current_name=None):
    """Scansiona models_dir/*.engine, legge il sidecar .json se presente
    (vedi build_engine.sh --postprocess), altrimenti indovina dal nome file."""
    models = []
    for engine_path in sorted(Path(models_dir).glob("*.engine")):
        sidecar = engine_path.with_suffix(".json")
        meta, source = {}, "guessed"
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
                source = "sidecar"
            except (ValueError, OSError):
                meta, source = {}, "guessed"
        postprocess = meta.get("postprocess") or guess_postprocess_from_name(engine_path.name)
        models.append({
            "file": engine_path.name,
            "label": meta.get("label") or engine_path.stem,
            "postprocess": postprocess,
            "input_size": meta.get("input_size"),
            "metadata_source": source,
            "current": engine_path.name == current_name,
        })
    return models


def _make_handler(state, jtop_client, frame_store, models_dir, static_dir, names):
    static_dir = Path(static_dir).resolve()
    models_dir = Path(models_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # niente log delle richieste sul terminale
            pass

        # ---------- helper di risposta ----------
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

        # ---------- GET ----------
        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    self._send_file(static_dir / "dashboard.html", "text/html; charset=utf-8")
                elif self.path.startswith("/static/"):
                    self._serve_static(self.path[len("/static/"):])
                elif self.path == "/stream":
                    self._serve_stream()
                elif self.path == "/api/stats":
                    snapshot = state.get_stats_snapshot()
                    snapshot["system"] = jtop_client.get_snapshot()
                    self._send_json(snapshot)
                elif self.path == "/api/models":
                    current = state.get_model_info().get("name")
                    self._send_json(list_models(models_dir, current))
                elif self.path == "/api/classes":
                    self._send_json(list(names))
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

        def _serve_stream(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            frame_store.clients += 1
            seq = -1
            try:
                while True:
                    jpeg, seq = frame_store.wait_next(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n"
                                     b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            finally:
                frame_store.clients -= 1

        # ---------- POST ----------
        def do_POST(self):
            if self.path == "/api/params":
                self._handle_params()
            elif self.path == "/api/model":
                self._handle_model_switch()
            elif self.path == "/api/power":
                self._handle_power()
            elif self.path == "/api/power/max":
                self._handle_max_performance()
            elif self.path == "/api/fan":
                self._handle_fan()
            elif self.path == "/api/fan-service":
                self._handle_fan_service()
            elif self.path == "/api/fan-service/config":
                self._handle_fan_service_config()
            else:
                self._send_json({"error": "not found"}, status=404)

        def _handle_params(self):
            try:
                patch = self._read_json_body()
                if not isinstance(patch, dict):
                    raise ValueError("il corpo deve essere un oggetto JSON")
                result = state.update_params(patch)
                self._send_json(result)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            except (TypeError, json.JSONDecodeError) as e:
                self._send_json({"error": "corpo non valido: {}".format(e)},
                                status=400)

        def _handle_model_switch(self):
            try:
                body = self._read_json_body()
                engine_file = body.get("file") if isinstance(body, dict) else None
                if not engine_file or "/" in engine_file or "\\" in engine_file:
                    raise ValueError("nome file non valido")
                if not (models_dir / engine_file).is_file():
                    raise ValueError("modello non trovato: {}".format(engine_file))
                state.request_model_switch(engine_file)
                self._send_json({"status": "switching"}, status=202)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            except (TypeError, json.JSONDecodeError) as e:
                self._send_json({"error": "corpo non valido: {}".format(e)},
                                status=400)

        def _handle_power(self):
            try:
                body = self._read_json_body()
                mode_id = body.get("mode_id") if isinstance(body, dict) else None
                if not isinstance(mode_id, int):
                    raise ValueError("mode_id deve essere un intero")
                modes = jtop_client.get_snapshot().get("power_modes") or []
                if modes and not (0 <= mode_id < len(modes)):
                    raise ValueError("mode_id fuori range")
                jtop_client.request_command({"type": "power_mode", "id": mode_id})
                self._send_json({"status": "queued"}, status=202)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            except (TypeError, json.JSONDecodeError) as e:
                self._send_json({"error": "corpo non valido: {}".format(e)},
                                status=400)

        def _handle_max_performance(self):
            jtop_client.request_command({"type": "max_performance"})
            self._send_json({"status": "queued"}, status=202)

        def _handle_fan(self):
            try:
                body = self._read_json_body()
                if not isinstance(body, dict):
                    raise ValueError("il corpo deve essere un oggetto JSON")
                profile = body.get("profile")
                speed = body.get("speed")
                if profile is None and speed is None:
                    raise ValueError("serve almeno uno tra profile e speed")
                if profile is not None and profile not in ("manual", "temp_control"):
                    raise ValueError("profile deve essere 'manual' o 'temp_control'")
                if speed is not None:
                    speed = int(speed)
                    if not (0 <= speed <= 100):
                        raise ValueError("speed deve essere in [0,100]")
                jtop_client.request_command(
                    {"type": "fan", "profile": profile, "speed": speed})
                self._send_json({"status": "queued"}, status=202)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            except (TypeError, json.JSONDecodeError) as e:
                self._send_json({"error": "corpo non valido: {}".format(e)},
                                status=400)

        def _handle_fan_service(self):
            try:
                body = self._read_json_body()
                action = body.get("action") if isinstance(body, dict) else None
                if action not in ("start", "stop"):
                    raise ValueError("action deve essere 'start' o 'stop'")
                fan_service.control(action)
                self._send_json(fan_service.get_status())
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            except RuntimeError as e:
                self._send_json({"error": str(e)}, status=500)
            except (TypeError, json.JSONDecodeError) as e:
                self._send_json({"error": "corpo non valido: {}".format(e)},
                                status=400)

        def _handle_fan_service_config(self):
            try:
                patch = self._read_json_body()
                if not isinstance(patch, dict):
                    raise ValueError("il corpo deve essere un oggetto JSON")
                fan_service.write_config(patch)
                status = fan_service.get_status()
                if status.get("active"):
                    # fanctl.py legge il config solo all'avvio: senza
                    # restart la modifica non avrebbe effetto finche' non
                    # si ferma/riavvia il servizio a mano.
                    fan_service.control("restart")
                    status = fan_service.get_status()
                self._send_json(status)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            except RuntimeError as e:
                self._send_json({"error": str(e)}, status=500)
            except (TypeError, json.JSONDecodeError) as e:
                self._send_json({"error": "corpo non valido: {}".format(e)},
                                status=400)

    return Handler


class DashboardServer(object):
    def __init__(self, state, jtop_client, frame_store, models_dir,
                static_dir, names, port=8090):
        self.port = port
        handler = _make_handler(state, jtop_client, frame_store,
                                models_dir, static_dir, names)
        self._server = _ThreadedHTTPServer(("0.0.0.0", port), handler)
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True
        self._thread.start()
        print("[dashboard] http://{}:{}/".format(_local_ip(), self.port))

    def shutdown(self):
        self._server.shutdown()
