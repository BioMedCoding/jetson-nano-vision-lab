# -*- coding: utf-8 -*-
"""Stato condiviso thread-safe tra il loop di cattura/inferenza (demo_server.py)
e i thread delle richieste HTTP (dashboard_server.py).

Un solo lock per AppState: tutte le operazioni sotto lock sono O(1)/O(n
piccolo) (copie/validazioni di dict), nessuna chiamata cv2/CUDA/I/O nella
sezione critica -> nessuna contesa reale col loop a ~30 fps.
"""
import collections
import threading
import time

FILTER_NAMES = ("none", "gray", "clahe", "sharpen", "blur", "edge")

_DEFAULT_ROI = {"enabled": False, "x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}
_DEFAULT_PARAMS = {
    "conf": 0.4, "iou": 0.45,
    "brightness": 0.0, "contrast": 1.0, "filter": "none",
    "roi": dict(_DEFAULT_ROI),
    "show_trajectories": True, "count_classes": True,
    "infer_stride": 1, "jpeg_quality": 80, "class_filter": [],
    "detection_enabled": True,
}


class FrameStore(object):
    """Stesso pattern di common/viewer.py:_FrameStore, reso pubblico/riusabile."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self.clients = 0

    def update(self, jpeg):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_next(self, last_seq, timeout=1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq


def _validate_and_merge(target, patch):
    """Valida patch e la fonde in target (dict mutabile). Solleva ValueError
    con un messaggio leggibile se qualcosa non torna."""
    for key, value in patch.items():
        if key == "conf":
            value = float(value)
            if not (0.0 <= value <= 1.0):
                raise ValueError("conf deve essere in [0,1]")
            target["conf"] = value
        elif key == "iou":
            value = float(value)
            if not (0.0 <= value <= 1.0):
                raise ValueError("iou deve essere in [0,1]")
            target["iou"] = value
        elif key == "brightness":
            value = float(value)
            if not (-100.0 <= value <= 100.0):
                raise ValueError("brightness deve essere in [-100,100]")
            target["brightness"] = value
        elif key == "contrast":
            value = float(value)
            if not (0.5 <= value <= 3.0):
                raise ValueError("contrast deve essere in [0.5,3.0]")
            target["contrast"] = value
        elif key == "filter":
            if value not in FILTER_NAMES:
                raise ValueError("filter deve essere uno di {}".format(FILTER_NAMES))
            target["filter"] = value
        elif key == "roi":
            if not isinstance(value, dict):
                raise ValueError("roi deve essere un oggetto")
            roi = dict(target["roi"])
            roi.update(value)
            enabled = bool(roi.get("enabled", False))
            x1, y1 = float(roi.get("x1", 0)), float(roi.get("y1", 0))
            x2, y2 = float(roi.get("x2", 1)), float(roi.get("y2", 1))
            if enabled:
                for v in (x1, y1, x2, y2):
                    if not (0.0 <= v <= 1.0):
                        raise ValueError("coordinate roi devono essere in [0,1]")
                if not (x1 < x2 and y1 < y2):
                    raise ValueError("roi: serve x1<x2 e y1<y2")
            target["roi"] = {"enabled": enabled, "x1": x1, "y1": y1,
                             "x2": x2, "y2": y2}
        elif key == "show_trajectories":
            target["show_trajectories"] = bool(value)
        elif key == "count_classes":
            target["count_classes"] = bool(value)
        elif key == "infer_stride":
            value = int(value)
            if not (1 <= value <= 10):
                raise ValueError("infer_stride deve essere in [1,10]")
            target["infer_stride"] = value
        elif key == "jpeg_quality":
            value = int(value)
            if not (10 <= value <= 95):
                raise ValueError("jpeg_quality deve essere in [10,95]")
            target["jpeg_quality"] = value
        elif key == "class_filter":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError("class_filter deve essere una lista di nomi classe")
            target["class_filter"] = list(dict.fromkeys(value))  # dedup, ordine preservato
        elif key == "detection_enabled":
            target["detection_enabled"] = bool(value)
        else:
            raise ValueError("parametro sconosciuto: {}".format(key))


class AppState(object):
    def __init__(self, history_len=300):
        self._lock = threading.Lock()
        self._params = dict(_DEFAULT_PARAMS)
        self._params["roi"] = dict(_DEFAULT_ROI)
        self._start_time = time.time()
        self._frame_count = 0
        self._infer_ms_history = collections.deque(maxlen=history_len)
        self._fps = 0.0
        self._fps_history = collections.deque(maxlen=history_len)
        self._dets_frame = {}
        self._dets_cumulative = {}
        self._mean_conf = 0.0
        self._roi_enter_total = 0
        self._roi_exit_total = 0
        self._roi_inside_now = 0
        self._roi_inside_peak = 0
        self._roi_log = collections.deque(maxlen=20)  # piu' recente in testa
        self._active_tracks = 0
        self._model_info = {"name": None, "input_size": None,
                            "postprocess": None, "switching": False}
        self._system = {}
        self._model_switch_request = None

    # ---- scritti dal loop principale (demo_server.py) ----

    def push_stat(self, infer_ms, fps, dets_by_class, cumulative_by_class,
                  active_tracks, mean_conf=0.0, roi_events=None,
                  roi_log_entries=None):
        """infer_ms=None (detection in pausa) aggiorna comunque fps/frame
        count/detection ma NON appende alla history del tempo di inferenza:
        altrimenti degli zeri finti sporcherebbero media/percentili appena
        la detection riparte."""
        with self._lock:
            self._frame_count += 1
            if infer_ms is not None:
                self._infer_ms_history.append(infer_ms)
            self._fps = fps
            self._fps_history.append(fps)
            self._dets_frame = dict(dets_by_class)
            self._dets_cumulative = dict(cumulative_by_class)
            self._mean_conf = mean_conf
            self._active_tracks = active_tracks
            if roi_events:
                self._roi_enter_total += roi_events.get("enter", 0)
                self._roi_exit_total += roi_events.get("exit", 0)
                self._roi_inside_now = roi_events.get("inside_now", 0)
                self._roi_inside_peak = max(self._roi_inside_peak,
                                            self._roi_inside_now)
            if roi_log_entries:
                t = time.time()
                for entry in roi_log_entries:
                    self._roi_log.appendleft(
                        {"type": entry["type"], "label": entry["label"], "t": t})

    def set_model_info(self, name, input_size, postprocess, switching=False):
        with self._lock:
            self._model_info = {"name": name, "input_size": input_size,
                                "postprocess": postprocess, "switching": switching}

    def set_system(self, system_dict):
        with self._lock:
            self._system = dict(system_dict)

    def get_model_info(self):
        with self._lock:
            return dict(self._model_info)

    # ---- lettura per /api/stats (dashboard_server.py) ----

    def get_stats_snapshot(self):
        with self._lock:
            hist = list(self._infer_ms_history)
            fps_hist = list(self._fps_history)
            model_info = dict(self._model_info)
            system = dict(self._system)
            frame_count = self._frame_count
            fps = self._fps
            dets_frame = dict(self._dets_frame)
            dets_cum = dict(self._dets_cumulative)
            mean_conf = self._mean_conf
            roi_enter = self._roi_enter_total
            roi_exit = self._roi_exit_total
            roi_inside = self._roi_inside_now
            roi_peak = self._roi_inside_peak
            roi_log = list(self._roi_log)
            active_tracks = self._active_tracks
            params = dict(self._params)
            params["roi"] = dict(self._params["roi"])

        uptime = time.time() - self._start_time
        if hist:
            arr = sorted(hist)
            n = len(arr)
            mean = sum(arr) / n
            p50 = arr[n // 2]
            p95 = arr[min(n - 1, int(n * 0.95))]
            current = hist[-1]
            variance = sum((v - mean) ** 2 for v in arr) / n
            std = variance ** 0.5
            vmin, vmax = arr[0], arr[-1]
        else:
            mean = p50 = p95 = current = std = vmin = vmax = 0.0

        return {
            "fps": round(fps, 1),
            "fps_history": [round(v, 1) for v in fps_hist],
            "infer_ms": {
                "current": round(current, 1), "mean": round(mean, 1),
                "p50": round(p50, 1), "p95": round(p95, 1),
                "min": round(vmin, 1), "max": round(vmax, 1),
                "std": round(std, 1),
                "history": [round(v, 1) for v in hist],
            },
            "frame_count": frame_count,
            "uptime_s": round(uptime, 1),
            "detections": {"frame": dets_frame, "cumulative_unique": dets_cum,
                          "mean_conf_current": round(mean_conf, 3)},
            "roi": {"inside_now": roi_inside, "enter_total": roi_enter,
                    "exit_total": roi_exit, "inside_peak": roi_peak,
                    "log": roi_log[:8]},
            "tracker": {"active_tracks": active_tracks},
            "model": model_info,
            "system": system,
            "params": params,
            "server_time": time.time(),
        }

    # ---- parametri: letti dal loop, scritti dall'HTTP server ----

    def get_params(self):
        with self._lock:
            p = dict(self._params)
            p["roi"] = dict(self._params["roi"])
            return p

    def update_params(self, patch):
        with self._lock:
            new = dict(self._params)
            new["roi"] = dict(self._params["roi"])
            _validate_and_merge(new, patch)   # solleva ValueError se invalido
            self._params = new
            result = dict(new)
            result["roi"] = dict(new["roi"])
            return result

    # ---- switch modello: letto dal loop, scritto dall'HTTP server ----

    def request_model_switch(self, engine_file):
        with self._lock:
            self._model_switch_request = engine_file
            self._model_info["switching"] = True

    def pop_model_switch_request(self):
        with self._lock:
            req = self._model_switch_request
            self._model_switch_request = None
            return req
