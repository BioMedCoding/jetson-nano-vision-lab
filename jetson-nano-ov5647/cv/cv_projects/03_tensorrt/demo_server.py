#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dashboard sperimentale YOLOv5/TensorRT: evoluzione di trt_detect.py.

trt_detect.py resta invariato (referenziato da avvia_camera.sh/cv/README.md);
questo script vive in parallelo e aggiunge: pagina web a schede (sistema,
statistiche, controlli immagine/detection, funzionalita' demo), cambio
modello a caldo, regolazione live di soglie/filtri, tracker leggero con
zona ROI (conteggio ingressi/uscite) e traiettorie.

Uso:
    python demo_server.py --engine ../models/yolov5n_416_fp16.engine --headless
"""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2                                                        # noqa: E402
import numpy as np                                                 # noqa: E402

import common.yolo_utils as yolo_utils                             # noqa: E402
from common.camera import open_csi_camera                          # noqa: E402
from common.dashboard_server import (DashboardServer, list_models, # noqa: E402
                                     guess_postprocess_from_name)
from common.filters import apply_filters                           # noqa: E402
from common.jtop_client import JtopClient                          # noqa: E402
from common.state import AppState, FrameStore                      # noqa: E402
from common.tracker import CentroidTracker, update_roi_membership  # noqa: E402
from common.trt_runner import TrtRunner                            # noqa: E402
from common.yolo_utils import COCO_NAMES, draw_detections, preprocess  # noqa: E402


def _get_postprocess_fn(name):
    attr = "postprocess" if name == "v5" else "postprocess_{}".format(name)
    fn = getattr(yolo_utils, attr, None)
    if fn is None:
        raise RuntimeError(
            "decoder di post-processing '{}' non disponibile in "
            "yolo_utils.py".format(name))
    return fn


class ModelRunner(object):
    """Incapsula l'engine TensorRT attivo + il decoder di post-processing da
    usare, con hot-swap. Il cambio avviene SOLO chiamando switch() dal thread
    del loop principale (mai dal thread di una richiesta HTTP): il contesto
    CUDA di pycuda e' legato al thread che lo crea."""

    def __init__(self, engine_path, models_dir):
        self.models_dir = Path(models_dir)
        self.runner = None
        self.postprocess_fn = None
        self.postprocess_name = None
        self.input_size = None
        self.label = None
        self.engine_path = None
        self._load(Path(engine_path))

    def _resolve_postprocess(self, engine_path):
        entries = {m["file"]: m for m in list_models(self.models_dir)}
        meta = entries.get(engine_path.name)
        if meta:
            return meta["postprocess"], meta.get("label")
        return guess_postprocess_from_name(engine_path.name), None

    def _load(self, engine_path):
        postprocess_name, label = self._resolve_postprocess(engine_path)
        fn = _get_postprocess_fn(postprocess_name)     # puo' sollevare: nome
        runner = TrtRunner(str(engine_path))            # o file mancante/rotto
        # solo se tutto e' andato a buon fine si sovrascrive lo stato attivo:
        # in caso di eccezione il modello precedente resta valido e attivo.
        self.runner = runner
        self.postprocess_fn = fn
        self.postprocess_name = postprocess_name
        self.input_size = runner.size
        self.engine_path = engine_path
        self.label = label or engine_path.stem

    def switch(self, filename):
        self._load(self.models_dir / filename)


class CameraSource(object):
    """Wrapper su open_csi_camera con riconnessione automatica: dopo troppe
    letture fallite di fila, richiude e riapre il device con backoff. Copre
    il caso reale noto (il daemon della camera resta appeso), non un vero
    hot-unplug del device CSI fisso."""

    def __init__(self, width, height, fps, flip, max_consec_fail=15):
        self.width, self.height, self.fps, self.flip = width, height, fps, flip
        self.max_consec_fail = max_consec_fail
        self.cap = None
        self._fail_count = 0
        self._open()

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = open_csi_camera(self.width, self.height, self.fps, self.flip)
        self._fail_count = 0

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            self._fail_count += 1
            if self._fail_count >= self.max_consec_fail:
                print("[camera] {} letture fallite di fila, "
                     "riapro il device...".format(self._fail_count))
                backoff = 2.0
                while True:
                    try:
                        self._open()
                        print("[camera] riconnessa.")
                        break
                    except Exception as e:
                        print("[camera] riapertura fallita ({}), "
                             "ritento tra {:.0f}s...".format(e, backoff))
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 30.0)
            return False, None
        self._fail_count = 0
        return True, frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


def _cls_name(cls_id, names):
    return names[cls_id] if cls_id < len(names) else str(cls_id)


def _count_by_class(dets, names):
    counts = {}
    for _x1, _y1, _x2, _y2, _score, cls_id in dets:
        label = _cls_name(cls_id, names)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _cumulative_by_class(tracker, names):
    result = {}
    for cls_id, count in tracker.total_ids_by_class.items():
        result[_cls_name(cls_id, names)] = count
    return result


def _track_color(track_id):
    np.random.seed(track_id * 17 + 3)
    return tuple(int(c) for c in np.random.randint(60, 255, 3))


def _draw_trajectories(frame, tracks, min_hits=3):
    for track in tracks:
        if track.hits < min_hits or len(track.centroid_history) < 2:
            continue
        color = _track_color(track.id)
        pts = [(int(x), int(y)) for x, y in track.centroid_history]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)


ROI_COLOR_DEFAULT = (255, 200, 0)   # ciano/ambra, colore "a riposo"
ROI_COLOR_ENTER = (0, 200, 0)       # verde: qualcuno e' appena entrato
ROI_COLOR_EXIT = (0, 90, 230)       # arancio/rosso: qualcuno e' appena uscito
ROI_FLASH_DURATION_S = 0.6


def _draw_roi(frame, roi, color=ROI_COLOR_DEFAULT):
    h, w = frame.shape[:2]
    x1, y1 = int(roi["x1"] * w), int(roi["y1"] * h)
    x2, y2 = int(roi["x2"] * w), int(roi["y2"] * h)
    thickness = 2 if color == ROI_COLOR_DEFAULT else 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, help="file .engine iniziale")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--names", default=None,
                    help="file con un nome classe per riga (default: COCO)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--history-len", type=int, default=300)
    ap.add_argument("--models-dir", default=str(BASE / "models"))
    args = ap.parse_args()

    names = COCO_NAMES
    if args.names:
        with open(args.names) as f:
            names = [l.strip() for l in f if l.strip()]

    state = AppState(history_len=args.history_len)
    state.update_params({"conf": args.conf, "iou": args.iou})

    jtop_client = JtopClient()
    jtop_client.start()
    frame_store = FrameStore()

    print("Carico modello {}...".format(args.engine))
    model = ModelRunner(args.engine, args.models_dir)
    state.set_model_info(model.engine_path.name, model.input_size,
                         model.postprocess_name, switching=False)
    print("Modello pronto (input {0}x{0}, decoder {1}).".format(
        model.input_size, model.postprocess_name))

    dashboard = DashboardServer(state, jtop_client, frame_store,
                                args.models_dir, str(BASE / "common" / "static"),
                                names, port=args.port)
    dashboard.start()

    camera = CameraSource(args.width, args.height, args.fps, args.flip)
    tracker = CentroidTracker()

    show_window = False
    if not args.headless and os.environ.get("DISPLAY"):
        try:
            cv2.namedWindow("Jetson Nano - Dashboard", cv2.WINDOW_AUTOSIZE)
            show_window = True
        except cv2.error:
            show_window = False

    fps_ema, n = 0.0, 0
    frame_idx = 0
    last_dets, last_tracks = [], []
    last_infer_ms, last_mean_conf = 0.0, 0.0
    roi_flash_color, roi_flash_until = None, 0.0
    try:
        while True:
            t0 = time.time()
            ret, frame = camera.read()
            if not ret:
                continue  # CameraSource gestisce gia' backoff/riconnessione

            params = state.get_params()
            proc_frame = apply_filters(frame, params)

            switch_req = state.pop_model_switch_request()
            if switch_req:
                try:
                    model.switch(switch_req)
                    print("[model] passato a {}".format(switch_req))
                except Exception as e:
                    print("[model] cambio a {} fallito: {}".format(switch_req, e))
                state.set_model_info(model.engine_path.name, model.input_size,
                                     model.postprocess_name, switching=False)

            roi = params["roi"]
            roi_events = None
            detection_enabled = params.get("detection_enabled", True)

            if not detection_enabled:
                # Detection in pausa: lo stream camera continua (si vede
                # sempre il video live), ma niente preprocess/infer/NMS ne'
                # tracker.update -> risparmio reale di GPU/CPU/calore quando
                # non serve monitorare attivamente. Si azzerano anche le
                # detection "congelate" cosi' non restano box fantasma a
                # schermo, e last_dets/last_tracks cosi' un eventuale primo
                # frame "saltato" dallo stride subito dopo la riattivazione
                # non riesuma detection vecchie da prima della pausa.
                dets, tracks = [], []
                last_dets, last_tracks = [], []
                infer_ms = None
                last_mean_conf = 0.0
            else:
                # infer_stride: esegue l'inferenza vera solo ogni N frame,
                # per alzare l'fps di visualizzazione a scapito della
                # reattivita' delle detection sui frame intermedi (hint UI).
                stride = params.get("infer_stride", 1)
                do_infer = (frame_idx % stride == 0)
                frame_idx += 1

                if do_infer:
                    img, r, pad = preprocess(proc_frame, model.input_size)
                    t1 = time.time()
                    pred = model.runner.infer(img)
                    infer_ms = (time.time() - t1) * 1000
                    dets = model.postprocess_fn(pred, r, pad, proc_frame.shape,
                                                conf_thres=params["conf"],
                                                iou_thres=params["iou"])
                    last_dets, last_infer_ms = dets, infer_ms
                    last_mean_conf = (sum(d[4] for d in dets) / len(dets)
                                      if dets else 0.0)

                    tracks = tracker.update(dets)
                    last_tracks = tracks
                    if roi.get("enabled"):
                        roi_events = update_roi_membership(
                            tracks, roi, proc_frame.shape[1], proc_frame.shape[0])
                else:
                    # Frame saltato dallo stride: NIENTE tracker.update() qui,
                    # altrimenti ogni traccia matura "missed" anche quando
                    # semplicemente non abbiamo rieseguito l'inferenza,
                    # alterando l'aging delle tracce e falsando gli eventi
                    # ROI. Si riusano detection/tracce dell'ultima inferenza
                    # vera.
                    dets, tracks = last_dets, last_tracks
                    infer_ms = last_infer_ms

            # Filtro classi: solo su cosa viene disegnato/contato, il tracker
            # vede sempre tutte le classi (niente eventi ROI fantasma quando
            # il filtro cambia). Non e' una leva di velocita': NMS/postprocess
            # girano comunque su tutte le classi prima del filtro.
            class_filter = set(params.get("class_filter") or [])
            dets_shown = ([d for d in dets if _cls_name(d[5], names) in class_filter]
                         if class_filter else dets)

            draw_detections(proc_frame, dets_shown, names)
            if params.get("show_trajectories", True):
                _draw_trajectories(proc_frame, tracks)
            if roi.get("enabled"):
                now = time.time()
                events = (roi_events or {}).get("events", [])
                if events:
                    roi_flash_color = (ROI_COLOR_ENTER
                                       if any(e["type"] == "enter" for e in events)
                                       else ROI_COLOR_EXIT)
                    roi_flash_until = now + ROI_FLASH_DURATION_S
                roi_color = (roi_flash_color
                            if roi_flash_color and now < roi_flash_until
                            else ROI_COLOR_DEFAULT)
                _draw_roi(proc_frame, roi, color=roi_color)

            dt = time.time() - t0
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt

            dets_by_class = _count_by_class(dets_shown, names)
            cumulative_by_class = _cumulative_by_class(tracker, names)
            if class_filter:
                cumulative_by_class = {k: v for k, v in cumulative_by_class.items()
                                       if k in class_filter}

            roi_log_entries = None
            if roi_events and roi_events.get("events"):
                roi_log_entries = [{"type": e["type"],
                                    "label": _cls_name(e["cls_id"], names)}
                                   for e in roi_events["events"]]

            state.push_stat(infer_ms, fps_ema, dets_by_class, cumulative_by_class,
                            len(tracks), mean_conf=last_mean_conf,
                            roi_events=roi_events, roi_log_entries=roi_log_entries)

            n += 1
            if n % 30 == 0:
                print("FPS: {:.1f}  oggetti: {}  modello: {}".format(
                    fps_ema, len(dets), model.engine_path.name))

            if frame_store.clients > 0:
                q = int(params.get("jpeg_quality", 80))
                ok, jpeg = cv2.imencode(
                    ".jpg", proc_frame, [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    frame_store.update(jpeg.tobytes())

            if show_window:
                cv2.imshow("Jetson Nano - Dashboard", proc_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        dashboard.shutdown()
        jtop_client.stop()
        if show_window:
            cv2.destroyAllWindows()
        print("Chiuso. FPS medi: {:.1f}".format(fps_ema))


if __name__ == "__main__":
    main()
