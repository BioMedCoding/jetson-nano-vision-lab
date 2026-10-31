#!/usr/bin/env python3
"""Live detection dashboard: camera -> TensorRT -> boxes, plus a web page at
http://<jetson>:8090/ with system stats, inference statistics, live image
and detection controls, hot model switching, a lightweight tracker with an
ROI zone (enter/exit counting) and trajectories.

Usage:
    python3 apps/dashboard.py                        # default model, stream
    python3 apps/dashboard.py --model yolov8n-416
    python3 apps/dashboard.py --allow-system-control # enable power/fan controls

No authentication: run it on a trusted network (see jetson_cv/dashboard/server.py).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jetson_cv import runtime  # noqa: E402
runtime.ensure()

import argparse  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402

from jetson_cv import MODELS_DIR  # noqa: E402
from jetson_cv.camera import ReconnectingCamera  # noqa: E402
from jetson_cv.dashboard.server import DashboardServer  # noqa: E402
from jetson_cv.dashboard.state import AppState  # noqa: E402
from jetson_cv.display import FrameStore  # noqa: E402
from jetson_cv.drawing import draw_detections, draw_roi, draw_trajectories  # noqa: E402
from jetson_cv.filters import apply_filters  # noqa: E402
from jetson_cv.inference.detector import Detector  # noqa: E402
from jetson_cv.labels import class_name  # noqa: E402
from jetson_cv.models import resolve  # noqa: E402
from jetson_cv.system.jtop_client import JtopClient  # noqa: E402
from jetson_cv.tracker import CentroidTracker, update_roi_membership  # noqa: E402

ROI_COLOR_IDLE = (255, 200, 0)      # BGR: cyan/amber at rest
ROI_COLOR_ENTER = (0, 200, 0)       # green: something just entered
ROI_COLOR_EXIT = (0, 90, 230)       # orange/red: something just left
ROI_FLASH_S = 0.6
LOG_EVERY_S = 60.0


class ActiveModel(object):
    """The current Detector, with hot swap. switch() must only be called from
    the main loop's thread (pycuda contexts are thread-bound)."""

    def __init__(self, spec, models_dir):
        self.models_dir = models_dir
        self.detector = Detector(resolve(spec, models_dir))

    @property
    def info(self):
        return self.detector.info

    def switch(self, model_id):
        # Build the new detector first: if that fails, the old one stays active.
        self.detector = Detector(resolve(model_id, self.models_dir))


def _count_by_class(dets, names):
    counts = {}
    for d in dets:
        label = class_name(names, d[5])
        counts[label] = counts.get(label, 0) + 1
    return counts


def _publish_model(state, model):
    info = model.info
    state.set_model_info(info.id, info.label, model.detector.input_size,
                         info.decoder, switching=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov5n-416",
                    help="model id under nano/models, or a model folder")
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--flip", type=int, default=0,
                    help="0 none, 1 90 ccw, 2 180, 3 90 cw")
    ap.add_argument("--headless", action="store_true",
                    help="never open a local window, even with a display")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--bind", default="0.0.0.0",
                    help="address to listen on (127.0.0.1 = this board only)")
    ap.add_argument("--allow-system-control", action="store_true",
                    help="let the web page change power mode and fan settings")
    ap.add_argument("--history-len", type=int, default=300)
    args = ap.parse_args()

    state = AppState(history_len=args.history_len)
    state.update_params({"conf": args.conf, "iou": args.iou})

    jtop_client = JtopClient()
    jtop_client.start()
    frame_store = FrameStore()

    print("Loading model {}...".format(args.model))
    model = ActiveModel(args.model, args.models_dir)
    _publish_model(state, model)
    print("Model ready: {} (input {}, decoder {}).".format(
        model.info.id, model.detector.input_size, model.info.decoder))

    dashboard = DashboardServer(state, jtop_client, frame_store, args.models_dir,
                                get_names=lambda: model.detector.names,
                                port=args.port, bind=args.bind,
                                allow_system_control=args.allow_system_control)
    dashboard.start()

    camera = ReconnectingCamera(flip=args.flip)
    tracker = CentroidTracker()

    window = None
    if not args.headless and os.environ.get("DISPLAY"):
        try:
            cv2.namedWindow("Jetson Nano dashboard", cv2.WINDOW_AUTOSIZE)
            window = "Jetson Nano dashboard"
        except cv2.error:
            window = None

    fps_ema, frame_idx = 0.0, 0
    last_dets, last_tracks = [], []
    last_infer_ms, last_mean_conf = 0.0, 0.0
    roi_flash_color, roi_flash_until = None, 0.0
    next_log = time.time() + LOG_EVERY_S
    try:
        while True:
            t0 = time.time()
            ret, frame = camera.read()
            if not ret:
                continue  # ReconnectingCamera already handles backoff/reopen

            params = state.get_params()
            frame = apply_filters(frame, params)
            names = model.detector.names

            switch_req = state.pop_model_switch_request()
            if switch_req:
                try:
                    model.switch(switch_req)
                    tracker = CentroidTracker()  # class ids may mean something else now
                    last_dets, last_tracks = [], []
                    state.reset_timing_history()
                    fps_ema = 0.0
                    print("[model] switched to {}".format(switch_req))
                except Exception as e:
                    print("[model] switch to {} failed: {}".format(switch_req, e))
                _publish_model(state, model)
                names = model.detector.names

            roi = params["roi"]
            roi_events = None

            if not params.get("detection_enabled", True):
                # Detection paused: the video keeps streaming, but no
                # inference/NMS/tracking -> real GPU/CPU/heat savings. Clear
                # the remembered detections too, so no ghost boxes linger and
                # the first stride-skipped frame after resuming does not
                # bring back pre-pause detections.
                dets, tracks = [], []
                last_dets, last_tracks = [], []
                infer_ms = None
                last_mean_conf = 0.0
            else:
                # infer_stride: run real inference only every N frames, which
                # raises the displayed fps at the cost of detection latency.
                stride = params.get("infer_stride", 1)
                do_infer = (frame_idx % stride == 0)
                frame_idx += 1
                if do_infer:
                    dets, infer_ms = model.detector.detect(
                        frame, conf=params["conf"], iou=params["iou"])
                    last_dets, last_infer_ms = dets, infer_ms
                    last_mean_conf = (sum(d[4] for d in dets) / len(dets)) if dets else 0.0
                    tracks = tracker.update(dets)
                    last_tracks = tracks
                    if roi.get("enabled"):
                        roi_events = update_roi_membership(
                            tracks, roi, frame.shape[1], frame.shape[0])
                else:
                    # Stride-skipped frame: NO tracker.update() here, or every
                    # track would age as "missed" just because we did not run
                    # inference, corrupting track ageing and ROI events.
                    dets, tracks = last_dets, last_tracks
                    infer_ms = last_infer_ms

            # Class filter: affects only what is drawn and counted. The
            # tracker always sees every class (no phantom ROI events when the
            # filter changes). It is not a speed lever: decoding/NMS still
            # run over all classes before this point.
            class_filter = set(params.get("class_filter") or [])
            dets_shown = ([d for d in dets if class_name(names, d[5]) in class_filter]
                          if class_filter else dets)

            draw_detections(frame, dets_shown, names)
            if params.get("show_trajectories", True):
                draw_trajectories(frame, tracks)
            if roi.get("enabled"):
                now = time.time()
                events = (roi_events or {}).get("events", [])
                if events:
                    roi_flash_color = (ROI_COLOR_ENTER
                                       if any(e["type"] == "enter" for e in events)
                                       else ROI_COLOR_EXIT)
                    roi_flash_until = now + ROI_FLASH_S
                flashing = roi_flash_color is not None and now < roi_flash_until
                draw_roi(frame, roi, roi_flash_color if flashing else ROI_COLOR_IDLE,
                         thickness=3 if flashing else 2)

            dt = time.time() - t0
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt

            if params.get("count_classes", True):
                dets_by_class = _count_by_class(dets_shown, names)
                cumulative = {class_name(names, k): v
                              for k, v in tracker.total_ids_by_class.items()}
                if class_filter:
                    cumulative = {k: v for k, v in cumulative.items() if k in class_filter}
            else:
                dets_by_class, cumulative = {}, {}

            roi_log = None
            if roi_events and roi_events.get("events"):
                roi_log = [{"type": e["type"], "label": class_name(names, e["cls_id"])}
                           for e in roi_events["events"]]

            state.push_stat(infer_ms, fps_ema, dets_by_class, cumulative,
                            len(tracks), mean_conf=last_mean_conf,
                            roi_events=roi_events, roi_log_entries=roi_log)

            if time.time() >= next_log:
                next_log += LOG_EVERY_S
                print("[status] {:.1f} fps, {:.1f} ms inference, model {}".format(
                    fps_ema, last_infer_ms, model.info.id))

            if frame_store.clients > 0:
                q = int(params.get("jpeg_quality", 80))
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    frame_store.update(jpeg.tobytes())

            if window:
                cv2.imshow(window, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        dashboard.shutdown()
        jtop_client.stop()
        if window:
            cv2.destroyAllWindows()
        print("Stopped. Average fps: {:.1f}".format(fps_ema))


if __name__ == "__main__":
    main()
