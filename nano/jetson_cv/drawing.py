"""Overlays drawn on BGR frames: detection boxes, FPS text, ROI, tracks."""
import cv2
import numpy as np

from .labels import class_name


def _stable_color(seed):
    rng = np.random.RandomState(seed)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def class_color(cls_id):
    return _stable_color(cls_id + 7)


def track_color(track_id):
    return _stable_color(track_id * 17 + 3)


def draw_detections(frame, dets, names):
    """dets: [(x1, y1, x2, y2, score, class_id), ...] in frame coordinates."""
    for x1, y1, x2, y2, score, cls_id in dets:
        color = class_color(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(class_name(names, cls_id), score)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_fps(frame, fps, extra=""):
    txt = "FPS: {:.1f}{}".format(fps, "  " + extra if extra else "")
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_trajectories(frame, tracks, min_hits=3):
    for track in tracks:
        if track.hits < min_hits or len(track.centroid_history) < 2:
            continue
        color = track_color(track.id)
        pts = [(int(x), int(y)) for x, y in track.centroid_history]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)


def draw_roi(frame, roi, color, thickness=2):
    """roi: dict with normalized x1, y1, x2, y2 in [0, 1]."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (int(roi["x1"] * w), int(roi["y1"] * h)),
                  (int(roi["x2"] * w), int(roi["y2"] * h)), color, thickness)
