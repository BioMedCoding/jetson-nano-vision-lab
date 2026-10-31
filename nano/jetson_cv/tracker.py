# -*- coding: utf-8 -*-
"""Lightweight multi-object tracker for the dashboard demo features
(unique-ID counts, ROI enter/exit counting, trajectories).

Design: centroid tracker with IoU gating and greedy matching (NOT SORT /
Kalman, NOT scipy's Hungarian solver). Rationale: no new dependency to
verify offline on Python 3.6 aarch64, and with a few dozen boxes at 20-30
fps on an already busy Nano, a Kalman filter adds cost and numerical
fragility for little gain in a demo with slow-moving objects. IoU gating
(on top of plain centroid distance) prevents ID swaps when two objects of
the same class cross.
"""
import collections


def _centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track(object):
    def __init__(self, track_id, bbox, cls_id):
        self.id = track_id
        self.bbox = bbox
        self.cls_id = cls_id
        self.centroid_history = collections.deque(maxlen=64)
        self.centroid_history.append(_centroid(bbox))
        self.hits = 1
        self.missed = 0
        self.inside_roi = False

    def update(self, bbox):
        self.bbox = bbox
        self.centroid_history.append(_centroid(bbox))
        self.hits += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1


class CentroidTracker(object):
    def __init__(self, max_missed=10, max_center_dist=80.0, iou_weight=0.5):
        self.max_missed = max_missed
        self.max_center_dist = max_center_dist
        self.iou_weight = iou_weight
        self._tracks = {}             # id -> Track
        self._next_id = 1
        self.total_ids_by_class = {}  # cls_id -> number of distinct IDs ever created

    @property
    def tracks(self):
        return list(self._tracks.values())

    def update(self, dets):
        """dets: [(x1,y1,x2,y2,score,cls_id), ...] for the current frame.

        Returns the active Tracks (with their assigned ids) after matching.
        """
        det_bboxes = [d[:4] for d in dets]
        det_cls = [d[5] for d in dets]

        candidates = []
        for tid, track in self._tracks.items():
            for di, det_bbox in enumerate(det_bboxes):
                if det_cls[di] != track.cls_id:
                    continue
                iou = _iou(track.bbox, det_bbox)
                tc, dc = _centroid(track.bbox), _centroid(det_bbox)
                dist = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5
                if iou <= 0.0 and dist > self.max_center_dist:
                    continue  # not a plausible match
                norm_dist = min(1.0, dist / self.max_center_dist)
                cost = (self.iou_weight * (1.0 - iou) +
                       (1.0 - self.iou_weight) * norm_dist)
                candidates.append((cost, tid, di))
        candidates.sort(key=lambda c: c[0])

        matched_tracks, matched_dets = set(), set()
        for cost, tid, di in candidates:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)
            self._tracks[tid].update(det_bboxes[di])

        for tid, track in list(self._tracks.items()):
            if tid not in matched_tracks:
                track.mark_missed()
                if track.missed > self.max_missed:
                    del self._tracks[tid]

        for di, det_bbox in enumerate(det_bboxes):
            if di in matched_dets:
                continue
            cls_id = det_cls[di]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(tid, det_bbox, cls_id)
            self.total_ids_by_class[cls_id] = (
                self.total_ids_by_class.get(cls_id, 0) + 1)

        return list(self._tracks.values())


def point_in_roi(cx, cy, roi, frame_w, frame_h):
    if not roi.get("enabled"):
        return False
    x1, y1 = roi["x1"] * frame_w, roi["y1"] * frame_h
    x2, y2 = roi["x2"] * frame_w, roi["y2"] * frame_h
    return x1 <= cx <= x2 and y1 <= cy <= y2


def update_roi_membership(tracks, roi, frame_w, frame_h):
    """Update track.inside_roi and return the current frame's events:
    {'enter': n, 'exit': n, 'inside_now': n, 'events': [...]}. Enter/exit
    count one event per crossing (state change), not per frame. 'events' is
    the same information in detail ({'type', 'cls_id'}, same order), so the
    caller can turn it into a readable log without this module having to
    know class names."""
    enter = exit_ = inside_now = 0
    events = []
    enabled = roi.get("enabled", False)
    for track in tracks:
        if not enabled:
            track.inside_roi = False
            continue
        cx, cy = _centroid(track.bbox)
        now_inside = point_in_roi(cx, cy, roi, frame_w, frame_h)
        if now_inside and not track.inside_roi:
            enter += 1
            events.append({"type": "enter", "cls_id": track.cls_id})
        elif track.inside_roi and not now_inside:
            exit_ += 1
            events.append({"type": "exit", "cls_id": track.cls_id})
        track.inside_roi = now_inside
        if now_inside:
            inside_now += 1
    return {"enter": enter, "exit": exit_, "inside_now": inside_now,
            "events": events}
