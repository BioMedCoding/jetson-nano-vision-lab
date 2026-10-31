"""Decoders: raw network output -> detections in original-frame coordinates.

Every decoder has the same signature

    decode(pred, ratio, pad, orig_shape, conf_thres, iou_thres)
        -> [(x1, y1, x2, y2, score, class_id), ...]

and is registered under a family name. A model's model.json says which
family to use ("decoder": "yolo_v5"). Supporting a new output format means
adding one function here and registering it; nothing else changes.

Families:
  yolo_v5  anchor-based head WITH an objectness column: output (N, 5 + nc)
           = cx, cy, w, h, obj, class scores. YOLOv5 (non-"u") and YOLOv7
           exported with --grid both produce this.
  yolo_v8  anchor-free head WITHOUT objectness: output (4 + nc, N), i.e.
           transposed. YOLOv8 and later ultralytics heads (and yolov5 "u"
           variants) produce this.
"""
import cv2
import numpy as np


def _nms_and_scale(boxes_xywh, scores, cls_ids, r, pad, orig_shape,
                   conf_thres, iou_thres):
    """NMS (OpenCV) + map letterboxed coordinates back to the original frame.
    Shared by all families: they only differ in how they get to
    (boxes_xywh, scores, cls_ids)."""
    xy, wh = boxes_xywh[:, :2], boxes_xywh[:, 2:4]
    boxes_tl = np.concatenate([xy - wh / 2.0, wh], 1)
    idxs = cv2.dnn.NMSBoxes(boxes_tl.tolist(), scores.tolist(),
                            float(conf_thres), float(iou_thres))
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten()

    dets = []
    oh, ow = orig_shape[:2]
    for i in idxs:
        x, y, w, h = boxes_tl[i]
        x1 = int(max((x - pad[0]) / r, 0))
        y1 = int(max((y - pad[1]) / r, 0))
        x2 = int(min((x + w - pad[0]) / r, ow - 1))
        y2 = int(min((y + h - pad[1]) / r, oh - 1))
        dets.append((x1, y1, x2, y2, float(scores[i]), int(cls_ids[i])))
    return dets


def decode_yolo_v5(pred, r, pad, orig_shape, conf_thres=0.4, iou_thres=0.45):
    pred = pred.reshape(-1, pred.shape[-1])
    pred = pred[pred[:, 4] > conf_thres]           # objectness pre-filter
    if not len(pred):
        return []
    scores_all = pred[:, 5:] * pred[:, 4:5]        # score = obj * cls
    cls_ids = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls_ids)), cls_ids]
    keep = scores > conf_thres
    pred, cls_ids, scores = pred[keep], cls_ids[keep], scores[keep]
    if not len(pred):
        return []
    return _nms_and_scale(pred[:, :4], scores, cls_ids, r, pad, orig_shape,
                          conf_thres, iou_thres)


def decode_yolo_v8(pred, r, pad, orig_shape, conf_thres=0.4, iou_thres=0.45):
    if pred.ndim > 2:
        pred = pred.reshape(pred.shape[-2], -1)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                              # (4+nc, N) -> (N, 4+nc)
    boxes = pred[:, :4]
    scores_all = pred[:, 4:]                       # no objectness column
    if not len(boxes):
        return []
    cls_ids = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls_ids)), cls_ids]
    keep = scores > conf_thres
    boxes, cls_ids, scores = boxes[keep], cls_ids[keep], scores[keep]
    if not len(boxes):
        return []
    return _nms_and_scale(boxes, scores, cls_ids, r, pad, orig_shape,
                          conf_thres, iou_thres)


DECODERS = {
    "yolo_v5": decode_yolo_v5,
    "yolo_v8": decode_yolo_v8,
}

# Names used by older metadata files and by --postprocess flags.
ALIASES = {"v5": "yolo_v5", "v7": "yolo_v5", "v8": "yolo_v8"}


def canonical_name(name):
    return ALIASES.get(name, name)


def get_decoder(name):
    key = canonical_name(name)
    if key not in DECODERS:
        raise KeyError("unknown decoder '{}' (known: {})".format(
            name, ", ".join(sorted(DECODERS))))
    return DECODERS[key]
