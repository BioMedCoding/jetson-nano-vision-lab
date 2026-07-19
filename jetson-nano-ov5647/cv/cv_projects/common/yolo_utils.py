# -*- coding: utf-8 -*-
"""Utilita' condivise per pre/post-processing YOLOv5 senza dipendere da torch.

Usate dallo script TensorRT e dal benchmark: letterbox, decodifica dell'output
(1, N, 85) gia' passato per sigmoid/griglia (formato dell'export ONNX v6.0),
NMS via OpenCV e disegno dei box.
"""
import cv2
import numpy as np

COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']


def letterbox(img, size=416, color=(114, 114, 114)):
    """Ridimensiona mantenendo l'aspect ratio e riempie fino a size x size.

    Ritorna (immagine, rapporto di scala, (pad_x, pad_y)).
    """
    h, w = img.shape[:2]
    r = min(float(size) / h, float(size) / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    dh, dw = (size - nh) / 2.0, (size - nw) / 2.0
    if (h, w) != (nh, nw):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(frame_bgr, size=416):
    """BGR -> tensore (1,3,S,S) float32 normalizzato, piu' i parametri letterbox."""
    img, r, pad = letterbox(frame_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return img[None], r, pad


def _nms_and_scale(boxes_xywh, scores, cls_ids, r, pad, orig_shape,
                   conf_thres, iou_thres):
    """NMS (cv2.dnn) + rimappatura da coordinate letterbox a quelle del frame
    originale. boxes_xywh: (N,4) centro-formato (cx,cy,w,h).

    Comune a tutti i decoder (v5/v7/v8): cambia solo come si arriva a
    (boxes_xywh, scores, cls_ids), non cosa succede dopo. Ritorna
    [(x1, y1, x2, y2, score, class_id), ...].
    """
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
        # riporta dalle coordinate letterbox a quelle del frame originale
        x1 = int(max((x - pad[0]) / r, 0))
        y1 = int(max((y - pad[1]) / r, 0))
        x2 = int(min((x + w - pad[0]) / r, ow - 1))
        y2 = int(min((y + h - pad[1]) / r, oh - 1))
        dets.append((x1, y1, x2, y2, float(scores[i]), int(cls_ids[i])))
    return dets


def postprocess(pred, r, pad, orig_shape, conf_thres=0.4, iou_thres=0.45):
    """Da output YOLOv5 (N, 85) a lista di detection sul frame originale.

    Ritorna [(x1, y1, x2, y2, score, class_id), ...].
    """
    pred = pred.reshape(-1, pred.shape[-1])
    pred = pred[pred[:, 4] > conf_thres]           # filtro sull'objectness
    if not len(pred):
        return []
    scores_all = pred[:, 5:] * pred[:, 4:5]        # conf = obj * cls
    cls_ids = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls_ids)), cls_ids]
    keep = scores > conf_thres
    pred, cls_ids, scores = pred[keep], cls_ids[keep], scores[keep]
    if not len(pred):
        return []
    return _nms_and_scale(pred[:, :4], scores, cls_ids, r, pad, orig_shape,
                          conf_thres, iou_thres)


def postprocess_v8(pred, r, pad, orig_shape, conf_thres=0.4, iou_thres=0.45):
    """Da output YOLOv8 (export ultralytics, opset12) a lista di detection sul
    frame originale. Formato reale verificato sull'ONNX esportato:
    (1, 4+nc, N) -> TrtRunner lo restituisce gia' come (4+nc, N) (righe=canali,
    colonne=predizioni), TRASPOSTO rispetto al (N, 85) di v5/v7. Niente
    objectness esplicito (a differenza di v5): lo score di classe e' usato
    cosi' com'e', senza moltiplicarlo per un obj-score che qui non esiste.

    Ritorna [(x1, y1, x2, y2, score, class_id), ...].
    """
    if pred.ndim > 2:
        pred = pred.reshape(pred.shape[-2], -1)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # (4+nc, N) -> (N, 4+nc)
    boxes = pred[:, :4]
    scores_all = pred[:, 4:]
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


def _class_color(cls_id):
    np.random.seed(cls_id + 7)
    return tuple(int(c) for c in np.random.randint(60, 255, 3))


def draw_detections(frame, dets, names=COCO_NAMES):
    for x1, y1, x2, y2, score, cls_id in dets:
        color = _class_color(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(names[cls_id] if cls_id < len(names)
                                   else str(cls_id), score)
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
