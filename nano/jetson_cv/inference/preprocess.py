"""Image -> network input. Letterboxing keeps the aspect ratio (the camera
is 4:3, the networks are square) and is undone again by the decoders."""
import cv2
import numpy as np


def letterbox(img, size=416, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to size x size.

    Returns (image, scale ratio, (pad_x, pad_y)).
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
    """BGR frame -> float32 (1, 3, S, S) in [0, 1], RGB, plus letterbox params."""
    img, r, pad = letterbox(frame_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return img[None], r, pad
