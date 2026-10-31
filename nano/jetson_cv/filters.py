"""Live image adjustments (brightness / contrast / filters) for the dashboard.

apply_filters() is applied to the frame that is BOTH displayed AND fed to
the network: one pipeline, so you see what the model sees. The expected
drop in detection quality with filters such as gray or edge is the point:
it is an educational test bench.
"""
import cv2
import numpy as np

FILTER_NAMES = ("none", "gray", "clahe", "sharpen", "blur", "edge")


def _brightness_contrast(frame, brightness, contrast):
    if brightness == 0 and contrast == 1.0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)


def _to_bgr(gray_or_bgr):
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr


def _filter_gray(frame):
    return _to_bgr(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


def _filter_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _filter_sharpen(frame):
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)


def _filter_blur(frame):
    return cv2.GaussianBlur(frame, (9, 9), sigmaX=0)


def _filter_edge(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return _to_bgr(edges)


_FILTER_FUNCS = {
    "none": lambda f: f,
    "gray": _filter_gray,
    "clahe": _filter_clahe,
    "sharpen": _filter_sharpen,
    "blur": _filter_blur,
    "edge": _filter_edge,
}


def apply_filters(frame_bgr, params):
    """Apply brightness/contrast, then the filter named in params['filter'].

    Always returns a 3-channel BGR frame (even for gray/edge), so the rest
    of the pipeline (preprocessing, drawing) needs no knowledge of filters.
    """
    frame = _brightness_contrast(
        frame_bgr, float(params.get("brightness", 0.0)),
        float(params.get("contrast", 1.0)))
    fn = _FILTER_FUNCS.get(params.get("filter", "none"), _FILTER_FUNCS["none"])
    return np.ascontiguousarray(fn(frame))
