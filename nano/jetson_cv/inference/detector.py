"""Detector = TensorRT engine + preprocessing + the decoder named by the
model's model.json. This is the one object apps use to run a model."""
import time

from .decoders import get_decoder
from .preprocess import preprocess
from .trt_runner import TrtRunner


class Detector(object):
    def __init__(self, model_info):
        if model_info.task != "detect":
            raise ValueError("model {} has task '{}'; only 'detect' is implemented"
                             .format(model_info.id, model_info.task))
        if not model_info.has_engine:
            raise RuntimeError(
                "model {} has no TensorRT engine yet ({}). Build it on the Nano:\n"
                "  nano/apps/build_engine.sh {}".format(
                    model_info.id, model_info.engine_path, model_info.dir))
        self.info = model_info
        self.decode = get_decoder(model_info.decoder)  # fail fast on a bad name
        self.runner = TrtRunner(model_info.engine_path)
        self.input_size = self.runner.size
        if self.input_size != model_info.input_size:
            print("[detector] warning: {} says input_size={} but the engine "
                  "expects {}; using the engine's".format(
                      model_info.id, model_info.input_size, self.input_size))
        self.names = model_info.names()

    def detect(self, frame_bgr, conf=0.4, iou=0.45):
        """Returns (detections, inference_ms). Detections are
        [(x1, y1, x2, y2, score, class_id), ...] in frame coordinates;
        inference_ms covers only the TensorRT call."""
        img, ratio, pad = preprocess(frame_bgr, self.input_size)
        t0 = time.time()
        pred = self.runner.infer(img)
        infer_ms = (time.time() - t0) * 1000.0
        dets = self.decode(pred, ratio, pad, frame_bgr.shape,
                           conf_thres=conf, iou_thres=iou)
        return dets, infer_ms
