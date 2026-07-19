# -*- coding: utf-8 -*-
"""Esecuzione di engine TensorRT (costruiti con 03_tensorrt/build_engine.sh).

Richiede: tensorrt (gia' nel sistema JetPack) + pycuda (requirements_trt.txt).
Non dipende da torch: utile per inferenza leggera e per modelli custom
addestrati sull'host e portati qui come ONNX -> engine.
"""
import numpy as np


class TrtRunner(object):
    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (inizializza il contesto CUDA)
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Engine non valido: {}".format(engine_path))
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self._host = []      # buffer host page-locked, uno per binding
        self._dev = []       # buffer device
        self._is_input = []
        self._shapes = []
        self.input_shape = None
        self._out_idx = None

        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(abs(np.prod(shape)))
            host = cuda.pagelocked_empty(size, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            self._host.append(host)
            self._dev.append(dev)
            self._shapes.append(shape)
            self._is_input.append(self.engine.binding_is_input(i))
            if self.engine.binding_is_input(i):
                self.input_shape = shape          # (1, 3, S, S)

        # L'export ONNX v6.0 puo' produrre piu' output: il tensore delle
        # predizioni e' quello con ultima dimensione 85 (4 box + obj + 80 classi)
        # e piu' elementi totali.
        best = -1
        for i, shape in enumerate(self._shapes):
            if self._is_input[i] or len(shape) < 2:
                continue
            if shape[-1] >= 6 and int(np.prod(shape)) > best:
                best = int(np.prod(shape))
                self._out_idx = i
        if self._out_idx is None:
            raise RuntimeError("Nessun binding di output riconosciuto")
        self.size = self.input_shape[-1]

    def infer(self, img_nchw):
        """img_nchw: np.float32 (1,3,S,S) normalizzato 0..1.

        Ritorna l'output principale come array (N, 85).
        """
        cuda = self._cuda
        in_idx = self._is_input.index(True)
        np.copyto(self._host[in_idx],
                  np.ascontiguousarray(img_nchw, dtype=self._host[in_idx].dtype).ravel())
        cuda.memcpy_htod_async(self._dev[in_idx], self._host[in_idx], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self._host[self._out_idx],
                               self._dev[self._out_idx], self.stream)
        self.stream.synchronize()
        out_shape = self._shapes[self._out_idx]
        return self._host[self._out_idx].reshape(out_shape).reshape(-1, out_shape[-1])
