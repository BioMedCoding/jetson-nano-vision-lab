"""Run a serialized TensorRT engine (built by apps/build_engine.sh).

Needs tensorrt (system package on JetPack) and pycuda (vendored wheel); no
torch at runtime. The pycuda CUDA context belongs to the thread that creates
the runner: create, use and replace runners from one thread only.
"""
import numpy as np


class TrtRunner(object):
    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        self._cuda = cuda

        # TensorRT 8.2 on the Nano prints "Using an engine plan file across
        # different models of devices is not recommended" even for engines
        # built on this very board; it is harmless here.
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Invalid TensorRT engine: {} (built with a different "
                               "TensorRT/JetPack version?)".format(engine_path))
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self._host = []      # page-locked host buffers, one per binding
        self._dev = []       # device buffers
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

        # YOLOv5 v6.0 exports have several outputs (the concatenated
        # predictions plus one raw map per scale). The one we want is the
        # largest output with >= 2 dims.
        best = -1
        for i, shape in enumerate(self._shapes):
            if self._is_input[i] or len(shape) < 2:
                continue
            if int(np.prod(shape)) > best:
                best = int(np.prod(shape))
                self._out_idx = i
        if self._out_idx is None:
            raise RuntimeError("No usable output binding in {}".format(engine_path))
        self.size = self.input_shape[-1]
        self.output_shape = self._shapes[self._out_idx]

    def infer(self, img_nchw):
        """img_nchw: float32 (1, 3, S, S). Returns the main output with its
        batch dimension removed, e.g. (N, 85) for yolo_v5, (84, N) for yolo_v8."""
        cuda = self._cuda
        in_idx = self._is_input.index(True)
        np.copyto(self._host[in_idx],
                  np.ascontiguousarray(img_nchw, dtype=self._host[in_idx].dtype).ravel())
        cuda.memcpy_htod_async(self._dev[in_idx], self._host[in_idx], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self._host[self._out_idx],
                               self._dev[self._out_idx], self.stream)
        self.stream.synchronize()
        out = self._host[self._out_idx].reshape(self.output_shape)
        return out.reshape(out.shape[-2], out.shape[-1])
