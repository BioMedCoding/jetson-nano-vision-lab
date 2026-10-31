"""Inference: preprocessing, TensorRT execution, output decoders.

Import submodules explicitly; trt_runner/detector pull in tensorrt and pycuda,
which apps that only need the camera should not have to load."""
