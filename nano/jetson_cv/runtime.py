"""Process-level workarounds for JetPack 4 on the Jetson Nano.

Two known traps, both of which must be handled before the process imports
numpy / cv2 / torch:

- OPENBLAS_CORETYPE=ARMV8: without it numpy's bundled OpenBLAS dies with
  "Illegal instruction (core dumped)" on the Nano's Cortex-A57.
- LD_PRELOAD=libgomp: importing cv2 after torch fails with "cannot allocate
  memory in static TLS block". LD_PRELOAD is only honoured at process start,
  so if it is missing we re-exec the interpreter with it set.

The venv's activate script and the systemd unit already export both; this
module makes `python apps/foo.py` work even when they were not.
"""
import os
import platform
import sys

LIBGOMP = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"


def ensure():
    if platform.machine() != "aarch64":
        return
    os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
    preload = os.environ.get("LD_PRELOAD", "")
    if "libgomp" in preload or not os.path.exists(LIBGOMP):
        return
    if os.environ.get("_JETSON_CV_REEXEC"):
        return  # never loop if the preload somehow does not stick
    os.environ["LD_PRELOAD"] = (LIBGOMP + " " + preload).strip()
    os.environ["_JETSON_CV_REEXEC"] = "1"
    # `python -u` is an interpreter flag that execv would drop. Under -u,
    # Python 3.6's stdout has no buffered layer (no `.raw` underneath).
    stdout_buffer = getattr(sys.stdout, "buffer", None)
    if stdout_buffer is not None and not hasattr(stdout_buffer, "raw"):
        os.environ["PYTHONUNBUFFERED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
