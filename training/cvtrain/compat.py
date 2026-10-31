"""filelock <-> ultralytics compatibility shim.

filelock is pinned below 3.12 (training/setup/requirements.lock) because
nnictl (NNI 3.0) deadlocks silently with filelock >= 3.12: NNI's internal
SimplePreemptiveLock subclasses filelock.SoftFileLock, whose behaviour
changed incompatibly (a legitimate lock became permanently unacquirable,
even past its staleness threshold; it hangs at "Starting web server...").

ultralytics.data.converter, however, does `from filelock import
AsyncFileLock, Timeout` at module level (AsyncFileLock appeared after 3.12),
so merely *importing* ultralytics fails with the pinned filelock, although
nothing here uses the async-locked code path. The name only has to exist.

Import this module before anything imports ultralytics (directly, or via
yolov5's utils/general.py). The vendored yolov5/train.py and export.py carry
the same shim (vendor/yolov5/LOCAL_PATCHES.diff).
"""
import filelock

if not hasattr(filelock, "AsyncFileLock"):
    filelock.AsyncFileLock = filelock.FileLock
