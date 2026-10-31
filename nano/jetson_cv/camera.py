"""OV5647 capture on the Jetson Nano via raw V4L2 ioctls.

Why not GStreamer / nvarguscamerasrc: Argus is closed source with a fixed
sensor whitelist and will never accept this driver (it is a plain V4L2
driver, not a tegracam one). Instead we talk to /dev/video0 directly: the
sensor streams Bayer RAW10 at a fixed 640x480, we mmap the buffers and
demosaic with OpenCV. The object returned by open_camera() mimics
cv2.VideoCapture (read / isOpened / release) so callers don't care.

Constraints inherited from the kernel driver (one fixed mode):
  - sensor resolution is 640x480; other sizes are a software resize
  - frame rate is not settable (the sensor free-runs at ~60 fps)
  - exposure/gain: the sensor's own AEC/AGC, converges in ~1-3 s
  - Bayer order on the wire is GRBG, whatever the driver/DT label says
    (BGGR); verified with an object of known colour, see
    camera/docs/CAMERA_DEBUG.md
"""
import ctypes
import fcntl
import mmap
import os
import select
import struct
import time

import cv2
import numpy as np

SENSOR_W, SENSOR_H = 640, 480
DEMOSAIC_CODE = cv2.COLOR_BayerGR2BGR

_FMT_SIZE = 208  # sizeof(struct v4l2_format): 8-byte aligned union


def _IOWR(t, nr, size):
    return (3 << 30) | (size << 16) | (ord(t) << 8) | nr


def _IOW(t, nr, size):
    return (1 << 30) | (size << 16) | (ord(t) << 8) | nr


def _fourcc(s):
    return s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24)


_VIDIOC_S_FMT = _IOWR('V', 5, _FMT_SIZE)
_VIDIOC_REQBUFS = _IOWR('V', 8, 20)
_VIDIOC_QUERYBUF = _IOWR('V', 9, 88)
_VIDIOC_QBUF = _IOWR('V', 15, 88)
_VIDIOC_DQBUF = _IOWR('V', 17, 88)
_VIDIOC_STREAMON = _IOW('V', 18, 4)
_VIDIOC_STREAMOFF = _IOW('V', 19, 4)
_VIDIOC_S_CTRL = _IOWR('V', 28, 8)
_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
_V4L2_MEMORY_MMAP = 1
# Must stay 0 (the default): with bypass=1 the Tegra VI never programs the
# capture channel and no frame ever arrives.
_TEGRA_CAMERA_CID_VI_BYPASS_MODE = 0x009a2000 + 100


class _timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _timecode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
                ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]


class _v4l2_buffer(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("field", ctypes.c_uint32), ("timestamp", _timeval),
                ("timecode", _timecode), ("sequence", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("m_offset", ctypes.c_uint32),
                ("m_pad", ctypes.c_uint32),
                ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32)]


class V4L2RawCamera(object):
    """Raw V4L2 capture + demosaic, cv2.VideoCapture-compatible interface."""

    def __init__(self, device="/dev/video0", width=SENSOR_W, height=SENSOR_H,
                 flip=0, n_buffers=4, warmup_frames=3, read_timeout=2.0):
        self._fd = -1
        self._buffers = []
        self._streaming = False
        self._out_size = (int(width), int(height))
        self._flip = int(flip)
        self._timeout = float(read_timeout)

        fd = os.open(device, os.O_RDWR)
        self._fd = fd

        ctrl = bytearray(struct.pack('Ii', _TEGRA_CAMERA_CID_VI_BYPASS_MODE, 0))
        fcntl.ioctl(fd, _VIDIOC_S_CTRL, ctrl)

        fmt = bytearray(_FMT_SIZE)
        struct.pack_into('I', fmt, 0, _V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into('IIII', fmt, 4, SENSOR_W, SENSOR_H,
                         _fourcc(b'BG10'), 1)
        fcntl.ioctl(fd, _VIDIOC_S_FMT, fmt)

        req = bytearray(struct.pack('IIIII', n_buffers,
                                    _V4L2_BUF_TYPE_VIDEO_CAPTURE,
                                    _V4L2_MEMORY_MMAP, 0, 0))
        fcntl.ioctl(fd, _VIDIOC_REQBUFS, req)
        count = struct.unpack_from('I', req, 0)[0]
        if count < 2:
            raise RuntimeError("REQBUFS: only {} buffer(s)".format(count))

        for i in range(count):
            qb = _v4l2_buffer()
            qb.type = _V4L2_BUF_TYPE_VIDEO_CAPTURE
            qb.memory = _V4L2_MEMORY_MMAP
            qb.index = i
            fcntl.ioctl(fd, _VIDIOC_QUERYBUF, qb)
            mm = mmap.mmap(fd, qb.length, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE,
                           offset=qb.m_offset)
            self._buffers.append(mm)
            fcntl.ioctl(fd, _VIDIOC_QBUF, qb)

        typ = ctypes.c_uint32(_V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(fd, _VIDIOC_STREAMON, typ)
        self._streaming = True

        for _ in range(warmup_frames):
            self.read()

    def isOpened(self):
        return self._streaming

    def read(self):
        if not self._streaming:
            return False, None
        r, _, _ = select.select([self._fd], [], [], self._timeout)
        if not r:
            return False, None
        db = _v4l2_buffer()
        db.type = _V4L2_BUF_TYPE_VIDEO_CAPTURE
        db.memory = _V4L2_MEMORY_MMAP
        try:
            fcntl.ioctl(self._fd, _VIDIOC_DQBUF, db)
        except OSError:
            return False, None

        mm = self._buffers[db.index]
        raw16 = np.frombuffer(mm, dtype='<u2',
                              count=SENSOR_W * SENSOR_H).reshape(
                                  (SENSOR_H, SENSOR_W))
        # >>2: 10 bit -> 8 bit. This copies the data, so the buffer can be
        # handed back to the driver right away.
        raw8 = (raw16 >> 2).astype(np.uint8)
        fcntl.ioctl(self._fd, _VIDIOC_QBUF, db)

        frame = cv2.cvtColor(raw8, DEMOSAIC_CODE)
        # flip: same meaning as nvvidconv's flip-method
        # (0 = none, 1 = 90 ccw, 2 = 180, 3 = 90 cw); other values ignored
        if self._flip == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self._flip == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self._flip == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if frame.shape[1::-1] != self._out_size:
            frame = cv2.resize(frame, self._out_size,
                               interpolation=cv2.INTER_LINEAR)
        return True, frame

    def release(self):
        if self._fd < 0:
            return
        if self._streaming:
            self._streaming = False
            typ = ctypes.c_uint32(_V4L2_BUF_TYPE_VIDEO_CAPTURE)
            try:
                fcntl.ioctl(self._fd, _VIDIOC_STREAMOFF, typ)
            except OSError:
                pass
        for mm in self._buffers:
            try:
                mm.close()
            except (OSError, ValueError):
                pass
        self._buffers = []
        os.close(self._fd)
        self._fd = -1

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def open_camera(width=SENSOR_W, height=SENSOR_H, flip=0, device="/dev/video0"):
    """Open the CSI camera. Raises RuntimeError with troubleshooting hints."""
    try:
        return V4L2RawCamera(device=device, width=width, height=height, flip=flip)
    except OSError as e:
        raise RuntimeError(
            "Cannot open the CSI camera via V4L2 ({}).\n"
            "  Checks:\n"
            "    1) does the device exist?     ls {}\n"
            "    2) is the driver loaded?      lsmod | grep ov5647\n"
            "    3) kernel errors?             dmesg | grep -iE 'ov5647|vi |csi'\n"
            "    4) camera busy (e.g. the dashboard service)?\n"
            "                                  fuser {}   /   systemctl status cv-demo"
            .format(e, device, device))


class ReconnectingCamera(object):
    """open_camera() plus automatic reopen after too many consecutive failed
    reads, with exponential backoff. Covers the real failure seen on this
    board (the capture pipeline wedges); the CSI camera cannot be hot-
    unplugged. Used by long-running apps such as the dashboard."""

    def __init__(self, width=SENSOR_W, height=SENSOR_H, flip=0, max_consec_fail=15,
                 log=print):
        self.width, self.height, self.flip = width, height, flip
        self.max_consec_fail = max_consec_fail
        self._log = log
        self.cap = None
        self._fail_count = 0
        self._open()

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = open_camera(self.width, self.height, self.flip)
        self._fail_count = 0

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            self._fail_count = 0
            return True, frame
        self._fail_count += 1
        if self._fail_count >= self.max_consec_fail:
            self._log("[camera] {} failed reads in a row, reopening the device..."
                      .format(self._fail_count))
            backoff = 2.0
            while True:
                try:
                    self._open()
                    self._log("[camera] reconnected.")
                    break
                except Exception as e:
                    self._log("[camera] reopen failed ({}), retrying in {:.0f}s..."
                              .format(e, backoff))
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 30.0)
        return False, None

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
