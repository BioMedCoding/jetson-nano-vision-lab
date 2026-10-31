# -*- coding: utf-8 -*-
"""Helper per la camera CSI (OV5647) su Jetson Nano via V4L2 raw.

NOTA: questo modulo NON usa piu' GStreamer/nvarguscamerasrc. Argus ha una
whitelist chiusa di sensori e non riconoscera' mai questo OV5647 (driver
V4L2 puro, non tegracam). La cattura avviene con ioctl V4L2 dirette su
/dev/video0: il sensore emette Bayer RAW10 (ordine reale GRBG, verificato
con oggetto di colore noto) a 640x480 fisso, il demosaic e' fatto qui con
OpenCV. L'interfaccia esposta imita cv2.VideoCapture (read/isOpened/release)
cosi' gli script esistenti non cambiano.

Vincoli ereditati dal driver kernel (una sola modalita' fissa):
  - risoluzione sensore 640x480 (richieste diverse -> resize software)
  - fps non impostabile (il sensore gira a ~60fps; il parametro e' ignorato)
  - esposizione/gain: AEC/AGC automatici del sensore, converge in ~2-3 s
"""
import ctypes
import fcntl
import mmap
import os
import select
import struct

import cv2
import numpy as np

SENSOR_W, SENSOR_H = 640, 480
# Ordine Bayer reale del flusso (le etichette driver/DT dicono BGGR ma i
# dati sono GRBG: verificato empiricamente, vedi OV5647_DEBUG_HANDOFF.md).
DEMOSAIC_CODE = cv2.COLOR_BayerGR2BGR

_FMT_SIZE = 208  # sizeof(struct v4l2_format): union allineata a 8 byte


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
# Deve restare 0 (default): con bypass=1 il VI Tegra non programma il
# canale di cattura e non arriva mai nessun frame.
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
    """Cattura V4L2 raw + demosaic, interfaccia compatibile cv2.VideoCapture."""

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
            raise RuntimeError("REQBUFS: solo {} buffer".format(count))

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
        # >>2: 10 bit -> 8 bit; copia i dati, dopo si puo' rimettere in coda
        raw8 = (raw16 >> 2).astype(np.uint8)
        fcntl.ioctl(self._fd, _VIDIOC_QBUF, db)

        frame = cv2.cvtColor(raw8, DEMOSAIC_CODE)
        # flip: stessa semantica di flip-method di nvvidconv (0=no, 1=90 ccw,
        # 2=180, 3=90 cw); altri valori ignorati
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


def open_csi_camera(width=640, height=480, fps=30, flip=0):
    """Apre la camera CSI. Solleva RuntimeError con suggerimenti se fallisce.

    `fps` e' accettato per compatibilita' ma ignorato (modalita' sensore fissa).
    """
    try:
        return V4L2RawCamera(width=width, height=height, flip=flip)
    except OSError as e:
        raise RuntimeError(
            "Impossibile aprire la camera CSI via V4L2 ({}).\n"
            "  Verifiche:\n"
            "    1) il device esiste?      ls /dev/video0\n"
            "    2) modulo caricato?       lsmod | grep ov5647\n"
            "    3) errori kernel?         dmesg | grep -iE 'ov5647|vi |csi'\n"
            "    4) altro processo che usa la camera? fuser /dev/video0".format(e)
        )
