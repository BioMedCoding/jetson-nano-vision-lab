import fcntl, struct, os, select, mmap, ctypes

def IOWR(t, nr, size):
    return (3 << 30) | (size << 16) | (ord(t) << 8) | nr
def IOW(t, nr, size):
    return (1 << 30) | (size << 16) | (ord(t) << 8) | nr
def fourcc(s):
    return s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24)

FMT_SIZE = 208   # corrected: struct v4l2_format is 8-byte aligned -> 208, not 204
VIDIOC_S_FMT     = IOWR('V', 5, FMT_SIZE)
VIDIOC_G_FMT     = IOWR('V', 4, FMT_SIZE)
VIDIOC_REQBUFS   = IOWR('V', 8, 20)
VIDIOC_QUERYBUF  = IOWR('V', 9, 88)
VIDIOC_QBUF      = IOWR('V', 15, 88)
VIDIOC_DQBUF     = IOWR('V', 17, 88)
VIDIOC_STREAMON  = IOW('V', 18, 4)
VIDIOC_STREAMOFF = IOW('V', 19, 4)
VIDIOC_S_CTRL    = IOWR('V', 28, 8)
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
TEGRA_CAMERA_CID_VI_BYPASS_MODE = 0x009a2000 + 100

class timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]

class timecode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
                ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]

class v4l2_buffer(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("field", ctypes.c_uint32), ("timestamp", timeval),
                ("timecode", timecode), ("sequence", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("m_offset", ctypes.c_uint32),
                ("m_pad", ctypes.c_uint32),
                ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32)]

fd = os.open('/dev/video0', os.O_RDWR)

sc = bytearray(struct.pack('Ii', TEGRA_CAMERA_CID_VI_BYPASS_MODE, 0))
try:
    fcntl.ioctl(fd, VIDIOC_S_CTRL, sc)
    print('set VI_BYPASS_MODE=0 (SWITCH_OFF) ok')
except OSError as e:
    print('S_CTRL VI_BYPASS_MODE failed:', e)

buf = bytearray(FMT_SIZE)
struct.pack_into('I', buf, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
struct.pack_into('IIII', buf, 4, 640, 480, fourcc(b'BG10'), 1)
fcntl.ioctl(fd, VIDIOC_S_FMT, buf)
w, h, pf, field, bpl, sizeimg = struct.unpack_from('IIIIII', buf, 4)
print('S_FMT ->', w, h, struct.pack('I', pf), 'bpl=', bpl, 'sizeimage=', sizeimg)

gbuf = bytearray(FMT_SIZE)
struct.pack_into('I', gbuf, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
fcntl.ioctl(fd, VIDIOC_G_FMT, gbuf)
w, h, pf, field, bpl, sizeimg = struct.unpack_from('IIIIII', gbuf, 4)
print('G_FMT (chan->format) ->', w, h, struct.pack('I', pf), 'bpl=', bpl, 'sizeimage=', sizeimg)

rb = bytearray(struct.pack('IIIII', 4, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP, 0, 0))
fcntl.ioctl(fd, VIDIOC_REQBUFS, rb)
count = struct.unpack_from('I', rb, 0)[0]
print('REQBUFS -> count=', count)

buffers = []
for i in range(count):
    qb = v4l2_buffer()
    qb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    qb.memory = V4L2_MEMORY_MMAP
    qb.index = i
    fcntl.ioctl(fd, VIDIOC_QUERYBUF, qb)
    print('QUERYBUF', i, 'offset=', qb.m_offset, 'length=', qb.length)
    mm = mmap.mmap(fd, qb.length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=qb.m_offset)
    buffers.append(mm)
    fcntl.ioctl(fd, VIDIOC_QBUF, qb)

typ = ctypes.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE)
fcntl.ioctl(fd, VIDIOC_STREAMON, typ)
print('STREAMON ok')

frames = []
for i in range(300):
    r, _, _ = select.select([fd], [], [], 2.0)
    if not r:
        print('timeout waiting for frame', i)
        break
    db = v4l2_buffer()
    db.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    db.memory = V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, VIDIOC_DQBUF, db)
    data = buffers[db.index].read(db.bytesused if db.bytesused else 640 * 480 * 2)
    buffers[db.index].seek(0)
    None if i % 20 else print(i, "index=", db.index, "bytesused=", db.bytesused, "first bytes:", data[:8].hex())
    frames.append(data)
    fcntl.ioctl(fd, VIDIOC_QBUF, db)

fcntl.ioctl(fd, VIDIOC_STREAMOFF, typ)
os.close(fd)

if frames:
    import numpy as np, cv2
    raw16 = np.frombuffer(frames[-1], dtype='<u2')[:640 * 480].reshape((480, 640))
    print('raw16 stats: min', raw16.min(), 'max', raw16.max(), 'mean', raw16.mean())
    raw8 = (raw16 >> 2).astype(np.uint8)  # 10-bit -> 8-bit, keeps Bayer pattern intact
    for name, code in [('bggr2bgr', cv2.COLOR_BayerBG2BGR), ('rggb2bgr', cv2.COLOR_BayerRG2BGR),
                        ('gbrg2bgr', cv2.COLOR_BayerGB2BGR), ('grbg2bgr', cv2.COLOR_BayerGR2BGR)]:
        bgr = cv2.cvtColor(raw8, code)
        cv2.imwrite('/tmp/ov5647_%s.jpg' % name, bgr)
    print('saved snapshots')
