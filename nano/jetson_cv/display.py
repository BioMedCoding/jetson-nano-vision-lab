"""Show frames either in a local window or as an MJPEG stream in a browser.

With an X11 display an OpenCV window opens (quit with 'q' or ESC). Without
one (SSH / headless) an MJPEG stream is served at http://<jetson>:8090/
(quit with Ctrl+C).

    viewer = Viewer("title")      # picks the mode automatically
    while viewer.show(frame):     # BGR frame
        ...
    viewer.close()

FrameStore and stream_mjpeg() are shared with the dashboard server.
"""
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

DEFAULT_PORT = 8090

_PAGE = ("<html><head><title>{title}</title></head>"
         "<body style='margin:0;background:#111;text-align:center'>"
         "<img src='/stream' style='max-width:100%;height:auto'>"
         "</body></html>")


class FrameStore(object):
    """Latest JPEG frame, shared between the capture loop and HTTP clients.
    `clients` lets the producer skip JPEG encoding when nobody watches."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self.clients = 0

    def update(self, jpeg):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_next(self, last_seq, timeout=1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq


def stream_mjpeg(handler, store):
    """Serve `store` as multipart/x-mixed-replace on a BaseHTTPRequestHandler."""
    handler.send_response(200)
    handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
    handler.end_headers()
    store.clients += 1
    seq = -1
    try:
        while True:
            jpeg, seq = store.wait_next(seq)
            if jpeg is None:
                continue
            handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
            handler.wfile.write(jpeg)
            handler.wfile.write(b"\r\n")
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        store.clients -= 1


def local_ip():
    """Best-effort LAN address, only used to print a clickable URL."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # no packet is sent for a UDP connect
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler(store, title):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def do_GET(self):
            if self.path == "/stream":
                stream_mjpeg(self, store)
                return
            body = _PAGE.format(title=title).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class Viewer(object):
    def __init__(self, title="preview", headless=None, port=DEFAULT_PORT,
                 jpeg_quality=80):
        self.title = title
        self._quality = jpeg_quality
        self._server = None
        self._store = None
        if headless is None:
            headless = not os.environ.get("DISPLAY")
        if not headless:
            try:
                cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            except cv2.error:
                headless = True
        self.headless = headless
        if headless:
            self._store = FrameStore()
            self._server = ThreadedHTTPServer(("0.0.0.0", port),
                                              _make_handler(self._store, title))
            t = threading.Thread(target=self._server.serve_forever)
            t.daemon = True
            t.start()
            print("[viewer] No display: MJPEG stream at http://{}:{}/ "
                  "(Ctrl+C to quit)".format(local_ip(), port))

    def show(self, frame):
        """Show a BGR frame. Returns False when the user asks to quit."""
        return self.poll_key(frame) not in (ord("q"), 27)

    def poll_key(self, frame):
        """Show a frame and return the key pressed in the window (-1 if none
        or headless). For apps with extra hotkeys (snapshot, capture)."""
        if self.headless:
            if self._store.clients > 0:  # only encode when someone watches
                ok, jpeg = cv2.imencode(".jpg", frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, self._quality])
                if ok:
                    self._store.update(jpeg.tobytes())
            return -1
        cv2.imshow(self.title, frame)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        if self._server is not None:
            self._server.shutdown()
        if not self.headless:
            cv2.destroyAllWindows()
