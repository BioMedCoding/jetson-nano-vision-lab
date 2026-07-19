# -*- coding: utf-8 -*-
"""Visualizzazione con fallback automatico.

Con un display X11 disponibile apre una finestra OpenCV (uscita con 'q' o ESC).
Senza display (SSH/headless) avvia uno stream MJPEG raggiungibile dal browser
del PC host su http://<ip-jetson>:8090/ (uscita con Ctrl+C).

Uso:
    from common.viewer import Viewer
    v = Viewer("titolo")            # auto-rileva la modalita'
    while v.show(frame):            # frame BGR (formato OpenCV)
        ...
    v.close()
"""
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

MJPEG_PORT = 8090

_PAGE = ("<html><head><title>{title}</title></head>"
         "<body style='margin:0;background:#111;text-align:center'>"
         "<img src='/stream' style='max-width:100%;height:auto'>"
         "</body></html>")


class _FrameStore(object):
    """Ultimo frame JPEG condiviso tra il loop di cattura e i client HTTP."""

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


def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler(store, title):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # niente log delle richieste sul terminale
            pass

        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                store.clients += 1
                seq = -1
                try:
                    while True:
                        jpeg, seq = store.wait_next(seq)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\n"
                                         b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    store.clients -= 1
            else:
                body = _PAGE.format(title=title).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return Handler


class Viewer(object):
    def __init__(self, title="preview", headless=None, port=MJPEG_PORT,
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
            self._store = _FrameStore()
            self._server = _ThreadedHTTPServer(
                ("0.0.0.0", port), _make_handler(self._store, title))
            t = threading.Thread(target=self._server.serve_forever)
            t.daemon = True
            t.start()
            print("[viewer] Nessun display: stream MJPEG su http://{}:{}/ "
                  "(Ctrl+C per uscire)".format(_local_ip(), port))

    def show(self, frame):
        """Mostra un frame BGR. Ritorna False quando l'utente chiede di uscire."""
        if self.headless:
            # codifica solo se qualcuno sta guardando, per non sprecare CPU
            if self._store.clients > 0:
                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
                if ok:
                    self._store.update(jpeg.tobytes())
            return True
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self):
        if self._server is not None:
            self._server.shutdown()
        if not self.headless:
            cv2.destroyAllWindows()
