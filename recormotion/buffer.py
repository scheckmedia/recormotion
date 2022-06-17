import time
from threading import Lock, Thread

import cv2

from recormotion.config import Configuration


class CaptureBuffer:
    def __init__(self):
        self._cfg = Configuration().config.video

        self._buffer_size = self._cfg.buffer_size
        self._buffer_pos = 0
        self._buffer = [None] * self._buffer_size

        self._running = False
        self._cam = cv2.VideoCapture(self._cfg.input)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.resolution.width)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.resolution.height)

        self._wait_time = 1 / self._cfg.max_fps

        self._lock = Lock()
        self._t = Thread(target=self._process)
        self._t.start()

    def _process(self):
        self._running = True
        while self._running:
            time.sleep(self._wait_time)
            success, img = self._cam.read()

            if not success:
                continue

            with self._lock:
                self._buffer_pos = (self._buffer_pos + 1) % self._buffer_size
                self._buffer[self._buffer_pos] = img

    @property
    def frame(self):
        with self._lock:
            return self._buffer[self._buffer_pos]
