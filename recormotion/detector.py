import time
from threading import Lock, Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests

from recormotion.buffer import CaptureBuffer
from recormotion.config import Configuration

cm = plt.get_cmap("rainbow")


class RemoteMotionDetector:
    def __init__(self, buffer: CaptureBuffer):
        self._cfg = Configuration().config.detection
        self._lock = Lock()
        self._busy = False
        self._request_uri = self._cfg.torchserve_url
        self._running = False
        self._buffer = buffer

        self._debug = self._cfg.debug
        self._last_frame = None
        self._last_match = -1
        self._wait_time = 1 / self._cfg.sampling_rate

        colors = list(
            (cm(1.0 * i / 90))
            for i in range(len(Configuration().config.detection.labels))
        )
        colors = (np.array(colors) * 255).astype(np.uint8)[..., :3]
        self._color_mapping = {
            l: tuple(c.tolist())
            for l, c in zip(Configuration().config.detection.labels.values(), colors)
        }

        self._t = Thread(target=self._process)
        self._t.start()

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def frame(self):
        return self._last_frame

    def _process(self):
        self._running = True
        while self._running:
            time.sleep(self._wait_time)

            frame = self._buffer.frame

            if not isinstance(frame, np.ndarray):
                time.sleep(0.1)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            _, encoded = cv2.imencode(".jpg", frame)
            res = requests.post(self._request_uri, files={"data": encoded})
            if res.status_code != 200:
                # todo: log warning
                continue

            self._last_frame = frame
            detections = res.json()

            self._parse_detections(detections)

    def _parse_detections(self, detections):
        for detection in detections:
            (class_name, coords), (_, score) = detection.items()
            x1, y1, x2, y2 = [int(x) for x in coords]
            if class_name not in self._cfg.trigger_classes:
                continue

            if self._debug:
                color = self._color_mapping[class_name]
                self._last_frame = cv2.rectangle(
                    self._last_frame, (x1, y1), (x2, y2), color, 2
                )
