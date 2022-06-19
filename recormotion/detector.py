import time
from logging import getLogger
from threading import Lock, Thread

import cv2
import numpy as np
import requests

from recormotion.buffer import CaptureBuffer
from recormotion.config import Configuration
from recormotion.utils import Visualizer

logger = getLogger(__file__)


class RemoteMotionDetector:
    """Object that handles motion detection using TorchServe to
    detect objects of interesets that triggers a recording.

    :param buffer: buffer object that produces frames and handles recording
    :type buffer: CaptureBuffer
    """

    def __init__(self, buffer: CaptureBuffer):
        self._cfg = Configuration().config.detection
        self._lock = Lock()
        self._busy = False
        self._request_uri = self._cfg.torchserve_url
        self._running = False
        self._buffer = buffer

        self._debug = self._cfg.debug
        self._last_frame = None
        self._last_match = 0
        self._invalidate_duration_ns = self._cfg.trigger_invalidate_duration * 1e9
        self._wait_time = 1 / self._cfg.sampling_rate

        self._visualizer = Visualizer(self._cfg.labels)

        self._t = Thread(target=self._process)
        self._t.start()

    @property
    def debug(self) -> bool:
        """Getter of debugging state

        :return: true if debugging is active otherwise false
        :rtype: bool
        """
        return self._debug

    @debug.setter
    def debug(self, value: bool):
        """Setter to enable/disable debugging
        If debugging is enabled, bounding boxes and
        labels of potential trigger objects will be shown

        :param value: debug state
        :type value: bool
        """
        self._debug = value

    @property
    def frame(self) -> np.ndarray:
        """Getter to read the current frame from the motion detection
        where an trigger event occured

        :return: frame where a trigger event occured
        :rtype: np.ndarray
        """
        return self._last_frame

    def _process(self):
        self._running = True
        logger.info("Start motion detection")

        while self._running:
            time.sleep(self._wait_time)
            frame = self._buffer.frame

            if not isinstance(frame, np.ndarray):
                time.sleep(0.1)
                self._buffer.recording = False
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            _, encoded = cv2.imencode(".jpg", frame)
            res = requests.post(self._request_uri, files={"data": encoded})
            if res.status_code != 200:
                logger.error(
                    "TorchServe error: %s",
                    res.json(),
                )
                time.sleep(0.1)
                continue

            self._last_frame = frame
            detections = res.json()

            if self._detections_has_trigger(detections):
                logger.info(
                    "trigger detected at %d, already recording: %r",
                    self._last_match,
                    self._buffer.recording,
                )
                self._last_match = time.time_ns()
                self._buffer.recording = True

            if self._buffer.recording:
                match_delta = time.time_ns() - self._last_match
                if match_delta >= self._invalidate_duration_ns:
                    self._buffer.recording = False
                    logger.info("stop recording, last trigger duration outdated")

    def _detections_has_trigger(self, detections):
        logger.debug("Parse and process %d detections", len(detections))

        detected_classes = set()
        has_trigger = False
        for detection in detections:
            (class_name, coords), (_, score) = detection.items()
            if (
                class_name not in self._cfg.trigger_classes
                or score < self._cfg.trigger_detection_threshold
            ):
                continue

            has_trigger = True
            if self._debug:
                x1, y1, x2, y2 = [int(x) for x in coords]
                self._last_frame = self._visualizer.draw_detection(
                    self._last_frame, class_name, x1, y1, x2, y2
                )
                detected_classes.add(class_name)

        if len(detected_classes) > 0:
            self._last_frame = self._visualizer.draw_legend(
                self._last_frame, detected_classes
            )

        return has_trigger
