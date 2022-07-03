import time
from datetime import datetime
from logging import getLogger
from threading import Lock, Thread

import cv2
import numpy as np
import requests

from recormotion.config import Configuration
from recormotion.io import CaptureBuffer
from recormotion.utils import Visualizer

logger = getLogger(__file__)


class RemoteMotionDetector:
    """Object that handles motion detection using TorchServe to
    detect objects of interesets that triggers a recording.

    Args:
        buffer (CaptureBuffer): buffer object that produces frames and handles recording
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
        self._wait_time = 1 / self._cfg.sampling_rate
        self._recording_started = None

        self._visualizer = Visualizer(self._cfg.labels)

        self._t = Thread(target=self._process)
        self._t.start()

    @property
    def debug(self) -> bool:
        """Getter of debugging state

        Returns:
            bool: true if debugging is active otherwise false
        """
        return self._debug

    @debug.setter
    def debug(self, value: bool):
        """Setter to enable/disable debugging
        If debugging is enabled, bounding boxes and
        labels of potential trigger objects will be shown

        Args:
            value (bool): debug state
        """
        self._debug = value

    @property
    def frame(self) -> np.ndarray:
        """Getter to read the current frame from the motion detection
        where an trigger event occured

        Returns:
            np.ndarray: frame where a trigger event occured
        """
        return self._last_frame

    def _process(self):
        self._running = True
        logger.info("Start motion detection")

        while self._running:
            # throttle num requests to torchserve
            time.sleep(self._wait_time)
            self._last_frame = self._buffer.frame

            if not isinstance(self._last_frame, np.ndarray):
                self._buffer.recording = False
                logger.warning("invalid buffered frame")
                time.sleep(0.1)
                continue

            detections = self._get_detections(self._last_frame)
            if not detections:
                time.sleep(0.1)
                continue

            detected_classes = self._detection_triggers(detections)
            if len(detected_classes) > 0:
                self._last_match = time.time()
                logger.info(
                    "trigger for class %s detected at %d, already recording: %r",
                    detected_classes,
                    self._last_match,
                    self._buffer.recording,
                )
                if not self._buffer.recording:
                    self._recording_started = self._last_match
                    if self._cfg.trigger_save:
                        filename = datetime.now().strftime(self._cfg.trigger_path)
                        cv2.imwrite(filename, self._last_frame)

                self._buffer.recording = True

            if self._buffer.recording:
                match_delta = time.time() - self._last_match
                recording_delta = time.time() - self._recording_started
                logger.info(
                    "match_delta: %s, exceeded: %r, recording_delta: %s, exceeded: %r",
                    match_delta,
                    match_delta >= self._cfg.trigger_invalidate_duration,
                    recording_delta,
                    recording_delta >= self._cfg.max_recording_time,
                )
                if (
                    match_delta >= self._cfg.trigger_invalidate_duration
                    or recording_delta >= self._cfg.max_recording_time
                ):
                    self._buffer.recording = False
                    logger.info("stop recording")

        logger.info("exit detection thread")

    def _get_detections(self, frame):
        _, encoded = cv2.imencode(".jpg", frame)
        try:
            logger.debug("send request to TorchServe")
            res = requests.post(self._request_uri, files={"data": encoded}, timeout=10)
            logger.debug(
                "receive response from TorchServe with status code: %d", res.status_code
            )
            if res.status_code == 200:
                return res.json()

            logger.error("TorchServe error: %s", res.json())
        except requests.exceptions.RequestException as ex:
            logger.warning("TorchServe timeout during connecting %s", ex)

        return None

    def _detection_triggers(self, detections):
        logger.debug("Parse and process %d detections", len(detections))

        detected_classes = set()
        for detection in detections:
            (class_name, coords), (_, score) = detection.items()
            if (
                class_name not in self._cfg.trigger_classes
                or score < self._cfg.trigger_detection_threshold
            ):
                continue

            detected_classes.add(class_name)
            if self._debug or self._cfg.trigger_save:
                x1, y1, x2, y2 = [int(x) for x in coords]
                self._last_frame = self._visualizer.draw_detection(
                    self._last_frame, class_name, x1, y1, x2, y2
                )

        if self._debug or self._cfg.trigger_save:
            self._last_frame = self._visualizer.draw_legend(
                self._last_frame, detected_classes
            )

        return detected_classes
