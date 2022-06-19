import subprocess
import time
from datetime import datetime
from logging import getLogger
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

import cv2
import numpy as np

from recormotion.config import Configuration
from recormotion.utils import Visualizer

logger = getLogger(__file__)


class CaptureBuffer:
    """Object that captures frames from a camera or source stream and
    buffers various frames to have a history. As soon as a write event is
    triggered, the history will be written first into the target stream.
    """

    def __init__(self):
        self._cfg = Configuration().config.video.input

        self._buffer_size = self._cfg.buffer_size
        self._buffer_pos = 0
        self._buffer = [None] * self._buffer_size

        self._running = False
        self._cam = cv2.VideoCapture(self._cfg.source)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.resolution.width)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.resolution.height)

        if self._cfg.fourcc:
            fourcc = cv2.VideoWriter_fourcc(*self._cfg.fourcc)
            self._cam.set(cv2.CAP_PROP_FOURCC, fourcc)

        logger.info(
            "Initialize video capturing at source %s for resolution %dx%d",
            self._cfg.source,
            self._cfg.resolution.width,
            self._cfg.resolution.height,
        )

        self._max_frame_time = 1 / self._cfg.max_fps
        self._write_buffer = FfmpegWrite()
        self._recording = False
        self._history_copied = False

        self._lock = Lock()
        self._t = Thread(target=self._process)
        self._t.start()

    @property
    def frame(self) -> np.ndarray:
        """Getter for current captured frame

        :return: Current frame in the buffer
        :rtype: np.ndarray
        """

        with self._lock:
            frame = self._buffer[self._buffer_pos]
            if isinstance(frame, np.ndarray):
                return frame.copy()

            return frame

    @property
    def recording(self) -> bool:
        """Getter for current recording state

        :return: Recording state, if true the frames will be recorded
        :rtype: bool
        """
        return self._recording

    @recording.setter
    def recording(self, value: bool):
        """Setter for recording state

        If recording is requested (true state) the history buffer
        is copied first to the write buffer otherwsie the recording
        stops

        :param value: recording state
        :type value: bool
        """
        if not value:
            self._write_buffer.stop()
            self._history_copied = False
        elif value and not self._history_copied:
            self._copy_buffer_history()

        self._recording = value

    def _copy_buffer_history(self):
        with self._lock:
            i = (self._buffer_pos + 1) % self._buffer_size
            while i != self._buffer_pos:
                if isinstance(self._buffer[i], np.ndarray):
                    self._write_buffer.write(self._buffer[i].copy())

                i = (i + 1) % self._buffer_size

            logger.debug("copy historic frames to write buffer done")
            self._history_copied = True

    def _process(self):
        self._running = True
        while self._running:
            tick = time.time()
            success, img = self._cam.read()

            logger.debug("capture frame success: %r", success)

            if not success:
                logger.warning("read frame not successful")
                # invalidate last frame to signal end of stream
                with self._lock:
                    self._buffer[self._buffer_pos] = None
                time.sleep(1)
                continue

            logger.debug(f"FPS after reading: {1 / (time.time() - tick)}")
            if self._cfg.timecode:
                img = Visualizer.draw_timecode(img)
                logger.debug(f"FPS after timecode: {1 / (time.time() - tick)}")

            with self._lock:
                self._buffer_pos = (self._buffer_pos + 1) % self._buffer_size
                self._buffer[self._buffer_pos] = img
                logger.debug("update buffer at position %d", self._buffer_pos)

            if self._recording:
                self._write_buffer.write(self.frame)
                logger.debug(f"FPS after write: {1 / (time.time() - tick)}")

            delta = self._max_frame_time - (time.time() - tick)
            logger.debug(f"FPS delta: {1/delta}")

            if delta > 0:
                logger.debug(f"sleep for {delta} seconds")
                time.sleep(delta)


class FfmpegWrite:
    """Object that writes a video file using ffmpegs PIPEs"""

    def __init__(self) -> None:
        self._cfg = Configuration().config.video.output
        self._recording = False
        self._pipe_out = None
        self._out_file = ""
        self._queue = Queue(1000)

    def _build_cmd(self):
        filename = datetime.now().strftime(self._cfg.filename)
        out_folder = Path(self._cfg.folder)
        out_folder.mkdir(parents=True, exist_ok=True)
        self._out_file = out_folder / filename

        # fmt: off
        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-framerate", f"{self._cfg.fps}",
            "-video_size", f"{self._cfg.resolution.width}x{self._cfg.resolution.height}",
            "-pixel_format", "bgr24",
            "-i", "pipe:0",  # The input comes from a pipe
            "-an",  # Tells FFMPEG not to expect any audio
            "-pix_fmt", "yuv420p",
            "-c:v", f"{self._cfg.encoder}",
        ]
        # fmt: on

        if self._cfg.ffmpeg_args and len(self._cfg.ffmpeg_args):
            cmd += self._cfg.ffmpeg_args

        cmd += [f"{self._out_file}"]

        logger.debug("build video writer command: %s", " ".join(cmd))
        return cmd

    def _process(self):
        logger.info("start recording into file %s", self._out_file)
        cmd = self._build_cmd()
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as p:
            self._pipe_out = p.stdin
            while self._recording:
                if self._queue.empty():
                    time.sleep(0.1)
                    continue

                frame = self._queue.get(timeout=0.1)
                frame = cv2.resize(
                    frame, (self._cfg.resolution.width, self._cfg.resolution.height)
                )
                self._pipe_out.write(frame.tobytes())
                logger.debug("write frame")
        with self._queue.mutex:
            self._queue.queue.clear()
        logger.info("stop recording")

    def write(self, frame: np.ndarray):
        """Writes a frame into the active video stream
        If the stream is not opened yet, this will be
        done first

        :param frame: frame / image to write into the video stream
        :type frame: np.ndarray
        """
        if not self._recording:
            self._recording = True
            Thread(target=self._process).start()

        self._queue.put(frame)

    def stop(self):
        """Method to stop an active recording"""
        self._recording = False
