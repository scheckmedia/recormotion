import queue
import subprocess
import time
from datetime import datetime
from logging import getLogger
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import List

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
        self._moving_average_size = 25
        self._moving_average = []

        self._running = False
        self._cam = cv2.VideoCapture(self._cfg.source)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.resolution.width)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.resolution.height)
        self._cam.set(cv2.CAP_PROP_FPS, self._cfg.max_fps)

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
        self._lock_history = Lock()
        self._t = Thread(target=self._process)
        self._t.start()

    @property
    def frame(self) -> np.ndarray:
        """Getter for current captured frame

        Returns:
            np.ndarray: Current frame in the buffer
        """

        with self._lock:
            frame = self._buffer[self._buffer_pos]
            if isinstance(frame, np.ndarray):
                return frame.copy()

            return frame

    @property
    def output_fps(self) -> int:
        """Getter for output stream frame rate

        Returns:
            int: output stream frame rate
        """
        return self._write_buffer.fps

    @property
    def recording(self) -> bool:
        """Getter for current recording state

        Returns:
            bool: Recording state, if true the frames will be recorded
        """
        return self._recording

    @recording.setter
    def recording(self, value: bool):
        """Setter for recording state

        If recording is requested (true state) the history buffer
        is copied first to the write buffer otherwsie the recording
        stops

        Args:
            value (bool): recording state
        """
        self._recording = value

        if not value:
            self._write_buffer.stop()
            with self._lock_history:
                self._history_copied = False

    def _copy_buffer_history(self):
        history = []
        with self._lock:
            i = (self._buffer_pos + 1) % self._buffer_size
            while i != self._buffer_pos:
                if isinstance(self._buffer[i], np.ndarray):
                    history.append(self._buffer[i].copy())

                i = (i + 1) % self._buffer_size

        logger.debug("copy historic frames to write buffer done")
        with self._lock_history:
            self._history_copied = True
        return history

    def _process(self):
        self._running = True
        logger.info("Start frame acquesition")

        while self._running:
            tick = time.time()
            success, img = self._cam.read()

            if not success:
                logger.warning("read frame not successful")
                # invalidate last frame to signal end of stream
                with self._lock:
                    self._buffer[self._buffer_pos] = None
                time.sleep(1)
                continue

            if self._cfg.timecode:
                img = Visualizer.draw_timecode(img)

            with self._lock:
                self._buffer_pos = (self._buffer_pos + 1) % self._buffer_size
                self._buffer[self._buffer_pos] = img

            if self._recording:
                history = None
                if not self._history_copied:
                    history = self._copy_buffer_history()

                self._write_buffer.write(self.frame, history)

            self._moving_average.append(1 / (time.time() - tick))
            if len(self._moving_average) >= self._moving_average_size:
                avg = np.mean(self._moving_average).astype(np.uint32)
                self._write_buffer.fps = avg
                self._moving_average = []

                logger.debug(f"Moving Average FPS: {avg}")

            delta = self._max_frame_time - (time.time() - tick)

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
        self._history = []
        self._fps = self._cfg.fps
        self._active_thread = None

    @property
    def fps(self) -> int:
        """Getter of the current frame rate for the output stream

        Returns:
            int: output stream frame rate
        """
        return self._fps

    @fps.setter
    def fps(self, value: int):
        """Setter to update the FPS for output stream

        Args:
            value (int): frame rate for the output stream
        """
        if value <= 0 or self._cfg.fps > 0:
            return

        self._fps = value

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
            "-framerate", f"{self._fps if self._fps > 0 else 25}",
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

    def _prepare_and_write_frame(self, frame: np.ndarray):
        frame = cv2.resize(
            frame, (self._cfg.resolution.width, self._cfg.resolution.height)
        )
        self._pipe_out.write(frame.tobytes())

    def _process(self, history: List[np.ndarray] = None):
        cmd = self._build_cmd()
        logger.info("start recording into file %s", self._out_file)

        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as p:
            self._pipe_out = p.stdin

            if history:
                tick = time.time()
                for frame in history:
                    self._prepare_and_write_frame(frame)
                    logger.debug("write history frame")

                logger.info(
                    f"history written to output stream with { 1 / (time.time() - tick)} FPS"
                )

            while True:
                try:
                    frame = self._queue.get_nowait()
                    tick = time.time()
                    self._prepare_and_write_frame(frame)
                    logger.debug(f"write frame with { 1 / (time.time() - tick)} FPS")
                except queue.Empty:
                    if self._recording:
                        time.sleep(0.1)
                    else:
                        logger.info("recording queue is empty")
                        break

    def write(self, frame: np.ndarray, history: List[np.ndarray] = None):
        """Writes a frame into the active video stream
        If the stream is not opened yet, this will be
        done first

        Args:
            frame (np.ndarray): frame / image to write into the video stream
            history (List[np.ndarray]): list of historic frames, to be written before frame
        """

        if not isinstance(frame, np.ndarray):
            logger.error("attempt to write invalid frame of type %s", type(frame))
            return

        self._queue.put(frame)

        if not self._recording and self._active_thread is None:
            self._recording = True
            self._active_thread = Thread(target=self._process, args=(history,))
            self._active_thread.start()

    def stop(self):
        """Method to stop an active recording"""
        if not self._recording:
            return

        self._recording = False
        if self._active_thread is not None:
            self._active_thread.join()
            self._active_thread = None

        logger.info("stop recording")
