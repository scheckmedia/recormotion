import subprocess
import time
from datetime import datetime
from logging import getLogger
from pathlib import Path
from threading import Lock, Thread

import cv2
import numpy as np

from recormotion.config import Configuration

logger = getLogger(__file__)


class CaptureBuffer:
    def __init__(self):
        self._cfg = Configuration().config.video.input

        self._buffer_size = self._cfg.buffer_size
        self._buffer_pos = 0
        self._buffer = [None] * self._buffer_size

        self._running = False
        self._cam = cv2.VideoCapture(self._cfg.source)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.resolution.width)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.resolution.height)

        logger.info(
            "Initialize video capturing at source %s for resolution %dx%d",
            self._cfg.source,
            self._cfg.resolution.width,
            self._cfg.resolution.height,
        )

        self._wait_time = 1 / self._cfg.max_fps
        self._write_buffer = FfmpegWriteBuffer()
        self._recording = False

        self._lock = Lock()
        self._t = Thread(target=self._process)
        self._t.start()

    @property
    def frame(self):
        with self._lock:
            return self._buffer[self._buffer_pos].copy()

    @property
    def recording(self):
        return self._recording

    @recording.setter
    def recording(self, value: bool):
        if not value:
            self._write_buffer.stop()

        self._recording = value

    def _process(self):
        self._running = True
        while self._running:
            time.sleep(self._wait_time)
            success, img = self._cam.read()

            logger.debug("capture frame success: %r", success)

            if not success:
                logger.warning("read frame not successful")
                time.sleep(0.1)
                continue

            with self._lock:
                self._buffer_pos = (self._buffer_pos + 1) % self._buffer_size
                self._buffer[self._buffer_pos] = img
                logger.debug("update buffer at position %d", self._buffer_pos)

            if self._recording:
                self._write_buffer.write(self.frame)


class FfmpegWriteBuffer:
    def __init__(self) -> None:
        self._cfg = Configuration().config.video.output
        self._recording = False
        self._pipe_out = None
        self._out_file = ""

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

    def _init_pipe(self):
        cmd = self._build_cmd()
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # pylint: disable=R1732
        self._pipe_out = p.stdin
        self._recording = True

    def write(self, frame: np.ndarray):
        if not self._recording:
            self._init_pipe()
            logger.info("start recording into file %s", self._out_file)

        frame = cv2.resize(
            frame, (self._cfg.resolution.width, self._cfg.resolution.height)
        )
        self._pipe_out.write(frame.tobytes())
        logger.debug("write frame")

    def stop(self):
        if self._recording:
            self._pipe_out.flush()
            self._pipe_out.close()
            self._recording = False

            logger.debug("stop recording")
