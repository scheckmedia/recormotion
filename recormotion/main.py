import time
from typing import Iterator

import cv2
import numpy as np
from flask import Flask, Response, request

from recormotion.config import Configuration
from recormotion.detector import RemoteMotionDetector
from recormotion.helper import setup_logger
from recormotion.io import CaptureBuffer

setup_logger()
app = Flask(__name__)
cfg = Configuration().config


capture_buffer = CaptureBuffer()
detector = RemoteMotionDetector(capture_buffer)


@app.route("/stream")
def root():
    """Default route to show the current mjpeg stream"""
    detector.debug = request.args.get("debug", False)

    return """
    <body style="background: black;">
        <img src="/" />
    </body>
    """


def gather_img() -> Iterator[bytes]:
    """Function to generate a mjpeg valid stream based on raw images

    Yields:
        Iterator[bytes]: iterator that holds a mjpeg frame
    """
    while True:
        time.sleep(1 / cfg.video.output.fps)

        if detector.debug:
            img = detector.frame
        else:
            img = capture_buffer.frame

        if not isinstance(img, np.ndarray):
            continue

        if detector.debug:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, frame = cv2.imencode(".jpg", img)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n"
        )


@app.route("/")
def mjpeg() -> Response:
    """entry point for mjpeg stream

    Returns:
        Response: http response with mjpeg stream
    """
    return Response(gather_img(), mimetype="multipart/x-mixed-replace; boundary=frame")


app.run(**cfg.http)
