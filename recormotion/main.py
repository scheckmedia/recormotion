import time
from typing import Iterator

import cv2
import numpy as np
from flask import Flask, Response, request

from recormotion.buffer import CaptureBuffer
from recormotion.config import Configuration
from recormotion.detector import RemoteMotionDetector
from recormotion.helper import setup_logger

setup_logger()
app = Flask(__name__)
cfg = Configuration().config


capture_buffer = CaptureBuffer()
detector = RemoteMotionDetector(capture_buffer)


@app.route("/")
def root():
    """Default route to show the current mjpeg stream"""
    detector.debug = request.args.get("debug", False)

    return """
    <body style="background: black;">
        <img src="/mjpeg" />
    </body>
    """


def gather_img() -> Iterator[bytes]:
    """Function to generate a mjpeg valid stream based on raw images

    :yield: iterator that holds a mjpeg frame
    :rtype: Iterator[bytes]
    """
    while True:
        time.sleep(1 / cfg.video.output.fps)
        # time.sleep(0.1)
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


@app.route("/mjpeg")
def mjpeg() -> Response:
    """entry point for mjpeg stream

    :return: http response with mjpeg stream
    :rtype: Response
    """
    return Response(gather_img(), mimetype="multipart/x-mixed-replace; boundary=frame")


app.run(host="0.0.0.0", threaded=True)
