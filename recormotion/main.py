import time

import cv2
import numpy as np
from flask import Flask, Response, request

from recormotion.buffer import CaptureBuffer
from recormotion.config import Configuration
from recormotion.detector import RemoteMotionDetector

app = Flask(__name__)
cfg = Configuration().config


capture_buffer = CaptureBuffer()
detector = RemoteMotionDetector(capture_buffer)


@app.route("/")
def root():
    detector.debug = request.args.get("debug", False)

    return """
    <body style="background: black;">
        <img src="/mjpeg" />
    </body>
    """


def gather_img():
    while True:
        time.sleep(0.1)
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
def mjpeg():
    return Response(gather_img(), mimetype="multipart/x-mixed-replace; boundary=frame")


app.run(host="0.0.0.0", threaded=True)
