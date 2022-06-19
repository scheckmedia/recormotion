from datetime import datetime
from typing import Dict

import cv2
import numpy as np
from matplotlib.pyplot import get_cmap

from recormotion.config import Configuration

cm = get_cmap("rainbow")


class Visualizer:
    """Simple visualization object to provide drawing functionality

    :param labels: Label mapping of all possible detections key is the id and value the class name
    :type labels: Dict[int, str]
    """

    def __init__(self, labels: Dict[int, str]) -> None:
        self._labels = labels
        colors = list((cm(1.0 * i / 90)) for i in range(len(labels)))
        colors = (np.array(colors) * 255).astype(np.uint8)[..., :3]
        self._color_mapping = {
            l: tuple(c.tolist()) for l, c in zip(labels.values(), colors)
        }

    def draw_detection(
        self,
        img: np.ndarray,
        class_name: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        thickness: int = 2,
    ) -> np.ndarray:
        """Helper function to draw bounding boxes around for a detection into a frame

        :param img: image to draw the detection
        :type img: np.ndarray
        :param class_name: name of the class the detection belongs to
        :type class_name: str
        :param x1: top left x postion of the bounding box
        :type x1: int
        :param y1: top left y position of the bounding box
        :type y1: int
        :param x2: bottom right x position of the bounding box
        :type x2: int
        :param y2: bottom right y position of the bounding box
        :type y2: int
        :param thickness: thicknes for the bounding box line, defaults to 2
        :type thickness: int, optional
        :return: image containing the visualized bounding box
        :rtype: np.ndarray
        """
        color = self._color_mapping[class_name]
        return cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def draw_legend(self, img: np.ndarray, detected_classes: set) -> np.ndarray:
        """Function to draw a legend for a list of detections

        :param img: image the legend should be drawn
        :type img: np.ndarray
        :param detected_classes: classes that should be part of the legend
        :type detected_classes: set
        :return: image with the drawn legend
        :rtype: np.ndarray
        """
        padding = 5
        x1 = y1 = padding
        fnt = cv2.FONT_HERSHEY_DUPLEX
        fnt_scale = 1.0
        fnt_thickness = 1

        for class_name in detected_classes:
            color = self._color_mapping[class_name]
            (tw, th), _ = cv2.getTextSize(class_name, fnt, fnt_scale, fnt_thickness)
            w = tw + padding
            h = th + padding

            img = cv2.rectangle(
                img,
                (x1 + padding, y1),
                (x1 + w, y1 + h),
                color,
                -1,
            )

            img = cv2.putText(
                img,
                class_name,
                (x1 + padding, y1 + h // 2 + 2 * padding),
                fnt,
                fnt_scale,
                (255, 255, 255),
                fnt_thickness,
                cv2.LINE_AA,
            )

            x1 += w

        return img

    @staticmethod
    def draw_timecode(img: np.ndarray) -> np.ndarray:
        """Helper function to draw a timecode into a frame
        The timecode contains the source of the frame (device or file),
        the date in format 2022-06-19 and the time 11:32:50.003

        :param img: image to draw the timecode
        :type img: np.ndarray
        :return: image with the timecode drawn
        :rtype: np.ndarray
        """
        camera = Configuration().config.video.input.source
        camera = "camera {camera}" if isinstance(camera, int) else camera
        dt = datetime.now()
        date_string = dt.strftime("%Y-%m-%d")
        time_string = dt.strftime("%H:%M:%S.%f")[:-3]
        messages = [f"Source: {camera}", date_string, time_string]

        fnt = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fnt_scale = 0.4

        message_dims = np.zeros((len(messages), 2), dtype=np.int32)
        for i, msg in enumerate(messages):
            dim, _ = cv2.getTextSize(msg, fnt, fnt_scale, 1)
            message_dims[i] = dim

        opacity = 0.3
        padding = 4
        overlay_max_width = message_dims[:, 0].max() + padding
        overlay_max_height = message_dims[:, 1].sum() + len(message_dims) * padding
        overlayer_y = img.shape[0] - padding - overlay_max_height
        overlayer_x = img.shape[1] - padding - overlay_max_width

        patch = (
            img[
                overlayer_y : overlayer_y + overlay_max_height,
                overlayer_x : overlayer_x + overlay_max_width,
            ].astype(np.float32)
            * (1 - opacity)
            + 255.0 * opacity
        )

        img[
            overlayer_y : overlayer_y + overlay_max_height,
            overlayer_x : overlayer_x + overlay_max_width,
        ] = patch.astype(np.uint8)

        for row, msg in enumerate(messages):
            tw, th = message_dims[row]
            th += padding // 2
            img = cv2.putText(
                img,
                msg,
                (
                    overlayer_x + overlay_max_width - tw - padding,
                    overlayer_y + th + row * th,
                ),
                fnt,
                fnt_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return img
