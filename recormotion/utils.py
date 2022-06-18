import cv2
import numpy as np
from matplotlib.pyplot import get_cmap

cm = get_cmap("rainbow")


class Visualizer:
    def __init__(self, labels) -> None:
        self._labels = labels
        colors = list((cm(1.0 * i / 90)) for i in range(len(labels)))
        colors = (np.array(colors) * 255).astype(np.uint8)[..., :3]
        self._color_mapping = {
            l: tuple(c.tolist()) for l, c in zip(labels.values(), colors)
        }

    def draw_detection(self, img, class_name, x1, y1, x2, y2, thickness=2):
        color = self._color_mapping[class_name]
        return cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def draw_legend(self, img: np.ndarray, detected_classes: set):
        padding = 5
        x1 = y1 = padding
        fnt = cv2.FONT_HERSHEY_SIMPLEX
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
