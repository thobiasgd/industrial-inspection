import cv2
import numpy as np


def create_anomaly_visualization(
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
):

    image_height, image_width = image_bgr.shape[:2]

    resized_map = cv2.resize(
        anomaly_map,
        (image_width, image_height),
        interpolation=cv2.INTER_LINEAR,
    )

    display_map = cv2.normalize(
        resized_map,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    heatmap_bgr = cv2.applyColorMap(
        display_map,
        cv2.COLORMAP_INFERNO,
    )

    overlay_bgr = cv2.addWeighted(
        image_bgr,
        0.6,
        heatmap_bgr,
        0.4,
        0.0,
    )

    return heatmap_bgr, overlay_bgr

def create_bounding_box_visualization(
    image_bgr: np.ndarray,
    bounding_boxes: list[dict],
) -> np.ndarray:

    output_image = image_bgr.copy()

    for box in bounding_boxes:
        x = box["x"]
        y = box["y"]

        box_width = box["width"]
        box_height = box["height"]

        x2 = x + box_width
        y2 = y + box_height

        cv2.rectangle(
            output_image,
            (x, y),
            (x2, y2),
            (0, 0, 255),
            3,
        )

    return output_image