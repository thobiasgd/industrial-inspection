import cv2
import numpy as np


def create_anomaly_visualization(
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
):

    # Obtém as dimensões da imagem original.
    image_height, image_width = image_bgr.shape[:2]

    # Amplia o mapa de 32x32 para o tamanho da imagem original.
    resized_map = cv2.resize(
        anomaly_map,
        (image_width, image_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Converte os scores para uma escala de 0 a 255
    # usada apenas para visualização.
    display_map = cv2.normalize(
        resized_map,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    # Converte os valores em um mapa de cores.
    heatmap_bgr = cv2.applyColorMap(
        display_map,
        cv2.COLORMAP_INFERNO,
    )

    # Sobrepõe o heatmap à imagem original.
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

    # Cria uma cópia para não modificar a imagem original.
    output_image = image_bgr.copy()

    for box in bounding_boxes:
        x = box["x"]
        y = box["y"]

        box_width = box["width"]
        box_height = box["height"]

        x2 = x + box_width
        y2 = y + box_height

        # Desenha a caixa da região suspeita.
        cv2.rectangle(
            output_image,
            (x, y),
            (x2, y2),
            (0, 0, 255),
            3,
        )

    return output_image