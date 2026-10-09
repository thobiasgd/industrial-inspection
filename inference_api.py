import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from anomaly_detector import AnomalyDetector
from visualization import create_anomaly_visualization


class BoundingBoxResponse(BaseModel):
    x: int
    y: int
    width: int
    height: int


class InspectionResponse(BaseModel):
    score: float
    threshold: float
    decision: str

    image_width: int
    image_height: int

    bounding_boxes: list[BoundingBoxResponse]

    heatmap_base64: str
    overlay_base64: str


def encode_image_to_base64(
    image_bgr: np.ndarray,
) -> str:
    # Codifica a imagem OpenCV como PNG em memória.
    success, encoded_image = cv2.imencode(
        ".png",
        image_bgr,
    )

    if not success:
        raise RuntimeError(
            "Could not encode image to PNG."
        )

    # Converte os bytes do PNG para uma string Base64.
    return base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")


project_root = Path(__file__).resolve().parent


app = FastAPI(
    title="Inspection Inference API",
    version="1.0.0",
)


# Inicializa o modelo apenas uma vez.
detector = AnomalyDetector(
    project_root
)


@app.get("/health")
def health():
    return {
        "status": "ready",
        "device": str(detector.device),
    }


@app.post(
    "/inspect",
    response_model=InspectionResponse,
)
async def inspect_image(
    image: UploadFile = File(...),
) -> InspectionResponse:

    # Lê a imagem enviada pela API.
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Image is empty.",
        )

    # Converte os bytes para uma matriz NumPy.
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    # Decodifica a imagem para o formato BGR do OpenCV.
    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image.",
        )

    # Executa a inferência.
    result = detector.inspect(
        image_bgr
    )

    # Cria o heatmap e o overlay.
    heatmap_bgr, overlay_bgr = (
        create_anomaly_visualization(
            image_bgr,
            result["anomaly_map"],
        )
    )

    # Obtém as dimensões da imagem original.
    image_height, image_width = (
        image_bgr.shape[:2]
    )

    return InspectionResponse(
        score=result["score"],
        threshold=result["threshold"],
        decision=result["decision"],

        image_width=image_width,
        image_height=image_height,

        bounding_boxes=result[
            "bounding_boxes"
        ],

        heatmap_base64=(
            encode_image_to_base64(
                heatmap_bgr
            )
        ),

        overlay_base64=(
            encode_image_to_base64(
                overlay_bgr
            )
        ),
    )