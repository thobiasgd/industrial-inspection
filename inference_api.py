from pathlib import Path

import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from visualization import create_anomaly_visualization
from anomaly_detector import AnomalyDetector

def encode_image_to_base64(image_bgr):
    success, encoded_image = cv2.imencode(".png", image_bgr)

    if not success:
        raise RuntimeError("Could not encode image to PNG.")

    return base64.b64encode(encoded_image.tobytes()).decode("utf-8")


class InspectionResponse(BaseModel):
    score: float
    threshold: float
    decision: str
    heatmap_base64: str
    overlay_base64: str


project_root = Path(__file__).resolve().parent

app = FastAPI(
    title="Inspection Inference API",
    version="1.0.0",
)

# Carrega ResNet, memory bank e threshold apenas uma vez.
detector = AnomalyDetector(project_root)


@app.get("/health")
def health():
    return {
        "status": "ready",
        "device": str(detector.device),
    }


@app.post("/inspect", response_model=InspectionResponse)
async def inspect_image(
    image: UploadFile = File(...),
) -> InspectionResponse:

    # Lê os bytes enviados pelo cliente.
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

    # Decodifica a imagem no formato BGR usado pelo OpenCV.
    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image.",
        )

    # Executa a inferência real na GPU.
    result = detector.inspect(image_bgr)

    heatmap_bgr, overlay_bgr = create_anomaly_visualization(
        image_bgr,
        result["anomaly_map"],
    )

    heatmap_base64 = encode_image_to_base64(heatmap_bgr)
    overlay_base64 = encode_image_to_base64(overlay_bgr)

    return InspectionResponse(
        score=result["score"],
        threshold=result["threshold"],
        decision=result["decision"],
        heatmap_base64=heatmap_base64,
        overlay_base64=overlay_base64,
    )