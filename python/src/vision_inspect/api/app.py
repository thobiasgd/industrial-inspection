import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from vision_inspect.api.schemas import InspectionResponse
from vision_inspect.detector import AnomalyDetector
from vision_inspect.visualization import create_anomaly_visualization


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = AnomalyDetector()
    try:
        yield
    finally:
        del app.state.detector


def encode_image_to_base64(
    image_bgr: np.ndarray,
) -> str:
    success, encoded_image = cv2.imencode(
        ".png",
        image_bgr,
    )

    if not success:
        raise RuntimeError(
            "Could not encode image to PNG."
        )

    return base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")


app = FastAPI(
    title="Inspection Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ready",
        "device": str(request.app.state.detector.device),
    }


@app.post(
    "/inspect",
    response_model=InspectionResponse,
)
async def inspect_image(
    request: Request,
    image: UploadFile = File(...),
) -> InspectionResponse:

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Image is empty.",
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image.",
        )

    result = request.app.state.detector.inspect(
        image_bgr
    )

    heatmap_bgr, overlay_bgr = (
        create_anomaly_visualization(
            image_bgr,
            result["anomaly_map"],
        )
    )

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
