from pydantic import BaseModel


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
