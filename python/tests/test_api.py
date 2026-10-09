import base64
from unittest.mock import Mock

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from vision_inspect.api import app as api


@pytest.fixture
def client(monkeypatch):
    detector = Mock(device="cuda:0")
    detector.inspect.return_value = {
        "score": 2.7,
        "threshold": 2.2,
        "decision": "REJECTED",
        "anomaly_map": np.arange(16, dtype=np.float32).reshape(4, 4),
        "bounding_boxes": [{"x": 1, "y": 2, "width": 3, "height": 4}],
    }
    factory = Mock(return_value=detector)
    monkeypatch.setattr(api, "AnomalyDetector", factory)

    with TestClient(api.app) as http:
        factory.assert_called_once_with()
        yield http, detector
        factory.assert_called_once_with()

    assert not hasattr(api.app.state, "detector")


def test_health(client):
    http, detector = client
    response = http.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "device": "cuda:0"}
    detector.inspect.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "detail"),
    [(b"", "Image is empty."), (b"not-an-image", "Invalid image.")],
)
def test_invalid_upload_is_rejected(client, payload, detail):
    http, detector = client
    response = http.post("/inspect", files={"image": ("image.png", payload, "image/png")})
    assert response.status_code == 400
    assert response.json() == {"detail": detail}
    detector.inspect.assert_not_called()


def test_inspection_preserves_http_contract_and_image_dimensions(client):
    http, detector = client
    source = np.zeros((12, 20, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", source)
    assert success

    response = http.post("/inspect", files={"image": ("image.png", encoded.tobytes(), "image/png")})

    assert response.status_code == 200
    result = response.json()
    assert set(result) == {
        "score", "threshold", "decision", "image_width", "image_height",
        "bounding_boxes", "heatmap_base64", "overlay_base64",
    }
    assert result["score"] == 2.7
    assert result["threshold"] == 2.2
    assert result["decision"] == "REJECTED"
    assert result["image_width"] == 20
    assert result["image_height"] == 12
    assert result["bounding_boxes"] == [{"x": 1, "y": 2, "width": 3, "height": 4}]
    detector.inspect.assert_called_once()
    np.testing.assert_array_equal(detector.inspect.call_args.args[0], source)

    for field in ("heatmap_base64", "overlay_base64"):
        image = cv2.imdecode(np.frombuffer(base64.b64decode(result[field]), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == source.shape
