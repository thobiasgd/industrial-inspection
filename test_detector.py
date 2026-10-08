from pathlib import Path

import cv2

from anomaly_detector import AnomalyDetector
from visualization import create_anomaly_visualization

# Localiza a raiz do projeto.
project_root = Path(__file__).resolve().parent

# Inicializa o detector.
detector = AnomalyDetector(project_root)


# Usa a contaminação que anteriormente era um falso negativo
# quando utilizávamos somente o maior patch.
image_path = (
    project_root
    / "dados"
    / "mvtec"
    / "bottle"
    / "test"
    / "contamination"
    / "000.png"
)

image_bgr = cv2.imread(
    str(image_path),
    cv2.IMREAD_COLOR,
)

if image_bgr is None:
    raise RuntimeError(
        f"Could not read image: {image_path}"
    )


# Executa a inspeção.
result = detector.inspect(image_bgr)


print(f"Score: {result['score']:.4f}")
print(f"Threshold: {result['threshold']:.4f}")
print("Decision:", result["decision"])
print("Anomaly map shape:", result["anomaly_map"].shape)

# Gera o heatmap e a sobreposição visual.
heatmap_bgr, overlay_bgr = create_anomaly_visualization(
    image_bgr,
    result["anomaly_map"],
)

# Monta uma visualização lado a lado.
preview = cv2.hconcat([
    image_bgr,
    heatmap_bgr,
    overlay_bgr,
])

window_name = (
    f"{result['decision']} | "
    f"Score: {result['score']:.4f} | "
    f"Threshold: {result['threshold']:.4f}"
)

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL,
)

cv2.resizeWindow(
    window_name,
    1200,
    400,
)

cv2.imshow(
    window_name,
    preview,
)

cv2.waitKey(0)
cv2.destroyAllWindows()