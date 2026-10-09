from vision_inspect.config import PROJECT_ROOT

import cv2

from vision_inspect.detector import AnomalyDetector
from vision_inspect.visualization import (
    create_anomaly_visualization,
    create_bounding_box_visualization,
)


def main() -> None:
    project_root = PROJECT_ROOT

    detector = AnomalyDetector(project_root)


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


    result = detector.inspect(image_bgr)


    print(f"Score: {result['score']:.4f}")
    print(f"Threshold: {result['threshold']:.4f}")
    print("Decision:", result["decision"])
    print("Anomaly map shape:", result["anomaly_map"].shape)

    print(
        "Bounding boxes:",
        result["bounding_boxes"],
    )

    heatmap_bgr, overlay_bgr = create_anomaly_visualization(
        image_bgr,
        result["anomaly_map"],
    )

    boxes_bgr = create_bounding_box_visualization(
        image_bgr,
        result["bounding_boxes"],
    )

    preview = cv2.hconcat([
        image_bgr,
        heatmap_bgr,
        overlay_bgr,
        boxes_bgr,
    ])

    window_name = (
        f"{result['decision']} | "
        f"Score: {result['score']:.4f} | "
        "Original | Heatmap | Overlay | Bounding boxes"
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


if __name__ == "__main__":
    main()
