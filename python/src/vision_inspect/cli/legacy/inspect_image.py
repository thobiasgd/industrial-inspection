from vision_inspect.config import PROJECT_ROOT

import cv2
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


def main() -> None:
    project_root = PROJECT_ROOT
    bank_path = project_root / "artifacts" / "memory_bank.pt"
    image_path = (
        project_root / "dados" / "mvtec" / "bottle"
        / "test" / "good" / "000.png"
    )

    threshold_path = project_root / "artifacts" / "threshold.pt"

    if not threshold_path.is_file():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("The GPU is not available to PyTorch.")

    if not bank_path.is_file():
        raise FileNotFoundError(f"Memory bank not found: {bank_path}")

    device = torch.device("cuda:0")

    memory_bank = torch.load(
        bank_path,
        map_location=device,
        weights_only=True,
    )

    threshold_data = torch.load(
        threshold_path,
        map_location="cpu",
        weights_only=True,
    )

    threshold = threshold_data["threshold"]

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    feature_extractor = create_feature_extractor(
        model,
        return_nodes={"layer2": "features"},
    )
    feature_extractor = feature_extractor.to(device).eval()

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"Could not read the image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = preprocess(image_rgb).unsqueeze(0).to(device)

    with torch.inference_mode():
        feature_map = feature_extractor(input_batch)["features"]
        _, channels, height, width = feature_map.shape

        test_features = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)

        if (
            memory_bank.ndim != 2
            or memory_bank.shape[0] == 0
            or memory_bank.shape[1] != channels
        ):
            raise ValueError("The memory bank has an incompatible shape.")

        distances = torch.cdist(test_features, memory_bank, p=2)

        patch_scores = distances.min(dim=1).values

        sorted_scores = torch.sort(
            patch_scores,
            descending=True,
        ).values

        top_1_count = max(1, int(len(sorted_scores) * 0.01))
        top_5_count = max(1, int(len(sorted_scores) * 0.05))
        top_10_count = max(1, int(len(sorted_scores) * 0.10))

        max_score = sorted_scores[0].item()

        top_1_mean = (
            sorted_scores[:top_1_count]
            .mean()
            .item()
        )

        top_5_mean = (
            sorted_scores[:top_5_count]
            .mean()
            .item()
        )

        top_10_mean = (
            sorted_scores[:top_10_count]
            .mean()
            .item()
        )

        mean_score = patch_scores.mean().item()

        print("\nPatch score statistics:")
        print(f"Maximum: {max_score:.4f}")
        print(f"Top 1% mean: {top_1_mean:.4f}")
        print(f"Top 5% mean: {top_5_mean:.4f}")
        print(f"Top 10% mean: {top_10_mean:.4f}")
        print(f"Global mean: {mean_score:.4f}")

        anomaly_map = patch_scores.reshape(height, width)

        anomaly_score = patch_scores.max().item()

    if anomaly_score > threshold:
        decision = "REJECTED"
    else:
        decision = "APPROVED"

    print("Distances shape:", distances.shape)
    print("Patch scores shape:", patch_scores.shape)
    print("Anomaly map shape:", anomaly_map.shape)
    print(f"Anomaly score: {anomaly_score:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("Decision:", decision)

    anomaly_array = anomaly_map.cpu().numpy()

    image_height, image_width = image_bgr.shape[:2]

    resized_map = cv2.resize(
        anomaly_array,
        (image_width, image_height),
        interpolation=cv2.INTER_LINEAR
    )

    display_map = cv2.normalize(
        resized_map,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

    heatmap_bgr = cv2.applyColorMap(display_map, cv2.COLORMAP_INFERNO)

    overlay_bgr = cv2.addWeighted(
        image_bgr, 0.6,
        heatmap_bgr, 0.4,
        0.0
    )

    print("Inspected image:", image_path)

    mask_path = (
        project_root / "dados" / "mvtec" / "bottle"
        / "ground_truth" / "contamination" / "000_mask.png"
    )

    ground_truth = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if ground_truth is None:
        raise RuntimeError(
            f"Could not read ground-truth mask: {mask_path}"
        )

    ground_truth_bgr = cv2.cvtColor(
        ground_truth,
        cv2.COLOR_GRAY2BGR,
    )

    preview = cv2.hconcat([
        image_bgr,
        ground_truth_bgr,
        heatmap_bgr,
        overlay_bgr,
    ])

    window_name = (
        f"{decision} | "
        f"Score: {anomaly_score:.4f} | "
        f"Threshold: {threshold:.4f}"
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 400)
    cv2.imshow(window_name, preview)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
