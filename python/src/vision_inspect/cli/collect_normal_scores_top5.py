from vision_inspect.config import PROJECT_ROOT

import cv2
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


def main() -> None:
    project_root = PROJECT_ROOT

    train_dir = (
        project_root
        / "dados"
        / "mvtec"
        / "bottle"
        / "train"
        / "good"
    )

    bank_path = project_root / "artifacts" / "memory_bank.pt"
    output_path = project_root / "artifacts" / "normal_scores_top5.pt"


    if not torch.cuda.is_available():
        raise RuntimeError("The GPU is not available to PyTorch.")

    device = torch.device("cuda:0")


    memory_bank = torch.load(
        bank_path,
        map_location=device,
        weights_only=True,
    )


    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


    model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )

    feature_extractor = create_feature_extractor(
        model,
        return_nodes={"layer2": "features"},
    )

    feature_extractor = feature_extractor.to(device).eval()


    reference_image_count = 20

    all_image_paths = sorted(train_dir.glob("*.png"))

    calibration_paths = all_image_paths[reference_image_count:]

    if not calibration_paths:
        raise RuntimeError("No calibration images were found.")


    normal_scores: list[float] = []


    with torch.inference_mode():

        for index, image_path in enumerate(calibration_paths, start=1):

            image_bgr = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR,
            )

            if image_bgr is None:
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                )

            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB,
            )

            input_batch = (
                preprocess(image_rgb)
                .unsqueeze(0)
                .to(device)
            )

            feature_map = feature_extractor(
                input_batch
            )["features"]

            channels = feature_map.shape[1]

            patch_features = (
                feature_map
                .permute(0, 2, 3, 1)
                .reshape(-1, channels)
            )

            distances = torch.cdist(
                patch_features,
                memory_bank,
                p=2,
            )

            patch_scores = (
                distances
                .min(dim=1)
                .values
            )

            sorted_scores = torch.sort(
                patch_scores,
                descending=True,
            ).values

            top_5_count = max(
                1,
                int(len(sorted_scores) * 0.05),
            )

            top_5_score = (
                sorted_scores[:top_5_count]
                .mean()
                .item()
            )

            normal_scores.append(top_5_score)

            print(
                f"Processed: {index}/{len(calibration_paths)} "
                f"- {image_path.name} "
                f"- Top 5%: {top_5_score:.4f}"
            )


    score_tensor = torch.tensor(
        normal_scores,
        dtype=torch.float32,
    )


    torch.save(
        {
            "scores": score_tensor,
            "image_names": [
                path.name
                for path in calibration_paths
            ],
        },
        output_path,
    )


    print("\nCalibration summary:")
    print("Images processed:", len(calibration_paths))
    print(f"Minimum score: {score_tensor.min().item():.4f}")
    print(f"Mean score: {score_tensor.mean().item():.4f}")
    print(f"Maximum score: {score_tensor.max().item():.4f}")
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
