from vision_inspect.config import PROJECT_ROOT

import cv2
import torch
from torchvision import transforms

from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


def main() -> None:
    project_root = PROJECT_ROOT
    image_path = project_root / "dados" / "mvtec" / "bottle" / "train" / "good" / "000.png"

    if not torch.cuda.is_available():
        raise RuntimeError("The GPU is not available to PyTorch.")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"Could not read the image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image_tensor = preprocess(image_rgb)

    input_batch = image_tensor.unsqueeze(0).to("cuda:0")

    print("Tensor shape:", image_tensor.shape)
    print("Input shape:", input_batch.shape)
    print("Data type:", input_batch.dtype)
    print("Device:", input_batch.device)
    print("Image prepared successfully!")

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    feature_extractor = create_feature_extractor(
        model,
        return_nodes={"layer2": "features"}
    )

    feature_extractor = feature_extractor.to(input_batch.device)

    feature_extractor.eval()

    with torch.inference_mode():
        outputs = feature_extractor(input_batch)
        feature_map = outputs["features"]

    print("Feature map shape:", feature_map.shape)
    print("Feature map device:", feature_map.device)
    print("Feature extraction completed successfully!")

    batch_size, channels, height, width = feature_map.shape

    features_by_position = feature_map.permute(0, 2, 3, 1)

    patch_features = features_by_position.reshape(-1, channels)

    print("Reordered shape:", features_by_position.shape)
    print("Patch features shape:", patch_features.shape)
    print("First patch shape:", patch_features[0].shape)
    print("Patch features device:", patch_features.device)

    train_dir = project_root / "dados" / "mvtec" / "bottle" / "train" / "good"
    max_images = 20
    image_paths = sorted(train_dir.glob("*.png"))[:max_images]

    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in: {train_dir}")

    all_features: list[torch.Tensor] = []

    feature_extractor.eval()

    with torch.inference_mode():
        for index, reference_path in enumerate(image_paths, start=1):
            image_bgr = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)

            if image_bgr is None:
                raise RuntimeError(f"Could not read the image: {reference_path}")

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            reference_batch = preprocess(image_rgb).unsqueeze(0)
            reference_batch = reference_batch.to(input_batch.device)

            reference_map = feature_extractor(reference_batch)["features"]

            reference_patches = reference_map.permute(0, 2, 3, 1).reshape(
                -1, reference_map.shape[1]
            )

            all_features.append(reference_patches.cpu())

            print(f"Processed: {index}/{len(image_paths)} - {reference_path.name}")

    memory_bank = torch.cat(all_features, dim=0)

    output_dir = project_root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "memory_bank.pt"
    torch.save(memory_bank, output_path)

    print("Images processed:", len(image_paths))
    print("Memory bank shape:", memory_bank.shape)
    print("Memory bank device:", memory_bank.device)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
