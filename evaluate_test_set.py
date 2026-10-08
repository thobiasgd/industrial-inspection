from pathlib import Path

import cv2
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


# Define os caminhos principais do projeto.
project_root = Path(__file__).resolve().parent

test_dir = project_root / "dados" / "mvtec" / "bottle" / "test"
bank_path = project_root / "artifacts" / "memory_bank.pt"
threshold_path = project_root / "artifacts" / "threshold.pt"


if not torch.cuda.is_available():
    raise RuntimeError("The GPU is not available to PyTorch.")

device = torch.device("cuda:0")


# Carrega o banco de características.
memory_bank = torch.load(
    bank_path,
    map_location=device,
    weights_only=True,
)


# Carrega o threshold calculado anteriormente.
threshold_data = torch.load(
    threshold_path,
    map_location="cpu",
    weights_only=True,
)

threshold = threshold_data["threshold"]


# Define o mesmo preprocessamento usado anteriormente.
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Carrega a ResNet18 e cria o extrator de características.
model = resnet18(
    weights=ResNet18_Weights.IMAGENET1K_V1
)

feature_extractor = create_feature_extractor(
    model,
    return_nodes={"layer2": "features"},
)

feature_extractor = feature_extractor.to(device).eval()


# Contadores da matriz de confusão.
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


# Lista para guardar resultados individuais.
results = []


categories = sorted(
    path for path in test_dir.iterdir()
    if path.is_dir()
)


with torch.inference_mode():

    for category_dir in categories:

        image_paths = sorted(
            category_dir.glob("*.png")
        )

        # A categoria "good" representa imagens normais.
        expected_defective = category_dir.name != "good"

        for image_path in image_paths:

            # Lê a imagem.
            image_bgr = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR,
            )

            if image_bgr is None:
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                )

            # Converte de BGR para RGB.
            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB,
            )

            # Prepara a imagem.
            input_batch = (
                preprocess(image_rgb)
                .unsqueeze(0)
                .to(device)
            )

            # Extrai as características.
            feature_map = feature_extractor(
                input_batch
            )["features"]

            channels = feature_map.shape[1]

            patch_features = (
                feature_map
                .permute(0, 2, 3, 1)
                .reshape(-1, channels)
            )

            # Calcula as distâncias.
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

            # Obtém a pontuação final da imagem.
            anomaly_score = (
                patch_scores
                .max()
                .item()
            )

            predicted_defective = (
                anomaly_score > threshold
            )


            # Atualiza a matriz de confusão.
            if expected_defective and predicted_defective:
                true_positive += 1

            elif not expected_defective and not predicted_defective:
                true_negative += 1

            elif not expected_defective and predicted_defective:
                false_positive += 1

            elif expected_defective and not predicted_defective:
                false_negative += 1


            results.append({
                "image": str(image_path),
                "category": category_dir.name,
                "score": anomaly_score,
                "expected_defective": expected_defective,
                "predicted_defective": predicted_defective,
            })

            decision = (
                "REJECTED"
                if predicted_defective
                else "APPROVED"
            )

            print(
                f"{category_dir.name}/{image_path.name}"
                f" | Score: {anomaly_score:.4f}"
                f" | {decision}"
            )


print("\nEvaluation summary:")

print("True positives:", true_positive)
print("True negatives:", true_negative)
print("False positives:", false_positive)
print("False negatives:", false_negative)

# Seleciona somente imagens defeituosas que foram aprovadas incorretamente.
false_negatives = [
    result
    for result in results
    if (
        result["expected_defective"]
        and not result["predicted_defective"]
    )
]

print("\nFalse negatives:")

for result in false_negatives:
    print(
        f'{result["category"]}/{Path(result["image"]).name}'
        f' | Score: {result["score"]:.4f}'
        f' | Threshold: {threshold:.4f}'
    )

# Agrupa os resultados por categoria.
categories_summary = {}

for result in results:
    category = result["category"]

    if category not in categories_summary:
        categories_summary[category] = {
            "total": 0,
            "correct": 0,
            "rejected": 0,
            "approved": 0,
        }

    categories_summary[category]["total"] += 1

    if result["predicted_defective"]:
        categories_summary[category]["rejected"] += 1
    else:
        categories_summary[category]["approved"] += 1

    expected_defective = result["expected_defective"]
    predicted_defective = result["predicted_defective"]

    if expected_defective == predicted_defective:
        categories_summary[category]["correct"] += 1


print("\nPerformance by category:")

for category, summary in categories_summary.items():
    accuracy = summary["correct"] / summary["total"] * 100

    print(
        f"{category}: "
        f"{summary['correct']}/{summary['total']} correct "
        f"({accuracy:.1f}%)"
    )