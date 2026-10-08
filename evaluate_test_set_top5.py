from pathlib import Path
import time

import cv2
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


# Define os caminhos principais do projeto.
project_root = Path(__file__).resolve().parent

test_dir = project_root / "dados" / "mvtec" / "bottle" / "test"
bank_path = project_root / "artifacts" / "memory_bank.pt"
threshold_path = project_root / "artifacts" / "threshold_top5.pt"

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

# Guarda o tempo de processamento de cada imagem.
processing_times = []

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

            # Garante que operações anteriores da GPU terminaram antes de iniciar o relógio.
            torch.cuda.synchronize()

            start_time = time.perf_counter()

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

            # Ordena os patches do mais anômalo para o menos anômalo.
            sorted_scores = torch.sort(
                patch_scores,
                descending=True,
            ).values

            # Calcula quantos patches correspondem a 5% do mapa.
            top_5_count = max(
                1,
                int(len(sorted_scores) * 0.05),
            )

            # Usa a média dos 5% patches mais anômalos
            # como pontuação final da imagem.
            anomaly_score = (
                sorted_scores[:top_5_count]
                .mean()
                .item()
            )

            predicted_defective = (
                anomaly_score > threshold
            )

            # Aguarda a GPU concluir todo o processamento desta imagem.
            torch.cuda.synchronize()

            end_time = time.perf_counter()

            processing_time_ms = (
                end_time - start_time
            ) * 1000

            processing_times.append(processing_time_ms)


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

# Calcula a quantidade total de imagens avaliadas.
total = (
    true_positive
    + true_negative
    + false_positive
    + false_negative
)

# Percentual total de classificações corretas.
accuracy = (
    true_positive + true_negative
) / total

# Percentual dos defeitos que conseguimos detectar.
recall = (
    true_positive
    / (true_positive + false_negative)
)

# Entre tudo que rejeitamos, quanto realmente era defeituoso.
precision = (
    true_positive
    / (true_positive + false_positive)
)

# Percentual das peças boas corretamente aprovadas.
specificity = (
    true_negative
    / (true_negative + false_positive)
)

# Percentual de peças boas rejeitadas incorretamente.
false_positive_rate = (
    false_positive
    / (false_positive + true_negative)
)

# Combina precision e recall em uma única métrica.
f1_score = (
    2 * precision * recall
    / (precision + recall)
)


print("\nMetrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Recall / Defect detection rate: {recall * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")
print(f"False positive rate: {false_positive_rate * 100:.2f}%")
print(f"F1 score: {f1_score * 100:.2f}%")

# Ignora as primeiras imagens para reduzir o efeito do aquecimento inicial da GPU.
warmup_images = 5
measured_times = processing_times[warmup_images:]

average_time = sum(measured_times) / len(measured_times)
minimum_time = min(measured_times)
maximum_time = max(measured_times)

# Estimativa de quantas imagens poderiam ser processadas por segundo,
# considerando o tempo médio de uma inspeção.
fps = 1000 / average_time


print("\nProcessing time:")
print(f"Average: {average_time:.2f} ms")
print(f"Minimum: {minimum_time:.2f} ms")
print(f"Maximum: {maximum_time:.2f} ms")
print(f"Estimated throughput: {fps:.2f} images/s")