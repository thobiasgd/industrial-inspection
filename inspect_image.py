from pathlib import Path

import cv2
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

# Define os caminhos do banco e de uma imagem de teste.
project_root = Path(__file__).resolve().parent
bank_path = project_root / "artifacts" / "memory_bank.pt"
# Seleciona uma imagem boa que não foi usada para construir o banco.
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

# Carrega o banco que criamos e coloca seus dados na GPU.
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

# Mantém exatamente a preparação usada para construir o banco.
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Recria o mesmo extrator, com os mesmos pesos e bloco de saída.
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

feature_extractor = create_feature_extractor(
    model,
    return_nodes={"layer2": "features"},
)
feature_extractor = feature_extractor.to(device).eval()

# Lê e prepara a imagem que queremos inspecionar.
image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

if image_bgr is None:
    raise RuntimeError(f"Could not read the image: {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
input_batch = preprocess(image_rgb).unsqueeze(0).to(device)

with torch.inference_mode():
    # Extrai e organiza as características da imagem de teste.
    feature_map = feature_extractor(input_batch)["features"]
    _, channels, height, width = feature_map.shape

    test_features = feature_map.permute(0, 2, 3, 1).reshape(-1, channels)

    # Confere se o banco tem características compatíveis com o extrator.
    if (
        memory_bank.ndim != 2
        or memory_bank.shape[0] == 0
        or memory_bank.shape[1] != channels
    ):
        raise ValueError("The memory bank has an incompatible shape.")

    # Compara cada vetor da imagem com todos os vetores do banco.
    distances = torch.cdist(test_features, memory_bank, p=2)

    # Para cada posição da imagem, guarda a menor distância encontrada.
    patch_scores = distances.min(dim=1).values

    # Ordena os patches do mais anômalo para o menos anômalo.
    sorted_scores = torch.sort(
        patch_scores,
        descending=True,
    ).values

    # Calcula quantos patches correspondem a diferentes percentuais do mapa.
    top_1_count = max(1, int(len(sorted_scores) * 0.01))
    top_5_count = max(1, int(len(sorted_scores) * 0.05))
    top_10_count = max(1, int(len(sorted_scores) * 0.10))

    # Calcula diferentes formas de resumir o mapa em um único valor.
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

    # Reorganiza as pontuações na grade espacial da imagem.
    anomaly_map = patch_scores.reshape(height, width)

    # Usa a maior pontuação local como pontuação da imagem inteira.
    anomaly_score = patch_scores.max().item()

# Decide se o produto está dentro do padrão esperado.
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

# Copia o mapa para a CPU e converte para uma matrix Numpy.
anomaly_array = anomaly_map.cpu().numpy()

# Obtém a altura e a largura da imagem original.
image_height, image_width = image_bgr.shape[:2]

# Amplia o mapa para permitir a sobreposição na imagem original
resized_map = cv2.resize(
    anomaly_array,
    (image_width, image_height),
    interpolation=cv2.INTER_LINEAR
)

# Ajusta os valores para visualização, sem alterar o mapa original
display_map = cv2.normalize(
    resized_map,
    None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_8U
)

# Transforma os balores em cores
heatmap_bgr = cv2.applyColorMap(display_map, cv2.COLORMAP_INFERNO)

# Mistura a imagem original com o mapa colorido
overlay_bgr = cv2.addWeighted(
    image_bgr, 0.6,
    heatmap_bgr, 0.4,
    0.0
)

# Identifica no terminal qual imagem foi inspecionada.
print("Inspected image:", image_path)

# Carrega a máscara real do defeito apenas para comparação.
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

# Converte de 1 canal para 3 canais para permitir o hconcat.
ground_truth_bgr = cv2.cvtColor(
    ground_truth,
    cv2.COLOR_GRAY2BGR,
)

# Mostra a imagem original, o mapa de calor e a sobreposição.
preview = cv2.hconcat([
    image_bgr,
    ground_truth_bgr,
    heatmap_bgr,
    overlay_bgr,
])

# Exibe a pontuação no título da janela.
window_name = (
    f"{decision} | "
    f"Score: {anomaly_score:.4f} | "
    f"Threshold: {threshold:.4f}"
)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 400)
cv2.imshow(window_name, preview)

# Aguarda uma tecla e fecha a janela.
cv2.waitKey(0)
cv2.destroyAllWindows()
