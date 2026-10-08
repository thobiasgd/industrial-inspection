from pathlib import Path

import cv2
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


class AnomalyDetector:

    def __init__(self, project_root: Path):

        # Define o dispositivo usado pelo detector.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        self.device = torch.device("cuda:0")

        # Define os arquivos produzidos durante nossa calibração.
        artifacts_dir = project_root / "artifacts"

        bank_path = artifacts_dir / "memory_bank.pt"
        threshold_path = artifacts_dir / "threshold_top5.pt"

        if not bank_path.is_file():
            raise FileNotFoundError(
                f"Memory bank not found: {bank_path}"
            )

        if not threshold_path.is_file():
            raise FileNotFoundError(
                f"Threshold not found: {threshold_path}"
            )

        # Carrega o banco de características diretamente na GPU.
        self.memory_bank = torch.load(
            bank_path,
            map_location=self.device,
            weights_only=True,
        )

        # Carrega o threshold da métrica Top 5%.
        threshold_data = torch.load(
            threshold_path,
            map_location="cpu",
            weights_only=True,
        )

        self.threshold = threshold_data["threshold"]

        # Define o mesmo preprocessamento usado na construção do banco.
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (256, 256),
                antialias=True,
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Carrega a ResNet18 pré-treinada.
        model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        # Utiliza somente as características produzidas pela layer2.
        self.feature_extractor = create_feature_extractor(
            model,
            return_nodes={
                "layer2": "features"
            },
        )

        self.feature_extractor = (
            self.feature_extractor
            .to(self.device)
            .eval()
        )

    def inspect(self, image_bgr):

        # Verifica se recebemos uma imagem válida.
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("The image is empty or invalid.")

        # Converte a imagem do padrão BGR do OpenCV para RGB.
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # Aplica o mesmo preprocessamento usado durante a calibração.
        input_batch = (
            self.preprocess(image_rgb)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():

            # Extrai as características da layer2.
            feature_map = self.feature_extractor(
                input_batch
            )["features"]

            _, channels, height, width = feature_map.shape

            # Transforma o mapa em uma lista de vetores de características.
            patch_features = (
                feature_map
                .permute(0, 2, 3, 1)
                .reshape(-1, channels)
            )

            # Compara cada patch com o banco de referências normais.
            distances = torch.cdist(
                patch_features,
                self.memory_bank,
                p=2,
            )

            # Para cada patch, mantém a distância até a referência mais próxima.
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

            # Seleciona os 5% patches com maiores scores.
            top_5_count = max(
                1,
                int(len(sorted_scores) * 0.05),
            )

            # Calcula a pontuação final da imagem.
            anomaly_score = (
                sorted_scores[:top_5_count]
                .mean()
                .item()
            )

            # Recupera a organização espacial do mapa.
            anomaly_map = patch_scores.reshape(
                height,
                width,
            )

        # Aplica o threshold calibrado.
        decision = (
            "REJECTED"
            if anomaly_score > self.threshold
            else "APPROVED"
        )

        return {
            "score": anomaly_score,
            "threshold": self.threshold,
            "decision": decision,
            "anomaly_map": anomaly_map.cpu().numpy(),
        }