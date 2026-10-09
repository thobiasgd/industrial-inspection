from pathlib import Path

import cv2
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

from vision_inspect.config import PROJECT_ROOT


class AnomalyDetector:

    def __init__(self, project_root: Path = PROJECT_ROOT):

        # Define o dispositivo usado pelo detector.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        self.device = torch.device("cuda:0")

        artifacts_dir = project_root / "artifacts"

        bank_path = (
            artifacts_dir
            / "memory_bank.pt"
        )

        threshold_path = (
            artifacts_dir
            / "threshold_top5.pt"
        )

        if not bank_path.is_file():
            raise FileNotFoundError(
                f"Memory bank not found: {bank_path}"
            )

        if not threshold_path.is_file():
            raise FileNotFoundError(
                f"Threshold not found: {threshold_path}"
            )

        # Carrega o banco de características na GPU.
        self.memory_bank = torch.load(
            bank_path,
            map_location=self.device,
            weights_only=True,
        )

        # Carrega o threshold calibrado.
        threshold_data = torch.load(
            threshold_path,
            map_location="cpu",
            weights_only=True,
        )

        self.threshold = threshold_data["threshold"]

        # Mesmo preprocessamento usado durante
        # a criação do memory bank.
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
            weights=ResNet18_Weights.IMAGENET1K_V1,
        )

        # Utiliza a saída da layer2.
        self.feature_extractor = create_feature_extractor(
            model,
            return_nodes={
                "layer2": "features",
            },
        )

        self.feature_extractor = (
            self.feature_extractor
            .to(self.device)
            .eval()
        )


    def inspect(self, image_bgr):

        # Verifica se a imagem recebida é válida.
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError(
                "The image is empty or invalid."
            )

        # OpenCV utiliza BGR.
        # A ResNet foi preparada para receber RGB.
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # Prepara a imagem e envia para a GPU.
        input_batch = (
            self.preprocess(image_rgb)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():

            # Extrai o mapa de características.
            feature_map = self.feature_extractor(
                input_batch
            )["features"]

            _, channels, height, width = (
                feature_map.shape
            )

            # Organiza uma posição espacial por linha.
            patch_features = (
                feature_map
                .permute(0, 2, 3, 1)
                .reshape(
                    -1,
                    channels,
                )
            )

            # Compara cada patch da imagem
            # com o banco de referências normais.
            distances = torch.cdist(
                patch_features,
                self.memory_bank,
                p=2,
            )

            # Mantém a distância da referência
            # mais próxima para cada patch.
            patch_scores = (
                distances
                .min(dim=1)
                .values
            )

            # Ordena do patch mais anômalo
            # para o menos anômalo.
            sorted_scores = torch.sort(
                patch_scores,
                descending=True,
            ).values

            # Seleciona os 5% patches mais anômalos.
            top_5_count = max(
                1,
                int(
                    len(sorted_scores)
                    * 0.05
                ),
            )

            # Score final da imagem.
            anomaly_score = (
                sorted_scores[
                    :top_5_count
                ]
                .mean()
                .item()
            )

            # O menor valor ainda pertencente
            # aos 5% mais anômalos é usado
            # para localização.
            localization_threshold = (
                sorted_scores[
                    top_5_count - 1
                ]
                .item()
            )

            # Recupera a grade espacial 32x32.
            anomaly_map = (
                patch_scores
                .reshape(
                    height,
                    width,
                )
            )

            # Cria a máscara das regiões mais anômalas.
            binary_map = (
                anomaly_map
                >= localization_threshold
            )

            binary_map_np = (
                binary_map
                .to(torch.uint8)
                .cpu()
                .numpy()
            )


        # Decide se a peça é aprovada ou rejeitada.
        decision = (
            "REJECTED"
            if anomaly_score > self.threshold
            else "APPROVED"
        )

        bounding_boxes = []


        # Só procura defeitos localizados
        # quando a peça foi rejeitada.
        if decision == "REJECTED":

            # Une pequenas lacunas dentro
            # das regiões suspeitas.
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 3),
            )

            cleaned_map = cv2.morphologyEx(
                binary_map_np,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1,
            )

            # Encontra regiões conectadas.
            (
                component_count,
                _,
                stats,
                _,
            ) = cv2.connectedComponentsWithStats(
                cleaned_map,
                connectivity=8,
            )

            image_height, image_width = (
                image_bgr.shape[:2]
            )

            # Converte posições da grade 32x32
            # para pixels da imagem original.
            scale_x = (
                image_width
                / width
            )

            scale_y = (
                image_height
                / height
            )

            # Remove regiões muito pequenas.
            min_component_area = 6


            for component_index in range(
                1,
                component_count,
            ):

                x = stats[
                    component_index,
                    cv2.CC_STAT_LEFT,
                ]

                y = stats[
                    component_index,
                    cv2.CC_STAT_TOP,
                ]

                box_width = stats[
                    component_index,
                    cv2.CC_STAT_WIDTH,
                ]

                box_height = stats[
                    component_index,
                    cv2.CC_STAT_HEIGHT,
                ]

                component_area = stats[
                    component_index,
                    cv2.CC_STAT_AREA,
                ]


                if (
                    component_area
                    < min_component_area
                ):
                    continue


                x1 = int(
                    round(
                        x * scale_x
                    )
                )

                y1 = int(
                    round(
                        y * scale_y
                    )
                )

                x2 = int(
                    round(
                        (
                            x
                            + box_width
                        )
                        * scale_x
                    )
                )

                y2 = int(
                    round(
                        (
                            y
                            + box_height
                        )
                        * scale_y
                    )
                )


                # Mantém as coordenadas
                # dentro dos limites da imagem.
                x1 = max(
                    0,
                    min(
                        x1,
                        image_width - 1,
                    ),
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        image_height - 1,
                    ),
                )

                x2 = max(
                    x1 + 1,
                    min(
                        x2,
                        image_width,
                    ),
                )

                y2 = max(
                    y1 + 1,
                    min(
                        y2,
                        image_height,
                    ),
                )


                bounding_boxes.append({
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                })


        return {
            "score": anomaly_score,
            "threshold": self.threshold,
            "decision": decision,

            "anomaly_map": (
                anomaly_map
                .cpu()
                .numpy()
            ),

            "bounding_boxes": bounding_boxes,
        }
