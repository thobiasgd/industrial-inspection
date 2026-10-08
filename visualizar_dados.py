from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Licaliza os dados a partir da pasta deste script
pasta = Path(__file__).resolve().parent / "dados" / "mvtec" / "bottle"

# Seleciona uma imagem boa, uma defeituosa e a máscara correspondente.
arquivos = [
    ("Sem defeito", pasta / "train" / "good" / "000.png"),
    ("Com defeito", pasta / "test" / "broken_large" / "000.png"),
    ("Máscara do defeito", pasta / "ground_truth" / "broken_large" / "000_mask.png"),
]

# Confere os caminhos antes de tentar abrir as imagens.
for titulo, caminho in arquivos:
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

# Prepara uma janela com três imagens lado a lado.
figura, eixos = plt.subplots(1, 3, figsize=(14, 4))

for eixo, (titulo, caminho) in zip(eixos, arquivos):
    with Image.open(caminho) as arquivo:
        imagem = arquivo.convert("RGB")

    eixo.imshow(imagem)
    eixo.set_title(titulo)
    eixo.axis("off")

    print(f"{titulo} - tamanho (largura, altura): {imagem.size}")

plt.tight_layout()
plt.show()