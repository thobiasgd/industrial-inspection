from vision_inspect.config import PROJECT_ROOT


def main() -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    pasta = PROJECT_ROOT / "dados" / "mvtec" / "bottle"

    arquivos = [
        ("Sem defeito", pasta / "train" / "good" / "000.png"),
        ("Com defeito", pasta / "test" / "broken_large" / "000.png"),
        ("Máscara do defeito", pasta / "ground_truth" / "broken_large" / "000_mask.png"),
    ]

    for titulo, caminho in arquivos:
        if not caminho.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

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


if __name__ == "__main__":
    main()
