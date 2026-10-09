import torch
import torchvision


def main() -> None:
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("CUDA do PyTorch:", torch.version.cuda)

    gpu_disponivel = torch.cuda.is_available()
    print("GPU disponivel: ", gpu_disponivel)

    if not gpu_disponivel:
        raise RuntimeError("O PyTorch não conseguiu acessar a GPUO")

    print("Nome da GPU: ", torch.cuda.get_device_name(0))

    numeros = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")

    resultado = numeros * 2

    print("Resultado:", resultado.cpu().tolist())

    print("Dispositivo do resultado:", resultado.device)

    print("Teste concluído com sucesso!")

    numeros = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")


if __name__ == "__main__":
    main()
