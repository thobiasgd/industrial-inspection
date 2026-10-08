import torch
import torchvision

# Mostra quais versões foram carregadas
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA do PyTorch:", torch.version.cuda)

# Verifica se o PyTorch consegue acessar a GPU
gpu_disponivel = torch.cuda.is_available()
print("GPU disponivel: ", gpu_disponivel)

#Interrompe o teste caso a GPU não esteja acessível.
if not gpu_disponivel:
    raise RuntimeError("O PyTorch não conseguiu acessar a GPUO")

print("Nome da GPU: ", torch.cuda.get_device_name(0))

# Cria um tensor com três números na GPU.
numeros = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")

# Multiplica cada número por 2 na GPU.
resultado = numeros * 2

# Copia o resultado para a CPU para exibi-lo como uma lista.
print("Resultado:", resultado.cpu().tolist())

# Confirma onde o tensor original do resultado está armazenado.
print("Dispositivo do resultado:", resultado.device)

print("Teste concluído com sucesso!")

# Cria três números diretamente na primeira GPU.
numeros = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")