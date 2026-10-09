# VisionInspect — serviço Python

Este projeto concentra a inferência, a API FastAPI e os comandos de preparação e avaliação do detector. O backend NestJS acessa este serviço por HTTP.

## Estrutura

```text
python/
├── pyproject.toml           # Metadados, dependências e configuração dos testes
├── requirements.txt        # Atalho para instalar o pacote em modo editável
├── src/vision_inspect/
│   ├── config.py           # Caminho central dos dados e artefatos
│   ├── detector.py         # Detector de anomalias com PyTorch
│   ├── visualization.py    # Heatmap, overlay e bounding boxes
│   ├── api/
│   │   ├── app.py          # Aplicação FastAPI e ciclo de vida do modelo
│   │   └── schemas.py      # Contratos das respostas HTTP
│   └── cli/
│       ├── build_memory_bank.py
│       ├── collect_normal_scores_top5.py
│       ├── calculate_threshold_top5.py
│       ├── evaluate_test_set_top5.py
│       ├── check_gpu.py
│       ├── preview_inspection.py
│       ├── view_dataset.py
│       └── legacy/         # Experimentos anteriores com o score máximo
├── tests/                  # Testes automatizados sem dependência de GPU
├── dados/                  # Dataset local, ignorado pelo Git
└── artifacts/              # Modelos e calibração locais, ignorados pelo Git
```

Os módulos de `cli/` têm uma função `main()` e são executados com `python -m`. Importá-los não inicia calibração, inferência nem janelas de visualização. A API carrega o detector uma vez por processo ao iniciar o servidor.

## Instalação

Requer Python 3.11 ou superior e uma GPU compatível com CUDA para executar o detector. Partindo da raiz do repositório, no PowerShell:

```powershell
cd python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[dev,visualization]"
python -m vision_inspect.cli.check_gpu
```

As dependências são definidas em `pyproject.toml`. O extra `dev` instala pytest e o cliente HTTP de testes; `visualization` instala as bibliotecas usadas para explorar o dataset. Para instalar apenas o serviço, use `python -m pip install -e .` ou `python -m pip install -r requirements.txt` dentro desta pasta, após instalar o PyTorch com CUDA.

Se já existir um ambiente `.venv` na raiz do repositório, ele pode ser reutilizado: ative-o e execute `python -m pip install -e "./python[dev,visualization]"` a partir da raiz. Ambientes virtuais não devem ser movidos, pois seus executáveis podem conter caminhos absolutos.

## Dados e artefatos

O dataset Bottle deve estar em `python/dados/mvtec/bottle/`, com os diretórios `train`, `test` e `ground_truth`. Os arquivos `.pt` ficam em `python/artifacts/`.

Esses caminhos são resolvidos em `config.py` e não dependem da pasta atual do terminal. Na instalação editável, a base padrão é esta pasta `python/`. Para usar outra base, defina uma variável de ambiente antes de executar os comandos:

```powershell
$env:VISION_INSPECT_HOME = "D:\VisionInspect"
```

Nesse exemplo, o serviço utiliza `D:\VisionInspect\dados` e `D:\VisionInspect\artifacts`. Em uma instalação por wheel, configure essa variável explicitamente. A aplicação não carrega arquivos `.env` automaticamente.

## Preparação e calibração

Com o pacote instalado, execute em sequência:

```powershell
python -m vision_inspect.cli.build_memory_bank
python -m vision_inspect.cli.collect_normal_scores_top5
python -m vision_inspect.cli.calculate_threshold_top5
python -m vision_inspect.cli.evaluate_test_set_top5
```

Os três primeiros comandos geram `memory_bank.pt`, `normal_scores_top5.pt` e `threshold_top5.pt`. A avaliação usa o conjunto de teste e imprime métricas e tempos de processamento. Os parâmetros e o cálculo do score Top 5% foram preservados.

## API

```powershell
python -m uvicorn vision_inspect.api.app:app --host 127.0.0.1 --port 8000
```

- `GET /health`: disponibilidade do detector e dispositivo utilizado.
- `POST /inspect`: recebe o campo multipart `image` e retorna score, decisão e visualizações.
- `GET /docs`: documentação interativa do contrato HTTP.

A configuração do NestJS continua sendo `INFERENCE_API_URL=http://127.0.0.1:8000`.

## Diagnóstico e visualização

```powershell
python -m vision_inspect.cli.check_gpu
python -m vision_inspect.cli.preview_inspection
python -m vision_inspect.cli.view_dataset
```

Os dois últimos comandos abrem janelas e precisam de uma sessão gráfica. `view_dataset` requer o extra `visualization`.

Os experimentos antigos foram preservados em `vision_inspect.cli.legacy`: `calculate_threshold`, `evaluate_test_set` e `inspect_image`. Eles usam `threshold.pt` e, para a calibração, `normal_scores.pt`; esses arquivos são distintos dos artefatos Top 5% utilizados pela API. O comando antigo `calculate_treshhold.py` passou a se chamar `calculate_threshold`.

## Testes

Dentro de `python/`, com o extra `dev` instalado:

```powershell
python -m pytest
```

Os testes verificam a resolução de caminhos, a importação dos comandos sem execução e o contrato HTTP com um detector simulado. Não requerem CUDA, download de pesos nem acesso ao dataset. A avaliação com imagens reais continua disponível pelo comando `evaluate_test_set_top5`.
