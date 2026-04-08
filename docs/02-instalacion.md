# 02 - Instalacion

## Requisitos

- Linux con Python 3.10+
- Webcam integrada o externa
- GPU NVIDIA recomendada para mejor rendimiento
- Uno de estos runtimes:
  - `LM Studio`
  - `Ollama`

## Preparar entorno

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## LM Studio

1. Abre `LM Studio`
2. Carga un modelo multimodal, por ejemplo `google/gemma-4-26b-a4b`
3. Habilita el servidor local compatible con OpenAI
4. Verifica que responda en `http://localhost:1234/v1/models`

## Ollama

```bash
ollama pull gemma4:e4b
ollama serve
```

Verificacion:

```bash
curl http://localhost:11434/api/tags
```
