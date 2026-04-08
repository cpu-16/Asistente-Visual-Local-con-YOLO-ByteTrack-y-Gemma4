# 05 - Troubleshooting

## El VLM responde lento

Usa una configuracion mas agresiva:

```bash
python main.py --vlm-runtime ollama --vlm-model gemma4:e4b --scene-cooldown 10 --scene-change-min-interval 4 --vlm-image-max-edge 320 --vlm-jpeg-quality 45 --vlm-max-labels 6
```

## LM Studio no responde

Verifica:

```bash
curl http://localhost:1234/v1/models
```

## Ollama no responde

Verifica:

```bash
curl http://localhost:11434/api/tags
```

## ByteTrack falla por dependencias

Instala:

```bash
pip install lap
```

## No abre la camara correcta

Prueba otro indice:

```bash
python main.py --camera-index 1
```

## La UI muestra advertencias de fuentes Qt

Son advertencias de OpenCV/Qt en algunos entornos Linux. No impiden el funcionamiento del pipeline.
