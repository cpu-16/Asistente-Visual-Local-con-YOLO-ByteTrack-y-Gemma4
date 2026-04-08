# 03 - Uso y runtimes

## Ejecutar con Ollama

```bash
python main.py --vlm-runtime ollama --vlm-model gemma4:e4b --scene-cooldown 10 --scene-change-min-interval 4
```

## Ejecutar con LM Studio

```bash
python main.py --vlm-runtime lmstudio --vlm-model google/gemma-4-26b-a4b
```

## Parametros utiles

- `--camera-index`: selecciona camara
- `--cam-width` y `--cam-height`: resolucion de captura
- `--scene-cooldown`: tiempo minimo entre descripciones
- `--scene-change-min-interval`: evita saturar el VLM por cambios menores
- `--vlm-image-max-edge`: reduce la resolucion enviada al VLM
- `--vlm-jpeg-quality`: compresion de imagen

## Recomendacion practica

- `Ollama + gemma4:e4b` para uso diario
- `LM Studio + gemma-4-26b-a4b` para comparar calidad
