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

- `--camera-source`: indice local o URL del stream
- `--camera-index`: selecciona camara
- `--cam-width` y `--cam-height`: resolucion de captura
- `--camera-open-timeout-ms`: timeout al abrir stream IP
- `--camera-read-timeout-ms`: timeout de lectura del stream IP
- `--camera-reconnect-delay`: espera antes de reconectar stream IP
- `--display-scale`: escala de la ventana mostrada en pantalla
- `--process-every-n-frames`: procesa cada N frames para aliviar CPU/GPU en streams IP
- `--scene-cooldown`: tiempo minimo entre descripciones
- `--scene-change-min-interval`: evita saturar el VLM por cambios menores
- `--vlm-image-max-edge`: reduce la resolucion enviada al VLM
- `--vlm-jpeg-quality`: compresion de imagen

## Recomendacion practica

- `Ollama + gemma4:e4b` para uso diario
- `LM Studio + gemma-4-26b-a4b` para comparar calidad

## Ejemplos de fuente de video

Webcam local:

```bash
python main.py --camera-source 0 --vlm-runtime ollama --vlm-model gemma4:e4b
```

Celular por HTTP:

```bash
python main.py --camera-source "http://192.168.1.50:8080/video" --vlm-runtime ollama --vlm-model gemma4:e4b --camera-open-timeout-ms 8000 --camera-read-timeout-ms 12000 --camera-reconnect-delay 2.0
```

Celular por HTTP con modo mas fluido:

```bash
python main.py --camera-source "http://192.168.1.50:8080/video" --vlm-runtime ollama --vlm-model gemma4:e4b --scene-cooldown 8 --scene-change-min-interval 2.5 --camera-open-timeout-ms 8000 --camera-read-timeout-ms 12000 --camera-reconnect-delay 2.0 --display-scale 1.2 --process-every-n-frames 2
```

Celular por RTSP:

```bash
python main.py --camera-source "rtsp://192.168.1.50:8554/live" --vlm-runtime ollama --vlm-model gemma4:e4b
```
