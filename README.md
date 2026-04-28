# Asistente Visual Local con YOLO, ByteTrack y Gemma 4

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO11](https://img.shields.io/badge/YOLO11-Detection-111111?style=for-the-badge)
![ByteTrack](https://img.shields.io/badge/ByteTrack-Tracking-1F6FEB?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local_VLM-111111?style=for-the-badge)
![LM Studio](https://img.shields.io/badge/LM_Studio-OpenAI_Compatible-00D9FF?style=for-the-badge)

**Vision artificial local en tiempo real con deteccion, tracking, memoria temporal, contexto automatico y razonamiento multimodal**

[Demo](#demo) · [Arquitectura](#arquitectura) · [Inicio-rápido](#inicio-rápido) · [Cámara-del-celular](#cámara-del-celular) · [Documentación](#documentación)

</div>

---

## Demo

Video de prueba incluido en el repo:

- [Ver demo local del proyecto](demo-yolo-gemma4.mp4)

El demo muestra el flujo base del sistema con:

- deteccion en tiempo real
- tracking persistente
- descripciones con `Gemma 4`
- senales humanas en overlay
- uso flexible de camara local o stream desde celular

---

## Idea general

Este proyecto implementa un asistente visual local que combina percepcion rapida y razonamiento semantico local.

En lugar de usar un VLM para todos los frames, la arquitectura separa responsabilidades:

- `YOLO11s` detecta objetos y personas en tiempo real
- `ByteTrack` mantiene continuidad temporal
- una memoria corta genera eventos y contexto
- `Gemma 4` describe e interpreta la escena solo cuando aporta valor

El resultado es una base mas realista para construir un sistema de:

- asistencia visual local
- monitoreo de actividad
- apoyo a procedimientos
- analitica contextual sobre una estacion de trabajo

---

## Estado actual

Actualmente el sistema ya puede:

- detectar objetos y personas con `YOLO11s`
- seguir objetos entre frames con `ByteTrack`
- describir la escena en espanol con `Gemma 4`
- cambiar entre `LM Studio` y `Ollama`
- usar webcam local o camara del celular por red local
- reconstruir contexto de escena:
  - `escritorio`
  - `actividad`
  - `procedimiento`
- mantener memoria temporal de entidades y eventos
- detectar senales humanas relevantes:
  - manos visibles
  - conteo de dedos con landmarks
  - posible llamada telefonica
  - ojos posiblemente cerrados
  - posible somnolencia
- recuperarse mejor ante cortes del stream del celular

---

## Arquitectura

```text
Fuente de video -> OpenCV -> YOLO11s -> ByteTrack -> SceneMemory -> Context Engine -> Human Signals -> Gemma 4 -> Overlay
```

### Componentes

- `OpenCV`
  - captura webcam local o stream IP/RTSP
- `YOLO11s`
  - deteccion de objetos y personas
- `ByteTrack`
  - tracking persistente de entidades
- `SceneMemory`
  - historial corto y eventos recientes
- `Context Engine`
  - clasifica la escena en escritorio, actividad o procedimiento
- `Human Signals`
  - manos, dedos, rostro, ojos, telefono
- `Gemma 4`
  - descripcion semantica y razonamiento breve

---

## Capacidades principales

### Percepcion visual

- deteccion continua con `YOLO11s`
- tracking con `ByteTrack`
- soporte para camara laptop, USB o celular por red

### Razonamiento local

- integracion con `LM Studio`
- integracion con `Ollama`
- prompts optimizados para respuestas breves en espanol

### Contexto y eventos

- memoria temporal de entidades
- eventos como aparicion y salida de objetos
- inferencia de contexto automatico

### Senales humanas

- gestos manuales visibles
- conteo de dedos por mano cuando los landmarks son detectables
- llamada telefonica probable
- ojos cerrados
- somnolencia heuristica

### Robustez

- reconexion automatica de stream IP
- timeouts configurables
- modo mas ligero para celular con procesamiento cada `N` frames

---

## Documentación

| Sección | Descripción |
|---------|-------------|
| [01 - Arquitectura](docs/01-arquitectura.md) | Cómo se combinan YOLO, ByteTrack, memoria temporal y Gemma 4 |
| [02 - Instalación](docs/02-instalacion.md) | Entorno, dependencias y primer arranque |
| [03 - Uso y runtimes](docs/03-uso-y-runtimes.md) | Ejecución con LM Studio/Ollama y modos webcam/celular |
| [04 - Contexto y eventos](docs/04-contexto-y-eventos.md) | Contexto automático, señales y límites actuales |
| [05 - Troubleshooting](docs/05-troubleshooting.md) | Errores comunes, latencia y red local |
| [06 - Resumen del proyecto](docs/06-resumen-del-proyecto.md) | Resumen técnico y hoja de ruta hacia casos reales |

---

## Estructura del proyecto

```text
vision-assistant-local/
├── docs/
│   ├── 01-arquitectura.md
│   ├── 02-instalacion.md
│   ├── 03-uso-y-runtimes.md
│   ├── 04-contexto-y-eventos.md
│   ├── 05-troubleshooting.md
│   └── 06-resumen-del-proyecto.md
├── src/
│   └── vision_assistant/
│       ├── app.py
│       ├── config.py
│       ├── context.py
│       ├── signals.py
│       └── vlm.py
├── demo-yolo-gemma4.mp4
├── LICENSE
├── README.md
├── main.py
└── requirements.txt
```

---

## Stack

- `Python`
- `OpenCV`
- `Ultralytics`
- `YOLO11s`
- `ByteTrack`
- `MediaPipe Tasks HandLandmarker`
- `LM Studio`
- `Ollama`
- `Gemma 4`

---

## Inicio rápido

### 1. Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Arrancar un runtime local

#### Opción A: Ollama

```bash
ollama pull gemma4:e4b
ollama serve
```

#### Opción B: LM Studio

- Carga un modelo multimodal como `google/gemma-4-26b-a4b`
- Levanta el servidor local en `http://localhost:1234`

### 4. Ejecutar la aplicación

Modo recomendado en laptop:

```bash
python main.py --camera-source 0 --vlm-runtime ollama --vlm-model gemma4:e4b --display-scale 1.2 --process-every-n-frames 2
```

Modo recomendado con más fluidez para stream del celular:

```bash
python main.py --camera-source "http://IP_DEL_CELULAR:PUERTO/video" --vlm-runtime ollama --vlm-model gemma4:e4b --scene-cooldown 8 --scene-change-min-interval 2.5 --camera-open-timeout-ms 8000 --camera-read-timeout-ms 12000 --camera-reconnect-delay 2.0 --display-scale 1.2 --process-every-n-frames 2
```

Modo equivalente con `LM Studio`:

```bash
python main.py --camera-source 0 --vlm-runtime lmstudio --vlm-model google/gemma-4-e2b --display-scale 1.2 --process-every-n-frames 2
```

---

## Cámara del celular

La aplicación acepta dos tipos de fuente de video:

- índice local: `0`, `1`, `2`
- URL de stream: `http://...` o `rtsp://...`

La ruta más práctica es usar una app como `IP Webcam` en Android y pasar el stream por red local.

Ejemplo:

```bash
python main.py --camera-source "http://192.168.0.4:8080/video" --vlm-runtime ollama --vlm-model gemma4:e2b --scene-cooldown 8 --scene-change-min-interval 2.5 --camera-open-timeout-ms 8000 --camera-read-timeout-ms 12000 --camera-reconnect-delay 2.0 --display-scale 1.2 --process-every-n-frames 2
```

Recomendaciones para el celular:

- usar la misma red Wi-Fi que la laptop
- bajar la resolucion del stream a `640x480` o `800x600`
- usar `15 FPS` si buscas estabilidad
- quitar ahorro de bateria a la app de camara
- mantener la app abierta en primer plano durante la prueba

---

## Controles

- `Q`: salir
- `P`: pausar video
- `S`: guardar screenshot
- `D`: forzar descripción

---

## Ajustes útiles

- `--display-scale`
  - agranda la ventana mostrada
- `--process-every-n-frames`
  - reduce carga de CPU/GPU para streams IP
- `--camera-open-timeout-ms`
  - timeout al abrir stream IP
- `--camera-read-timeout-ms`
  - timeout de lectura del stream IP
- `--camera-reconnect-delay`
  - espera antes de reconectar
- `--vlm-image-max-edge`
  - baja resolución enviada al VLM
- `--vlm-jpeg-quality`
  - comprime imagen para respuesta más rápida

---

## Casos de uso reales

### Asistente de escritorio

- detectar objetos visibles
- describir cambios en la mesa
- observar señales de atención o actividad

### Monitor local de actividad

- presencia
- interacción con objetos
- llamada telefónica
- somnolencia o inactividad visible

### Asistente de procedimientos

- observar manipulación de objetos
- resumir la escena de trabajo
- servir como base para reglas de pasos o validaciones futuras

---

## Próximos pasos

- zonas configurables en pantalla
- historial consultable de `30-120` segundos
- preguntas sobre la escena reciente
- reglas persona-objeto más fuertes
- pose corporal y atención más finas
- API local o panel web
- integración futura vía MCP

---

## Repositorio

- GitHub: [cpu-16/Asistente-Visual-Local-con-YOLO-ByteTrack-y-Gemma4](https://github.com/cpu-16/Asistente-Visual-Local-con-YOLO-ByteTrack-y-Gemma4)

---

## Licencia

Este proyecto se publica bajo licencia `Apache-2.0`.

---

## Agradecimientos

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [MediaPipe](https://ai.google.dev/edge/mediapipe)
- [Ollama](https://ollama.com/)
- [LM Studio](https://lmstudio.ai/)
- [Gemma](https://ai.google.dev/gemma)

---

<div align="center">

**Construido localmente por `cpu-16` y OpenCode.**

Hecho con enfoque practico, privacidad local y ganas de llevar vision artificial al mundo real.

</div>
