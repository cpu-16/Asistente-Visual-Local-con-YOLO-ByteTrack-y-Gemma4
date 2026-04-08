# Asistente Visual Local con YOLO, ByteTrack y Gemma 4

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Real_Time_Detection-111111?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local_VLM-111111?style=for-the-badge)
![LM Studio](https://img.shields.io/badge/LM_Studio-OpenAI_Compatible-00D9FF?style=for-the-badge)

**Asistente visual local en tiempo real con deteccion, tracking, memoria temporal y contexto automatico**

</div>

---

## Documentacion

| Seccion | Descripcion |
|---------|-------------|
| [01 - Arquitectura](docs/01-arquitectura.md) | Como se combinan YOLO, ByteTrack, memoria temporal y Gemma 4 |
| [02 - Instalacion](docs/02-instalacion.md) | Preparacion del entorno, dependencias y primer arranque |
| [03 - Uso y runtimes](docs/03-uso-y-runtimes.md) | Ejecucion con LM Studio u Ollama y ajustes de rendimiento |
| [04 - Contexto y eventos](docs/04-contexto-y-eventos.md) | Como infiere escritorio, actividad y procedimiento |
| [05 - Troubleshooting](docs/05-troubleshooting.md) | Errores comunes, latencia y verificacion del entorno |

---

## Idea general

Este proyecto implementa un sistema de vision artificial local que usa la webcam de la laptop o una camara externa para:

- detectar objetos en tiempo real con `YOLO11s`
- seguirlos entre frames con `ByteTrack`
- recordar eventos recientes en memoria temporal
- estimar automaticamente el contexto de la escena
- describir lo que ocurre usando `Gemma 4` corriendo localmente en `LM Studio` u `Ollama`

La idea no es usar un VLM para todos los frames, sino combinar:

- un detector rapido para mantener FPS
- un tracker para continuidad temporal
- un VLM local para razonamiento semantico bajo demanda

---

## Capacidades actuales

- Deteccion de objetos/personas en tiempo real con webcam
- Tracking persistente usando `ByteTrack`
- Soporte para `LM Studio` y `Ollama`
- Descripciones de escena en espanol
- Memoria temporal de objetos detectados
- Eventos recientes como aparicion o salida de objetos
- Inferencia automatica de contexto:
  - `escritorio`
  - `actividad`
  - `procedimiento`
- Overlay con FPS, runtime, latencia VLM, contexto y eventos

---

## Arquitectura

```text
Webcam -> OpenCV -> YOLO11s -> ByteTrack -> SceneMemory -> Context Engine -> Gemma 4 -> Overlay
```

### Roles de cada componente

- `YOLO11s`: detecta que objetos hay y donde estan
- `ByteTrack`: mantiene IDs para seguir objetos entre frames
- `SceneMemory`: guarda entidades y eventos recientes
- `Context Engine`: estima si la escena parece escritorio, actividad o procedimiento
- `Gemma 4`: describe y razona sobre la escena con contexto extra

---

## Estructura del proyecto

```text
vision-assistant-local/
├── docs/
│   ├── 01-arquitectura.md
│   ├── 02-instalacion.md
│   ├── 03-uso-y-runtimes.md
│   ├── 04-contexto-y-eventos.md
│   └── 05-troubleshooting.md
├── src/
│   └── vision_assistant/
│       ├── app.py
│       ├── config.py
│       ├── context.py
│       └── vlm.py
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

---

## Stack

- `Python`
- `OpenCV`
- `Ultralytics YOLO`
- `ByteTrack`
- `LM Studio` o `Ollama`
- `Gemma 4`

---

## Inicio rapido

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

#### Opcion A: LM Studio

- Carga un modelo multimodal compatible, por ejemplo `google/gemma-4-26b-a4b`
- Habilita el servidor local compatible con OpenAI en `http://localhost:1234`

#### Opcion B: Ollama

```bash
ollama pull gemma4:e4b
ollama serve
```

### 4. Ejecutar la aplicacion

Con `Ollama`:

```bash
python main.py --vlm-runtime ollama --vlm-model gemma4:e4b --scene-cooldown 10 --scene-change-min-interval 4
```

Con `LM Studio`:

```bash
python main.py --vlm-runtime lmstudio --vlm-model google/gemma-4-26b-a4b
```

---

## Controles

- `Q`: salir
- `P`: pausar video
- `S`: guardar screenshot
- `D`: forzar descripcion de escena

---

## Contexto automatico

El sistema intenta inferir el tipo de escena usando objetos presentes, estabilidad visual, movimiento y eventos recientes.

### Contextos actuales

- `escritorio`
  - objetos estables como laptop, teclado, mouse, telefono, libros
- `actividad`
  - personas, entradas/salidas y dinamica de movimiento
- `procedimiento`
  - manipulacion de objetos y cambios frecuentes

Este mecanismo no depende solo del VLM. Usa una arquitectura hibrida:

- reglas simples de contexto
- memoria temporal
- tracking persistente
- VLM solo para descripcion y razonamiento breve

---

## Casos de uso realistas

### 1. Asistente de escritorio inteligente

- localizar objetos visibles
- describir el estado de la mesa
- recordar cambios recientes

### 2. Monitor local de actividad

- detectar presencia
- observar entradas o salidas de escena
- resumir eventos relevantes

### 3. Asistente de procedimientos

- observar secuencias de manipulacion
- detectar cambios de objetos
- inferir que la escena parece una tarea o proceso

---

## Rendimiento

En hardware con GPU dedicada, la parte de deteccion y tracking puede mantenerse fluida. El cuello de botella suele estar en el VLM.

Recomendacion practica:

- usa `YOLO11s` para tiempo real
- usa `Ollama + gemma4:e4b` para respuestas mas rapidas
- deja `LM Studio + 26B` para pruebas de mayor calidad semantica

---

## Roadmap

- Zonas configurables en pantalla
- Historial consultable de 30-60 segundos
- Preguntas sobre la escena actual y reciente
- Reglas de interaccion persona-objeto
- Deteccion de pasos/procedimientos mas explicita
- Exportacion de eventos

---

## Contribuir

1. Fork el proyecto
2. Crea tu rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Añade nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

---

## Licencia

Este proyecto se publica bajo licencia `Apache-2.0`.

---

## Agradecimientos

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Ollama](https://ollama.com/)
- [LM Studio](https://lmstudio.ai/)
- [Gemma](https://ai.google.dev/gemma)

---

<div align="center">

**Vision artificial local con contexto, memoria y razonamiento semantico**

</div>
