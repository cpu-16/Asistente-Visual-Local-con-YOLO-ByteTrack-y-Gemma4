# 06 - Resumen del proyecto

## Objetivo general

Se construyo un asistente visual local en tiempo real capaz de usar webcam de laptop o stream de camara desde celular para:

- detectar objetos y personas
- hacer tracking entre frames
- describir la escena con un modelo local
- inferir contexto de uso
- detectar algunas senales humanas relevantes

La idea principal fue evitar depender solo de un VLM para todos los frames y, en su lugar, combinar percepcion rapida con razonamiento semantico local.

---

## Arquitectura implementada

```text
Fuente de video -> OpenCV -> YOLO -> ByteTrack -> Memoria temporal -> Senales/contexto -> Gemma 4 -> Overlay
```

### Capas del sistema

1. Captura de video
- webcam local por indice
- stream remoto por URL `http://...` o `rtsp://...`

2. Deteccion
- `YOLO11s` como detector principal

3. Tracking
- `ByteTrack` para mantener continuidad temporal de objetos

4. Memoria y eventos
- historial corto de entidades
- eventos como aparicion y salida de objetos

5. Senales humanas
- manos y conteo de dedos con `MediaPipe Tasks HandLandmarker`
- rostro y heuristicas de ojos cerrados/somnolencia
- heuristica de posible llamada telefonica

6. Razonamiento semantico
- `Gemma 4` local usando `LM Studio` u `Ollama`

7. Interfaz
- overlay con FPS, contexto, eventos, senales y descripcion

---

## Tecnologias usadas

### Vision y video
- `OpenCV`
- `Ultralytics`
- `YOLO11s`
- `ByteTrack`

### Modelos locales
- `LM Studio`
- `Ollama`
- `Gemma 4 26B A4B`
- `gemma4:e4b`
- `gemma4:e2b`

### Senales humanas
- `MediaPipe Tasks`
- `HandLandmarker`
- `Haar cascades` de OpenCV para rostro/ojos

### Infraestructura y publicacion
- `git`
- `GitHub`
- `gh`

---

## Lo que se construyo

### 1. Base funcional del proyecto

Se creo el proyecto `vision-assistant-local` con:

- estructura modular en `src/vision_assistant`
- punto de entrada `main.py`
- configuracion centralizada
- README profesional
- documentacion en `docs/`

### 2. Pipeline de vision en tiempo real

Se implemento un pipeline que:

- abre camara local
- detecta objetos con YOLO
- sigue objetos con ByteTrack
- dibuja overlay con informacion en vivo

### 3. Integracion de VLM local

Se creo una capa de adaptacion para dos runtimes:

- `LM Studio`
- `Ollama`

Con esto el sistema puede cambiar de backend sin reescribir el pipeline principal.

### 4. Optimizaciones de rendimiento

Se redujo la carga del VLM mediante:

- compresion de imagen
- reduccion de resolucion enviada al modelo
- menos tokens
- cooldown entre descripciones
- descripciones disparadas por eventos y cambios, no por cada frame

### 5. Correcciones para Gemma 4 local

Se resolvieron problemas reales como:

- respuestas vacias de `LM Studio` por `reasoning_content`
- respuestas vacias de `Ollama` por `thinking`
- forzado de salida final visible en espanol

### 6. Contexto automatico

Se agrego inferencia de contexto basada en objetos, movimiento y eventos:

- `escritorio`
- `actividad`
- `procedimiento`
- `indeterminado`

### 7. Memoria temporal

Se implemento una memoria corta para:

- entidades detectadas
- eventos recientes
- persistencia minima de contexto

### 8. Senales humanas y comportamiento

Se agregaron senales para enriquecer la interpretacion:

- conteo de dedos por mano visible
- gestos manuales visibles
- posible llamada telefonica
- ojos posiblemente cerrados
- persona con senales de somnolencia
- resumen de objetos visibles

### 9. Soporte para camara del celular

Se amplio la app para aceptar:

- webcam por indice local
- stream IP del celular por HTTP o RTSP

Tambien se agrego:

- reconexion automatica si el stream se cae
- timeouts configurables para fuentes IP
- estado visual de captura en el overlay

### 10. Publicacion del proyecto

Se creo y publico el repositorio:

- `https://github.com/cpu-16/Asistente-Visual-Local-con-YOLO-ByteTrack-y-Gemma4`

---

## Problemas que resolvimos durante el desarrollo

### Rendimiento del VLM
- `Gemma 4 26B` era demasiado pesado para descripcion frecuente
- se probaron variantes mas ligeras
- se dejo soporte para `e2b`, `e4b` y `26B`

### Respuestas vacias del modelo
- `LM Studio` devolvia `reasoning_content`
- `Ollama` devolvia `thinking`
- se corrigio el manejo de esas respuestas

### Compatibilidad de deteccion de manos
- `mediapipe solutions` no estaba disponible en el entorno actual
- se migro a `MediaPipe Tasks HandLandmarker`

### Conectividad de celular por WiFi
- se diagnostico la red local
- se identificaron factores como IP real, MAC privada y aislamiento
- se logro conectar el celular por stream IP

### Estabilidad del stream MJPEG
- `IP Webcam` por HTTP podia degradarse o expirar
- se agrego reconexion automatica y tolerancia a timeout

---

## Estado actual del sistema

Actualmente el sistema ya puede:

- detectar objetos y personas en tiempo real
- seguirlos entre frames
- describir la escena en espanol con Gemma 4 local
- cambiar entre `LM Studio` y `Ollama`
- usar webcam local o camara del celular por red
- inferir contexto de la escena
- detectar algunas senales humanas relevantes

No es todavia un sistema completo de comprension de comportamiento humano, pero ya paso claramente de una demo basica a una base seria de producto/prototipo.

---

## Aplicaciones reales posibles

### 1. Asistente de escritorio inteligente
- detectar objetos sobre el escritorio
- describir el estado de la mesa
- recordar cambios recientes
- responder preguntas simples sobre lo visible

### 2. Monitor local de actividad
- detectar presencia
- ver entradas y salidas de escena
- alertar sobre uso de telefono o somnolencia
- resumir actividad reciente

### 3. Asistente de procedimientos
- observar secuencias de manipulacion
- detectar que una tarea parece estar en progreso
- validar senales manuales o cambios de objetos

### 4. Soporte de accesibilidad
- describir la escena en lenguaje natural
- enfatizar acciones humanas y objetos cercanos

### 5. Supervison de estaciones de trabajo
- detectar actividad o inactividad
- ver objetos usados alrededor de una persona
- servir como base para metricas de atencion o flujo de trabajo

---

## Que se puede hacer despues

### Bloque 1. Mejor comprension humana
- pose corporal completa
- landmarks faciales mas finos
- mejor deteccion de ojos cerrados y atencion
- mano cerca de cara, mano cerca de objeto, mano en teclado, etc.

### Bloque 2. Memoria mas fuerte
- historial de 30 a 120 segundos
- preguntas sobre lo ocurrido recientemente
- resumen temporal por persona u objeto

### Bloque 3. Reglas mas utiles
- zonas de interes
- objeto abandonado
- persona inactiva mucho tiempo
- deteccion de uso de herramientas o telefono

### Bloque 4. Analitica realista
- inferencia de productividad por contexto
- actividad de escritorio
- tiempo en tarea
- atencion visual y manipulacion de objetos

### Bloque 5. Integracion externa
- API local
- panel web
- MCP para exponer la camara y eventos a un agente
- app movil dedicada en vez de depender de IP Webcam

### Bloque 6. Robustez de despliegue
- reconexion mas avanzada
- configuracion persistente
- grabacion de clips o snapshots por evento
- exportacion de eventos

---

## Recomendacion de ruta para llevarlo al mundo real

Orden sugerido:

1. fortalecer estabilidad de camara del celular
2. mejorar deteccion de acciones humanas
3. agregar memoria temporal consultable
4. exponer API local para integraciones
5. construir caso de uso especifico

Los mejores casos iniciales para producto real serian:

- asistente de escritorio
- monitor de actividad personal/local
- asistente de procedimientos en laboratorio o taller

---

## Resumen final

Se construyo una plataforma local de vision artificial con razonamiento semantico que ya combina:

- percepcion rapida
- tracking temporal
- memoria corta
- interpretacion por VLM local
- senales humanas
- soporte de camara flexible

El proyecto ya es una base valida para evolucionar hacia un sistema de asistencia visual, monitoreo inteligente o analitica contextual aplicada al mundo real.
