# 01 - Arquitectura

## Pipeline principal

```text
Camara -> OpenCV -> YOLO11s -> ByteTrack -> SceneMemory -> Context Engine -> Gemma 4 -> Overlay
```

## Componentes

### OpenCV

- Captura video desde webcam interna o externa
- Muestra overlay con estado del sistema

### YOLO11s

- Detecta objetos y personas en cada frame
- Devuelve clases, confianza y bounding boxes

### ByteTrack

- Reutiliza las detecciones de YOLO
- Mantiene IDs persistentes entre frames
- Permite asociar movimiento y continuidad temporal

### SceneMemory

- Mantiene un registro corto de entidades activas
- Genera eventos como:
  - objeto nuevo detectado
  - objeto que sale de escena

### Context Engine

- Calcula puntajes para:
  - escritorio
  - actividad
  - procedimiento
- Usa labels, estabilidad, movimiento y eventos recientes

### Gemma 4

- Recibe una imagen comprimida con contexto adicional
- Genera una descripcion corta en espanol
- No analiza todos los frames: solo frames relevantes

## Por que esta arquitectura

- `YOLO` resuelve velocidad
- `ByteTrack` resuelve continuidad
- `Gemma 4` resuelve interpretacion semantica
- `SceneMemory` evita tratar cada frame como algo aislado
