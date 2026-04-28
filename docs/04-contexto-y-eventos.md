# 04 - Contexto y eventos

## Eventos actuales

La memoria temporal genera eventos simples basados en tracking:

- nuevo objeto detectado
- objeto que salio de escena

Estos eventos se guardan en una cola corta y se muestran en el overlay.

## Senales adicionales

### Conteo de dedos

La app usa `MediaPipe Tasks HandLandmarker` para estimar landmarks de mano y contar dedos extendidos por cada mano visible.

Esto sirve para casos como:

- gestos simples
- conteo con dedos
- enriquecer la descripcion semantica del VLM

El modelo `hand_landmarker.task` se descarga automaticamente la primera vez en `models/hand_landmarker.task`.

### Posible llamada telefonica

La app tambien aplica una heuristica ligera:

- detecta `person` con YOLO
- detecta `cell phone` con YOLO
- si el telefono aparece cerca de la parte superior de la caja de la persona, marca `posible persona hablando por telefono`

No es un clasificador perfecto, pero es util como primera senal contextual.

## Contextos estimados

### Escritorio

Sube cuando hay objetos tipicos de mesa y cierta estabilidad:

- laptop
- keyboard
- mouse
- book
- cell phone

### Actividad

Sube cuando hay personas y dinamica de movimiento o cambios.

### Procedimiento

Sube cuando hay manipulacion de objetos y variacion frecuente de posiciones.

## Limitaciones actuales

- No hay zonas configurables todavia
- El conteo de dedos depende de manos visibles y orientacion razonable
- No hay memoria de largo plazo
- El contexto es heuristico, no un clasificador entrenado
- La deteccion de llamada es una heuristica espacial, no un modelo especializado

## Siguientes mejoras naturales

- zonas de interes
- historial consultable
- preguntas sobre eventos pasados
- interacciones persona-objeto
