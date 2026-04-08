# 04 - Contexto y eventos

## Eventos actuales

La memoria temporal genera eventos simples basados en tracking:

- nuevo objeto detectado
- objeto que salio de escena

Estos eventos se guardan en una cola corta y se muestran en el overlay.

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
- No hay reconocimiento de manos
- No hay memoria de largo plazo
- El contexto es heuristico, no un clasificador entrenado

## Siguientes mejoras naturales

- zonas de interes
- historial consultable
- preguntas sobre eventos pasados
- interacciones persona-objeto
