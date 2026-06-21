# Investigación: mejorar la calidad del audio procedural

> Origen: research de Jack (`info.txt`, movido aquí desde Downloads el 2026-06-21).
> Objetivo: el motor actual suena demasiado sintético. Esta nota recoge las
> librerías/técnicas candidatas y cómo aplicarlas al stack actual.

## Diagnóstico del problema actual

El motor (`audio_engine.py` + `generators/*`) sintetiza todo a mano con NumPy:
osciladores sine/triangle, biquads y envolventes exponenciales. Suena sintético por:

1. **Osciladores puros sin contenido inarmónico vivo** — sine+triangle estáticos.
   La realidad acústica tiene microvariación de pitch, beating y ruido de respiración.
2. **Envolventes demasiado limpias** — ataques/decays matemáticos perfectos.
3. **Sin modelado físico** — los impactos, crujidos y resonancias se aproximan con
   ruido filtrado, no con simulación de cuerpos resonantes.
4. **Falta de movimiento espectral** — los timbres no evolucionan en el tiempo.

## Candidatos por categoría (de la investigación)

### Síntesis de alto rendimiento (reemplazo del core)
- **SignalFlow** (`ideoforms/signalflow`) — core C++, API Python. Grafo de nodos
  para síntesis compleja en tiempo real. El mejor candidato para reescribir el motor
  manteniendo bajo consumo de CPU en `star` (4 núcleos, sin GPU).
- **Pyo** — DSP en C para Python. Maduro, muchos efectos y generadores.
- **Pippi** — manipulación de sonido + osciladores + efectos, orientado a composición.

### Modelado físico (el mayor salto de realismo)
- **tao_synth** (`lucasw/tao_synth`) — red virtual de masas y resortes. Produce
  instrumentos/objetos con resonancia orgánica. Ideal para crujidos de metal/hielo,
  golpes de casco de submarino, estructuras bajo presión (escenas SCP/array/reactor).
- **AudioPG** (arXiv) — primitivas paramétricas (oscilaciones armónicas + ráfagas
  transitorias) para eventos acústicos realistas sin grabaciones.
- **PySynth** — síntesis sustractiva simple (cuerdas frotadas).

### Movimiento y "vida" en sonidos sintéticos
- **tones** (`eriknyquist/tones`) — pitch-bending, vibrato, polifonía. Barato,
  Python puro. Sirve para dar microvariación a drones y pads (atacar el punto 1).
- **procgen** (`jcarlosroldan/procgen`) — Perlin/Simplex noise para texturas
  orgánicas. Ya usamos un Perlin LFO en el pad; ampliarlo a más capas.
- **enginesound** (`DasEtwas/enginesound`) — generador de motores con loops sin
  costura por RPM. Referencia directa para `_traffic_chunk`/`_moto_chunk` de gen_urban.

### Efectos / post (ya parcialmente en uso)
- **Pedalboard** (Spotify) — VST3/AU + efectos nativos. **Ya se usa** en el master bus
  (`pedalboard.Reverb/Chorus/Compressor/Lowpass`). Se puede explotar más: convolución
  con impulsos de sala reales daría reverb mucho menos artificial que la algorítmica.

### Análisis (MIR) — para que los agentes "entiendan" el audio que generan
- **librosa / audioFlux / mirflex** — extraer chroma, MFCC, tempo, picos espectrales
  a JSON. Útil para un loop de auto-evaluación: generar → analizar → comparar contra
  referencias (p.ej. Cryo Chamber) → ajustar parámetros. No mejora el sonido por sí
  solo pero permite medir objetivamente "qué tan sintético" suena.

## Plan de adopción propuesto (incremental, sin romper producción)

1. **Quick win (Python puro, bajo riesgo):** añadir microvariación de pitch (vibrato
   lento + jitter Perlin) a drones y pad → ataca la causa #1 sin nuevas dependencias.
2. **Reverb por convolución:** reemplazar la reverb algorítmica del master por
   convolución con un impulso de sala/caverna (Pedalboard ya soporta `Convolution`).
   Salto grande de realismo, costo CPU moderado.
3. **Modelado físico para impactos:** prototipar `tao_synth` (o port de masas-resortes
   en NumPy) para crujidos/golpes de las escenas industriales. Validar CPU en `star`.
4. **Evaluar SignalFlow** como motor de segunda generación si 1-3 no bastan.
5. **Pipeline MIR opcional:** script offline que puntúe los chunks generados.

## Restricciones de `star` a respetar
- 4 núcleos, sin aceleración GPU → descartar deep-learning en tiempo real.
- El motor corre en bloques (chunks); cualquier librería nueva debe trabajar por
  bloques sin glitches en los límites (mismo cuidado que con los biquads `zi`).
- Mantener salida PCM s16le mono a 44100 Hz hacia `audio_supervisor.py`.
