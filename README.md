# Star Ambient Stream

Stream 24/7 de audio ambiental procedural con overlays HUD en tiempo real.
Transmite a YouTube Live desde un servidor Linux con Docker.

## Qué hace

- **6 escenas de video** que rotan cada 2 horas (orbital, nieve, bosque, submarino, montaña, desierto)
- **Audio 100% procedural** generado en tiempo real con Python — no hay loops de audio
- Cada escena tiene un perfil de audio radicalmente distinto (acordes, reverb, capas de ambiente)
- El audio evoluciona en tiempo real según:
  - Hora del día en Santiago de Chile (6 fases: madrugada → noche)
  - Clima en vivo via Open-Meteo
  - Dispositivos activos en red local (netalertx SQLite)
  - Carga del servidor (CPU, RAM)
  - Viewers en el directo (YouTube API)
- **HUD overlay en video**: hora, escena activa, CPU/RAM, dispositivos, viewers
- Flash de notificación al cambiar escena

## Arquitectura

```
audio_engine.py  →  stdout PCM s16le  →  ffmpeg stdin
                                              ↓
                              video .mp4 (loop) + drawtext HUD
                                              ↓
                                    RTMP → YouTube Live
```

`stream_manager.py` orquesta todo: crea el broadcast en YouTube, rota escenas, monitorea procesos y reinicia si alguno muere.

## Escenas y perfiles de audio

| Escena | Archivo | Audio |
|--------|---------|-------|
| `orbital` | `scenes/orbital.mp4` | Dark station — drones profundos, radio walkie-talkie, ventiladores |
| `nieve` | `scenes/nieve.mp4` | Arctic outpost — cristalino, viento intenso, reverb amplio |
| `bosque` | `scenes/bosque.mp4` | Forest station — cálido, orgánico, agua fluyendo |
| `submarina` | `scenes/submarina.mp4` | Deep sea base — graves extremos, burbujas, sonar lento |
| `montana` | `scenes/montana.mp4` | Mountain base — viento fuerte, acordes elevados |
| `desierto` | `scenes/desierto.mp4` | Desert heat — seco, mínimo reverb, disperso |

## Setup

### 1. Credenciales de YouTube

1. Crea un proyecto en [Google Cloud Console](https://console.cloud.google.com)
2. Habilita YouTube Data API v3
3. Descarga las credenciales OAuth como `client_secrets.json` (ver `client_secrets.json.example`)
4. Genera el token (desde una máquina con navegador):
   ```bash
   pip install google-auth-oauthlib
   python auth.py
   ```
5. Copia `token.pickle` al servidor

### 2. Videos de escena

Coloca tus videos MP4 en la carpeta `scenes/`:
```
scenes/orbital.mp4
scenes/nieve.mp4
scenes/bosque.mp4
scenes/submarina.mp4
scenes/montana.mp4
scenes/desierto.mp4
```

Deben ser videos cortos (8–30s) que ffmpeg loopea con `-stream_loop -1`.
Generados con Veo3 o cualquier herramienta de video.

### 3. netalertx (opcional)

Para que el conteo de dispositivos en red funcione, el contenedor necesita acceso a:
```
/opt/stacks/netalertx/data/db/app.db
```
Agrega el volumen en `docker-compose.yml` si lo tienes corriendo.

### 4. Levantar

```bash
docker compose build
docker compose up -d
```

## Dependencias Python

Instaladas en el Dockerfile:
- `google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`
- `numpy`, `scipy` — síntesis vectorizada
- `pedalboard` (Spotify) — efectos de audio: reverb, chorus, compressor, filtros

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `audio_engine.py` | Generador de audio procedural. Escribe PCM s16le a stdout. |
| `stream_manager.py` | Orquestador: YouTube API, rotación de escenas, monitoreo de procesos. |
| `hud_updater.py` | Actualiza `/tmp/star_hud{1,2,3}.txt` cada segundo para el overlay de ffmpeg. |
| `generate_weather_ambient.py` | Utilidad standalone para generar WAV de ambiente (no usado en streaming). |
| `check_status.py` | Diagnóstico rápido del estado del stream. |
| `auth.py` | Genera `token.pickle` via OAuth. Ejecutar una vez. |
| `Dockerfile` | Imagen basada en `python:3.12-slim` con ffmpeg y todas las deps. |

## Comunicación entre procesos (IPC via /tmp)

| Archivo | Escrito por | Leído por |
|---------|-------------|-----------|
| `/tmp/star_scene.txt` | `stream_manager` | `audio_engine` |
| `/tmp/star_state.txt` | `audio_engine` | `hud_updater` |
| `/tmp/star_viewers.txt` | `stream_manager` (poll API) | `audio_engine`, `hud_updater` |
| `/tmp/star_hud{1,2,3}.txt` | `hud_updater` | `ffmpeg` (drawtext reload) |

## Configuración rápida

Variables en `stream_manager.py`:
```python
SCENE_DURATION = 2 * 3600    # segundos por escena (default: 2h)
CYCLE_DURATION = 11.5 * 3600 # duración del episodio antes de crear nuevo broadcast
```

Variables en `audio_engine.py`:
- `SCENE_PROFILES` — perfil de audio por escena (reverb, sub, gains de capas)
- `CHORD_SETS` — acordes por tipo de escena
- `day_phase()` — brightness/density/energy según hora del día
