#!/usr/bin/env python3
"""
stream_manager.py — Orchestrador del stream 24/7 con rotación de escenas.

Arquitectura: RTMP key fija desde YouTube Studio — sin API, sin quota.
Ciclo: 11h por episodio → 60s pausa → nuevo episodio (YouTube crea VOD nuevo).
Escenas rotan cada SCENE_DURATION segundos dentro del episodio.
"""

import os, sys, time, subprocess, threading, json

import hud_updater

RTMP_KEY       = "yfh2-rb5k-10b6-akw7-9thu"
RTMP_URL       = f"rtmp://a.rtmp.youtube.com/live2/{RTMP_KEY}"

# ── Escenas: orden y duración ──────────────────────────────
SCENE_ORDER = ['electrica', 'montana', 'orbital', 'nieve', 'bosque', 'submarina', 'desierto', 'volcanica', 'reactor', 'tormenta', 'array_suiza', 'latam_noche', 'scp_exterior', 'scp_contencion']
SCENE_FILES = {
    'orbital':       '/app/scenes/orbital_2h.mp4',
    'nieve':         '/app/scenes/nieve_2h.mp4',
    'bosque':        '/app/scenes/bosque_2h.mp4',
    'submarina':     '/app/scenes/submarina_2h.mp4',
    'montana':       '/app/scenes/montana_2h.mp4',
    'desierto':      '/app/scenes/desierto_2h.mp4',
    'electrica':     '/app/scenes/electrica_2h.mp4',
    'volcanica':     '/app/scenes/volcanica_2h.mp4',
    'reactor':       '/app/scenes/reactor_2h.mp4',
    'tormenta':      '/app/scenes/tormenta_2h.mp4',
    'array_suiza':   '/app/scenes/array_suiza_2h.mp4',
    'latam_noche':   '/app/scenes/latam_noche_2h.mp4',
    'scp_exterior':  '/app/scenes/scp_exterior_2h.mp4',
    'scp_contencion':'/app/scenes/scp_contencion_2h.mp4',
}
SCENE_DURATION  = 45 * 60        # 45 minutos por escena
CYCLE_DURATION  = 11 * 3600      # 11h → corte limpio del episodio
EPISODE_PAUSE   = 60             # 60s fuera del aire → YouTube crea nuevo VOD

AUDIO_ENGINE  = "audio_supervisor.py"
VIEWERS_FILE  = "/tmp/star_viewers.txt"
SCENE_FILE    = "/tmp/star_scene.txt"
CRASH_FILE    = "/tmp/star_crashes.txt"
FONT          = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
HUD1          = "/tmp/star_hud1.txt"
HUD2          = "/tmp/star_hud2.txt"
HUD3          = "/tmp/star_hud3.txt"
LOGO_FILE     = "/app/logo.png"
COMMAND_FILE  = "/tmp/star_chat_cmd.txt"   # escrito por chat_listener.py

# Ruta estable para persistir episodio (sobrevive reboots)
EPISODE_FILE  = '/app/episode.json'

# Backoff en segundos segun numero de crashes consecutivos
_CRASH_BACKOFF = [1, 2, 5, 10, 20, 30]


# ── Titulos dinamicos por escena ──────────────────────────────────────────
SCENE_LABELS = {
    'montana':   ('Mountain Base',  'Refugio de montana -- viento, lluvia y crujidos lejanos'),
    'orbital':   ('Orbital Station', 'Estacion orbital -- drones, radio walkie y ventiladores'),
    'nieve':     ('Arctic Outpost',  'Avanzada artica -- blizzard, hielo y crujidos estructurales'),
    'bosque':    ('Forest Station',   'Estacion forestal -- lluvia, agua fluyendo y campanas'),
    'submarina': ('Deep Sea Base',    'Base submarina -- graves extremos, sonar y burbujas'),
    'desierto':  ('Desert Heat',      'Calor del desierto -- viento seco y campanas metalicas'),
    'electrica': ('Electric Field',    'Campo electrico -- zumbido 50/60Hz, EMF y crujidos metalicos'),
    'volcanica':     ('Geothermal Outpost',      'Base geotermal -- magma, vapor y sismos'),
    'reactor':       ('Nuclear Reactor',          'Reactor nuclear -- Geiger, refrigerante y valvulas'),
    'tormenta':      ('Storm Bunker',              'Bunker de tormenta -- truenos, lluvia y relampagos'),
    'array_suiza':   ('Signal Array — Alps',       'Array de telescopios en los Alpes -- viento, telemetria y EMF sub-sonico'),
    'latam_noche':   ('LATAM Noche',               'Azotea latinoamericana de noche -- ciudad, perros y motos lejanas'),
    'scp_exterior':  ('SCP Foundation — Exterior', 'Exterior SCP -- grillos, hojas, zumbido de lampara y EMF'),
    'scp_contencion':('SCP Containment Platform',  'Plataforma de contencion SCP -- cadenas, niebla y bosque nocturno'),
}

_BASE_DESC = (
    "🔊 Audio ambiental 100% procedural — generado en tiempo real con Python.\n"
    "Sin samples ni loops. Cada segundo es único, irrepetible.\n\n"
    "El audio muta en vivo según: 🌡️ clima real · 📡 dispositivos en red · 🕐 hora del día · 👥 viewers\n"
    "Tech: Python 3.12 · numpy · scipy · pedalboard · ffmpeg · Docker\n\n"
    "🗺️ ESCENAS DISPONIBLES\n"
    "  ⚡ Eléctrica   — Torre de alta tensión, corona EMF, chichirreo de ionización\n"
    "  🏔️ Montaña     — Viento alpino, harpa eólica, crujidos glaciares, avalanchas\n"
    "  🛸 Orbital     — Propulsores RCS, beeps de telemetría, vibración de casco\n"
    "  ❄️ Nieve       — Blizzard ártico, crujidos de hielo, silbidos de cable tensado\n"
    "  🌲 Bosque      — Lluvia, copa de árboles multicapa, agua fluyendo, campanas\n"
    "  🌊 Submarina   — Graves extremos, sonar activo, gemido de casco, burbujas\n"
    "  🌵 Desierto    — Viento seco, torbellinos de arena, campanas metálicas\n"
    "  🌋 Volcánica   — Magma sub-bass, vapor, sismos\n"
    "  ☢️  Reactor     — Contador Geiger, refrigerante, válvulas de presión\n"
    "  ⛈️  Tormenta    — Truenos físicos, lluvia pesada, relámpagos\n"
    "  📡 Array Suiza — Telescopios en los Alpes, telemetría, EMF\n"
    "  🌆 LATAM Noche — Azotea nocturna, ciudad, perros, motos\n"
    "  🏚️ SCP Exterior — Logo SCP, hiedra, grillos, zumbido eléctrico\n"
    "  ⛓️  SCP Contencion — Plataforma de cadenas, niebla, bosque\n\n"
    "═══════════════════════════════════════\n"
    "📋 COMANDOS DE CHAT (solo propietario / moderadores)\n"
    "═══════════════════════════════════════\n\n"
    "CONTROL DE ESCENA\n"
    "  cambia / !skip          → Rota a la siguiente escena\n"
    "  orbital / nieve / ...   → Salta directo a esa escena\n"
    "  escena bosque / ...     → Idem con prefijo opcional\n\n"
    "INFORMACIÓN\n"
    "  !status                 → Escena actual + siguiente + viewers + crashes\n"
    "  !escenas                → Lista todas las escenas disponibles\n"
    "  !siguiente              → Qué escena viene después (sin cambiarla)\n"
    "  !hora                   → Hora actual del servidor (Santiago)\n"
    "  !viewers                → Viewers en vivo\n"
    "  !crash                  → Contador de crashes del episodio\n\n"
    "CONTROL DEL STREAM\n"
    "  !reinicia               → Reinicia FFmpeg sin cortar episodio (fix lag)\n\n"
    "⚠️ Los comandos solo funcionan si eres el propietario del canal o moderador.\n"
    "Los mensajes del resto del chat son completamente ignorados por el bot.\n"
)


def make_title(episode, scene):
    label, _ = SCENE_LABELS.get(scene, ('Deep Space', ''))
    return f"Deep Space Outpost -- {label} | 24/7 Ambient Sci-Fi EP {episode}"


def make_description(scene):
    _, scene_desc = SCENE_LABELS.get(scene, ('', 'Ambiente procedural'))
    return f"▶️ Escena actual: {scene_desc}\n\n" + _BASE_DESC


def load_episode():
    try:
        return json.load(open(EPISODE_FILE))['episode']
    except Exception:
        return 1

def save_episode(n):
    try:
        json.dump({'episode': n}, open(EPISODE_FILE, 'w'))
    except Exception:
        pass


def build_drawtext(textfile, y_expr, fontsize, color='white', alpha='0.85'):
    return (
        f"drawtext="
        f"fontfile={FONT}:"
        f"textfile={textfile}:"
        f"reload=1:"
        f"fix_bounds=1:"
        f"expansion=none:"
        f"fontcolor={color}@{alpha}:"
        f"fontsize={fontsize}:"
        f"x=20:"
        f"y={y_expr}:"
        f"shadowcolor=black@0.9:"
        f"shadowx=2:"
        f"shadowy=2"
    )


def write_scene(scene):
    """Publica la escena activa para que audio_engine cambie su perfil."""
    try:
        with open(SCENE_FILE, 'w') as f:
            f.write(scene)
    except Exception:
        pass


def read_chat_command():
    """Lee y consume el comando de chat. Devuelve (cmd, value) o (None, None)."""
    try:
        if not os.path.exists(COMMAND_FILE):
            return None, None
        with open(COMMAND_FILE) as f:
            obj = json.load(f)
        # Consumir el archivo (borrarlo para no releerlo)
        try:
            os.remove(COMMAND_FILE)
        except Exception:
            pass
        # Ignorar comandos con mas de 30s de antiguedad
        if time.time() - obj.get('ts', 0) > 30:
            return None, None
        return obj.get('cmd'), obj.get('value')
    except Exception:
        return None, None


def write_crash_count(count):
    """Publica crashes consecutivos para el HUD sin afectar el stream si falla."""
    try:
        with open(CRASH_FILE, 'w') as f:
            f.write(str(count))
    except Exception:
        pass


def start_ffmpeg_audio(video_path, _ingestion_url=None, _stream_name=None):
    """Lanza audio_engine + ffmpeg hacia RTMP_URL fija."""
    print(f"[INFO] Iniciando ffmpeg → {RTMP_URL} | video={video_path}", flush=True)

    for f in (HUD1, HUD2, HUD3):
        if not os.path.exists(f):
            open(f, 'w').write(' ')

    vf = ",".join([
        "fps=30",
        build_drawtext(HUD1, "h-88", fontsize=21, color='0x00EEFF'),
        build_drawtext(HUD2, "h-60", fontsize=18, color='0xAAFFAA'),
        build_drawtext(HUD3, "h-32", fontsize=17, color='0xFFDD44'),
    ])
    filter_complex = f"[0:v]{vf}[base];[2:v]scale=220:-1[logo];[base][logo]overlay=18:18[v]"

    audio_proc = subprocess.Popen(
        ['python', '-u', AUDIO_ENGINE],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
    )

    ffmpeg_proc = subprocess.Popen(
        [
            'ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'warning',
            # Video input
            '-re', '-stream_loop', '-1', '-fflags', '+genpts', '-i', video_path,
            # Audio input desde pipe del supervisor
            '-thread_queue_size', '4096',
            '-f', 's16le', '-ar', '44100', '-ac', '1', '-i', 'pipe:0',
            # Logo transparente para tapar watermark del video fuente
            '-loop', '1', '-i', LOGO_FILE,
            '-filter_complex', filter_complex,
            '-map', '[v]', '-map', '1:a',
            # Video encoding
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-profile:v', 'baseline',
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            '-g', '60',
            '-keyint_min', '60',
            '-sc_threshold', '0',
            '-b:v', '1800k',
            '-maxrate', '1800k',
            '-bufsize', '3600k',
            # Audio encoding
            '-af', 'aresample=async=1000:first_pts=0',
            '-c:a', 'aac',
            '-b:a', '96k',
            '-ar', '44100',
            '-ac', '2',
            # RTMP output con reconexion automatica ante drops de red
            '-flvflags', 'no_duration_filesize',
            '-reconnect', '1',
            '-reconnect_at_eof', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '15',
            '-f', 'flv',
            RTMP_URL,
        ],
        stdin=audio_proc.stdout,
        stderr=sys.stderr,
    )
    audio_proc.stdout.close()
    return audio_proc, ffmpeg_proc


def kill_procs(*procs):
    for p in procs:
        try:
            p.terminate()
            p.wait(timeout=8)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def run_episode(episode, scene_idx_start=0):
    """Corre un episodio. Retorna el índice de escena para el siguiente."""
    scene_idx     = scene_idx_start
    scene_start   = time.time()
    episode_start = time.time()
    current_scene = SCENE_ORDER[scene_idx % len(SCENE_ORDER)]
    crash_count   = 0

    write_scene(current_scene)
    write_crash_count(crash_count)
    print(f"[INFO] --- Episodio {episode} | Escena: {current_scene} | {CYCLE_DURATION//3600}h ---", flush=True)

    audio_proc, ffmpeg_proc = start_ffmpeg_audio(SCENE_FILES[current_scene])

    try:
        while time.time() - episode_start < CYCLE_DURATION:
            now = time.time()

            chat_cmd, chat_val = read_chat_command()
            if chat_cmd == 'goto' and chat_val in SCENE_FILES:
                if chat_val != current_scene:
                    scene_idx     = SCENE_ORDER.index(chat_val) if chat_val in SCENE_ORDER else scene_idx
                    current_scene = chat_val
                    print(f"[CHAT] Saltando a escena → {current_scene}", flush=True)
                    write_scene(current_scene)
                    kill_procs(ffmpeg_proc, audio_proc)
                    time.sleep(2)
                    audio_proc, ffmpeg_proc = start_ffmpeg_audio(SCENE_FILES[current_scene])
                    scene_start = now
                    crash_count = 0
                    write_crash_count(crash_count)

            elif chat_cmd == 'reinicia':
                print("[CHAT] Reiniciando ffmpeg...", flush=True)
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(2)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(SCENE_FILES[current_scene])

            elif now - scene_start >= SCENE_DURATION:
                scene_idx     = (scene_idx + 1) % len(SCENE_ORDER)
                current_scene = SCENE_ORDER[scene_idx]
                print(f"[INFO] Rotando escena → {current_scene}", flush=True)
                write_scene(current_scene)
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(2)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(SCENE_FILES[current_scene])
                scene_start = now
                crash_count = 0
                write_crash_count(crash_count)

            ffmpeg_ret = ffmpeg_proc.poll()
            audio_ret  = audio_proc.poll()
            if ffmpeg_ret is not None or audio_ret is not None:
                reason  = "ffmpeg" if ffmpeg_ret is not None else "audio_engine"
                backoff = _CRASH_BACKOFF[min(crash_count, len(_CRASH_BACKOFF) - 1)]
                crash_count += 1
                write_crash_count(crash_count)
                print(f"[WARN] {reason} cayó (crash #{crash_count}), backoff {backoff}s...", flush=True)
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(backoff)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(SCENE_FILES[current_scene])

            time.sleep(2)

    except KeyboardInterrupt:
        print("[INFO] Interrupcion manual.", flush=True)
    finally:
        kill_procs(ffmpeg_proc, audio_proc)

    print(f"[INFO] Episodio {episode} completado.", flush=True)
    return (scene_idx + 1) % len(SCENE_ORDER)


def main():
    hud_updater.start(youtube=None, broadcast_id=None)

    episode   = load_episode()
    scene_idx = 0

    while True:
        scene_idx = run_episode(episode, scene_idx)
        episode  += 1
        save_episode(episode)
        print(f"[INFO] Pausa {EPISODE_PAUSE}s antes de EP {episode}...", flush=True)
        time.sleep(EPISODE_PAUSE)


if __name__ == "__main__":
    main()
