#!/usr/bin/env python3
"""
stream_manager.py — Orchestrador del stream 24/7 con rotación de escenas.

Ciclo de un episodio: 11.5h → pausa 90s → siguiente episodio.
Dentro de cada episodio, las escenas rotan cada SCENE_DURATION segundos.
Al rotar escena: ffmpeg + audio_engine se reinician con el nuevo video.
La escena activa se publica en /tmp/star_scene.txt para audio_engine.
"""

import os, sys, time, pickle, subprocess, threading, json
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError

import hud_updater

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
SCENE_DURATION  = 45 * 60        # 45 minutos por escena (14 escenas × 45min = 10.5h)
# 11.5h: justo bajo el límite de 12h de VOD de YouTube → cada episodio se archiva
# completo. La rotación de episodios depende de la quota API (ver fix en chat_listener).
CYCLE_DURATION  = 11.5 * 3600    # duracion total del episodio

TOKEN_PICKLE  = "token.pickle"
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


def update_broadcast_metadata(youtube, broadcast_id, episode, scene, scheduled_start):
    """Actualiza titulo y descripcion del broadcast en YouTube."""
    try:
        new_title = make_title(episode, scene)
        new_desc  = make_description(scene)
        youtube.liveBroadcasts().update(
            part='snippet',
            body={
                'id': broadcast_id,
                'snippet': {
                    'title': new_title,
                    'description': new_desc,
                    'scheduledStartTime': scheduled_start,
                },
            }
        ).execute()
        print(f"[INFO] Titulo actualizado -> {new_title}", flush=True)
    except Exception as e:
        print(f"[WARN] No se pudo actualizar titulo: {e}", flush=True)


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


def get_youtube_client():
    creds = None
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[INFO] Renovando credenciales...")
            creds.refresh(Request())
            with open(TOKEN_PICKLE, 'wb') as f:
                pickle.dump(creds, f)
        else:
            print("[ERROR] Credenciales invalidas.")
            sys.exit(1)
    return build('youtube', 'v3', credentials=creds)


def create_broadcast(youtube, title, description):
    print(f"[INFO] Creando broadcast: '{title}'")
    scheduled_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 120))
    body = {
        'snippet': {
            'title': title, 'description': description,
            'scheduledStartTime': scheduled_start,
        },
        'status': {'privacyStatus': 'public', 'selfDeclaredCreativeCommons': False},
        'contentDetails': {
            'enableAutoStart': True, 'enableAutoStop': True,
            'monitorStream': {'enableMonitorStream': False},
        },
    }
    broadcast_id = youtube.liveBroadcasts().insert(part='snippet,status,contentDetails', body=body).execute()['id']
    return broadcast_id, scheduled_start


def create_stream(youtube, title):
    print("[INFO] Creando live stream...")
    body = {
        'snippet': {'title': f"{title} -- Stream"},
        'cdn': {'frameRate': '30fps', 'ingestionType': 'rtmp', 'resolution': '720p'},
    }
    resp = youtube.liveStreams().insert(part='snippet,cdn', body=body).execute()
    return resp['id'], resp['cdn']['ingestionInfo']['streamName'], resp['cdn']['ingestionInfo']['ingestionAddress']


def bind_broadcast(youtube, broadcast_id, stream_id):
    youtube.liveBroadcasts().bind(id=broadcast_id, part='id,contentDetails', streamId=stream_id).execute()


def poll_viewers(youtube, broadcast_id, stop_event):
    while not stop_event.is_set():
        try:
            resp  = youtube.liveBroadcasts().list(part='statistics', id=broadcast_id).execute()
            items = resp.get('items', [])
            if items:
                viewers = int(items[0].get('statistics', {}).get('concurrentViewers', 0))
                with open(VIEWERS_FILE, 'w') as f:
                    f.write(str(viewers))
        except Exception as e:
            print(f"[WARN] Viewers: {e}", file=sys.stderr)
        stop_event.wait(60)


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


def start_ffmpeg_audio(video_path, ingestion_url, stream_name):
    """Lanza audio_engine + ffmpeg. Devuelve (audio_proc, ffmpeg_proc)."""
    rtmp = f"{ingestion_url}/{stream_name}"
    print(f"[INFO] Iniciando ffmpeg -> {ingestion_url} | video={video_path}", flush=True)

    for f in (HUD1, HUD2, HUD3):
        if not os.path.exists(f):
            open(f, 'w').write(' ')

    vf = ",".join([
        "fps=30",
        build_drawtext(HUD1, "h-88", fontsize=21, color='0x00EEFF'),
        build_drawtext(HUD2, "h-60", fontsize=18, color='0xAAFFAA'),
        build_drawtext(HUD3, "h-32", fontsize=17, color='0xFFDD44'),
    ])
    filter_complex = f"[0:v]{vf}[base];[base][2:v]overlay=W-w-15:H-h-15[v]"

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
            rtmp,
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


def run_episode(youtube, episode):
    _init_scene = SCENE_ORDER[0]
    title       = make_title(episode, _init_scene)
    description = make_description(_init_scene)

    broadcast_id, scheduled_start = create_broadcast(youtube, title, description)
    stream_id, stream_name, ingestion_url = create_stream(youtube, title)
    bind_broadcast(youtube, broadcast_id, stream_id)
    print(f"[INFO] Broadcast ID: {broadcast_id}", flush=True)

    hud_updater.start(youtube=youtube, broadcast_id=broadcast_id)

    stop_viewers = threading.Event()
    viewers_thread = threading.Thread(
        target=poll_viewers, args=(youtube, broadcast_id, stop_viewers), daemon=True
    )
    viewers_thread.start()

    episode_start  = time.time()
    scene_idx      = 0
    scene_start    = time.time()
    current_scene  = SCENE_ORDER[scene_idx % len(SCENE_ORDER)]
    crash_count    = 0

    write_scene(current_scene)
    write_crash_count(crash_count)
    # update_broadcast_metadata(youtube, broadcast_id, episode, current_scene, scheduled_start) # Redundant at startup, prevents API timeouts
    print(f"[INFO] Escena inicial: {current_scene}", flush=True)
    audio_proc, ffmpeg_proc = start_ffmpeg_audio(
        SCENE_FILES[current_scene], ingestion_url, stream_name
    )

    try:
        while time.time() - episode_start < CYCLE_DURATION:
            now = time.time()

            # ── Comando de chat (cambia / escena X / reinicia) ──
            chat_cmd, chat_val = read_chat_command()
            if chat_cmd == 'goto' and chat_val in SCENE_FILES:
                new_scene = chat_val
                if new_scene != current_scene:
                    scene_idx = SCENE_ORDER.index(new_scene) if new_scene in SCENE_ORDER else scene_idx
                    current_scene = new_scene
                    print(f"[CHAT] Saltando a escena -> {current_scene}", flush=True)
                    write_scene(current_scene)
                    update_broadcast_metadata(youtube, broadcast_id, episode, current_scene, scheduled_start)
                    kill_procs(ffmpeg_proc, audio_proc)
                    time.sleep(2)
                    audio_proc, ffmpeg_proc = start_ffmpeg_audio(
                        SCENE_FILES[current_scene], ingestion_url, stream_name
                    )
                    scene_start = now
                    crash_count = 0
                    write_crash_count(crash_count)

            elif chat_cmd == 'reinicia':
                # Reinicia el pipeline de video sin cambiar escena (fix lag/cortes)
                print("[CHAT] Reiniciando ffmpeg por comando de chat...", flush=True)
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(2)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(
                    SCENE_FILES[current_scene], ingestion_url, stream_name
                )

            # ── Rotacion automatica de escena ──
            elif now - scene_start >= SCENE_DURATION:
                scene_idx     = (scene_idx + 1) % len(SCENE_ORDER)
                current_scene = SCENE_ORDER[scene_idx]
                print(f"[INFO] Rotando escena -> {current_scene}", flush=True)
                write_scene(current_scene)
                # No actualizamos metadata del broadcast en cada rotación automática:
                # ahorra ~50 unidades de quota cada 45min. El título solo refleja la
                # escena inicial del episodio; el HUD del video ya muestra la escena actual.
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(2)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(
                    SCENE_FILES[current_scene], ingestion_url, stream_name
                )
                scene_start = now
                crash_count = 0
                write_crash_count(crash_count)

            # ── Monitoreo de procesos ──
            ffmpeg_ret = ffmpeg_proc.poll()
            audio_ret  = audio_proc.poll()
            if ffmpeg_ret is not None or audio_ret is not None:
                reason  = "ffmpeg" if ffmpeg_ret is not None else "audio_engine"
                backoff = _CRASH_BACKOFF[min(crash_count, len(_CRASH_BACKOFF) - 1)]
                crash_count += 1
                write_crash_count(crash_count)
                print(f"[WARN] {reason} termino (crash #{crash_count}) con codigos (ffmpeg:{ffmpeg_ret}, audio:{audio_ret}). Backoff {backoff}s...", flush=True)
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(backoff)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(
                    SCENE_FILES[current_scene], ingestion_url, stream_name
                )

            time.sleep(2)  # polling cada 2s

    except KeyboardInterrupt:
        print("[INFO] Interrupcion manual.")
    finally:
        stop_viewers.set()
        kill_procs(ffmpeg_proc, audio_proc)


def main():
    youtube = get_youtube_client()
    episode = load_episode()
    while True:
        try:
            print(f"\n[INFO] --- Episodio {episode} ---", flush=True)
            run_episode(youtube, episode)
            episode += 1
            save_episode(episode)
            print("[INFO] Pausa 90s...", flush=True)
            time.sleep(90)
        except HttpError as e:
            print(f"[ERROR] YouTube API: {e}")
            time.sleep(300)
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(300)


if __name__ == "__main__":
    main()
