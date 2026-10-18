#!/usr/bin/env python3
"""
stream_manager.py — Orquestador del stream 24/7 con rotación de escenas y ciclo de episodios.

Arquitectura: stream RTMP fijo + YouTube API para crear/bindear broadcasts.
Ciclo: 11h por episodio → 120s pausa (VOD processing) → nuevo episodio/broadcast automático.
Escenas rotan cada SCENE_DURATION segundos dentro del episodio.
"""

import os, sys, time, subprocess, threading, json, pickle
from datetime import datetime, timedelta, timezone

import hud_updater
try:
    from thumbnail_generator import render_thumbnail
except ImportError:
    render_thumbnail = None
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

RTMP_KEY       = os.getenv("RTMP_KEY", "").strip()
RTMP_URL       = f"rtmp://a.rtmp.youtube.com/live2/{RTMP_KEY}"

# ── Escenas: orden y duración ──────────────────────────────
SCENE_ORDER = [
    'electrica', 'montana', 'orbital', 'nieve', 'bosque',
    'submarina', 'desierto', 'volcanica', 'reactor', 'tormenta',
    'array_suiza', 'latam_noche', 'scp_exterior', 'scp_contencion'
]
SCENE_FILES = {
    'veo3_manana':   '/app/scenes/veo3_manana.mp4',
    'veo3_tarde':    '/app/scenes/veo3_tarde.mp4',
    'veo3_noche':    '/app/scenes/veo3_noche.mp4',
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

CAFE_SCENE_ORDER = ['veo3_manana', 'veo3_tarde', 'veo3_noche']
OUTPOST_SCENE_ORDER = [
    'orbital', 'nieve', 'scp_contencion', 'submarina',
    'array_suiza', 'tormenta', 'reactor', 'latam_noche',
    'montana', 'bosque', 'desierto', 'volcanica'
]

def get_current_scene_for_time():
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("America/Santiago")
    except Exception:
        tz = None
    now = datetime.now(tz) if tz else datetime.now()
    h = now.hour
    if 6 <= h < 13:
        return 'veo3_manana'
    elif 13 <= h < 20:
        return 'veo3_tarde'
    else:
        return 'veo3_noche'

SCENE_DURATION  = 45 * 60         # 45 minutos por escena
CYCLE_DURATION  = int(11 * 3600)  # 11h → corte limpio para archivar VOD antes de 12h
EPISODE_PAUSE   = 120             # 2 min fuera del aire → YouTube asienta el VOD y libera RTMP

AUDIO_ENGINE  = "audio_supervisor.py"
VIEWERS_FILE  = "/tmp/star_viewers.txt"
SCENE_FILE    = "/tmp/star_scene.txt"
CRASH_FILE    = "/tmp/star_crashes.txt"
FONT          = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
HUD1          = "/tmp/star_hud1.txt"
HUD2          = "/tmp/star_hud2.txt"
HUD3          = "/tmp/star_hud3.txt"
LOGO_FILE     = "/app/logo_jackstar_circle.png"   # Jackstar — top-left
LOGO_FILE2    = "/app/logo_kika_circle.png"        # Kika — top-left junto a jackstar
LOGO_FILE3    = "/app/logo_king.png"               # Rey — bottom-right original
COMMAND_FILE  = "/tmp/star_chat_cmd.txt"           # escrito por chat_listener.py
TOKEN_PICKLE  = "/app/token.pickle"
BROADCAST_STATE_FILE = "/app/broadcast_state.json"

# Ruta estable para persistir episodio (sobrevive reboots)
EPISODE_FILE  = '/app/episode.json'

# Backoff en segundos segun numero de crashes consecutivos
_CRASH_BACKOFF = [1, 2, 5, 10, 20, 30]
_BROADCAST_RETRY_INITIAL = 15 * 60
_BROADCAST_RETRY_MAX = 2 * 60 * 60


def parse_iso_time(iso_str):
    """Parsea una cadena ISO 8601 a timestamp POSIX."""
    if not iso_str:
        return None
    try:
        iso_clean = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_clean)
        return dt.timestamp()
    except Exception:
        return None


def get_youtube_client():
    """Carga credenciales OAuth y crea el cliente YouTube."""
    creds = None
    if not os.path.exists(TOKEN_PICKLE) or os.path.getsize(TOKEN_PICKLE) == 0:
        raise RuntimeError("token.pickle vacío o no encontrado")
    with open(TOKEN_PICKLE, "rb") as f:
        creds = pickle.load(f)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_PICKLE, "wb") as f:
            pickle.dump(creds, f)
    return build("youtube", "v3", credentials=creds)


def get_fixed_stream_id(youtube):
    """Busca el live stream asociado al RTMP key persistente."""
    resp = youtube.liveStreams().list(
        part="id,snippet,cdn,status",
        mine=True,
        maxResults=50,
    ).execute()
    for item in resp.get("items", []):
        stream_name = item.get("cdn", {}).get("ingestionInfo", {}).get("streamName")
        if stream_name == RTMP_KEY:
            return item["id"]
    raise RuntimeError("No se encontró el live stream del RTMP key fijo.")


def create_broadcast(youtube, episode, scene):
    """Crea un broadcast nuevo para el episodio actual."""
    start_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    body = {
        "snippet": {
            "title": make_title(episode, scene),
            "description": make_description(scene),
            "scheduledStartTime": start_at,
            "tags": [
                "sci fi ambient", "scifi ambient", "space ambient", "dark ambient",
                "ambient live", "24 7 live stream", "study music ambient",
                "sleep ambient", "focus ambient", "deep space ambience",
                "space station ambient", "futuristic ambient", "background for studying",
                "relaxing ambient stream", "generative ambient"
            ]
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
        "contentDetails": {
            "enableAutoStart": True,
            "enableAutoStop": True,
            "recordFromStart": True,
            "startWithSlate": False,
            "monitorStream": {
                "enableMonitorStream": True,
            },
        },
    }
    resp = youtube.liveBroadcasts().insert(
        part="snippet,status,contentDetails",
        body=body,
    ).execute()
    broadcast_id = resp["id"]
    print(f"[INFO] Broadcast creado: {broadcast_id} | Titulo: {body['snippet']['title']}", flush=True)
    return resp


def upload_thumbnail(youtube, broadcast_id, thumbnail_path):
    """Sube la miniatura para el broadcast/video."""
    if not os.path.exists(thumbnail_path):
        print(f"[WARN] No se encontro archivo de miniatura en {thumbnail_path}", flush=True)
        return
    try:
        request = youtube.thumbnails().set(
            videoId=broadcast_id,
            media_body=MediaFileUpload(thumbnail_path)
        )
        request.execute()
        print(f"[INFO] Miniatura subida con éxito para broadcast {broadcast_id}", flush=True)
    except Exception as e:
        print(f"[WARN] Fallo la subida de miniatura: {e}", flush=True)


def list_broadcasts(youtube, status="all"):
    """Lista broadcasts del canal."""
    resp = youtube.liveBroadcasts().list(
        part="id,snippet,status,contentDetails",
        broadcastStatus=status,
        broadcastType="all",
        maxResults=50,
    ).execute()
    return resp.get("items", [])


def get_broadcast(youtube, broadcast_id):
    """Obtiene un broadcast puntual."""
    resp = youtube.liveBroadcasts().list(
        part="id,snippet,status,contentDetails",
        id=broadcast_id,
    ).execute()
    items = resp.get("items", [])
    return items[0] if items else None


def load_broadcast_state():
    try:
        with open(BROADCAST_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_broadcast_state(episode, scene, broadcast_id):
    try:
        with open(BROADCAST_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "episode": episode,
                "scene": scene,
                "broadcast_id": broadcast_id,
                "updated_at": int(time.time()),
            }, f)
    except Exception:
        pass


def clear_broadcast_state():
    try:
        if os.path.exists(BROADCAST_STATE_FILE):
            os.remove(BROADCAST_STATE_FILE)
    except Exception:
        pass


def delete_broadcast(youtube, broadcast_id):
    """Borra un upcoming duplicado."""
    youtube.liveBroadcasts().delete(id=broadcast_id).execute()
    print(f"[INFO] Broadcast duplicado eliminado: {broadcast_id}", flush=True)


def choose_reusable_broadcast(youtube, episode, scene):
    """Reutiliza el broadcast del episodio si ya existe y no ha expirado."""
    wanted_title = make_title(episode, scene)
    saved = load_broadcast_state()
    candidate_ids = []
    now = time.time()

    # 1. Comprobar si hay broadcasts activos 'live'
    active_live = list_broadcasts(youtube, status="active")
    for chosen_live in active_live:
        start_iso = chosen_live.get("snippet", {}).get("actualStartTime")
        start_ts = parse_iso_time(start_iso)
        if start_ts:
            elapsed = now - start_ts
            if elapsed >= CYCLE_DURATION:
                print(
                    f"[WARN] Broadcast activo {chosen_live['id']} ya cumplió {elapsed/3600:.2f}h "
                    f"(>= {CYCLE_DURATION/3600:.1f}h). Transicionando a COMPLETE...",
                    flush=True,
                )
                try:
                    safe_transition(youtube, chosen_live['id'], "complete")
                except Exception as e:
                    print(f"[ERROR] Error completando broadcast expirado: {e}", flush=True)
                continue  # Descartado porque ya terminó su ciclo
            else:
                print(
                    f"[INFO] Reutilizando live activo del canal: {chosen_live['id']} "
                    f"(antiguedad: {elapsed/3600:.2f}h)",
                    flush=True,
                )
                return chosen_live
        else:
            print(f"[INFO] Reutilizando live activo del canal (sin actualStartTime): {chosen_live['id']}", flush=True)
            return chosen_live

    # 2. Comprobar si hay guardado local correspondiente al mismo episodio
    if saved.get("episode") == episode and saved.get("broadcast_id"):
        candidate_ids.append(saved["broadcast_id"])

    # 3. Comprobar upcoming
    for item in list_broadcasts(youtube, status="upcoming"):
        if item.get("snippet", {}).get("title") == wanted_title:
            candidate_ids.append(item["id"])

    seen = set()
    ordered_ids = []
    for bid in candidate_ids:
        if bid and bid not in seen:
            ordered_ids.append(bid)
            seen.add(bid)

    for bid in ordered_ids:
        item = get_broadcast(youtube, bid)
        if not item:
            continue
        life_cycle = item.get("status", {}).get("lifeCycleStatus")
        if life_cycle in ("complete", "revoked"):
            continue
        print(f"[INFO] Reutilizando broadcast existente: {bid} ({life_cycle})", flush=True)
        return item

    return None


def cleanup_duplicate_upcoming_broadcasts(youtube, episode, scene, keep_broadcast_id):
    """Elimina upcoming duplicados del mismo episodio/titulo."""
    wanted_title = make_title(episode, scene)
    duplicates = []
    for item in list_broadcasts(youtube, status="upcoming"):
        if item.get("snippet", {}).get("title") != wanted_title:
            continue
        if item["id"] == keep_broadcast_id:
            continue
        duplicates.append(item)

    duplicates.sort(key=lambda item: item.get("snippet", {}).get("scheduledStartTime", ""))
    for item in duplicates:
        try:
            delete_broadcast(youtube, item["id"])
        except Exception as e:
            print(f"[WARN] No se pudo borrar duplicado {item['id']}: {e}", flush=True)


def get_or_create_broadcast(youtube, episode, scene, stream_id):
    """Obtiene un broadcast reutilizable o crea uno nuevo con miniatura SVG/PNG única. Retorna (broadcast_id, item)."""
    existing = choose_reusable_broadcast(youtube, episode, scene)
    if existing:
        broadcast_id = existing["id"]
        bind_broadcast(youtube, broadcast_id, stream_id)
        save_broadcast_state(episode, scene, broadcast_id)
        cleanup_duplicate_upcoming_broadcasts(youtube, episode, scene, broadcast_id)
        if render_thumbnail:
            try:
                png_path, _ = render_thumbnail(episode, scene)
                upload_thumbnail(youtube, broadcast_id, png_path)
            except Exception as e:
                print(f"[WARN] Error generando miniatura procedural ({e}), fallback a PNG estatico", flush=True)
                upload_thumbnail(youtube, broadcast_id, "/app/thumbnail_scifi_live_v1.png")
        else:
            upload_thumbnail(youtube, broadcast_id, "/app/thumbnail_scifi_live_v1.png")
        return broadcast_id, existing

    resp = create_broadcast(youtube, episode, scene)
    broadcast_id = resp["id"]
    bind_broadcast(youtube, broadcast_id, stream_id)
    save_broadcast_state(episode, scene, broadcast_id)
    cleanup_duplicate_upcoming_broadcasts(youtube, episode, scene, broadcast_id)
    if render_thumbnail:
        try:
            png_path, _ = render_thumbnail(episode, scene)
            upload_thumbnail(youtube, broadcast_id, png_path)
        except Exception as e:
            print(f"[WARN] Error generando miniatura procedural ({e}), fallback a PNG estatico", flush=True)
            upload_thumbnail(youtube, broadcast_id, "/app/thumbnail_scifi_live_v1.png")
    else:
        upload_thumbnail(youtube, broadcast_id, "/app/thumbnail_scifi_live_v1.png")
    return broadcast_id, resp


def bind_broadcast(youtube, broadcast_id, stream_id):
    """Vincula el broadcast al stream RTMP persistente."""
    try:
        youtube.liveBroadcasts().bind(
            part="id,contentDetails",
            id=broadcast_id,
            streamId=stream_id,
        ).execute()
        print(f"[INFO] Broadcast {broadcast_id} vinculado a stream {stream_id}", flush=True)
    except HttpError as e:
        text = str(e)
        if "redundantTransition" in text or "alreadyBound" in text or "invalidTransition" in text:
            print(f"[INFO] Broadcast {broadcast_id} ya vinculado a stream {stream_id}", flush=True)
        else:
            raise


def get_stream_status(youtube, stream_id):
    """Devuelve estado del feed RTMP visto por YouTube."""
    resp = youtube.liveStreams().list(part="status", id=stream_id).execute()
    items = resp.get("items", [])
    if not items:
        return None, None
    status = items[0].get("status", {})
    return status.get("streamStatus"), status.get("healthStatus", {}).get("status")


def get_broadcast_state(youtube, broadcast_id):
    """Devuelve el lifecycle status del broadcast."""
    resp = youtube.liveBroadcasts().list(
        part="status",
        id=broadcast_id,
    ).execute()
    items = resp.get("items", [])
    if not items:
        return None
    return items[0].get("status", {}).get("lifeCycleStatus")


def safe_transition(youtube, broadcast_id, target_state):
    """Intenta una transición ignorando estados redundantes."""
    try:
        youtube.liveBroadcasts().transition(
            broadcastStatus=target_state,
            id=broadcast_id,
            part="status",
        ).execute()
        print(f"[INFO] Broadcast {broadcast_id} -> {target_state}", flush=True)
        return True
    except HttpError as e:
        text = str(e)
        print(f"[WARN] Transition {target_state} rechazada: {text}", flush=True)
        if "redundantTransition" in text or "invalidTransition" in text:
            return False
        raise


def wait_stream_inactive(youtube, stream_id, timeout=90):
    """Espera a que YouTube deje de ver el feed RTMP como activo."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        stream_status, _ = get_stream_status(youtube, stream_id)
        if stream_status != "active":
            return True
        time.sleep(5)
    return False


def ensure_broadcast_live(youtube, broadcast_id, stream_id, timeout=150):
    """Espera feed activo y empuja el broadcast hasta estado live."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        life_cycle = get_broadcast_state(youtube, broadcast_id)
        stream_status, health = get_stream_status(youtube, stream_id)
        print(
            f"[INFO] YouTube state | broadcast={life_cycle} stream={stream_status} health={health}",
            flush=True,
        )
        if life_cycle == "live":
            return True
        if stream_status == "active":
            if life_cycle == "ready":
                safe_transition(youtube, broadcast_id, "testing")
                time.sleep(3)
                life_cycle = get_broadcast_state(youtube, broadcast_id)
            if life_cycle in ("testing", "liveStarting", "testStarting", "ready"):
                safe_transition(youtube, broadcast_id, "live")
                time.sleep(3)
            if get_broadcast_state(youtube, broadcast_id) == "live":
                return True
        time.sleep(5)
    return False


# ── Titulos dinámicos por escena ──────────────────────────────────────────
SCENE_LABELS = {
    'veo3_manana':   ('Cozy Study', 'Lo-Fi Chill'),
    'veo3_tarde':    ('Greenhouse Botanical', 'Plant Care'),
    'veo3_noche':    ('Midnight Cafe', 'Rainy Night Lo-Fi'),
    'montana':       ('Mountain Base',             'Refugio de montana -- viento, lluvia y crujidos lejanos'),
    'orbital':       ('Orbital Station',            'Estacion orbital -- drones, radio walkie y ventiladores'),
    'nieve':         ('Arctic Outpost',             'Avanzada artica -- blizzard, hielo y crujidos estructurales'),
    'bosque':        ('Forest Station',              'Estacion forestal -- lluvia, agua fluyendo y campanas'),
    'submarina':     ('Deep Sea Base',               'Base submarina -- graves extremos, sonar y burbujas'),
    'desierto':      ('Desert Heat',                 'Calor del desierto -- viento seco y campanas metalicas'),
    'electrica':     ('Electric Field',              'Campo electrico -- zumbido 50/60Hz, EMF y crujidos metalicos'),
    'volcanica':     ('Geothermal Outpost',         'Base geotermal -- magma, vapor y sismos'),
    'reactor':       ('Nuclear Reactor',             'Reactor nuclear -- Geiger, refrigerante y valvulas'),
    'tormenta':      ('Storm Bunker',                 'Bunker de tormenta -- truenos, lluvia y relampagos'),
    'array_suiza':   ('Signal Array — Alps',          'Array de telescopios en los Alpes -- viento, telemetria y EMF sub-sonico'),
    'latam_noche':   ('LATAM Noche',                  'Azotea latinoamericana de noche -- ciudad, perros y motos lejanas'),
    'scp_exterior':  ('SCP Foundation — Exterior',    'Exterior SCP -- grillos, hojas, zumbido de lampara y EMF'),
    'scp_contencion':('SCP Containment Platform',     'Plataforma de contencion SCP -- cadenas, niebla y bosque nocturno'),
}

_BASE_DESC = (
    "24/7 sci-fi ambient live stream for sleep, study, focus and deep work. "
    "Dark space station atmosphere, slow cinematic backgrounds and generative audio running live.\n\n"
    "🏰 COMUNIDAD OFICIAL — DRAKESCRAFT NETWORK (MINECRAFT 1.21.11)\n"
    "Este canal es la transmision oficial del servidor de Minecraft DrakesCraft:\n"
    "• IP Servidor: mc.drakescraft.cl (Java y Bedrock)\n"
    "• Web Oficial: https://drakescraft.cl | https://web.drakescraft.cl\n"
    "• Discord Oficial: https://discord.gg/drakescraft\n"
    "• Modalidades: Slimefun 4, OneBlock, SkyBlock, Supervivencia y Economia.\n\n" 
    "Scenes rotate through orbital station, arctic outpost, deep sea base, electric field, storm bunker, volcanic outpost and more.\n\n"
    "If you like ambient worlds, subscribe and come back when new scenes go live.\n\n"
    "#Ambient #SciFi #DarkAmbient\n\n"
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
    "  ⛓️  SCP Contencion — Plataforma de cadenas, niebla, bosque\n"
)


def make_title(episode, scene):
    scene_name, _ = SCENE_LABELS.get(scene, ('Station', 'Ambiente'))
    # Opción B: Alternancia por episodios de 11h
    # Episodios pares -> Cozy Study & Chill Lo-Fi
    # Episodios impares -> Deep Space Outpost & Dark Sci-Fi Lo-Fi
    if episode % 2 == 0:
        return f"24/7 Lo-Fi Chill Beats -- {scene_name} | Study & Focus Radio [Pure DSP] EP {episode}"
    else:
        return f"Deep Space Outpost -- {scene_name} | 24/7 Dark Sci-Fi Ambient Lo-Fi EP {episode}"


def make_description(scene):
    _, scene_desc = SCENE_LABELS.get(scene, ('', 'Ambiente procedural'))
    return f"▶️ Escena actual: {scene_desc}\n\n" + _BASE_DESC


def load_episode():
    try:
        return json.load(open(EPISODE_FILE))['episode']
    except Exception:
        return 13


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
        try:
            os.remove(COMMAND_FILE)
        except Exception:
            pass
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
    print(f"[INFO] Iniciando ffmpeg → rtmp://a.rtmp.youtube.com/live2/*** | video={video_path}", flush=True)

    for f in (HUD1, HUD2, HUD3):
        if not os.path.exists(f):
            open(f, 'w').write(' ')

    vf = ",".join([
        "fps=30",
        build_drawtext(HUD1, "h-88", fontsize=21, color='0x00EEFF'),
        build_drawtext(HUD2, "h-60", fontsize=18, color='0xAAFFAA'),
        build_drawtext(HUD3, "h-32", fontsize=17, color='0xFFDD44'),
    ])
    # Espectrómetro reactivo de audio integrado sobre la barra HUD
    filter_complex = (
        f"[1:a]aresample=async=1000:first_pts=0,asplit=2[a_stream][a_spec];"
        f"[a_spec]showfreqs=s=460x42:mode=bar:ascale=log:fscale=log:win_size=1024:colors=0x00EEFF|0x00BBDD|0x88FFAA|0xFFDD44,format=rgba,colorchannelmixer=aa=0.75[spec];"
        f"[0:v]{vf}[base];"
        f"[base][spec]overlay=20:H-148[b0];"
        f"[2:v]scale=110:-1[l1];"
        f"[3:v]scale=110:-1[l2];"
        f"[b0][l1]overlay=15:15[b2];"
        f"[b2][l2]overlay=133:15[b3];"
        f"[b3][4:v]overlay=W-w-15:H-h-15[v]"
    )

    audio_proc = subprocess.Popen(
        [sys.executable, '-u', AUDIO_ENGINE],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        stdin=subprocess.DEVNULL,
    )

    ffmpeg_proc = subprocess.Popen(
        [
            'ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'warning',
            '-re', '-stream_loop', '-1', '-fflags', '+genpts', '-i', video_path,
            '-thread_queue_size', '4096',
            '-f', 's16le', '-ar', '44100', '-ac', '2', '-i', 'pipe:0',
            '-loop', '1', '-i', LOGO_FILE,
            '-loop', '1', '-i', LOGO_FILE2,
            '-loop', '1', '-i', LOGO_FILE3,
            '-filter_complex', filter_complex,
            '-map', '[v]', '-map', '[a_stream]',
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
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '44100',
            '-ac', '2',
            # RTMP output con reconexión automática
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
                p.wait(timeout=5)  # recoge el proceso para no dejar zombis
            except Exception:
                pass


def run_episode(youtube, stream_id, episode, scene_idx_start=0):
    """Retorna (próxima escena, si el broadcast de YouTube quedó live)."""
    scene_idx      = scene_idx_start
    is_cafe_mode   = (episode % 2 == 0)
    active_order   = CAFE_SCENE_ORDER if is_cafe_mode else OUTPOST_SCENE_ORDER
    if is_cafe_mode:
        current_scene = get_current_scene_for_time()
    else:
        current_scene = active_order[scene_idx % len(active_order)]
    crash_count    = 0
    broadcast_id   = None
    broadcast_item = None
    broadcast_ready = False
    retry_delay = _BROADCAST_RETRY_INITIAL
    next_broadcast_retry = time.time() + retry_delay

    write_scene(current_scene)
    write_crash_count(crash_count)
    print(f"[INFO] ==================================================", flush=True)
    print(f"[INFO] --- Episodio {episode} | Escena inicial: {current_scene} | Ciclo: {CYCLE_DURATION//3600}h ---", flush=True)
    print(f"[INFO] ==================================================", flush=True)

    try:
        if youtube and stream_id:
            broadcast_id, broadcast_item = get_or_create_broadcast(youtube, episode, current_scene, stream_id)
        else:
            print("[WARN] YouTube API no disponible; se mantiene solo el envio RTMP.", flush=True)
    except Exception as e:
        print(f"[ERROR] No se pudo preparar broadcast: {e}", flush=True)

    audio_proc, ffmpeg_proc = start_ffmpeg_audio(SCENE_FILES[current_scene])

    if broadcast_id:
        try:
            if not ensure_broadcast_live(youtube, broadcast_id, stream_id):
                print("[ERROR] Broadcast no paso a live dentro del timeout.", flush=True)
            else:
                broadcast_item = get_broadcast(youtube, broadcast_id)
                broadcast_ready = True
        except Exception as e:
            print(f"[ERROR] Fallo activando broadcast: {e}", flush=True)

    # Sincronización precisa con actualStartTime de YouTube
    actual_start_ts = None
    if broadcast_item:
        actual_start_ts = parse_iso_time(broadcast_item.get("snippet", {}).get("actualStartTime"))

    if actual_start_ts:
        episode_start = actual_start_ts
        elapsed_min = (time.time() - episode_start) / 60.0
        print(f"[INFO] Episodio sincronizado con YouTube actualStartTime: emitido hace {elapsed_min:.1f} min", flush=True)
    else:
        episode_start = time.time()
        print(f"[INFO] Episodio iniciado localmente en t={datetime.now().strftime('%H:%M:%S')}", flush=True)

    scene_start   = time.time()
    last_log_time = time.time()
    last_broadcast_check = time.time()
    last_broadcast_check = time.time()

    try:
        while time.time() - episode_start < CYCLE_DURATION:
            now = time.time()

            # Una cuota temporal agotada no debe consumir el numero del VOD: reintenta
            # crear/activar el mismo episodio con backoff, sin martillar la API.
            if youtube and stream_id and not broadcast_ready and now >= next_broadcast_retry:
                try:
                    retry_id, retry_item = get_or_create_broadcast(
                        youtube, episode, current_scene, stream_id
                    )
                    # Si el feed ya estaba activo al vincular, YouTube no dispara
                    # enableAutoStart y rechaza testing/live con invalidTransition
                    # (EP 21, 06/10). Cortar y relanzar el RTMP para que vea
                    # inactive -> active; sin broadcast público no hay audiencia.
                    if retry_id and get_broadcast_state(youtube, retry_id) == "ready":
                        print(f"[INFO] Relanzando RTMP para que {retry_id} arranque solo...", flush=True)
                        kill_procs(ffmpeg_proc, audio_proc)
                        if not wait_stream_inactive(youtube, stream_id):
                            print("[WARN] YouTube sigue viendo el feed activo; se relanza igual.", flush=True)
                        audio_proc, ffmpeg_proc = start_ffmpeg_audio(SCENE_FILES[current_scene])
                        scene_start = time.time()
                        crash_count = 0
                        write_crash_count(crash_count)
                    if retry_id and ensure_broadcast_live(youtube, retry_id, stream_id):
                        broadcast_id = retry_id
                        broadcast_item = get_broadcast(youtube, broadcast_id)
                        broadcast_ready = True
                        actual_start_ts = parse_iso_time(
                            (broadcast_item or {}).get("snippet", {}).get("actualStartTime")
                        )
                        if actual_start_ts:
                            episode_start = actual_start_ts
                        print(f"[SUCCESS] Broadcast EP {episode} activo tras reintento.", flush=True)
                    else:
                        broadcast_id, broadcast_item = retry_id, retry_item
                        print(f"[WARN] EP {episode} aún no está live; conservar número y reintentar.", flush=True)
                except Exception as e:
                    # No imprimir el objeto de excepción completo: algunos proveedores
                    # pueden incluir datos de solicitud en mensajes de error.
                    print(
                        f"[WARN] Reintento de broadcast EP {episode} falló ({type(e).__name__}); "
                        "se conservará el número y se reintentará.",
                        flush=True,
                    )
                next_broadcast_retry = time.time() + retry_delay
                retry_delay = min(retry_delay * 2, _BROADCAST_RETRY_MAX)

            # Telemetría de avance cada 15 minutos en logs
            if now - last_log_time >= 900:
                elapsed_h = (now - episode_start) / 3600.0
                remaining_h = (CYCLE_DURATION - (now - episode_start)) / 3600.0
                print(
                    f"[CYCLE] EP {episode} | Avance: {elapsed_h:.2f}h / {CYCLE_DURATION/3600:.1f}h "
                    f"(Restante: {remaining_h:.2f}h) | Escena: {current_scene}",
                    flush=True,
                )
                last_log_time = now

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
                is_cafe_mode = (episode % 2 == 0)
                active_order = CAFE_SCENE_ORDER if is_cafe_mode else OUTPOST_SCENE_ORDER
                if is_cafe_mode:
                    current_scene = get_current_scene_for_time()
                else:
                    scene_idx     = (scene_idx + 1) % len(active_order)
                    current_scene = active_order[scene_idx]
                print(f"[INFO] Rotando escena → {current_scene}", flush=True)
                write_scene(current_scene)
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(2)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(SCENE_FILES[current_scene])
                scene_start = now
                crash_count = 0
                write_crash_count(crash_count)

            # Vigilancia de estado en YouTube: si finalizó o cayó el broadcast remotamente, avanzar de inmediato
            # Intervalo aumentado a 300s (5 min) para proteger cuota de API
            if youtube and broadcast_id and (now - last_broadcast_check >= 300):
                last_broadcast_check = now
                try:
                    b_state = get_broadcast_state(youtube, broadcast_id)
                    if b_state == 'complete':
                        print(f"[WARN] YouTube marco el broadcast {broadcast_id} como 'complete'. Avanzando episodio inmediatamente...", flush=True)
                        break
                except Exception:
                    pass

            ffmpeg_ret = ffmpeg_proc.poll()
            audio_ret  = audio_proc.poll()
            if ffmpeg_ret is not None or audio_ret is not None:
                reason  = "ffmpeg" if ffmpeg_ret is not None else "audio_engine"
                retcode = ffmpeg_ret if ffmpeg_ret is not None else audio_ret
                backoff = _CRASH_BACKOFF[min(crash_count, len(_CRASH_BACKOFF) - 1)]
                crash_count += 1
                write_crash_count(crash_count)
                print(
                    f"[WARN] {reason} cayó con código {retcode} (crash #{crash_count}), backoff {backoff}s...",
                    flush=True,
                )
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(backoff)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(SCENE_FILES[current_scene])

            time.sleep(2)

    except KeyboardInterrupt:
        print("[INFO] Interrupcion manual detectada.", flush=True)
    finally:
        print(f"[INFO] Deteniendo flujo RTMP de EP {episode}...", flush=True)
        kill_procs(ffmpeg_proc, audio_proc)
        if youtube and broadcast_id:
            print(f"[INFO] Transicionando broadcast {broadcast_id} a COMPLETE en YouTube...", flush=True)
            try:
                safe_transition(youtube, broadcast_id, "complete")
            except Exception as e:
                print(f"[WARN] Error al marcar broadcast {broadcast_id} como complete: {e}", flush=True)
        clear_broadcast_state()

    if broadcast_ready:
        print(f"[INFO] Episodio {episode} completado exitosamente. VOD disponible para archivo en YouTube.", flush=True)
    else:
        print(
            f"[WARN] EP {episode} terminó sin broadcast público; el contador no avanzará.",
            flush=True,
        )
    return (scene_idx + 1) % len(SCENE_ORDER), broadcast_ready


def main():
    try:
        youtube = get_youtube_client()
        stream_id = get_fixed_stream_id(youtube)
        print(f"[INFO] YouTube API inicializada. Stream ID: {stream_id}", flush=True)
    except Exception as e:
        print(f'[WARN] YouTube API no disponible ({e}). Streaming sin API.', flush=True)
        youtube = None
        stream_id = None
    hud_updater.start(youtube=youtube, broadcast_id=None)

    episode   = load_episode()
    scene_idx = 0

    while True:
        scene_idx, published = run_episode(youtube, stream_id, episode, scene_idx)
        if published:
            episode += 1
            save_episode(episode)
            print(f"[INFO] Pausa {EPISODE_PAUSE}s antes de EP {episode} para procesar y asentar VOD...", flush=True)
        else:
            print(
                f"[WARN] EP {episode} no se publicó; se conserva el contador y se reintentará.",
                flush=True,
            )
        time.sleep(EPISODE_PAUSE)


if __name__ == "__main__":
    main()
