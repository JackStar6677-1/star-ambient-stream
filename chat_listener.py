#!/usr/bin/env python3
"""
chat_listener.py — Listener de chat en directo con comandos del propietario.

Solo el propietario del canal y moderadores pueden usar los comandos.
Los viewers normales son ignorados completamente.

COMANDOS DISPONIBLES (solo propietario / moderadores):
────────────────────────────────────────────────────────
ESCENAS
  cambia / !skip / siguiente / next   → Rota a la escena siguiente
  bosque / orbital / nieve / ...      → Salta directamente a esa escena
  escena bosque / escena orbital /... → Idem, con prefijo opcional

INFO
  !status / estado                    → Escena actual
  !escenas / escenas                  → Lista todas las escenas disponibles
  !siguiente / que sigue              → Dice cuál viene después (sin cambiar)
  !hora                               → Hora actual del servidor (UTC-4 Santiago)
  !viewers / espectadores             → Viewers en vivo

CONTROL
  !reinicia / reinicia                → Reinicia ffmpeg (fix lag/cortes de imagen)
  !crash / crashes                    → Muestra el contador de crashes del HUD
"""

import os, sys, json, time, pickle, threading, subprocess
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

TOKEN_PICKLE  = "token.pickle"
SCENE_FILE    = "/tmp/star_scene.txt"
CRASH_FILE    = "/tmp/star_crashes.txt"
VIEWERS_FILE  = "/tmp/star_viewers.txt"
COMMAND_FILE  = "/tmp/star_chat_cmd.txt"   # leído por stream_manager.py
LOG_PREFIX    = "[CHAT]"

SCENE_ORDER = ['electrica', 'montana', 'orbital', 'nieve', 'bosque', 'submarina', 'desierto', 'volcanica', 'reactor', 'tormenta', 'array_suiza', 'latam_noche', 'scp_exterior', 'scp_contencion']
SCENE_LABELS_ES = {
    'electrica':     'Eléctrica ⚡',
    'montana':       'Montaña 🏔️',
    'orbital':       'Orbital 🛸',
    'nieve':         'Nieve ❄️',
    'bosque':        'Bosque 🌲',
    'submarina':     'Submarina 🌊',
    'desierto':      'Desierto 🌵',
    'volcanica':     'Volcánica 🌋',
    'reactor':       'Reactor ⚛️',
    'tormenta':      'Tormenta ⛈️',
    'array_suiza':   'Array Suiza 📡',
    'latam_noche':   'LATAM Noche 🌆',
    'scp_exterior':  'SCP Exterior 🏚️',
    'scp_contencion':'SCP Contención ⛓️',
}

POLL_INTERVAL = 8   # segundos entre polls
MAX_RESULTS   = 50


def log(msg):
    print(f"{LOG_PREFIX} {msg}", flush=True)


def get_youtube_client():
    creds = None
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_PICKLE, 'wb') as f:
                pickle.dump(creds, f)
        else:
            log("ERROR: credenciales invalidas.")
            sys.exit(1)
    return build('youtube', 'v3', credentials=creds)


def get_live_chat_id(youtube):
    try:
        resp = youtube.liveBroadcasts().list(
            part='snippet',
            broadcastStatus='active',
            broadcastType='all',
        ).execute()
        items = resp.get('items', [])
        if not items:
            return None
        for item in items:
            chat_id = item.get('snippet', {}).get('liveChatId')
            if chat_id:
                return chat_id
        return None
    except Exception as e:
        log(f"Error obteniendo liveChatId: {e}")
        return None


def is_dead_live_chat_error(exc):
    """Detecta cuando YouTube invalida el liveChatId al cambiar/recrear directo."""
    if not isinstance(exc, HttpError):
        return False
    status = getattr(exc.resp, 'status', None)
    text = str(exc)
    return status == 404 and ('liveChatNotFound' in text or 'live chat' in text.lower())


def get_channel_owner_id(youtube):
    try:
        resp = youtube.channels().list(part='id', mine=True).execute()
        items = resp.get('items', [])
        return items[0]['id'] if items else None
    except Exception as e:
        log(f"Error obteniendo channelId: {e}")
        return None


def current_scene():
    try:
        with open(SCENE_FILE) as f:
            return f.read().strip()
    except Exception:
        return 'desconocida'


def crash_count():
    try:
        with open(CRASH_FILE) as f:
            return f.read().strip()
    except Exception:
        return '?'


def viewers_count():
    try:
        with open(VIEWERS_FILE) as f:
            return f.read().strip()
    except Exception:
        return '?'


def next_scene_name(current):
    try:
        idx = SCENE_ORDER.index(current)
        nxt = SCENE_ORDER[(idx + 1) % len(SCENE_ORDER)]
        return nxt
    except ValueError:
        return SCENE_ORDER[0]


def write_command(cmd_type, value=None):
    """Escribe un comando en COMMAND_FILE para que stream_manager lo lea."""
    obj = {'cmd': cmd_type, 'value': value, 'ts': time.time()}
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, COMMAND_FILE)
    log(f"Comando escrito: {cmd_type} {value or ''}")


def send_chat_message(youtube, live_chat_id, text):
    try:
        youtube.liveChatMessages().insert(
            part='snippet',
            body={
                'snippet': {
                    'liveChatId': live_chat_id,
                    'type': 'textMessageEvent',
                    'textMessageDetails': {'messageText': text[:200]},
                }
            }
        ).execute()
    except Exception as e:
        log(f"Error enviando mensaje al chat: {e}")


def server_time_str():
    """Hora actual del servidor en formato legible (Chile)."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")


SCENE_ALIASES = {
    # array_suiza
    'array': 'array_suiza', 'suiza': 'array_suiza', 'array suiza': 'array_suiza',
    'telescopios': 'array_suiza', 'alpes': 'array_suiza',
    # latam_noche
    'latam': 'latam_noche', 'latam noche': 'latam_noche',
    'noche': 'latam_noche', 'azotea': 'latam_noche', 'ciudad': 'latam_noche',
    # scp_exterior
    'scp': 'scp_exterior', 'scp exterior': 'scp_exterior',
    'exterior': 'scp_exterior', 'fundacion': 'scp_exterior',
    # scp_contencion
    'contencion': 'scp_contencion', 'scp contencion': 'scp_contencion',
    'cadenas': 'scp_contencion', 'plataforma': 'scp_contencion',
    # escenas existentes por si acaso
    'volcan': 'volcanica', 'volcanica': 'volcanica',
    'montana': 'montana', 'montaña': 'montana',
}

def parse_command(text):
    """
    Parsea el texto y devuelve (cmd, valor) o None.
    cmds: 'skip' | 'goto' | 'status' | 'list' | 'next_info' |
          'hora' | 'viewers' | 'reinicia' | 'crash_info'
    """
    t = text.strip().lower()

    # ── Cambio de escena: nombre directo o alias ──
    if t in SCENE_ORDER:
        return ('goto', t)
    if t in SCENE_ALIASES:
        return ('goto', SCENE_ALIASES[t])

    # ── Cambio de escena: con prefijo opcional ──
    for prefix in ('escena ', 'scene ', '!escena ', '!scene ', 'ir a ', 'pon '):
        if t.startswith(prefix):
            scene = t[len(prefix):].strip()
            if scene in SCENE_ORDER:
                return ('goto', scene)
            if scene in SCENE_ALIASES:
                return ('goto', SCENE_ALIASES[scene])
            return None

    # ── Skip / avanzar ──
    if t in ('cambia', '!skip', 'skip', 'siguiente', 'next', 'avanza', 'cambiar'):
        return ('skip', None)

    # ── Status ──
    if t in ('!status', 'status', '!estado', 'estado', 'que suena', 'que hay'):
        return ('status', None)

    # ── Lista de escenas ──
    if t in ('!escenas', 'escenas', '!lista', 'lista', 'que escenas hay', 'opciones'):
        return ('list', None)

    # ── Qué viene después (sin cambiar) ──
    if t in ('!siguiente', 'que sigue', 'que viene', 'proxima', 'próxima', '!next', 'despues', 'después'):
        return ('next_info', None)

    # ── Hora del servidor ──
    if t in ('!hora', 'hora', 'que hora es', 'time', '!time'):
        return ('hora', None)

    # ── Viewers ──
    if t in ('!viewers', 'viewers', 'espectadores', '!espectadores', 'cuantos hay', 'cuántos'):
        return ('viewers', None)

    # ── Reiniciar ffmpeg (fix lag) ──
    if t in ('!reinicia', 'reinicia', '!restart', 'restart', 'fix lag', '!fixlag', 'lag fix'):
        return ('reinicia', None)

    # ── Crashes ──
    if t in ('!crash', 'crash', 'crashes', '!crashes', '!errores', 'errores'):
        return ('crash_info', None)

    return None


def handle_command(youtube, live_chat_id, cmd, value):
    """Ejecuta el comando y responde en el chat."""
    scene_now = current_scene()
    label_now = SCENE_LABELS_ES.get(scene_now, scene_now.upper())

    if cmd == 'skip':
        nxt = next_scene_name(scene_now)
        label_nxt = SCENE_LABELS_ES.get(nxt, nxt.upper())
        write_command('goto', nxt)
        send_chat_message(youtube, live_chat_id,
            f"🎬 Cambiando escena → {label_nxt}")

    elif cmd == 'goto':
        if value == scene_now:
            send_chat_message(youtube, live_chat_id,
                f"🔄 Ya estás en: {label_now}")
            return
        label_new = SCENE_LABELS_ES.get(value, value.upper())
        write_command('goto', value)
        send_chat_message(youtube, live_chat_id,
            f"🎬 Saltando a → {label_new}")

    elif cmd == 'status':
        nxt = next_scene_name(scene_now)
        label_nxt = SCENE_LABELS_ES.get(nxt, nxt.upper())
        send_chat_message(youtube, live_chat_id,
            f"📡 Escena: {label_now} | Siguiente: {label_nxt} | "
            f"Viewers: {viewers_count()} | Crashes: {crash_count()}")

    elif cmd == 'list':
        names = " · ".join(SCENE_LABELS_ES.get(s, s) for s in SCENE_ORDER)
        send_chat_message(youtube, live_chat_id,
            f"🗺️ Escenas disponibles: {names}")

    elif cmd == 'next_info':
        nxt = next_scene_name(scene_now)
        label_nxt = SCENE_LABELS_ES.get(nxt, nxt.upper())
        send_chat_message(youtube, live_chat_id,
            f"⏭️ Después de {label_now} viene: {label_nxt}")

    elif cmd == 'hora':
        send_chat_message(youtube, live_chat_id,
            f"🕐 Hora servidor: {server_time_str()} (Santiago)")

    elif cmd == 'viewers':
        send_chat_message(youtube, live_chat_id,
            f"👥 Viewers en vivo: {viewers_count()}")

    elif cmd == 'reinicia':
        # Mata ffmpeg — stream_manager lo relanza automáticamente en ~2s
        write_command('reinicia', None)
        send_chat_message(youtube, live_chat_id,
            f"🔧 Reiniciando FFmpeg... el directo vuelve en segundos.")

    elif cmd == 'crash_info':
        send_chat_message(youtube, live_chat_id,
            f"💥 Crashes en episodio actual: {crash_count()}")


def poll_loop(youtube, live_chat_id, owner_id):
    processed_ids = set()
    log(f"Listener activo en liveChatId={live_chat_id[:20]}...")

    # Saltar mensajes históricos: hacer un poll inicial sin procesar
    # para obtener el nextPageToken del presente.
    try:
        resp = youtube.liveChatMessages().list(
            liveChatId=live_chat_id, part='id', maxResults=200
        ).execute()
        next_page_token = resp.get('nextPageToken')
        for item in resp.get('items', []):
            processed_ids.add(item['id'])
        log(f"Saltados {len(processed_ids)} mensajes históricos")
    except Exception as e:
        log(f"Warn al saltar histórico: {e}")
        if is_dead_live_chat_error(e):
            log("liveChatId inválido al iniciar poll; re-resolviendo broadcast activo.")
            return
        next_page_token = None

    while True:
        try:
            kwargs = dict(
                liveChatId=live_chat_id,
                part='snippet,authorDetails',
                maxResults=MAX_RESULTS,
            )
            if next_page_token:
                kwargs['pageToken'] = next_page_token

            resp = youtube.liveChatMessages().list(**kwargs).execute()
            next_page_token = resp.get('nextPageToken')
            polling_ms = resp.get('pollingIntervalMillis', POLL_INTERVAL * 1000)

            for item in resp.get('items', []):
                msg_id = item['id']
                if msg_id in processed_ids:
                    continue
                processed_ids.add(msg_id)

                author   = item['authorDetails']
                text     = item['snippet'].get('displayMessage', '')
                channel  = author.get('channelId', '')
                is_owner = (
                    author.get('isOwner', False) or
                    author.get('isChatModerator', False) or
                    channel == owner_id
                )

                if not is_owner:
                    continue

                result = parse_command(text)
                if result is None:
                    continue

                cmd, value = result
                log(f"Ejecutando: {cmd} {value or ''} (de {author.get('displayName','')})")
                handle_command(youtube, live_chat_id, cmd, value)

            # Evitar acumulación de IDs
            if len(processed_ids) > 2000:
                processed_ids = set(list(processed_ids)[-500:])

            wait = max(POLL_INTERVAL, polling_ms / 1000)
            time.sleep(wait)

        except Exception as e:
            log(f"Error en poll: {e}")
            if is_dead_live_chat_error(e):
                log("liveChatId expirado/no encontrado; buscando el chat activo nuevamente.")
                return
            # Quota excedida: esperar 1 hora en lugar de reintentar cada 30s
            if isinstance(e, HttpError) and 'quotaExceeded' in str(e):
                log("Quota de YouTube API excedida — pausando chat listener 3600s")
                time.sleep(3600)
            else:
                time.sleep(30)


def main():
    youtube  = get_youtube_client()
    owner_id = get_channel_owner_id(youtube)
    log(f"Canal propietario: {owner_id}")

    while True:
        live_chat_id = get_live_chat_id(youtube)
        if live_chat_id:
            poll_loop(youtube, live_chat_id, owner_id)
        else:
            log("Sin broadcast activo. Reintentando en 60s...")
            time.sleep(60)


if __name__ == "__main__":
    main()
