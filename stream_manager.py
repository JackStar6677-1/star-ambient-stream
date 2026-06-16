#!/usr/bin/env python3
"""
stream_manager.py — Orchestrador del stream 24/7 con rotación de escenas.

Ciclo de un episodio: 11.5h → pausa 90s → siguiente episodio.
Dentro de cada episodio, las escenas rotan cada SCENE_DURATION segundos.
Al rotar escena: ffmpeg + audio_engine se reinician con el nuevo video.
La escena activa se publica en /tmp/star_scene.txt para audio_engine.
"""

import os, sys, time, pickle, subprocess, threading
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError

import hud_updater

# ── Escenas: orden y duración ──────────────────────────────
SCENE_ORDER = ['orbital', 'nieve', 'bosque', 'submarina', 'montana', 'desierto']
SCENE_FILES = {
    'orbital':   '/app/scenes/orbital.mp4',
    'nieve':     '/app/scenes/nieve.mp4',
    'bosque':    '/app/scenes/bosque.mp4',
    'submarina': '/app/scenes/submarina.mp4',
    'montana':   '/app/scenes/montana.mp4',
    'desierto':  '/app/scenes/desierto.mp4',
}
SCENE_DURATION  = 2 * 3600       # 2 horas por escena
CYCLE_DURATION  = 11.5 * 3600    # duración total del episodio

TOKEN_PICKLE  = "token.pickle"
AUDIO_ENGINE  = "audio_engine.py"
VIEWERS_FILE  = "/tmp/star_viewers.txt"
SCENE_FILE    = "/tmp/star_scene.txt"
FONT          = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
HUD1          = "/tmp/star_hud1.txt"
HUD2          = "/tmp/star_hud2.txt"
HUD3          = "/tmp/star_hud3.txt"


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
            print("[ERROR] Credenciales inválidas.")
            sys.exit(1)
    return build('youtube', 'v3', credentials=creds)


def create_broadcast(youtube, title, description):
    print(f"[INFO] Creando broadcast: '{title}'")
    body = {
        'snippet': {
            'title': title, 'description': description,
            'scheduledStartTime': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 120)),
        },
        'status': {'privacyStatus': 'public', 'selfDeclaredCreativeCommons': False},
        'contentDetails': {
            'enableAutoStart': True, 'enableAutoStop': True,
            'monitorStream': {'enableMonitorStream': False},
        },
    }
    return youtube.liveBroadcasts().insert(part='snippet,status,contentDetails', body=body).execute()['id']


def create_stream(youtube, title):
    print("[INFO] Creando live stream...")
    body = {
        'snippet': {'title': f"{title} — Stream"},
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


def start_ffmpeg_audio(video_path, ingestion_url, stream_name):
    """Lanza audio_engine + ffmpeg. Devuelve (audio_proc, ffmpeg_proc)."""
    rtmp = f"{ingestion_url}/{stream_name}"
    print(f"[INFO] Iniciando ffmpeg → {ingestion_url} | video={video_path}", flush=True)

    for f in (HUD1, HUD2, HUD3):
        if not os.path.exists(f):
            open(f, 'w').close()

    vf = ",".join([
        build_drawtext(HUD1, "h-88", fontsize=21, color='0x00EEFF'),
        build_drawtext(HUD2, "h-60", fontsize=18, color='0xAAFFAA'),
        build_drawtext(HUD3, "h-32", fontsize=17, color='0xFFDD44'),
    ])

    audio_proc = subprocess.Popen(
        ['python', '-u', AUDIO_ENGINE],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
    )

    ffmpeg_proc = subprocess.Popen(
        [
            'ffmpeg', '-loglevel', 'warning',
            '-re', '-stream_loop', '-1', '-i', video_path,
            '-f', 's16le', '-ar', '44100', '-ac', '1', '-i', 'pipe:0',
            '-map', '0:v', '-map', '1:a',
            '-vf', vf,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-b:v', '2500k',
            '-c:a', 'aac', '-b:a', '128k',
            '-f', 'flv', rtmp,
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
    title = f"Deep Space Outpost — Ambient Sci-Fi 24/7 | EP {episode}"
    description = (
        "Transmisión ambiental 24/7 completamente automatizada desde servidor Star.\n\n"
        "El audio y la escena evolucionan en tiempo real:\n"
        "  • 6 escenas distintas rotando cada 2 horas (orbital, nieve, bosque, submarino, montaña, desierto)\n"
        "  • Cada escena tiene su propio perfil de audio completamente diferente\n"
        "  • El audio evoluciona según hora del día, clima en vivo, dispositivos en red y viewers\n\n"
        "HUD: hora Santiago, escena activa, CPU/RAM, dispositivos, viewers.\n\n"
        "Síntesis procedural Python • Servidor: Star @ Tailscale"
    )

    broadcast_id = create_broadcast(youtube, title, description)
    stream_id, stream_name, ingestion_url = create_stream(youtube, title)
    bind_broadcast(youtube, broadcast_id, stream_id)
    print(f"[INFO] Broadcast ID: {broadcast_id}", flush=True)

    hud_updater.start(youtube=youtube, broadcast_id=broadcast_id)

    stop_viewers = threading.Event()
    viewers_thread = threading.Thread(
        target=poll_viewers, args=(youtube, broadcast_id, stop_viewers), daemon=True
    )
    viewers_thread.start()

    episode_start = time.time()
    scene_idx     = 0
    scene_start   = time.time()
    current_scene = SCENE_ORDER[scene_idx % len(SCENE_ORDER)]

    write_scene(current_scene)
    print(f"[INFO] Escena inicial: {current_scene}", flush=True)
    audio_proc, ffmpeg_proc = start_ffmpeg_audio(
        SCENE_FILES[current_scene], ingestion_url, stream_name
    )

    try:
        while time.time() - episode_start < CYCLE_DURATION:
            now = time.time()

            # ── Rotación de escena ──
            if now - scene_start >= SCENE_DURATION:
                scene_idx    = (scene_idx + 1) % len(SCENE_ORDER)
                current_scene = SCENE_ORDER[scene_idx]
                print(f"[INFO] Rotando escena → {current_scene}", flush=True)
                write_scene(current_scene)
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(3)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(
                    SCENE_FILES[current_scene], ingestion_url, stream_name
                )
                scene_start = now

            # ── Monitoreo de procesos ──
            ffmpeg_dead = ffmpeg_proc.poll() is not None
            audio_dead  = audio_proc.poll()  is not None
            if ffmpeg_dead or audio_dead:
                reason = "ffmpeg" if ffmpeg_dead else "audio_engine"
                print(f"[WARN] {reason} terminó. Reiniciando...", flush=True)
                kill_procs(ffmpeg_proc, audio_proc)
                time.sleep(4)
                audio_proc, ffmpeg_proc = start_ffmpeg_audio(
                    SCENE_FILES[current_scene], ingestion_url, stream_name
                )

            time.sleep(10)

    except KeyboardInterrupt:
        print("[INFO] Interrupción manual.")
    finally:
        stop_viewers.set()
        kill_procs(ffmpeg_proc, audio_proc)


def main():
    youtube = get_youtube_client()
    episode = 1
    while True:
        try:
            print(f"\n[INFO] ─── Episodio {episode} ───", flush=True)
            run_episode(youtube, episode)
            episode += 1
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
