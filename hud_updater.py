#!/usr/bin/env python3
"""
hud_updater.py — Actualiza los archivos de texto del HUD en tiempo real.

Escribe cada segundo:
  /tmp/star_hud1.txt  → Línea 1: servidor, ubicación, hora, fecha
  /tmp/star_hud2.txt  → Línea 2: modo de audio, CPU, RAM, red, viewers de YouTube
  /tmp/star_hud3.txt  → Línea 3: notificación flash cuando cambia el modo

Se comunica con audio_engine mediante /tmp/star_state.txt
Se comunica con stream_manager mediante /tmp/star_viewers.txt
"""

import time, os, threading
os.environ['TZ'] = 'America/Santiago'
try:
    time.tzset()
except AttributeError:
    pass  # Windows no tiene tzset, pero en el contenedor Linux sí
STATE_FILE    = "/tmp/star_state.txt"    # escrito por audio_engine
VIEWERS_FILE  = "/tmp/star_viewers.txt"  # escrito por stream_manager
HUD1          = "/tmp/star_hud1.txt"
HUD2          = "/tmp/star_hud2.txt"
HUD3          = "/tmp/star_hud3.txt"

DAYS_ES   = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MONTHS_ES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

_last_mode    = ""
_notify_until = 0.0
_lock         = threading.Lock()


def read_file_safe(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return default


def parse_state(raw):
    """state: 'mode|cpu|ram_pct|devices' """
    parts = raw.split("|")
    mode     = parts[0] if len(parts) > 0 else "UNKNOWN"
    cpu      = parts[1] if len(parts) > 1 else "0.00"
    ram      = parts[2] if len(parts) > 2 else "0%"
    devices  = parts[3] if len(parts) > 3 else "0"
    return mode, cpu, ram, devices


def update_loop(youtube=None, broadcast_id=None):
    global _last_mode, _notify_until

    # Inicializar archivos vacíos
    for f in (HUD1, HUD2, HUD3):
        open(f, 'w').close()

    viewers_poll_interval = 60
    last_viewers_poll = 0
    cached_viewers = 0

    while True:
        now = time.time()
        lt  = time.localtime()

        # ── Hora y fecha en español ──
        hora  = time.strftime("%H:%M:%S", lt)
        dia   = DAYS_ES[lt.tm_wday]
        mes   = MONTHS_ES[lt.tm_mon - 1]
        fecha = f"{dia} {lt.tm_mday:02d} {mes} {lt.tm_year}"

        # ── Estado del audio_engine ──
        state_raw = read_file_safe(STATE_FILE, "UNKNOWN|0.00|0%|0")
        mode, cpu, ram, devices = parse_state(state_raw)

        # ── Viewers YouTube (de stream_manager) ──
        viewers_raw = read_file_safe(VIEWERS_FILE, "0")
        try:
            cached_viewers = int(viewers_raw)
        except Exception:
            pass

        # ── Notificación de cambio de modo ──
        with _lock:
            if mode != _last_mode and _last_mode != "":
                _notify_until = now + 10.0
                _last_mode = mode
            elif _last_mode == "":
                _last_mode = mode

        notify = ""
        if now < _notify_until:
            notify = f">> AUDIO: {mode}"

        # ── Escribir archivos HUD ──
        try:
            with open(HUD1, 'w') as f:
                f.write(f"[ STAR ]  Santiago, Chile  //  {hora}  //  {fecha}")
            with open(HUD2, 'w') as f:
                viewers_str = f"  Viewers:{cached_viewers}" if cached_viewers > 0 else ""
                # % causa problema en drawtext — usar 'pct'
                ram_safe = ram.replace('%', 'pct')
                f.write(f"[ {mode} ]  CPU:{cpu}  RAM:{ram_safe}  Disp:{devices}{viewers_str}")
            with open(HUD3, 'w') as f:
                f.write(notify)
        except Exception:
            pass

        time.sleep(1)


def start(youtube=None, broadcast_id=None):
    """Lanza el loop en un hilo demonio."""
    t = threading.Thread(target=update_loop, args=(youtube, broadcast_id), daemon=True)
    t.start()
    return t
