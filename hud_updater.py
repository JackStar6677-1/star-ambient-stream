#!/usr/bin/env python3
"""
hud_updater.py — HUD overlay para el stream 24/7.

HUD1 (cyan)   → Servidor / ubicación / hora / fecha
HUD2 (verde)  → Escena activa + métricas del sistema
HUD3 (amarillo) → Métricas del audio generativo + clima
"""

import time, os, threading

os.environ['TZ'] = 'America/Santiago'
try:
    time.tzset()
except AttributeError:
    pass

STATE_FILE  = "/tmp/star_state.txt"
VIEWERS_FILE= "/tmp/star_viewers.txt"
CRASH_FILE  = "/tmp/star_crashes.txt"
HUD1        = "/tmp/star_hud1.txt"
HUD2        = "/tmp/star_hud2.txt"
HUD3        = "/tmp/star_hud3.txt"

DAYS_ES   = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
MONTHS_ES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

# Nombres legibles para escenas
SCENE_NAMES = {
    "ELECTRIC FIELD":  "Campo Eléctrico",
    "ORBITAL":         "Órbita",
    "SNOW":            "Tormenta de Nieve",
    "FOREST":          "Bosque",
    "UNDERWATER":      "Submarina",
    "MOUNTAIN":        "Base Montana",
    "DESERT":          "Desierto",
    "VOLCANIC":        "Volcánica",
    "REACTOR":         "Reactor",
    "STORM":           "Tormenta",
    "SWISS ARRAY":     "Array Suiza",
    "LATAM NOCHE":     "LATAM Noche",
    "SCP EXTERIOR":    "SCP Exterior",
    "SCP CONTAINMENT": "Contención SCP",
}

# Causa del clima → descripción
CAUSE_LABELS = {
    "noche":      "Noche",
    "lluvia":     "Lluvia",
    "viento":     "Viento",
    "nieve":      "Nevada",
    "tormenta":   "Tormenta",
    "calor":      "Calor",
    "frio":       "Frío",
    "niebla":     "Niebla",
}

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
    """Parsea el state completo: mode|cpu|ram|devices|key=val|..."""
    defaults = {
        "chord": "---", "reverb": "0.00", "wet": "0.00",
        "cause": "---", "peak": "0.000", "rms": "0.000",
        "chunk": "0",   "layer": "---",  "gain": "0.00",
        "crashes": "0",
    }
    parts = raw.split("|")
    mode    = parts[0] if len(parts) > 0 else "UNKNOWN"
    cpu     = parts[1] if len(parts) > 1 else "0.00"
    ram     = parts[2] if len(parts) > 2 else "0%"
    devices = parts[3] if len(parts) > 3 else "0"
    for part in parts[4:]:
        if "=" in part:
            k, v = part.split("=", 1)
            if k in defaults:
                defaults[k] = v
    return mode, cpu, ram, devices, defaults


def fmt_gain(g):
    """Amplificación adaptativa: ×1.00"""
    try:
        return f"x{float(g):.2f}"
    except Exception:
        return g


def fmt_rms(r):
    """Volumen RMS como porcentaje del máximo."""
    try:
        pct = float(r) * 100
        return f"{pct:.1f}%"
    except Exception:
        return r


def fmt_chord(c):
    """'dark_station:2/4' → 'dark_station  2/4'"""
    return c.replace(":", "  ") if ":" in c else c


def update_loop(youtube=None, broadcast_id=None):
    global _last_mode, _notify_until

    for f in (HUD1, HUD2, HUD3):
        open(f, 'w').close()

    cached_viewers = 0

    while True:
        now = time.time()
        lt  = time.localtime()

        hora  = time.strftime("%H:%M:%S", lt)
        dia   = DAYS_ES[lt.tm_wday]
        fecha = f"{dia} {lt.tm_mday:02d} {MONTHS_ES[lt.tm_mon-1]} {lt.tm_year}"

        state_raw = read_file_safe(STATE_FILE, "UNKNOWN|0.00|0%|0")
        mode, cpu, ram, devices, m = parse_state(state_raw)

        try:
            cached_viewers = int(read_file_safe(VIEWERS_FILE, "0"))
        except Exception:
            pass

        crash_raw = read_file_safe(CRASH_FILE, m["crashes"])
        crashes   = "OK" if crash_raw in ("0", "") else f"!{crash_raw}"

        scene_name = SCENE_NAMES.get(mode, mode)
        cause_name = CAUSE_LABELS.get(m["cause"], m["cause"].capitalize())

        # Notificación flash al cambiar escena
        with _lock:
            if mode != _last_mode and _last_mode != "":
                _notify_until = now + 12.0
            _last_mode = mode

        ram_safe = ram.replace('%', '%')  # drawtext acepta % en textfile

        viewers_str = f"  |  Viewers: {cached_viewers}" if cached_viewers > 0 else ""

        try:
            # Línea 1: identidad + hora
            with open(HUD1, 'w') as f:
                f.write(f"[ STAR ]  Santiago, Chile  //  {hora}  //  {fecha}{viewers_str}")

            # Línea 2: escena + sistema
            with open(HUD2, 'w') as f:
                f.write(
                    f"[ {scene_name} ]"
                    f"  CPU: {cpu}%"
                    f"  MEM: {ram}"
                    f"  Disp: {devices}"
                    f"  Reverb: {m['reverb']}/{m['wet']}"
                    f"  Vol: {fmt_rms(m['rms'])}"
                    f"  Amp: {fmt_gain(m['gain'])}"
                )

            # Línea 3: audio generativo + clima
            if now < _notify_until:
                with open(HUD3, 'w') as f:
                    f.write(f">> Cambiando a: {scene_name}")
            else:
                with open(HUD3, 'w') as f:
                    f.write(
                        f"Acorde: {fmt_chord(m['chord'])}"
                        f"  |  Capa: {m['layer']}"
                        f"  |  Clima: {cause_name}"
                        f"  |  Pico: {m['peak']}"
                        f"  |  Estado: {crashes}"
                    )
        except Exception:
            pass

        time.sleep(1)


def start(youtube=None, broadcast_id=None):
    t = threading.Thread(target=update_loop, args=(youtube, broadcast_id), daemon=True)
    t.start()
    return t
