#!/usr/bin/env python3
"""
audio_engine.py — Generador de audio ambiental procedural 24/7.
Arquitectura de tiers fijos: pad > drone > textura > mecánico > eventos.
Sin normalize-by-peak. Mix limpio. 14 escenas.
"""

import sys, os, time, json, math, random, sqlite3, urllib.request
import numpy as np
from scipy import signal as sps

os.environ['TZ'] = 'America/Santiago'
try:
    time.tzset()
except AttributeError:
    pass

try:
    from pedalboard import Pedalboard, Reverb, Chorus, Compressor, LowpassFilter, HighpassFilter, Gain
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False

SR                = 44100
CHUNK             = SR * 4
WEATHER_INTERVAL  = 1800
NETWORK_INTERVAL  = 300
SYSTEM_INTERVAL   = 60
NETALERTX_DB      = "/opt/stacks/netalertx/data/db/app.db"
STATE_FILE        = "/tmp/star_state.txt"
VIEWERS_FILE      = "/tmp/star_viewers.txt"
SCENE_FILE        = "/tmp/star_scene.txt"
AUDIO_SNAPSHOT    = "/tmp/star_audio_snapshot.json"
SNAPSHOT_INTERVAL = 60
SNAPSHOT_MAX_AGE  = 6 * 3600
STARTUP_FADE_SEC  = 1.5

# ── TIERS DE AMPLITUD FIJOS ───────────────────────────────────────────────────
T_PAD      = 0.26
T_DRONE    = 0.20
T_WIND     = 0.13
T_RAIN     = 0.11
T_BLIZZARD = 0.09
T_WATER    = 0.08
T_FAN      = 0.055
T_RADIO    = 0.095
T_STEPS    = 0.038
T_BELLS    = 0.042
T_DINGS    = 0.035
T_CREAK    = 0.028
T_KNOCK    = 0.055
T_HUM      = 0.028
T_CITY     = 0.045
T_CLOCK    = 0.007
T_CASSETTE = 0.003

# ── ACORDES / CAMPANADAS / MELODÍAS ──────────────────────────────────────────
CHORD_SETS = {
    'dark_station': [
        [73.42, 146.83, 220.00, 261.63, 311.13],
        [98.00, 196.00, 233.08, 261.63, 392.00],
        [116.54, 174.61, 233.08, 293.66, 466.16],
        [110.00, 164.81, 220.00, 261.63, 329.63],
    ],
    'arctic': [
        [65.41, 196.00, 233.08, 293.66, 392.00],
        [103.83, 207.65, 261.63, 311.13, 392.00],
        [87.31, 174.61, 233.08, 261.63, 349.23],
        [73.42, 146.83, 207.65, 261.63, 311.13],
    ],
    'forest': [
        [65.41, 130.81, 196.00, 261.63, 329.63, 392.00],
        [73.42, 146.83, 220.00, 293.66, 369.99, 440.00],
        [87.31, 130.81, 174.61, 261.63, 349.23, 523.25],
        [98.00, 146.83, 196.00, 261.63, 311.13, 392.00],
    ],
    'desert': [
        [73.42, 220.00, 277.18, 329.63, 440.00],
        [77.78, 155.56, 233.08, 293.66, 440.00],
        [98.00, 196.00, 233.08, 293.66, 392.00],
        [110.00, 220.00, 293.66, 329.63, 440.00],
    ],
    'deep_sea': [
        [27.50, 55.00, 82.41, 110.00, 164.81, 220.00],
        [30.87, 61.74, 92.50, 130.81, 185.00, 246.94],
        [32.70, 65.41, 98.00, 130.81, 155.56, 233.08],
        [29.14, 58.27, 87.31, 116.54, 174.61, 233.08],
    ],
    'mountain': [
        [65.41, 164.81, 196.00, 261.63, 392.00, 523.25],
        [73.42, 146.83, 220.00, 261.63, 349.23, 440.00],
        [82.41, 164.81, 207.65, 261.63, 311.13, 415.30],
        [87.31, 174.61, 233.08, 261.63, 349.23, 466.16],
    ],
    'reactor': [
        [55.00, 110.00, 155.56, 207.65, 233.08],
        [61.74, 123.47, 174.61, 220.00, 246.94],
        [69.30, 138.59, 185.00, 246.94, 277.18],
        [58.27, 116.54, 164.81, 220.00, 261.63],
    ],
    'volcanic': [
        [24.50, 49.00, 73.42, 98.00, 138.59, 185.00],
        [27.50, 55.00, 77.78, 110.00, 155.56, 207.65],
        [23.12, 46.25, 69.30, 92.50, 130.81, 174.61],
        [25.96, 51.91, 73.42, 103.83, 146.83, 196.00],
    ],
}

BELL_SETS = {
    'dark_station': [146.83, 164.81, 185.00, 220.00, 246.94, 293.66],
    'arctic':       [155.56, 174.61, 196.00, 220.00, 246.94, 261.63],
    'forest':       [130.81, 164.81, 196.00, 220.00, 261.63, 293.66],
    'desert':       [110.00, 123.47, 138.59, 164.81, 185.00, 220.00],
    'deep_sea':     [55.00,  73.42,  87.31,  110.00, 130.81, 146.83],
    'mountain':     [146.83, 174.61, 196.00, 220.00, 261.63, 293.66],
    'reactor':      [220.00, 233.08, 246.94, 261.63, 277.18, 293.66],
    'volcanic':     [41.20,  55.00,  69.30,  82.41,  110.00, 146.83],
}

MELODY_SETS = {
    'dark_station': [146.83, 130.81, 110.00, 123.47, 164.81, 146.83, 130.81,  98.00],
    'arctic':       [196.00, 146.83, 130.81, 164.81, 146.83, 174.61, 155.56, 130.81],
    'forest':       [196.00, 164.81, 130.81, 164.81, 220.00, 196.00, 174.61, 146.83],
    'desert':       [220.00, 196.00, 164.81, 174.61, 146.83, 196.00, 220.00, 185.00],
    'deep_sea':     [ 55.00,  65.41,  82.41,  73.42,  61.74,  55.00,  49.00,  58.27],
    'mountain':     [196.00, 164.81, 146.83, 174.61, 196.00, 220.00, 164.81, 196.00],
    'reactor':      [103.83, 110.00, 116.54, 103.83,  92.50, 110.00, 123.47, 103.83],
    'volcanic':     [ 36.71,  41.20,  49.00,  43.65,  36.71,  41.20,  32.70,  41.20],
}

# ── PERFILES DE ESCENA — solo scalers 0.0–1.0 ─────────────────────────────────
SCENE_PROFILES = {
    'orbital': {
        'chord': 'dark_station', 'reverb_base': 0.91, 'wet_base': 0.50,
        'chorus_base': 0.36, 'sub_base': 0.016,
        'wind_gain': 0.40, 'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False, 'blizzard': False,
        'water_gain': 0.0, 'fan_gain': 0.90, 'radio_gain': 0.85, 'elec_hum': True,
        'creak_gain': 0.0, 'city_gain': 0.65, 'clock_night': True,  'cassette': 0.90,
        'bell_gain': 0.40, 'ding_gain': 0.80, 'step_gain': 0.70,
        'label': 'ORBITAL STATION',
    },
    'nieve': {
        'chord': 'arctic', 'reverb_base': 0.89, 'wet_base': 0.46,
        'chorus_base': 0.30, 'sub_base': 0.012,
        'wind_gain': 0.70, 'rain_base': 0.25, 'rain_wx_mul': 0.35, 'snow': True, 'blizzard': True,
        'water_gain': 0.0, 'fan_gain': 0.20, 'radio_gain': 0.35, 'elec_hum': False,
        'creak_gain': 0.70, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.50,
        'bell_gain': 0.55, 'ding_gain': 0.25, 'step_gain': 0.40,
        'label': 'ARCTIC OUTPOST',
    },
    'bosque': {
        'chord': 'forest', 'reverb_base': 0.77, 'wet_base': 0.36,
        'chorus_base': 0.26, 'sub_base': 0.010,
        'wind_gain': 0.75, 'rain_base': 0.50, 'rain_wx_mul': 0.85, 'snow': False, 'blizzard': False,
        'water_gain': 0.60, 'fan_gain': 0.0, 'radio_gain': 0.0, 'elec_hum': False,
        'creak_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.40,
        'bell_gain': 0.60, 'ding_gain': 0.0, 'step_gain': 0.20,
        'label': 'FOREST STATION',
    },
    'submarina': {
        'chord': 'deep_sea', 'reverb_base': 0.96, 'wet_base': 0.56,
        'chorus_base': 0.16, 'sub_base': 0.022,
        'wind_gain': 0.15, 'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False, 'blizzard': False,
        'water_gain': 1.0, 'fan_gain': 0.50, 'radio_gain': 0.35, 'elec_hum': True,
        'creak_gain': 0.55, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.40,
        'bell_gain': 0.50, 'ding_gain': 0.55, 'step_gain': 0.50,
        'label': 'DEEP SEA BASE',
    },
    'montana': {
        'chord': 'mountain', 'reverb_base': 0.90, 'wet_base': 0.52,
        'chorus_base': 0.22, 'sub_base': 0.013,
        'wind_gain': 0.90, 'rain_base': 0.45, 'rain_wx_mul': 0.55, 'snow': False, 'blizzard': True,
        'water_gain': 0.0, 'fan_gain': 0.0, 'radio_gain': 0.0, 'elec_hum': False,
        'creak_gain': 0.30, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.30,
        'bell_gain': 0.45, 'ding_gain': 0.0, 'step_gain': 0.25,
        'label': 'MOUNTAIN BASE',
    },
    'desierto': {
        'chord': 'desert', 'reverb_base': 0.66, 'wet_base': 0.26,
        'chorus_base': 0.16, 'sub_base': 0.008,
        'wind_gain': 0.60, 'rain_base': 0.0, 'rain_wx_mul': 0.06, 'snow': False, 'blizzard': False,
        'water_gain': 0.0, 'fan_gain': 0.0, 'radio_gain': 0.0, 'elec_hum': False,
        'creak_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.25,
        'bell_gain': 0.35, 'ding_gain': 0.0, 'step_gain': 0.0,
        'label': 'DESERT HEAT',
    },
    'electrica': {
        'chord': 'dark_station', 'reverb_base': 0.93, 'wet_base': 0.58,
        'chorus_base': 0.28, 'sub_base': 0.015,
        'wind_gain': 0.35, 'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False, 'blizzard': False,
        'water_gain': 0.0, 'fan_gain': 0.0, 'radio_gain': 0.20, 'elec_hum': True,
        'creak_gain': 0.35, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.30,
        'bell_gain': 0.20, 'ding_gain': 0.0, 'step_gain': 0.30,
        'label': 'ELECTRIC FIELD',
    },
    'reactor': {
        'chord': 'reactor', 'reverb_base': 0.86, 'wet_base': 0.42,
        'chorus_base': 0.16, 'sub_base': 0.014,
        'wind_gain': 0.0, 'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False, 'blizzard': False,
        'water_gain': 0.30, 'fan_gain': 1.0, 'radio_gain': 0.25, 'elec_hum': True,
        'creak_gain': 0.20, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.30,
        'bell_gain': 0.0, 'ding_gain': 0.50, 'step_gain': 0.60,
        'label': 'NUCLEAR REACTOR',
    },
    'tormenta': {
        'chord': 'mountain', 'reverb_base': 0.94, 'wet_base': 0.53,
        'chorus_base': 0.26, 'sub_base': 0.017,
        'wind_gain': 1.10, 'rain_base': 0.80, 'rain_wx_mul': 0.12, 'snow': False, 'blizzard': False,
        'water_gain': 0.0, 'fan_gain': 0.0, 'radio_gain': 0.30, 'elec_hum': False,
        'creak_gain': 0.25, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.0,
        'bell_gain': 0.15, 'ding_gain': 0.0, 'step_gain': 0.20,
        'label': 'STORM BUNKER',
    },
    'volcanica': {
        'chord': 'volcanic', 'reverb_base': 0.96, 'wet_base': 0.60,
        'chorus_base': 0.18, 'sub_base': 0.026,
        'wind_gain': 0.20, 'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False, 'blizzard': False,
        'water_gain': 0.30, 'fan_gain': 0.0, 'radio_gain': 0.20, 'elec_hum': False,
        'creak_gain': 0.90, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.0,
        'bell_gain': 0.0, 'ding_gain': 0.0, 'step_gain': 0.0,
        'label': 'GEOTHERMAL OUTPOST',
    },
    'array_suiza': {
        'chord': 'arctic', 'reverb_base': 0.90, 'wet_base': 0.50,
        'chorus_base': 0.20, 'sub_base': 0.010,
        'wind_gain': 0.75, 'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False, 'blizzard': True,
        'water_gain': 0.0, 'fan_gain': 0.0, 'radio_gain': 0.70, 'elec_hum': True,
        'creak_gain': 0.45, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.0,
        'bell_gain': 0.25, 'ding_gain': 0.0, 'step_gain': 0.20,
        'label': 'SIGNAL ARRAY — ALPS',
    },
    'latam_noche': {
        'chord': 'forest', 'reverb_base': 0.63, 'wet_base': 0.28,
        'chorus_base': 0.14, 'sub_base': 0.010,
        'wind_gain': 0.18, 'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False, 'blizzard': False,
        'water_gain': 0.0, 'fan_gain': 0.0, 'radio_gain': 0.10, 'elec_hum': False,
        'creak_gain': 0.0, 'city_gain': 0.90, 'clock_night': True, 'cassette': 1.0,
        'bell_gain': 0.20, 'ding_gain': 0.0, 'step_gain': 0.0,
        'label': 'LATAM NOCHE',
    },
    'scp_exterior': {
        'chord': 'arctic', 'reverb_base': 0.81, 'wet_base': 0.38,
        'chorus_base': 0.16, 'sub_base': 0.010,
        'wind_gain': 0.50, 'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False, 'blizzard': False,
        'water_gain': 0.0, 'fan_gain': 0.0, 'radio_gain': 0.30, 'elec_hum': True,
        'creak_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.50,
        'bell_gain': 0.0, 'ding_gain': 0.0, 'step_gain': 0.35,
        'label': 'SCP FOUNDATION — EXTERIOR',
    },
    'scp_contencion': {
        'chord': 'dark_station', 'reverb_base': 0.95, 'wet_base': 0.58,
        'chorus_base': 0.22, 'sub_base': 0.017,
        'wind_gain': 0.25, 'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False, 'blizzard': False,
        'water_gain': 0.0, 'fan_gain': 0.0, 'radio_gain': 0.20, 'elec_hum': True,
        'creak_gain': 0.75, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.50,
        'bell_gain': 0.0, 'ding_gain': 0.0, 'step_gain': 0.55,
        'label': 'SCP CONTAINMENT',
    },
}

DEFAULT_SCENE = 'orbital'


# ── DATOS EXTERNOS ─────────────────────────────────────────────────────────────

def fetch_weather():
    try:
        req = urllib.request.Request("http://ip-api.com/json/", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            d = json.loads(r.read().decode())
            lat, lon = d.get('lat', -33.45), d.get('lon', -70.66)
    except Exception:
        lat, lon = -33.45, -70.66
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code")
        with urllib.request.urlopen(
                urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=5) as r:
            w = json.loads(r.read().decode())['current']
            return w['temperature_2m'], w['relative_humidity_2m'], w['wind_speed_10m'], w['weather_code']
    except Exception:
        return 20.0, 50.0, 10.0, 0


def fetch_network_devices():
    try:
        con = sqlite3.connect(f"file:{NETALERTX_DB}?mode=ro", uri=True, timeout=3)
        row = con.execute("SELECT onlineDevices FROM Online_History ORDER BY rowid DESC LIMIT 1").fetchone()
        con.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def fetch_system_load():
    try:
        with open('/proc/loadavg') as f:
            load1 = float(f.read().split()[0])
        with open('/proc/meminfo') as f:
            lines = f.read().splitlines()
        mem = {l.split(':')[0]: int(l.split()[1]) for l in lines if ':' in l}
        used_pct = 1.0 - mem.get('MemAvailable', 1) / max(mem.get('MemTotal', 1), 1)
        return load1, used_pct
    except Exception:
        return 0.5, 0.5


def fetch_viewers():
    try:
        with open(VIEWERS_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def fetch_scene():
    try:
        with open(SCENE_FILE) as f:
            s = f.read().strip()
            return s if s in SCENE_PROFILES else DEFAULT_SCENE
    except Exception:
        return DEFAULT_SCENE


def weather_rain_intensity(w_code):
    if 51 <= w_code <= 57:  return 0.30
    if 61 <= w_code <= 67:  return 0.65
    if 71 <= w_code <= 77:  return 0.45
    if 80 <= w_code <= 82:  return 0.85
    if 95 <= w_code <= 99:  return 1.00
    return 0.0


def day_phase(hour):
    """Mínimo 0.28 — nunca inaudible."""
    if 22 <= hour or hour < 5:  return 0.28, 0.24, 0.26
    elif 5  <= hour < 8:        return 0.50, 0.42, 0.44
    elif 8  <= hour < 18:       return 0.82, 0.80, 0.76
    else:                       return 0.55, 0.52, 0.50


# ── ESTADO GLOBAL ──────────────────────────────────────────────────────────────

class AudioState:
    def __init__(self):
        self.scene       = DEFAULT_SCENE
        self.chord_set   = SCENE_PROFILES[DEFAULT_SCENE]['chord']
        self.brightness  = 0.40
        self.density     = 0.35
        self.energy      = 0.38
        self.sub_gain    = 0.016
        self.reverb_room = 0.90
        self.reverb_wet  = 0.50
        self.chorus_mix  = 0.36

        self.t_brightness = 0.40
        self.t_density    = 0.35
        self.t_energy     = 0.38
        self.t_sub        = 0.016
        self.t_reverb     = 0.90
        self.t_reverb_wet = 0.50
        self.t_chorus     = 0.36

        self.LERP         = 0.003
        self.fast_lerp    = False

        self.temp = 15.0; self.humidity = 50.0; self.wind_spd = 5.0; self.w_code = 0
        self.devices = 0; self.load1 = 0.5; self.viewers = 0; self.load_ram_pct = 0.0
        self.rain_intensity = 0.0

    def update_targets(self):
        new_scene = fetch_scene()
        if new_scene != self.scene:
            self.scene     = new_scene
            self.fast_lerp = True
            log(f"Escena → {new_scene}")

        prof = SCENE_PROFILES[self.scene]
        self.chord_set = prof['chord']

        hour = time.localtime().tm_hour
        b, d, e = day_phase(hour)

        device_bonus = min(self.devices / 30.0, 1.0) * 0.15
        viewer_bonus = min(self.viewers / 20.0, 1.0) * 0.10
        load_pen     = min(self.load1 / 4.0, 0.40)
        wind_factor  = min(self.wind_spd / 35.0, 1.0) * 0.18
        hum_factor   = (self.humidity - 50.0) / 100.0 * 0.06

        self.t_brightness = float(np.clip(b - load_pen * 0.08, 0.20, 1.0))
        self.t_density    = float(np.clip(d + device_bonus + viewer_bonus, 0.20, 1.0))
        self.t_energy     = float(np.clip(e + wind_factor, 0.20, 1.0))
        self.t_sub        = prof['sub_base'] * (1.0 + self.t_energy * 0.4)
        self.rain_intensity = weather_rain_intensity(self.w_code)
        self.t_reverb     = float(np.clip(prof['reverb_base'] + hum_factor, 0.50, 0.97))
        self.t_reverb_wet = float(np.clip(prof['wet_base'] + hum_factor * 0.6, 0.12, 0.82))
        self.t_chorus     = prof['chorus_base']

    def lerp_step(self, n):
        rate = self.LERP * (12.0 if self.fast_lerp else 1.0)
        k = 1.0 - (1.0 - rate) ** n
        self.brightness  += (self.t_brightness  - self.brightness)  * k
        self.density     += (self.t_density     - self.density)     * k
        self.energy      += (self.t_energy      - self.energy)      * k
        self.sub_gain    += (self.t_sub         - self.sub_gain)    * k
        self.reverb_room += (self.t_reverb      - self.reverb_room) * k
        self.reverb_wet  += (self.t_reverb_wet  - self.reverb_wet)  * k
        self.chorus_mix  += (self.t_chorus      - self.chorus_mix)  * k
        if self.fast_lerp:
            if (abs(self.reverb_room - self.t_reverb) + abs(self.brightness - self.t_brightness)) < 0.01:
                self.fast_lerp = False

    def export(self, chunk_count=None):
        prof  = SCENE_PROFILES.get(self.scene, SCENE_PROFILES[DEFAULT_SCENE])
        label = prof['label']
        ram   = f"{int(self.load_ram_pct * 100)}%"
        try:
            with open(STATE_FILE, 'w') as f:
                f.write(f"{label}|{self.load1:.2f}|{ram}|{self.devices}")
        except Exception:
            pass


# ── SNAPSHOT ───────────────────────────────────────────────────────────────────

def load_snapshot():
    try:
        if not os.path.exists(AUDIO_SNAPSHOT):
            return None
        with open(AUDIO_SNAPSHOT) as f:
            data = json.load(f)
        age = time.time() - float(data.get('wall_time', 0))
        return data if 0 < age < SNAPSHOT_MAX_AGE else None
    except Exception:
        return None


def save_snapshot(synth):
    tmp = AUDIO_SNAPSHOT + ".tmp"
    try:
        with open(tmp, 'w') as f:
            json.dump({
                'seed': synth.seed, 't_abs': synth.t_abs,
                'chord_t': synth.chord_t, 'chord_idx': synth.chord_idx,
                'chord_dur': synth.chord_dur, 'last_bell': synth.last_bell,
                'last_note': synth.last_note, 'wall_time': time.time(),
            }, f)
        os.replace(tmp, AUDIO_SNAPSHOT)
    except Exception:
        pass


# ── SÍNTESIS MUSICAL ────────────────────────────────────────────────────────────

class Synthesizer:
    def __init__(self):
        snap = load_snapshot()
        self.seed      = int(snap['seed']) if snap else int(time.time())
        self.t_abs     = float(snap['t_abs']) if snap else 0.0
        self.chord_t   = float(snap['chord_t']) if snap else 0.0
        self.chord_idx = int(snap['chord_idx']) if snap else 0
        self.chord_dur = float(snap['chord_dur']) if snap else 0.0
        self.last_bell = float(snap['last_bell']) if snap else 0.0
        self.last_note = float(snap['last_note']) if snap else 0.0

        self._prog_rng = random.Random(self.seed + 404)
        self._bell_rng = random.Random(self.seed + 17)
        self._mel_rng  = random.Random(self.seed + 31)
        self._mel_scene = None
        self._wind_rng  = np.random.RandomState(self.seed + 101)
        self._wind_zi1  = None
        self._wind_zi2  = None
        self._mel_phrase = []

        # Perlin-like LFO
        self._prng = np.random.RandomState(self.seed + 7777)
        self._pval = 0.0
        self._pvel = 0.0

        if not self.chord_dur or self.chord_dur <= 0:
            self.chord_dur = self._next_dur()

        self.ambient   = AmbientSounds(start_t=self.t_abs)
        self.fade_left = STARTUP_FADE_SEC

        if snap:
            log(f"Snapshot restaurado: t={self.t_abs:.0f}s chord={self.chord_idx}")

    def _next_dur(self):
        return self._prog_rng.uniform(65.0, 130.0)

    def _reset_mel_for_scene(self, scene):
        if scene == self._mel_scene:
            return
        h = sum((i+1)*ord(c) for i, c in enumerate(scene))
        self._mel_rng  = random.Random(self.seed + 31 + h)
        self._mel_phrase = []
        self._mel_scene = scene

    def generate_chunk(self, state):
        n = CHUNK
        self._reset_mel_for_scene(state.scene)
        t    = np.linspace(self.t_abs, self.t_abs + n / SR, n, endpoint=False)
        prof   = SCENE_PROFILES.get(state.scene, SCENE_PROFILES[DEFAULT_SCENE])
        chords = CHORD_SETS.get(state.chord_set, CHORD_SETS['dark_station'])
        bells  = BELL_SETS.get(state.chord_set, BELL_SETS['dark_station'])
        notes  = MELODY_SETS.get(state.chord_set, MELODY_SETS['dark_station'])

        # Perlin LFO
        dt = n / SR
        self._pvel += self._prng.normal(0, 0.007)
        self._pvel *= 0.993
        self._pval += self._pvel * dt
        self._pval  = float(np.clip(self._pval, -1.0, 1.0))
        p_prev = self._pval - self._pvel * dt
        plfo   = np.linspace(p_prev, self._pval, n)

        # FM pad
        FM_MAP = {
            'dark_station': (0.50, 0.22, 0.07),
            'arctic':       (0.50, 0.14, 0.05),
            'forest':       (0.75, 0.20, 0.07),
            'deep_sea':     (0.50, 0.25, 0.09),
            'desert':       (0.75, 0.18, 0.06),
            'mountain':     (0.50, 0.16, 0.05),
            'reactor':      (0.50, 0.22, 0.07),
            'volcanic':     (0.50, 0.28, 0.10),
        }
        fm_ratio, fm_base, fm_lfo_d = FM_MAP.get(state.chord_set, (0.5, 0.20, 0.07))
        fm_idx = fm_base + fm_lfo_d * plfo

        def _fm_pad(chord, tv):
            out = np.zeros(len(tv))
            for i, freq in enumerate(chord):
                g  = 0.125 if i == 0 else 0.082 / (i ** 0.6)
                fm = freq * fm_ratio
                ma = np.sin(2*np.pi*fm*0.9985*tv)
                ca = np.sin(2*np.pi*freq*0.9975*tv + fm_idx*ma)
                mb = np.sin(2*np.pi*fm*1.0015*tv)
                cb = np.sin(2*np.pi*freq*1.0025*tv + fm_idx*mb)
                sc = np.sin(2*np.pi*(freq*0.5)*tv) * 0.14
                out += (ca*0.52 + cb*0.48) * g + sc * g * 0.16
            return out

        ch_cur  = chords[self.chord_idx % len(chords)]
        ch_next = chords[(self.chord_idx + 1) % len(chords)]
        fade_d  = min(16.0, self.chord_dur * 0.20)
        pad     = _fm_pad(ch_cur,  t)
        pad_nx  = _fm_pad(ch_next, t)
        ct_arr  = self.chord_t + (t - t[0])
        bl      = np.clip((ct_arr - (self.chord_dur - fade_d)) / max(fade_d, 0.001), 0.0, 1.0)
        pad     = pad * np.cos(bl * np.pi * 0.5) + pad_nx * np.sin(bl * np.pi * 0.5)
        pad     = np.tanh(pad * 1.30) / np.tanh(np.array(1.30)) * 0.93
        pad    *= 0.86 + 0.14 * (0.5 + 0.5 * plfo)

        self.chord_t += n / SR
        if self.chord_t >= self.chord_dur:
            self.chord_t -= self.chord_dur
            self.chord_idx = (self.chord_idx + 1) % len(chords)
            self.chord_dur = self._next_dur()

        # Sub-drone
        root    = max(ch_cur[0] / 2.0, 55.0)
        vibrato = 1.0 + 0.004 * plfo + 0.002 * np.sin(2*np.pi*0.03*t)
        mod_sub = np.sin(2*np.pi*root*0.5*t) * 0.18
        drone   = (np.sin(2*np.pi*root*vibrato*t) * 0.72 +
                   np.sin(2*np.pi*root*vibrato*t + mod_sub) * 0.28)
        drone  *= np.clip(t / 12.0, 0.0, 1.0) * state.sub_gain

        # Viento
        wind_gain_mul = prof['wind_gain']
        fc = 300 + 1600 * state.brightness
        b_, a_ = _biquad_bp(fc, 0.70)
        noise = self._wind_rng.uniform(-1.0, 1.0, n)
        if self._wind_zi1 is None:
            self._wind_zi1 = np.zeros(2)
            self._wind_zi2 = np.zeros(2)
        y1, self._wind_zi1 = sps.lfilter(b_, a_, noise, zi=self._wind_zi1)
        wf, self._wind_zi2 = sps.lfilter(b_, a_, y1,   zi=self._wind_zi2)
        wind = wf * (0.028 + state.energy * 0.048) * wind_gain_mul

        # Melodía larga
        mel = np.zeros(n)
        gap = self._mel_rng.uniform(22.0, 50.0)
        if not self._mel_phrase and (self.t_abs - self.last_note) > gap:
            freq = self._mel_rng.choice(notes)
            if self._mel_rng.random() < 0.28:
                freq *= 0.5
            dur  = self._mel_rng.uniform(12.0, 25.0)
            att  = self._mel_rng.uniform(3.5, 6.5)
            rel  = self._mel_rng.uniform(4.5, 8.0)
            self._mel_phrase = [(self.t_abs + self._mel_rng.uniform(1.0, 2.5), freq, dur, att, rel)]
            self.last_note = self.t_abs

        still = []
        ages_base = np.arange(n) / SR + (self.t_abs - n/SR)
        for (ts, freq, dur, att, rel) in self._mel_phrase:
            total = dur + rel
            ages  = ages_base - ts
            mask  = (ages > 0) & (ages < total)
            if mask.any():
                amp = 0.09 * state.density
                env = np.where(ages < att, ages / max(att, 1e-6),
                      np.where(ages < dur, 1.0,
                               (total - ages) / max(rel, 1e-6)))
                env = np.where(mask, np.clip(env, 0.0, 1.0), 0.0)
                mel += np.sin(2*np.pi*freq*0.998*ages) * amp * env
            if ts + total > self.t_abs:
                still.append((ts, freq, dur, att, rel))
        self._mel_phrase = still

        # Bells
        b_out  = np.zeros(n)
        b_gain = prof['bell_gain']
        b_chance = 0.025 * state.density * b_gain
        b_gap    = max(1.5, 2.0 / max(b_chance, 0.001))
        if b_gain > 0 and (self.t_abs - self.last_bell) > b_gap and self._bell_rng.random() < b_chance:
            bf   = self._bell_rng.choice(bells)
            bamp = self._bell_rng.uniform(0.55, 1.0) * state.density
            bdec = 0.30 + bf / 700.0
            dur  = min(int(4.0 * SR), n)
            lt   = np.arange(dur) / SR
            b_out[:dur] += (np.sin(2*np.pi*bf*lt)*0.65 +
                            np.sin(2*np.pi*bf*2.14*lt)*0.35) * bamp * np.exp(-bdec*lt)
            self.last_bell = self.t_abs

        # Ambiente
        ambient = self.ambient.generate_chunk(self.t_abs, n, state, prof)

        mix = (
            pad     * T_PAD   +
            drone   * T_DRONE +
            wind    * T_WIND  +
            mel     * 0.80    +
            b_out   * T_BELLS +
            ambient
        )

        self.t_abs += n / SR
        return mix


def _biquad_bp(fc, Q, sr=SR):
    w0 = 2 * np.pi * fc / sr
    a  = np.sin(w0) / (2.0 * Q)
    c  = np.cos(w0)
    a0 = 1.0 + a
    b  = np.array([a, 0.0, -a]) / a0
    av = np.array([a0, -2.0*c, 1.0-a]) / a0
    return b, av


# ── CAPAS DE AMBIENTE ──────────────────────────────────────────────────────────

class AmbientSounds:
    def __init__(self, start_t=0.0):
        r = random.Random(9999)
        self.rng = r

        # Radio — buffer precomputado, sin loops Python sample-a-sample
        self._radio_next   = start_t + 20.0
        self._radio_buf    = None   # numpy array con la transmisión completa
        self._radio_buf_pos = 0
        self._radio_rng    = np.random.RandomState(42)

        # Fan
        self._fan_rng = np.random.RandomState(7)
        self._fan_zi1 = None; self._fan_zi2 = None

        # Steps
        self._step_next = start_t + 8.0
        self._step_q    = []

        # Dings
        self._ding_next = start_t + 18.0
        self._ding_q    = []

        # Water
        self._water_rng  = np.random.RandomState(42)
        self._water_zi   = None
        self._bubble_q   = []
        self._bubble_next = start_t + 2.0
        self._water_lfo_t = 0.0

        # Knock
        self._knock_next = start_t + 120.0
        self._knock_q    = []

        # Rain
        self._rain_rng   = np.random.RandomState(77)
        self._rain_zi    = None
        self._rain_lfo_t = 0.0
        self._drop_q     = []
        self._drop_next  = start_t + 1.0

        # Hum
        self._hum_ph = [0.0]*4

        # Blizzard
        self._gust_rng   = np.random.RandomState(13)
        self._gust_zi    = None
        self._gust_lfo_t = 0.0

        # Creak
        self._creak_next = start_t + 35.0
        self._creak_q    = []
        self._creak_rng  = random.Random(31337)

        # City
        self._city_rng = np.random.RandomState(55)
        self._city_zi  = None

        # Cassette
        self._cass_rng = np.random.RandomState(88)
        self._cass_zi  = None

    def _gen_radio_buf(self, dur_s):
        """Genera una transmisión de radio completa como array numpy — sin loops Python."""
        n   = int(dur_s * SR)
        nyq = SR / 2.0
        rng = self._radio_rng

        # Ruido base → static
        raw = rng.uniform(-1.0, 1.0, n).astype(np.float64)
        sos_st = sps.butter(1, np.clip(8000/nyq, 0.001, 0.499), btype='low', output='sos')
        static = sps.sosfilt(sos_st, raw) * 0.018

        # Envoltura de sílabas — imita ritmo del habla
        t     = np.arange(n) / SR
        rate  = rng.uniform(2.2, 4.0)
        syl   = np.abs(np.sin(np.pi * rate * t)) ** 0.55
        word  = np.clip(np.abs(np.sin(np.pi * rng.uniform(0.55, 0.90) * t)) * 3.5, 0, 1)
        env   = syl * word
        # fade in/out de la transmisión
        fi = min(int(0.15*SR), n)
        fo = min(int(0.12*SR), n)
        env[:fi]  *= np.linspace(0, 1, fi)
        env[-fo:] *= np.linspace(1, 0, fo)

        # Componente voiced: ruido a través de dos formantes
        f1 = rng.uniform(400, 850)
        f2 = rng.uniform(1100, 2000)
        vn = rng.uniform(-1.0, 1.0, n).astype(np.float64)
        def _bp(fc, bw):
            lo = np.clip((fc-bw)/nyq, 0.001, 0.499)
            hi = np.clip((fc+bw)/nyq, lo+0.001, 0.499)
            return sps.butter(2, [lo, hi], btype='band', output='sos')
        v1 = sps.sosfilt(_bp(f1, 100), vn)
        v2 = sps.sosfilt(_bp(f2, 160), vn)
        voice = np.tanh((v1*0.55 + v2*0.45) * 9.0) * 0.048

        # Bip de inicio
        beep_dur = min(int(0.12*SR), n)
        beep_t   = np.arange(beep_dur) / SR
        beep     = np.sin(2*np.pi*1080*beep_t) * 0.055 * np.linspace(1, 0, beep_dur)

        buf = voice * env + static
        buf[:beep_dur] += beep
        return buf

    def _radio_chunk(self, t_abs, n, radio_gain):
        if radio_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        if self._radio_buf is None and t_abs >= self._radio_next:
            dur = self.rng.uniform(7, 16)
            self._radio_buf     = self._gen_radio_buf(dur)
            self._radio_buf_pos = 0
            gap = self.rng.uniform(25, 70)
            self._radio_next = t_abs + dur + gap

        if self._radio_buf is not None:
            pos  = self._radio_buf_pos
            avail = len(self._radio_buf) - pos
            if avail <= 0:
                self._radio_buf = None
                return out
            take = min(n, avail)
            out[:take] = self._radio_buf[pos:pos+take] * radio_gain
            self._radio_buf_pos += take
            if self._radio_buf_pos >= len(self._radio_buf):
                self._radio_buf = None
        return out

    def _fan_chunk(self, n, fan_gain):
        if fan_gain <= 0:
            return np.zeros(n)
        noise = self._fan_rng.uniform(-1, 1, n).astype(np.float64)
        nyq   = SR / 2.0
        s1 = sps.butter(2, [np.clip(92/nyq,0.005,0.490), np.clip(132/nyq,0.01,0.495)], btype='band', output='sos')
        s2 = sps.butter(2, [np.clip(167/nyq,0.005,0.490), np.clip(207/nyq,0.01,0.495)], btype='band', output='sos')
        if self._fan_zi1 is None:
            self._fan_zi1 = sps.sosfilt_zi(s1)*noise[0]
            self._fan_zi2 = sps.sosfilt_zi(s2)*noise[0]
        f1, self._fan_zi1 = sps.sosfilt(s1, noise, zi=self._fan_zi1)
        f2, self._fan_zi2 = sps.sosfilt(s2, noise, zi=self._fan_zi2)
        t   = np.arange(n) / SR
        lfo = 0.90 + 0.10 * np.sin(2*np.pi*0.022*t)
        return (f1*0.55 + f2*0.45) * lfo * fan_gain

    def _step_chunk(self, t_abs, n, step_gain, density):
        if step_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        while self._step_next < t_abs + n/SR:
            gap  = self.rng.uniform(6, 16) / max(density * step_gain, 0.1)
            amp  = self.rng.uniform(0.65, 1.0)
            freq = self.rng.uniform(52, 88)
            dec  = self.rng.uniform(28, 45)
            self._step_q.append((self._step_next, amp, freq, dec))
            self._step_q.append((self._step_next + self.rng.uniform(0.18, 0.28),
                                  amp*0.70, freq*self.rng.uniform(0.88,1.12), dec))
            self._step_next += gap
        still = []
        for (ts, amp, freq, dec) in self._step_q:
            s = int((ts - t_abs) * SR)
            if s >= n: still.append((ts,amp,freq,dec)); continue
            if s < -int(2*SR): continue
            dur = min(int(1.5*SR), n-max(s,0))
            lt  = np.arange(dur) / SR
            out[max(s,0):max(s,0)+dur] += np.sin(2*np.pi*freq*lt)*amp*np.exp(-dec*lt)
            if s >= 0: still.append((ts,amp,freq,dec))
        self._step_q = [(ts,a,f,d) for (ts,a,f,d) in still if ts > t_abs-2.0]
        return out

    def _ding_chunk(self, t_abs, n, ding_gain, density):
        if ding_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        while self._ding_next < t_abs + n/SR:
            freq  = self.rng.uniform(900, 3200)
            amp   = self.rng.uniform(0.65, 1.0) * ding_gain
            decay = self.rng.uniform(4, 12)
            self._ding_q.append((self._ding_next, freq, amp, decay))
            self._ding_next += self.rng.uniform(25, 80) / max(density, 0.1)
        still = []
        for (ts, freq, amp, decay) in self._ding_q:
            s = int((ts - t_abs) * SR)
            if s >= n: still.append((ts,freq,amp,decay)); continue
            dur = min(int(3*SR), n-max(s,0))
            lt  = np.arange(dur) / SR
            out[max(s,0):max(s,0)+dur] += (np.sin(2*np.pi*freq*lt)*0.60 +
                                            np.sin(2*np.pi*freq*2.76*lt)*0.40)*amp*np.exp(-decay*lt)
        self._ding_q = [(ts,f,a,d) for (ts,f,a,d) in still]
        return out

    def _water_chunk(self, t_abs, n, water_gain):
        if water_gain <= 0:
            return np.zeros(n)
        noise = self._water_rng.uniform(-1, 1, n).astype(np.float64)
        nyq = SR / 2.0
        sos = sps.butter(2, [np.clip(250/nyq,0.005,0.490), np.clip(600/nyq,0.01,0.495)], btype='band', output='sos')
        if self._water_zi is None:
            self._water_zi = sps.sosfilt_zi(sos) * 0.0
        flow, self._water_zi = sps.sosfilt(sos, noise, zi=self._water_zi)
        t   = np.arange(n) / SR
        lfo = 0.60 + 0.40 * np.abs(np.sin(2*np.pi*0.04*(t+self._water_lfo_t)))
        self._water_lfo_t += n/SR
        out = flow * lfo * water_gain
        while self._bubble_next < t_abs + n/SR:
            freq = self.rng.uniform(180, 900)
            amp  = self.rng.uniform(0.70, 1.0) * water_gain
            dec  = self.rng.uniform(35, 100)
            self._bubble_q.append((self._bubble_next, freq, amp, dec))
            self._bubble_next += self.rng.uniform(0.08, 0.5) / max(water_gain, 0.1)
        still = []
        for (ts, freq, amp, dec) in self._bubble_q:
            s = int((ts - t_abs) * SR)
            if s >= n: still.append((ts,freq,amp,dec)); continue
            dur = min(int(0.08*SR), n-max(s,0))
            lt  = np.arange(dur) / SR
            out[max(s,0):max(s,0)+dur] += np.sin(2*np.pi*freq*lt)*amp*np.exp(-dec*lt)
        self._bubble_q = [(ts,f,a,d) for (ts,f,a,d) in still if ts > t_abs-0.5]
        return np.clip(out, -0.30, 0.30)

    def _knock_chunk(self, t_abs, n, scene):
        out = np.zeros(n)
        if t_abs < self._knock_next:
            return out
        if not self._knock_q:
            metal = scene in ('orbital','submarina','reactor','electrica','scp_contencion')
            fb    = self.rng.uniform(180,320) if metal else self.rng.uniform(80,180)
            ab    = self.rng.uniform(0.70, 1.0)
            dec   = 22 if metal else 14
            n_hits = self.rng.randint(2, 5)
            for i in range(n_hits):
                d = i * self.rng.uniform(0.28, 0.45)
                self._knock_q.append((t_abs+d, fb, ab*self.rng.uniform(0.7,1.0), dec))
            self._knock_next = t_abs + self.rng.uniform(120, 480)
        still = []
        for (ts, freq, amp, dec) in self._knock_q:
            s = int((ts - t_abs) * SR)
            if s >= n: still.append((ts,freq,amp,dec)); continue
            dur = min(int(0.35*SR), n-max(s,0))
            lt  = np.arange(dur) / SR
            out[max(s,0):max(s,0)+dur] += (np.sin(2*np.pi*freq*lt)*0.65 +
                                            np.sin(2*np.pi*freq*1.41*lt)*0.35)*amp*np.exp(-dec*lt)
        self._knock_q = [(ts,f,a,d) for (ts,f,a,d) in still]
        return out

    def _rain_chunk(self, t_abs, n, rain_gain, is_snow=False):
        if rain_gain <= 0:
            return np.zeros(n)
        noise = self._rain_rng.uniform(-1,1,n).astype(np.float64)
        nyq   = SR / 2.0
        lo = np.clip((150 if is_snow else 700)/nyq, 0.005, 0.490)
        hi = np.clip((2500 if is_snow else 12000)/nyq, lo+0.01, 0.495)
        sos = sps.butter(3, [lo,hi], btype='band', output='sos')
        if self._rain_zi is None:
            self._rain_zi = sps.sosfilt_zi(sos)*noise[0]
        hiss, self._rain_zi = sps.sosfilt(sos, noise, zi=self._rain_zi)
        t   = np.arange(n) / SR
        lfo = 0.75 + 0.25 * np.abs(np.sin(2*np.pi*0.022*(t+self._rain_lfo_t)))
        self._rain_lfo_t += n/SR
        out = hiss * lfo * (0.80 if is_snow else 1.0) * rain_gain
        if not is_snow:
            while self._drop_next < t_abs + n/SR:
                freq = self.rng.uniform(1000, 5000)
                amp  = self.rng.uniform(0.50, 1.0) * rain_gain
                dec  = self.rng.uniform(80, 300)
                self._drop_q.append((self._drop_next, freq, amp, dec))
                self._drop_next += self.rng.uniform(0.008, 0.06) / max(rain_gain, 0.1)
            still = []
            for (ts, freq, amp, dec) in self._drop_q:
                s = int((ts-t_abs)*SR)
                if s >= n: still.append((ts,freq,amp,dec)); continue
                st  = max(s,0)
                dur = min(int(0.04*SR), n-st)
                lt  = np.arange(dur)/SR
                out[st:st+dur] += np.sin(2*np.pi*freq*lt)*amp*np.exp(-dec*lt)
            self._drop_q = [(ts,f,a,d) for (ts,f,a,d) in still if ts > t_abs-0.1]
        return np.clip(out, -0.35, 0.35)

    def _hum_chunk(self, n):
        t    = np.arange(n) / SR
        out  = np.zeros(n)
        amps = [1.0, 0.55, 0.28, 0.14]
        for i, (f, a) in enumerate(zip([50.0,100.0,150.0,200.0], amps)):
            out += np.sin(self._hum_ph[i] + 2*np.pi*f*t) * a
            self._hum_ph[i] = (self._hum_ph[i] + 2*np.pi*f*n/SR) % (2*np.pi)
        return out * (0.86 + 0.14 * np.sin(2*np.pi*6.5*t))

    def _blizzard_chunk(self, n):
        noise = self._gust_rng.uniform(-1, 1, n).astype(np.float64)
        nyq   = SR / 2.0
        sos   = sps.butter(3, [np.clip(80/nyq,0.005,0.490), np.clip(950/nyq,0.01,0.495)], btype='band', output='sos')
        if self._gust_zi is None:
            self._gust_zi = sps.sosfilt_zi(sos)*noise[0]
        gust, self._gust_zi = sps.sosfilt(sos, noise, zi=self._gust_zi)
        t   = np.arange(n) / SR
        lfo = (0.45 + 0.35*np.abs(np.sin(2*np.pi*0.07*(t+self._gust_lfo_t))) +
               0.20*np.sin(2*np.pi*0.19*(t+self._gust_lfo_t))**2)
        self._gust_lfo_t += n/SR
        return gust * lfo

    def _creak_chunk(self, t_abs, n, creak_gain):
        if creak_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        while self._creak_next < t_abs + n/SR:
            freq = self._creak_rng.uniform(160, 380)
            amp  = self._creak_rng.uniform(0.65, 1.0) * creak_gain
            dur  = self._creak_rng.uniform(0.25, 1.0)
            self._creak_q.append((self._creak_next, freq, amp, dur))
            self._creak_next += self._creak_rng.uniform(12, 45)
        still = []
        for (ts, freq, amp, dur) in self._creak_q:
            s = int((ts-t_abs)*SR)
            if s >= n: still.append((ts,freq,amp,dur)); continue
            d  = min(int(dur*SR), n-max(s,0))
            if d <= 0: continue
            lt   = np.arange(d)/SR
            farr = freq * (1.0 - 0.28*lt/max(dur,0.001))
            ph   = 2*np.pi*np.cumsum(farr)/SR
            env  = np.sin(np.pi*lt/max(dur,0.001))**0.7
            out[max(s,0):max(s,0)+d] += np.sin(ph)*amp*env
        self._creak_q = [(ts,f,a,d) for (ts,f,a,d) in still if ts > t_abs-2.0]
        return out

    def _city_chunk(self, t_abs, n, city_gain):
        if city_gain <= 0:
            return np.zeros(n)
        noise = self._city_rng.uniform(-1,1,n).astype(np.float64)
        nyq   = SR / 2.0
        sos   = sps.butter(2, [np.clip(55/nyq,0.005,0.490), np.clip(220/nyq,0.01,0.495)], btype='band', output='sos')
        if self._city_zi is None:
            self._city_zi = sps.sosfilt_zi(sos)*0.0
        rumble, self._city_zi = sps.sosfilt(sos, noise, zi=self._city_zi)
        t   = np.arange(n)/SR
        lfo = 0.70 + 0.30*np.sin(2*np.pi*0.028*(t+t_abs))
        return rumble * lfo * city_gain

    def _clock_chunk(self, t_abs, n, clock_night):
        if not clock_night:
            return np.zeros(n)
        if 5 <= time.localtime().tm_hour < 22:
            return np.zeros(n)
        out = np.zeros(n)
        for beat in range(int(t_abs), int(t_abs + n/SR) + 2):
            s = int((float(beat) - t_abs) * SR)
            if 0 <= s < n:
                dur = min(int(0.014*SR), n-s)
                lt  = np.arange(dur)/SR
                out[s:s+dur] += np.sin(2*np.pi*3400*lt)*np.exp(-420*lt)
        return out

    def _cassette_chunk(self, n, cassette):
        if cassette <= 0:
            return np.zeros(n)
        if 5 <= time.localtime().tm_hour < 18:
            return np.zeros(n)
        noise = self._cass_rng.uniform(-1,1,n).astype(np.float64)
        nyq   = SR / 2.0
        sos   = sps.butter(2, [np.clip(3800/nyq,0.005,0.490), np.clip(11000/nyq,0.01,0.495)], btype='band', output='sos')
        if self._cass_zi is None:
            self._cass_zi = sps.sosfilt_zi(sos)*noise[0]
        hiss, self._cass_zi = sps.sosfilt(sos, noise, zi=self._cass_zi)
        return hiss * cassette

    def generate_chunk(self, t_abs, n, state, prof):
        rain_gain = prof['rain_base'] + state.rain_intensity * prof['rain_wx_mul']

        radio  = self._radio_chunk(t_abs, n, prof['radio_gain'])
        fan    = self._fan_chunk(n, prof['fan_gain'])
        steps  = self._step_chunk(t_abs, n, prof.get('step_gain', 0.0), state.density)
        dings  = self._ding_chunk(t_abs, n, prof.get('ding_gain', 0.0), state.density)
        water  = self._water_chunk(t_abs, n, prof['water_gain'])
        knock  = self._knock_chunk(t_abs, n, state.scene)
        rain   = self._rain_chunk(t_abs, n, rain_gain, prof['snow'])
        hum    = self._hum_chunk(n) if prof['elec_hum'] else np.zeros(n)
        bliz   = self._blizzard_chunk(n) if prof['blizzard'] else np.zeros(n)
        creak  = self._creak_chunk(t_abs, n, prof['creak_gain'])
        city   = self._city_chunk(t_abs, n, prof['city_gain'])
        clock  = self._clock_chunk(t_abs, n, prof['clock_night'])
        cass   = self._cassette_chunk(n, prof['cassette'])

        return (
            radio * T_RADIO  * prof['radio_gain'] +
            fan   * T_FAN    * prof['fan_gain']   +
            steps * T_STEPS                       +
            dings * T_DINGS                       +
            water * T_WATER                       +
            knock * T_KNOCK                       +
            rain  * T_RAIN                        +
            hum   * T_HUM                         +
            bliz  * T_BLIZZARD                    +
            creak * T_CREAK                       +
            city  * T_CITY                        +
            clock * T_CLOCK                       +
            cass  * T_CASSETTE
        )


# ── PEDALBOARD ─────────────────────────────────────────────────────────────────

def build_board(state):
    if not HAS_PEDALBOARD:
        return None
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=60.0),
        Chorus(rate_hz=0.22 + state.brightness*0.30,
               depth=0.16 + (1-state.brightness)*0.18,
               centre_delay_ms=7.0, feedback=0.08, mix=state.chorus_mix),
        Reverb(room_size=state.reverb_room, damping=0.32 + state.brightness*0.30,
               wet_level=state.reverb_wet, dry_level=1.0 - state.reverb_wet*0.60),
        Compressor(threshold_db=-16, ratio=2.5, attack_ms=120, release_ms=1000),
        LowpassFilter(cutoff_frequency_hz=5000 + state.brightness*3800),
        Gain(gain_db=1.2),
    ])


def update_board(board, state):
    if board is None:
        return
    board[1].rate_hz   = 0.22 + state.brightness*0.30
    board[1].depth     = 0.16 + (1-state.brightness)*0.18
    board[1].mix       = state.chorus_mix
    board[2].room_size = state.reverb_room
    board[2].damping   = 0.32 + state.brightness*0.30
    board[2].wet_level = state.reverb_wet
    board[2].dry_level = 1.0 - state.reverb_wet*0.60
    board[4].cutoff_frequency_hz = 5000 + state.brightness*3800


# ── MAIN ───────────────────────────────────────────────────────────────────────

def log(msg, level="INFO"):
    print(f"[{level}] {msg}", file=sys.stderr, flush=True)


def main():
    state = AudioState()
    synth = Synthesizer()
    board = None
    last_weather = last_network = last_system = last_viewers = last_snap = 0
    chunk_count  = 0
    _stage       = "init"
    master_gain  = 1.0

    log("audio_engine v2 iniciado — PCM s16le → stdout")

    try:
        while True:
            now    = time.time()
            _stage = "fetch"

            if now - last_weather > WEATHER_INTERVAL:
                try:
                    state.temp, state.humidity, state.wind_spd, state.w_code = fetch_weather()
                    log(f"Clima: {state.temp:.0f}°C hum={state.humidity:.0f}% "
                        f"viento={state.wind_spd:.0f}km/h código={state.w_code}")
                except Exception as e:
                    log(f"Clima falló: {e}", "WARN")
                last_weather = now

            if now - last_network > NETWORK_INTERVAL:
                try:
                    state.devices = fetch_network_devices()
                    log(f"Dispositivos: {state.devices}")
                except Exception as e:
                    log(f"Red falló: {e}", "WARN")
                last_network = now

            if now - last_system > SYSTEM_INTERVAL:
                try:
                    state.load1, state.load_ram_pct = fetch_system_load()
                except Exception as e:
                    log(f"Sistema falló: {e}", "WARN")
                last_system = now

            if now - last_viewers > 30:
                state.viewers = fetch_viewers()
                last_viewers  = now

            _stage = "update"
            state.update_targets()
            state.lerp_step(CHUNK)
            state.export(chunk_count=chunk_count)

            _stage = "board"
            if board is None:
                board = build_board(state)
                log(f"Pedalboard listo (HAS_PEDALBOARD={HAS_PEDALBOARD})")
            else:
                update_board(board, state)

            _stage = "generate"
            audio = synth.generate_chunk(state)

            _stage = "nan_check"
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                log(f"NaN/Inf en chunk {chunk_count} — limpiando", "WARN")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.8, neginf=-0.8)
            audio = np.clip(audio, -1.0, 1.0)

            _stage = "pedalboard"
            if board and HAS_PEDALBOARD:
                proc  = board(audio.astype(np.float32).reshape(1,-1), SR)
                audio = np.nan_to_num(proc.flatten().astype(np.float64), nan=0.0)

            # Ganancia adaptativa RMS — target 0.08, sin normalize-by-peak
            _stage = "gain"
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms > 1e-6:
                raw_gain    = 0.26 / rms
                master_gain += (float(np.clip(raw_gain, 0.40, 5.00)) - master_gain) * 0.04
            audio = np.tanh(audio * master_gain * 0.85)

            # Fade de arranque
            if synth.fade_left > 0:
                nn = CHUNK
                fade_samples = min(nn, int(synth.fade_left * SR))
                ramp = np.ones(nn)
                ramp[:fade_samples] = np.linspace(0, 1, fade_samples)
                audio *= ramp
                synth.fade_left = max(0.0, synth.fade_left - nn/SR)

            _stage = "write"
            peak = float(np.max(np.abs(audio)))
            sys.stdout.buffer.write((audio * 32000).astype(np.int16).tobytes())
            sys.stdout.buffer.flush()

            chunk_count += 1

            if chunk_count % 15 == 0:
                prof  = SCENE_PROFILES.get(state.scene, {})
                label = prof.get('label', state.scene)
                log(f"{label} | bright={state.brightness:.2f} density={state.density:.2f} "
                    f"energy={state.energy:.2f} peak={peak:.3f} rms={rms:.4f} "
                    f"mgain={master_gain:.2f} t={synth.t_abs:.0f}s chunk={chunk_count}")

            if now - last_snap > SNAPSHOT_INTERVAL:
                save_snapshot(synth)
                last_snap = now

    except BrokenPipeError:
        log("Pipe cerrado.")
    except KeyboardInterrupt:
        log("Detenido.")
    except Exception as e:
        import traceback
        log(f"CRASH en '{_stage}' chunk={chunk_count} — {type(e).__name__}: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")


if __name__ == "__main__":
    main()
