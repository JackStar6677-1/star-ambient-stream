#!/usr/bin/env python3
"""
audio_engine.py — Generador de audio ambiental para stream 24/7.

Escena activa: /tmp/star_scene.txt (orbital|nieve|bosque|submarina|montana|desierto)
Cada escena tiene un perfil de audio radicalmente distinto.
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

SR               = 44100
CHUNK            = SR * 4
WEATHER_INTERVAL = 1800
NETWORK_INTERVAL = 300
SYSTEM_INTERVAL  = 60
NETALERTX_DB     = "/opt/stacks/netalertx/data/db/app.db"
STATE_FILE       = "/tmp/star_state.txt"
VIEWERS_FILE     = "/tmp/star_viewers.txt"
SCENE_FILE       = "/tmp/star_scene.txt"

# ──────────────────────────────────────────────────────────
# Sets de acordes por escena
# ──────────────────────────────────────────────────────────

CHORD_SETS = {
    # Estación orbital: oscuro, tenso, electrónico
    'dark_station': [
        [73.42, 146.83, 220.00, 261.63, 311.13],
        [98.00, 196.00, 233.08, 261.63, 392.00],
        [116.54, 174.61, 233.08, 293.66, 466.16],
        [110.00, 164.81, 220.00, 261.63, 329.63],
    ],
    # Nieve/ártico: frío, cristalino, amplio
    'arctic': [
        [65.41, 196.00, 233.08, 293.66, 392.00],
        [103.83, 207.65, 261.63, 311.13, 392.00],
        [87.31, 174.61, 233.08, 261.63, 349.23],
        [73.42, 146.83, 207.65, 261.63, 311.13],
    ],
    # Bosque: cálido, orgánico, resonante
    'forest': [
        [65.41, 130.81, 196.00, 261.63, 329.63, 392.00],
        [73.42, 146.83, 220.00, 293.66, 369.99, 440.00],
        [87.31, 130.81, 174.61, 261.63, 349.23, 523.25],
        [98.00, 146.83, 196.00, 261.63, 311.13, 392.00],
    ],
    # Desierto: seco, caliente, disperso
    'desert': [
        [73.42, 220.00, 277.18, 329.63, 440.00],
        [77.78, 155.56, 233.08, 293.66, 440.00],
        [98.00, 196.00, 233.08, 293.66, 392.00],
        [110.00, 220.00, 293.66, 329.63, 440.00],
    ],
    # Submarino: grave, presión, resonancia profunda
    'deep_sea': [
        [27.50, 55.00, 82.41, 110.00, 164.81, 220.00],
        [30.87, 61.74, 92.50, 130.81, 185.00, 246.94],
        [32.70, 65.41, 98.00, 130.81, 155.56, 233.08],
        [29.14, 58.27, 87.31, 116.54, 174.61, 233.08],
    ],
    # Montaña: amplio, viento, elevado
    'mountain': [
        [65.41, 164.81, 196.00, 261.63, 392.00, 523.25],
        [73.42, 146.83, 220.00, 261.63, 349.23, 440.00],
        [82.41, 164.81, 207.65, 261.63, 311.13, 415.30],
        [87.31, 174.61, 233.08, 261.63, 349.23, 466.16],
    ],
}

BELL_SETS = {
    'dark_station': [587.33, 659.25, 739.99, 932.33, 1046.50, 1244.51],
    'arctic':       [622.25, 698.46, 783.99, 932.33, 1046.50, 1244.51],
    'forest':       [523.25, 659.25, 783.99, 880.00, 1046.50, 1318.51],
    'desert':       [440.00, 493.88, 554.37, 659.25, 739.99, 880.00],
    'deep_sea':     [146.83, 196.00, 233.08, 293.66, 369.99, 440.00],
    'mountain':     [587.33, 698.46, 830.61, 987.77, 1174.66, 1396.91],
}

MELODY_SETS = {
    'dark_station': [293.66, 233.08, 220.00, 261.63, 349.23, 293.66, 233.08, 196.00],
    'arctic':       [392.00, 293.66, 233.08, 392.00, 261.63, 311.13, 349.23, 293.66],
    'forest':       [392.00, 329.63, 261.63, 329.63, 440.00, 392.00, 349.23, 293.66],
    'desert':       [440.00, 554.37, 659.25, 739.99, 554.37, 659.25, 739.99, 880.00],
    'deep_sea':     [110.00, 130.81, 164.81, 146.83, 123.47, 110.00, 98.00, 116.54],
    'mountain':     [523.25, 392.00, 329.63, 392.00, 440.00, 523.25, 587.33, 523.25],
}

SONAR_FREQS = {
    'dark_station': 650, 'arctic': 900, 'forest': 1800,
    'desert': 1400, 'deep_sea': 280, 'mountain': 1200,
}

# Perfil de audio por escena de video
SCENE_PROFILES = {
    'orbital':   {
        'chord': 'dark_station', 'reverb_base': 0.94, 'wet_base': 0.55,
        'chorus_base': 0.42, 'sub_base': 0.018,
        'fan_gain': 1.6, 'radio_gain': 0.10, 'water_gain': 0.0,
        'wind_gain': 0.5, 'bell_gain': 0.05, 'ding_gain': 0.7,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': True, 'rain_glass': True, 'blizzard': False,
        'creak_gain': 0.0, 'city_gain': 0.12, 'clock_night': True, 'cassette': 0.003,
        'label': 'ORBITAL STATION',
    },
    'nieve':     {
        'chord': 'arctic', 'reverb_base': 0.90, 'wet_base': 0.48,
        'chorus_base': 0.34, 'sub_base': 0.012,
        'fan_gain': 0.7, 'radio_gain': 0.07, 'water_gain': 0.0,
        'wind_gain': 1.8, 'bell_gain': 0.08, 'ding_gain': 0.4,
        'rain_base': 0.28, 'rain_wx_mul': 0.4, 'snow': True,
        'elec_hum': False, 'rain_glass': True, 'blizzard': True,
        'creak_gain': 0.6, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'ARCTIC OUTPOST',
    },
    'bosque':    {
        'chord': 'forest', 'reverb_base': 0.78, 'wet_base': 0.38,
        'chorus_base': 0.28, 'sub_base': 0.010,
        'fan_gain': 0.0, 'radio_gain': 0.0, 'water_gain': 0.5,
        'wind_gain': 0.9, 'bell_gain': 0.10, 'ding_gain': 0.0,
        'rain_base': 0.55, 'rain_wx_mul': 0.9, 'snow': False,
        'elec_hum': False, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'FOREST STATION',
    },
    'submarina': {
        'chord': 'deep_sea', 'reverb_base': 0.97, 'wet_base': 0.58,
        'chorus_base': 0.18, 'sub_base': 0.022,
        'fan_gain': 0.4, 'radio_gain': 0.05, 'water_gain': 1.0,
        'wind_gain': 0.2, 'bell_gain': 0.12, 'ding_gain': 0.5,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': True, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.35, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'DEEP SEA BASE',
    },
    'montana':   {
        'chord': 'mountain', 'reverb_base': 0.85, 'wet_base': 0.44,
        'chorus_base': 0.28, 'sub_base': 0.010,
        'fan_gain': 0.0, 'radio_gain': 0.0, 'water_gain': 0.2,
        'wind_gain': 2.2, 'bell_gain': 0.07, 'ding_gain': 0.0,
        'rain_base': 0.30, 'rain_wx_mul': 0.8, 'snow': False,
        'elec_hum': False, 'rain_glass': False, 'blizzard': True,
        'creak_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'MOUNTAIN BASE',
    },
    'desierto':  {
        'chord': 'desert', 'reverb_base': 0.68, 'wet_base': 0.28,
        'chorus_base': 0.18, 'sub_base': 0.008,
        'fan_gain': 0.0, 'radio_gain': 0.0, 'water_gain': 0.0,
        'wind_gain': 0.7, 'bell_gain': 0.04, 'ding_gain': 0.0,
        'rain_base': 0.0, 'rain_wx_mul': 0.08, 'snow': False,
        'elec_hum': False, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'DESERT HEAT',
    },
}

DEFAULT_SCENE = 'orbital'


# ──────────────────────────────────────────────────────────
# Condiciones externas
# ──────────────────────────────────────────────────────────

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
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
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


def weather_rain_intensity(w_code):
    """Intensidad de lluvia 0–1 según weather_code de Open-Meteo."""
    if 51 <= w_code <= 57:  return 0.30   # llovizna
    if 61 <= w_code <= 67:  return 0.65   # lluvia moderada
    if 71 <= w_code <= 77:  return 0.45   # nieve
    if 80 <= w_code <= 82:  return 0.85   # chubascos
    if 95 <= w_code <= 99:  return 1.00   # tormenta
    return 0.0


def fetch_scene():
    try:
        with open(SCENE_FILE) as f:
            s = f.read().strip()
            return s if s in SCENE_PROFILES else DEFAULT_SCENE
    except Exception:
        return DEFAULT_SCENE


def day_phase(hour):
    """Multiplicadores de brightness/density/energy según hora del día."""
    if   0  <= hour < 5:  return 0.15, 0.10, 0.15
    elif 5  <= hour < 8:  return 0.30, 0.25, 0.25
    elif 8  <= hour < 13: return 0.70, 0.65, 0.60
    elif 13 <= hour < 18: return 0.80, 0.75, 0.70
    elif 18 <= hour < 21: return 0.45, 0.45, 0.40
    else:                 return 0.20, 0.20, 0.20


# ──────────────────────────────────────────────────────────
# Estado global
# ──────────────────────────────────────────────────────────

class AudioState:
    def __init__(self):
        self.scene       = DEFAULT_SCENE
        self.chord_set   = SCENE_PROFILES[DEFAULT_SCENE]['chord']
        self.brightness  = 0.3
        self.density     = 0.3
        self.energy      = 0.3
        self.sub_gain    = 0.05
        self.reverb_room = 0.90
        self.reverb_wet  = 0.60
        self.chorus_mix  = 0.40
        self.sonar_freq  = 650

        # Targets para lerp
        self.t_brightness = 0.3
        self.t_density    = 0.3
        self.t_energy     = 0.3
        self.t_sub        = 0.05
        self.t_reverb     = 0.90
        self.t_reverb_wet = 0.60
        self.t_chorus     = 0.40

        self.LERP      = 0.003   # velocidad normal
        self.fast_lerp = False   # True durante 60s al cambiar escena

        self.temp = 15.0; self.humidity = 50.0; self.wind_spd = 5.0; self.w_code = 0
        self.devices = 0; self.load1 = 0.5; self.viewers = 0
        self.load_ram_pct = 0.0
        self.rain_intensity = 0.0   # calculado de w_code en update_targets

    def update_targets(self):
        new_scene = fetch_scene()
        if new_scene != self.scene:
            self.scene      = new_scene
            self.fast_lerp  = True
            print(f"[INFO] Escena → {new_scene}", file=sys.stderr)

        prof = SCENE_PROFILES[self.scene]
        self.chord_set  = prof['chord']
        self.sonar_freq = SONAR_FREQS.get(self.chord_set, 800)

        hour = time.localtime().tm_hour
        b, d, e = day_phase(hour)

        device_bonus = min(self.devices / 30.0, 1.0) * 0.20
        viewer_bonus = min(self.viewers / 20.0, 1.0) * 0.12
        load_pen     = min(self.load1 / 4.0, 0.5)

        self.t_brightness = float(np.clip(b - load_pen * 0.1, 0.05, 1.0))
        self.t_density    = float(np.clip(d + device_bonus + viewer_bonus - load_pen * 0.15, 0.05, 1.0))
        self.t_energy     = float(np.clip(e + (self.wind_spd / 50.0) * 0.12 - load_pen * 0.1, 0.05, 1.0))
        self.t_sub        = prof['sub_base'] * (1.0 + self.t_energy * 0.5)
        self.rain_intensity = weather_rain_intensity(self.w_code)
        self.t_reverb     = prof['reverb_base']
        self.t_reverb_wet = prof['wet_base']
        self.t_chorus     = prof['chorus_base']

    def lerp_step(self, n):
        rate = self.LERP * (15.0 if self.fast_lerp else 1.0)
        k = 1.0 - (1.0 - rate) ** n
        self.brightness  += (self.t_brightness  - self.brightness)  * k
        self.density     += (self.t_density     - self.density)     * k
        self.energy      += (self.t_energy      - self.energy)      * k
        self.sub_gain    += (self.t_sub         - self.sub_gain)    * k
        self.reverb_room += (self.t_reverb      - self.reverb_room) * k
        self.reverb_wet  += (self.t_reverb_wet  - self.reverb_wet)  * k
        self.chorus_mix  += (self.t_chorus      - self.chorus_mix)  * k

        # Desactivar fast_lerp cuando params están cerca del target
        if self.fast_lerp:
            delta = (abs(self.reverb_room - self.t_reverb) +
                     abs(self.reverb_wet  - self.t_reverb_wet) +
                     abs(self.brightness  - self.t_brightness))
            if delta < 0.01:
                self.fast_lerp = False

    def export(self):
        prof  = SCENE_PROFILES.get(self.scene, SCENE_PROFILES[DEFAULT_SCENE])
        label = prof['label']
        ram_pct = f"{int(self.load_ram_pct * 100)}%"
        try:
            with open(STATE_FILE, 'w') as f:
                f.write(f"{label}|{self.load1:.2f}|{ram_pct}|{self.devices}")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────
# Síntesis principal
# ──────────────────────────────────────────────────────────

class Synthesizer:
    def __init__(self):
        self.t_abs     = 0.0
        self.chord_t   = 0.0
        self.chord_dur = 20.0
        self.chord_idx = 0
        self.bell_rng  = random.Random(int(time.time()))
        self.mel_rng   = random.Random(int(time.time()) + 1)
        self.last_bell = 0.0
        self.last_note = 0.0
        self.note_active = None
        self.wind_zi   = None
        self.ambient   = AmbientSounds()

    def generate_chunk(self, state):
        n = CHUNK
        t = np.linspace(self.t_abs, self.t_abs + n / SR, n, endpoint=False)
        chords  = CHORD_SETS.get(state.chord_set, CHORD_SETS['dark_station'])
        bells_n = BELL_SETS.get(state.chord_set, BELL_SETS['dark_station'])
        notes_n = MELODY_SETS.get(state.chord_set, MELODY_SETS['dark_station'])

        # ── Pad con crossfade suave ──
        ch_cur  = chords[self.chord_idx % len(chords)]
        ch_next = chords[(self.chord_idx + 1) % len(chords)]
        fade_dur = 6.0
        pad = np.zeros(n)
        for i, freq in enumerate(ch_cur):
            g = 0.16 if i == 0 else 0.10 / (i ** 0.5)
            pad += np.sin(2 * np.pi * freq * 0.998 * t) * g
            pad += np.sin(2 * np.pi * freq * 1.002 * t) * g * 0.65
        blend_pad = np.zeros(n)
        for i, freq in enumerate(ch_next):
            g = 0.16 if i == 0 else 0.10 / (i ** 0.5)
            blend_pad += np.sin(2 * np.pi * freq * 0.998 * t) * g
            blend_pad += np.sin(2 * np.pi * freq * 1.002 * t) * g * 0.65

        ct_arr = self.chord_t + (t - t[0])
        blend  = np.clip((ct_arr - (self.chord_dur - fade_dur)) / fade_dur, 0.0, 1.0)
        pad    = pad * (1 - blend) + blend_pad * blend
        pad    = np.tanh(pad * 0.80) * 0.34
        pad   *= 0.88 + 0.12 * np.sin(2 * np.pi * 0.07 * t)

        self.chord_t += n / SR
        if self.chord_t >= self.chord_dur:
            self.chord_t -= self.chord_dur
            self.chord_idx = (self.chord_idx + 1) % len(chords)

        # ── Sub-drone — mínimo 55 Hz para evitar infrasónicos inaudibles ──
        root_raw  = ch_cur[0] / 2.0
        root_freq = max(root_raw, 55.0)   # deep_sea tendría 13 Hz sin este clip
        vibrato   = 1.0 + 0.003 * np.sin(2 * np.pi * 0.05 * t)
        drone     = np.sin(2 * np.pi * root_freq * vibrato * t)
        drone    *= np.clip(t / 15.0, 0.0, 1.0) * state.sub_gain

        # ── Viento filtrado ──
        prof = SCENE_PROFILES.get(state.scene, SCENE_PROFILES[DEFAULT_SCENE])
        wind_gain_mul = prof['wind_gain']
        fc_mean = 300 + 1800 * state.brightness
        nyq = SR / 2.0
        lo  = np.clip(fc_mean * 0.55 / nyq, 0.005, 0.490)
        hi  = np.clip(fc_mean * 1.45 / nyq, lo + 0.010, 0.495)
        sos_w = sps.butter(3, [lo, hi], btype='band', output='sos')
        noise = np.random.RandomState(int(self.t_abs * 10) % 999983).uniform(-1.0, 1.0, n)
        if self.wind_zi is None:
            self.wind_zi = sps.sosfilt_zi(sos_w) * noise[0]
        wind_out, self.wind_zi = sps.sosfilt(sos_w, noise, zi=self.wind_zi)
        wind = wind_out * (0.035 + state.energy * 0.08) * wind_gain_mul

        # ── Melodía (ADSR) ──
        mel = np.zeros(n)
        note_interval = 3.5 + (1.0 - state.density) * 6.0
        if (self.t_abs - self.last_note) > note_interval:
            self.note_active = (self.mel_rng.choice(notes_n), self.t_abs, 2.5)
            self.last_note   = self.t_abs
        if self.note_active:
            freq_n, start_n, dur_n = self.note_active
            ATTACK, DECAY, SUS, RELEASE = 0.4, 0.5, 0.45, 1.2
            total = dur_n + RELEASE
            for i in range(n):
                age = (self.t_abs - n / SR + i / SR) - start_n
                if 0 < age < total:
                    if   age < ATTACK:          env = age / ATTACK
                    elif age < ATTACK + DECAY:  env = 1.0 - (1.0 - SUS) * (age - ATTACK) / DECAY
                    elif age < dur_n:           env = SUS
                    elif age < total:           env = SUS * (1.0 - (age - dur_n) / RELEASE)
                    else:                       env = 0.0
                    mel[i] = (math.sin(2*math.pi*freq_n*0.997*age) +
                              math.sin(2*math.pi*freq_n*1.003*age)) * 0.5 * env * 0.11 * state.density

        # ── Bells (bajas en amplitud) ──
        bells = np.zeros(n)
        bell_gain = prof['bell_gain']
        bell_chance = 0.03 * state.density
        min_gap = max(1.0, 1.5 / max(bell_chance, 0.001))
        if (self.t_abs - self.last_bell) > min_gap and self.bell_rng.random() < bell_chance:
            freq_b  = self.bell_rng.choice(bells_n)
            amp_b   = self.bell_rng.uniform(0.015, 0.055) * state.density * bell_gain
            decay_b = 0.35 + freq_b / 750.0
            dur_b   = min(int(4.5 * SR), n)
            lt      = np.arange(dur_b) / SR
            bells[:dur_b] += (np.sin(2*np.pi*freq_b*lt) * 0.65 +
                              np.sin(2*np.pi*freq_b*2.15*lt) * 0.35) * amp_b * np.exp(-decay_b * lt)
            self.last_bell = self.t_abs

        # ── Sonar ──
        period = 4.5 + (1.0 - state.energy) * 3.0
        pt  = t % period
        mask = pt < 0.30
        sonar = np.zeros(n)
        sonar[mask] = np.sin(pt[mask]*2*np.pi*state.sonar_freq) * np.exp(-20.0*pt[mask]) * 0.008

        ambient_wet, ambient_dry = self.ambient.generate_chunk(self.t_abs, n, state)

        wet = (
            pad          * 0.32 +
            drone        * 0.30 +
            wind         * 0.20 +
            mel          * 0.75 +
            bells        * 0.10 +
            sonar        * 0.04 +
            ambient_wet  * 0.85
        )
        self.t_abs += n / SR
        # Retorna (wet, dry) — wet va al pedalboard, dry se suma después
        return wet, ambient_dry


# ──────────────────────────────────────────────────────────
# Capas de ambiente
# ──────────────────────────────────────────────────────────

class AmbientSounds:
    def __init__(self):
        self.rng = random.Random(9999)

        # ── Radio walkie-talkie ──
        self.radio_next_start = 20.0
        self.radio_end        = 0.0
        self.radio_active     = False
        self.radio_trans_t    = 0.0
        self.radio_trans_dur  = 0.0
        # Estructura de frases para naturalidad
        self.syl_dur       = 0.0    # duración sílaba actual
        self.syl_elapsed   = 0.0    # tiempo transcurrido en sílaba
        self.gap_dur       = 0.0    # duración pausa entre sílabas
        self.gap_elapsed   = 0.0
        self.syl_count     = 0      # sílabas en frase actual
        self.syl_max       = 0      # sílabas por frase
        self.phrase_gap    = 0.0    # pausa entre frases
        self.phrase_gap_el = 0.0
        self.in_gap        = False
        self.in_phrase_gap = False
        # Formantes por frase (cambian entre frases)
        self.f1 = 520.0; self.f2 = 1300.0; self.f3 = 2100.0
        self.f0 = 135.0
        # Filtros formante (continuos entre chunks)
        self.f_lp  = [0.0, 0.0, 0.0]
        self.f_bp  = [0.0, 0.0, 0.0]
        self.sq_lp = 0.0; self.sq_bp = 0.0
        self.buzz_ph  = 0.0; self.beep_ph = 0.0
        self.static_level = 0.0     # nivel de estática, flota aleatoriamente
        self.static_target = 0.008

        # ── Ventiladores ──
        self.fan_zi1 = None; self.fan_zi2 = None
        self.fan_rng = np.random.RandomState(7)

        # ── Pasos ──
        self.step_next  = 8.0
        self.step_queue = []

        # ── Dings metálicos ──
        self.ding_next   = 18.0
        self.ding_active = []

        # ── Carrito ──
        self.cart_next   = 100.0
        self.cart_active = None

        # ── Agua (bosque/submarino) ──
        self.water_rng = np.random.RandomState(42)
        self.water_zi  = None
        self.bubble_queue = []      # [(t_abs, freq, amp, decay)]
        self.bubble_next  = 2.0
        self.water_lfo_t  = 0.0

        # ── Knock / golpe puerta ──
        self.knock_next   = 120.0
        self.knock_queue  = []
        self.knock_hit_i  = 0

        # ── Lluvia ──
        self.rain_rng   = np.random.RandomState(77)
        self.rain_zi    = None
        self.rain_lfo_t = 0.0
        self.drop_queue = []
        self.drop_next  = 1.0

        # ── Hum eléctrico (50 Hz + armónicos) ──
        self.hum_ph = [0.0, 0.0, 0.0, 0.0]   # fases para 50/100/150/200 Hz

        # ── Blizzard (viento en ráfagas) ──
        self.gust_rng   = np.random.RandomState(13)
        self.gust_zi    = None
        self.gust_lfo_t = 0.0

        # ── Crujido estructural ──
        self.creak_next  = 40.0
        self.creak_queue = []
        self.creak_rng   = random.Random(31337)

        # ── Murmullo distante ciudad/maquinaria ──
        self.city_rng = np.random.RandomState(55)
        self.city_zi  = None

        # ── Cassette hiss nocturno ──
        self.cass_rng = np.random.RandomState(88)
        self.cass_zi  = None

    # ── Radio walkie-talkie ─────────────────────────────────
    def _new_phrase(self):
        """Inicializa parámetros para una nueva 'frase' de radio."""
        self.syl_max       = self.rng.randint(3, 7)
        self.syl_count     = 0
        self.syl_dur       = self.rng.uniform(0.15, 0.45)
        self.syl_elapsed   = 0.0
        self.in_gap        = False
        # Formantes cambian por frase (distinta "palabra")
        self.f1 = self.rng.uniform(400, 800)
        self.f2 = self.rng.uniform(1100, 1900)
        self.f3 = self.rng.uniform(1900, 2800)
        self.f0 = self.rng.uniform(100, 180)
        self.static_target = self.rng.uniform(0.005, 0.015)

    def _radio_sample(self, t_global, trans_t, trans_dur):
        """Síntesis formante con estructura natural de frases/sílabas."""
        s = 0.0

        # Estática siempre presente (flota suavemente)
        self.static_level += (self.static_target - self.static_level) * 0.001
        static = self.rng.uniform(-1, 1) * self.static_level

        # Beep apertura
        if trans_t < 0.15:
            self.beep_ph = (self.beep_ph + 2*math.pi*1080/SR) % (2*math.pi)
            s = math.sin(self.beep_ph) * 0.06 * (1 - trans_t/0.15)
            s += self.rng.uniform(-1, 1) * 0.07 * math.exp(-40*trans_t)
            return s

        # Squelch cierre
        if trans_t > trans_dur - 0.12:
            ct = trans_t - (trans_dur - 0.12)
            return static + self.rng.uniform(-1, 1) * 0.08 * (1 - ct/0.12)

        # Estructura de frases
        if self.in_phrase_gap:
            self.phrase_gap_el += 1/SR
            if self.phrase_gap_el >= self.phrase_gap:
                self.in_phrase_gap = False
                self._new_phrase()
            return static * 0.5  # menos estática entre frases

        if self.in_gap:
            self.gap_elapsed += 1/SR
            if self.gap_elapsed >= self.gap_dur:
                self.in_gap = False
                self.syl_dur     = self.rng.uniform(0.15, 0.45)
                self.syl_elapsed = 0.0
            return static * 0.7  # estática reducida entre sílabas

        # Sílaba activa
        self.syl_elapsed += 1/SR
        if self.syl_elapsed >= self.syl_dur:
            self.syl_count += 1
            if self.syl_count >= self.syl_max:
                # Fin de frase: pausa larga
                self.in_phrase_gap = True
                self.phrase_gap    = self.rng.uniform(0.25, 0.65)
                self.phrase_gap_el = 0.0
            else:
                # Fin de sílaba: pausa corta
                self.in_gap      = True
                self.gap_dur     = self.rng.uniform(0.04, 0.18)
                self.gap_elapsed = 0.0
            return static

        # Síntesis de voz: pulso + formantes
        # f0 varía ligeramente en la sílaba (inflexión)
        syl_progress = self.syl_elapsed / max(self.syl_dur, 0.001)
        f0_mod = self.f0 * (1.0 + 0.08 * math.sin(math.pi * syl_progress))
        self.buzz_ph = (self.buzz_ph + 2*math.pi*f0_mod/SR) % (2*math.pi)
        # Onda de pulso (más realista que cuadrada pura)
        buzz = 0.50 if self.buzz_ph < 0.25*2*math.pi else (-0.50 if self.buzz_ph < 0.50*2*math.pi else 0.0)
        sib  = self.rng.uniform(-1, 1) * 0.10

        # Envolvente de sílaba (anti-clic)
        env = math.sin(math.pi * syl_progress) ** 0.5

        src = (buzz * 0.7 + sib * 0.3) * env
        fmix = 0.0
        for idx, (fc, fg) in enumerate([(self.f1, 0.44), (self.f2, 0.34), (self.f3, 0.22)]):
            fv = 2 * math.sin(math.pi * fc / SR)
            notch = src - 0.16 * self.f_bp[idx]
            self.f_lp[idx] += fv * self.f_bp[idx]
            hp = notch - self.f_lp[idx]
            self.f_bp[idx] = fv * hp + self.f_bp[idx]
            fmix += self.f_bp[idx] * fg

        voiced = math.tanh(fmix * 4.5) * 0.05

        # Filtro bandpass walkie (300–2600 Hz)
        f_r = 2 * math.sin(math.pi * 1450 / SR)
        notch_r = voiced - 0.5 * self.sq_bp
        self.sq_lp += f_r * self.sq_bp
        hp_r = notch_r - self.sq_lp
        self.sq_bp = f_r * hp_r + self.sq_bp

        return self.sq_bp * 0.88 + static

    def _radio_chunk(self, t_abs, n, radio_gain):
        if radio_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        for i in range(n):
            ta = t_abs + i / SR
            if not self.radio_active and ta >= self.radio_next_start:
                self.radio_active    = True
                self.radio_trans_dur = self.rng.uniform(7, 16)
                self.radio_end       = ta + self.radio_trans_dur
                self.radio_trans_t   = 0.0
                # Reset filtros y nueva frase inicial
                self.f_lp[:] = [0, 0, 0]; self.f_bp[:] = [0, 0, 0]
                self.sq_lp = self.sq_bp = 0
                self._new_phrase()
                # Gap entre transmisiones: 25-70s
                self.radio_next_start = self.radio_end + self.rng.uniform(25, 70)
            if self.radio_active:
                if ta >= self.radio_end:
                    self.radio_active = False
                else:
                    out[i] = self._radio_sample(ta, self.radio_trans_t, self.radio_trans_dur) * radio_gain
                    self.radio_trans_t += 1 / SR
        return out

    # ── Ventiladores ─────────────────────────────────────
    def _fan_chunk(self, n, fan_gain):
        if fan_gain <= 0:
            return np.zeros(n)
        noise = self.fan_rng.uniform(-1, 1, n).astype(np.float64)
        nyq = SR / 2.0

        def fan_sos(fc, bw=40):
            lo = np.clip((fc - bw/2) / nyq, 0.005, 0.490)
            hi = np.clip((fc + bw/2) / nyq, lo + 0.005, 0.495)
            return sps.butter(2, [lo, hi], btype='band', output='sos')

        sos1 = fan_sos(112); sos2 = fan_sos(187)
        if self.fan_zi1 is None:
            self.fan_zi1 = sps.sosfilt_zi(sos1) * noise[0]
            self.fan_zi2 = sps.sosfilt_zi(sos2) * noise[0]
        f1, self.fan_zi1 = sps.sosfilt(sos1, noise, zi=self.fan_zi1)
        f2, self.fan_zi2 = sps.sosfilt(sos2, noise, zi=self.fan_zi2)
        t = np.arange(n) / SR
        lfo = 0.9 + 0.10 * np.sin(2*np.pi*0.025*t)
        return (f1 * 0.55 + f2 * 0.45) * lfo * 0.018 * fan_gain

    # ── Pasos ─────────────────────────────────────────────
    def _step_chunk(self, t_abs, n, density):
        out = np.zeros(n)
        while self.step_next < t_abs + n / SR:
            gap  = self.rng.uniform(5, 14) / max(density, 0.1)
            amp  = self.rng.uniform(0.020, 0.048)
            freq = self.rng.uniform(52, 88)
            dec  = self.rng.uniform(28, 45)
            self.step_queue.append((self.step_next, amp, freq, dec))
            self.step_queue.append((self.step_next + self.rng.uniform(0.18, 0.28),
                                    amp * 0.70, freq * self.rng.uniform(0.88, 1.12), dec))
            self.step_next += gap
        still = []
        for (ts, amp, freq, dec) in self.step_queue:
            s = int((ts - t_abs) * SR)
            if s >= n:
                still.append((ts, amp, freq, dec)); continue
            if s < -int(2 * SR): continue
            dur = min(int(1.5 * SR), n - max(s, 0))
            lt = np.arange(dur) / SR
            wave = np.sin(2*np.pi*freq*lt) * amp * np.exp(-dec * lt)
            start = max(s, 0)
            out[start:start+dur] += wave[:dur]
            if s >= 0:
                still.append((ts, amp, freq, dec))
        self.step_queue = [(ts,a,f,d) for (ts,a,f,d) in still if ts > t_abs - 2.0]
        return out

    # ── Dings metálicos (amplitud reducida) ───────────────
    def _ding_chunk(self, t_abs, n, density, ding_gain):
        if ding_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        while self.ding_next < t_abs + n / SR:
            freq  = self.rng.uniform(900, 3200)
            amp   = self.rng.uniform(0.007, 0.022) * ding_gain
            decay = self.rng.uniform(4, 12)
            self.ding_active.append((self.ding_next, freq, amp, decay))
            # Mucho menos frecuentes: 20-70s entre dings
            self.ding_next += self.rng.uniform(20, 70) / max(density, 0.1)
        still = []
        for (ts, freq, amp, decay) in self.ding_active:
            s = int((ts - t_abs) * SR)
            if s >= n:
                still.append((ts, freq, amp, decay)); continue
            dur = min(int(3 * SR), n - max(s, 0))
            lt  = np.arange(dur) / SR
            wave = (np.sin(2*np.pi*freq*lt) * 0.60 +
                    np.sin(2*np.pi*freq*2.76*lt) * 0.40) * amp * np.exp(-decay * lt)
            start = max(s, 0)
            out[start:start+dur] += wave[:dur]
        self.ding_active = [(ts,f,a,d) for (ts,f,a,d) in still]
        return out

    # ── Carrito rodante ───────────────────────────────────
    def _cart_chunk(self, t_abs, n):
        out = np.zeros(n)
        if self.cart_active is None and t_abs >= self.cart_next:
            dur  = self.rng.uniform(3, 7)
            fb   = self.rng.uniform(38, 68)
            self.cart_active = (t_abs, dur, fb)
            self.cart_next   = t_abs + dur + self.rng.uniform(100, 280)
        if self.cart_active:
            ts, dur, fb = self.cart_active
            for i in range(n):
                ta  = t_abs + i / SR
                age = ta - ts
                if age < 0 or age > dur: continue
                if   age < 0.5:      env = age / 0.5
                elif age > dur - 0.5: env = (dur - age) / 0.5
                else:                 env = 1.0
                rumble = self.rng.uniform(-1, 1) * 0.030
                wheel  = math.sin(2*math.pi*(fb + 4*math.sin(age*0.6))*age) * 0.022
                out[i] += (rumble + wheel) * env * 0.45
            if t_abs > ts + dur:
                self.cart_active = None
        return out

    # ── Agua: burbujas + flujo (bosque / submarino) ───────
    def _water_chunk(self, t_abs, n, water_gain):
        if water_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)

        # Flujo continuo: ruido bandpass modulado (200-700 Hz para underwater, más alto para bosque)
        noise = self.water_rng.uniform(-1, 1, n).astype(np.float64)
        nyq = SR / 2.0
        if self.water_zi is None:
            lo = np.clip(250 / nyq, 0.005, 0.490)
            hi = np.clip(600 / nyq, lo + 0.01, 0.495)
            sos_f = sps.butter(2, [lo, hi], btype='band', output='sos')
            self.water_zi = sps.sosfilt_zi(sos_f) * 0.0

        lo = np.clip(250 / nyq, 0.005, 0.490)
        hi = np.clip(620 / nyq, lo + 0.01, 0.495)
        sos_f = sps.butter(2, [lo, hi], btype='band', output='sos')
        flow_raw, _ = sps.sosfilt(sos_f, noise, zi=self.water_zi.copy())

        # LFO lento para flujo de agua que varía
        t = np.arange(n) / SR
        lfo_flow = 0.6 + 0.4 * np.abs(np.sin(2*np.pi*0.04*(t + self.water_lfo_t)))
        self.water_lfo_t += n / SR
        out += flow_raw * lfo_flow * 0.015 * water_gain

        # Burbujas: click cortos con frecuencia aleatoria
        while self.bubble_next < t_abs + n / SR:
            freq  = self.rng.uniform(180, 900)
            amp   = self.rng.uniform(0.020, 0.055) * water_gain
            decay = self.rng.uniform(35, 100)
            self.bubble_queue.append((self.bubble_next, freq, amp, decay))
            # Más burbujas si el agua_gain es alto (submarino)
            self.bubble_next += self.rng.uniform(0.08, 0.5) / max(water_gain, 0.1)

        still = []
        for (ts, freq, amp, decay) in self.bubble_queue:
            s = int((ts - t_abs) * SR)
            if s >= n:
                still.append((ts, freq, amp, decay)); continue
            dur = min(int(0.08 * SR), n - max(s, 0))
            lt  = np.arange(dur) / SR
            wave = np.sin(2*np.pi*freq*lt) * amp * np.exp(-decay * lt)
            start = max(s, 0)
            out[start:start+dur] += wave[:dur]
        self.bubble_queue = [(ts,f,a,d) for (ts,f,a,d) in still if ts > t_abs - 0.5]
        return np.clip(out, -0.3, 0.3)

    # ── Knock / golpe de puerta (psicoseo ocasional) ──────
    def _knock_chunk(self, t_abs, n, scene):
        out = np.zeros(n)
        if t_abs < self.knock_next:
            return out
        if not self.knock_queue:
            # Programar 2-4 golpes
            n_hits = self.rng.randint(2, 5)
            is_metal = scene in ('orbital', 'submarina')
            freq_base = self.rng.uniform(180, 320) if is_metal else self.rng.uniform(80, 180)
            amp_base  = self.rng.uniform(0.04, 0.09)
            for i in range(n_hits):
                delay = i * self.rng.uniform(0.28, 0.45)
                amp   = amp_base * self.rng.uniform(0.7, 1.0)
                decay = 22 if is_metal else 14
                self.knock_queue.append((t_abs + delay, freq_base, amp, decay))
            # Próximo knock: 2-8 minutos
            self.knock_next = t_abs + self.rng.uniform(120, 480)

        still = []
        for (ts, freq, amp, decay) in self.knock_queue:
            s = int((ts - t_abs) * SR)
            if s >= n:
                still.append((ts, freq, amp, decay)); continue
            dur = min(int(0.35 * SR), n - max(s, 0))
            lt  = np.arange(dur) / SR
            wave = (np.sin(2*np.pi*freq*lt) * 0.65 +
                    np.sin(2*np.pi*freq*1.41*lt) * 0.35) * amp * np.exp(-decay * lt)
            start = max(s, 0)
            out[start:start+dur] += wave[:dur]
        self.knock_queue = [(ts,f,a,d) for (ts,f,a,d) in still]
        return out

    # ── Lluvia procedural ─────────────────────────────────
    def _rain_chunk(self, t_abs, n, rain_gain, is_snow=False, rain_glass=False):
        """Hiss de lluvia/nieve + gotas individuales + resonancia de vidrio."""
        if rain_gain <= 0:
            return np.zeros(n)
        noise = self.rain_rng.uniform(-1, 1, n).astype(np.float64)
        nyq = SR / 2.0

        lo = np.clip(150 / nyq if is_snow else 700 / nyq, 0.005, 0.490)
        hi = np.clip(2500 / nyq if is_snow else 14000 / nyq, lo + 0.01, 0.495)
        sos = sps.butter(3, [lo, hi], btype='band', output='sos')
        if self.rain_zi is None:
            self.rain_zi = sps.sosfilt_zi(sos) * noise[0]
        hiss, self.rain_zi = sps.sosfilt(sos, noise, zi=self.rain_zi)

        t = np.arange(n) / SR
        lfo = 0.75 + 0.25 * np.abs(np.sin(2*np.pi*0.022*(t + self.rain_lfo_t)))
        self.rain_lfo_t += n / SR

        out = hiss * lfo * (0.022 if is_snow else 0.035) * rain_gain

        if not is_snow:
            while self.drop_next < t_abs + n / SR:
                freq = self.rng.uniform(1000, 5000)
                amp  = self.rng.uniform(0.006, 0.020) * rain_gain
                dec  = self.rng.uniform(80, 300)
                self.drop_queue.append((self.drop_next, freq, amp, dec))
                self.drop_next += self.rng.uniform(0.008, 0.06) / max(rain_gain, 0.1)
            still = []
            for (ts, freq, amp, dec) in self.drop_queue:
                s = int((ts - t_abs) * SR)
                if s >= n:
                    still.append((ts, freq, amp, dec)); continue
                start = max(s, 0)
                dur = min(int(0.04 * SR), n - start)
                lt  = np.arange(dur) / SR
                out[start:start+dur] += np.sin(2*np.pi*freq*lt) * amp * np.exp(-dec * lt)
                if rain_glass:
                    g_freq = 1900 + freq * 0.22
                    g_dur  = min(int(0.10 * SR), n - start)
                    g_lt   = np.arange(g_dur) / SR
                    out[start:start+g_dur] += (np.sin(2*np.pi*g_freq*g_lt) *
                                               amp * 0.45 * np.exp(-55*g_lt))
            self.drop_queue = [(ts,f,a,d) for (ts,f,a,d) in still if ts > t_abs - 0.1]

        return np.clip(out, -0.35, 0.35)

    # ── Hum eléctrico (50 Hz + armónicos) ────────────────
    def _elec_hum_chunk(self, n, elec_hum):
        """Zumbido de transformador — orbital y submarina."""
        if not elec_hum:
            return np.zeros(n)
        t     = np.arange(n) / SR
        freqs = [50.0, 100.0, 150.0, 200.0]
        amps  = [0.011, 0.006, 0.003, 0.0015]
        out   = np.zeros(n)
        for i, (f, a) in enumerate(zip(freqs, amps)):
            out += np.sin(self.hum_ph[i] + 2*np.pi*f*t) * a
            self.hum_ph[i] = (self.hum_ph[i] + 2*np.pi*f*n/SR) % (2*np.pi)
        return out * (0.85 + 0.15 * np.sin(2*np.pi*6.7*t))

    # ── Blizzard: viento en ráfagas ───────────────────────
    def _blizzard_chunk(self, t_abs, n, wind_gain):
        """Overlay gusty sobre el viento base — nieve y montaña."""
        if wind_gain <= 0:
            return np.zeros(n)
        noise = self.gust_rng.uniform(-1, 1, n).astype(np.float64)
        nyq = SR / 2.0
        lo = np.clip(80 / nyq, 0.005, 0.490)
        hi = np.clip(950 / nyq, lo + 0.01, 0.495)
        sos = sps.butter(3, [lo, hi], btype='band', output='sos')
        if self.gust_zi is None:
            self.gust_zi = sps.sosfilt_zi(sos) * noise[0]
        gust_raw, self.gust_zi = sps.sosfilt(sos, noise, zi=self.gust_zi)
        t = np.arange(n) / SR
        lfo = (0.45 + 0.35 * np.abs(np.sin(2*np.pi*0.07*(t + self.gust_lfo_t))) +
               0.20 * np.sin(2*np.pi*0.19*(t + self.gust_lfo_t)) ** 2)
        self.gust_lfo_t += n / SR
        return gust_raw * lfo * 0.018 * wind_gain

    # ── Crujido estructural ────────────────────────────────
    def _creak_chunk(self, t_abs, n, creak_gain):
        """El edificio cruje con el viento — nieve / submarino."""
        if creak_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        while self.creak_next < t_abs + n / SR:
            freq = self.creak_rng.uniform(170, 380)
            amp  = self.creak_rng.uniform(0.025, 0.065) * creak_gain
            dur  = self.creak_rng.uniform(0.25, 1.0)
            self.creak_queue.append((self.creak_next, freq, amp, dur))
            self.creak_next += self.creak_rng.uniform(12, 40)
        still = []
        for (ts, freq, amp, dur) in self.creak_queue:
            s = int((ts - t_abs) * SR)
            if s >= n:
                still.append((ts, freq, amp, dur)); continue
            d = min(int(dur * SR), n - max(s, 0))
            if d <= 0: continue
            lt    = np.arange(d) / SR
            # Pitch glide descendente (un crujido real baja de tono)
            f_arr = freq * (1.0 - 0.28 * lt / max(dur, 0.001))
            phase = 2*np.pi * np.cumsum(f_arr) / SR
            env   = np.sin(np.pi * lt / max(dur, 0.001)) ** 0.7
            start = max(s, 0)
            out[start:start+d] += np.sin(phase) * amp * env
        self.creak_queue = [(ts,f,a,d) for (ts,f,a,d) in still if ts > t_abs - 2.0]
        return out

    # ── Murmullo distante ciudad / maquinaria ──────────────
    def _city_chunk(self, t_abs, n, city_gain):
        """Rumble distante de sala de máquinas — orbital."""
        if city_gain <= 0:
            return np.zeros(n)
        noise = self.city_rng.uniform(-1, 1, n).astype(np.float64)
        nyq = SR / 2.0
        lo = np.clip(55 / nyq, 0.005, 0.490)
        hi = np.clip(230 / nyq, lo + 0.01, 0.495)
        sos = sps.butter(2, [lo, hi], btype='band', output='sos')
        if self.city_zi is None:
            self.city_zi = sps.sosfilt_zi(sos) * 0.0
        rumble, self.city_zi = sps.sosfilt(sos, noise, zi=self.city_zi)
        t = np.arange(n) / SR
        lfo = 0.70 + 0.30 * np.sin(2*np.pi*0.031*(t + t_abs))
        return rumble * lfo * 0.016 * city_gain

    # ── Tick de reloj subliminal (4AM) ────────────────────
    def _clock_chunk(self, t_abs, n, clock_night):
        """Tick casi inaudible — solo entre las 22h y las 5AM."""
        if not clock_night:
            return np.zeros(n)
        if 5 <= time.localtime().tm_hour < 22:
            return np.zeros(n)
        out = np.zeros(n)
        for beat in range(int(t_abs), int(t_abs + n / SR) + 2):
            s = int((float(beat) - t_abs) * SR)
            if 0 <= s < n:
                dur = min(int(0.014 * SR), n - s)
                lt  = np.arange(dur) / SR
                out[s:s+dur] += np.sin(2*np.pi*3400*lt) * 0.007 * np.exp(-420*lt)
        return out

    # ── Cassette hiss nocturno ────────────────────────────
    def _cassette_chunk(self, n, cass_level):
        """Textura de cinta de cassette — solo de noche (18h-5AM)."""
        if cass_level <= 0:
            return np.zeros(n)
        noise = self.cass_rng.uniform(-1, 1, n).astype(np.float64)
        nyq = SR / 2.0
        lo = np.clip(3800 / nyq, 0.005, 0.490)
        hi = np.clip(11000 / nyq, lo + 0.01, 0.495)
        sos = sps.butter(2, [lo, hi], btype='band', output='sos')
        if self.cass_zi is None:
            self.cass_zi = sps.sosfilt_zi(sos) * noise[0]
        hiss, self.cass_zi = sps.sosfilt(sos, noise, zi=self.cass_zi)
        if 5 <= time.localtime().tm_hour < 18:
            return np.zeros(n)
        return hiss * cass_level

    # ── Mezcla ────────────────────────────────────────────
    def generate_chunk(self, t_abs, n, state):
        prof = SCENE_PROFILES.get(state.scene, SCENE_PROFILES[DEFAULT_SCENE])

        radio    = self._radio_chunk(t_abs, n, prof['radio_gain'])
        fan      = self._fan_chunk(n, prof['fan_gain'])
        steps    = self._step_chunk(t_abs, n, state.density)
        dings    = self._ding_chunk(t_abs, n, state.density, prof['ding_gain'])
        cart     = self._cart_chunk(t_abs, n)
        water    = self._water_chunk(t_abs, n, prof['water_gain'])
        knock    = self._knock_chunk(t_abs, n, state.scene)
        blizzard = self._blizzard_chunk(t_abs, n, prof['wind_gain']) if prof['blizzard'] else np.zeros(n)
        creak    = self._creak_chunk(t_abs, n, prof['creak_gain'])

        rain_gain = prof['rain_base'] + state.rain_intensity * prof['rain_wx_mul']
        rain = self._rain_chunk(t_abs, n, rain_gain,
                                is_snow=prof['snow'], rain_glass=prof['rain_glass'])

        # Señal que va al pedalboard (chorus + reverb)
        wet_mix = (radio + fan + steps + dings + cart + water + knock * 0.5 +
                   rain + blizzard + creak)

        # Señal DRY: bypass pedalboard — no chorus, no reverb, no cola de ruido
        hum      = self._elec_hum_chunk(n, prof['elec_hum'])
        city     = self._city_chunk(t_abs, n, prof['city_gain'])
        clock    = self._clock_chunk(t_abs, n, prof['clock_night'])
        cassette = self._cassette_chunk(n, prof['cassette'])
        dry_mix  = hum + city + clock + cassette

        return wet_mix, dry_mix


# ──────────────────────────────────────────────────────────
# Pedalboard — se actualiza in-place (no rebuild)
# ──────────────────────────────────────────────────────────

def build_board(state):
    if not HAS_PEDALBOARD:
        return None
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=55.0),   # corta sub-bass inaudible
        Chorus(rate_hz=0.25 + state.brightness * 0.35,
               depth=0.18 + (1 - state.brightness) * 0.22,
               centre_delay_ms=7.0, feedback=0.10, mix=state.chorus_mix),
        Reverb(room_size=state.reverb_room,
               damping=0.3 + state.brightness * 0.35,
               wet_level=state.reverb_wet,
               dry_level=1.0 - state.reverb_wet * 0.65),
        # Compressor más suave — menos pumping en señal ambient continua
        Compressor(threshold_db=-14, ratio=2.0, attack_ms=150, release_ms=1200),
        LowpassFilter(cutoff_frequency_hz=5200 + state.brightness * 4000),
        Gain(gain_db=1.5),
    ])


def update_board_inplace(board, state):
    """Actualiza parámetros sin reconstruir (board[0]=HPF fijo, no se toca)."""
    if board is None:
        return
    # board[0] = HighpassFilter — cutoff fijo, no cambia
    board[1].rate_hz   = 0.25 + state.brightness * 0.35
    board[1].depth     = 0.18 + (1 - state.brightness) * 0.22
    board[1].mix       = state.chorus_mix
    board[2].room_size = state.reverb_room
    board[2].damping   = 0.3 + state.brightness * 0.35
    board[2].wet_level = state.reverb_wet
    board[2].dry_level = 1.0 - state.reverb_wet * 0.65
    # board[3] = Compressor — parámetros fijos, no cambia
    board[4].cutoff_frequency_hz = 5200 + state.brightness * 4000


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    state = AudioState()
    synth = Synthesizer()
    last_weather = last_network = last_system = last_viewers = 0
    board = None
    chunk_count = 0

    print("[INFO] audio_engine iniciado. Escribiendo PCM s16le a stdout...", file=sys.stderr)

    try:
        while True:
            now = time.time()

            if now - last_weather > WEATHER_INTERVAL:
                try:
                    state.temp, state.humidity, state.wind_spd, state.w_code = fetch_weather()
                    print(f"[INFO] Clima: {state.temp}°C {state.humidity}% {state.wind_spd}km/h", file=sys.stderr)
                except Exception as e:
                    print(f"[WARN] Clima: {e}", file=sys.stderr)
                last_weather = now

            if now - last_network > NETWORK_INTERVAL:
                try:
                    state.devices = fetch_network_devices()
                except Exception:
                    pass
                last_network = now

            if now - last_system > SYSTEM_INTERVAL:
                try:
                    state.load1, state.load_ram_pct = fetch_system_load()
                except Exception:
                    pass
                last_system = now

            if now - last_viewers > 30:
                state.viewers = fetch_viewers()
                last_viewers  = now

            state.update_targets()
            state.lerp_step(CHUNK)
            state.export()

            # Inicializar board una sola vez; luego actualizar in-place
            if board is None:
                board = build_board(state)
            else:
                update_board_inplace(board, state)

            wet, dry = synth.generate_chunk(state)
            wet = np.nan_to_num(wet, nan=0.0, posinf=0.8, neginf=-0.8)
            wet = np.clip(wet, -1.0, 1.0)

            if board and HAS_PEDALBOARD:
                wet_f32 = np.clip(wet.astype(np.float32), -1.0, 1.0).reshape(1, -1)
                processed = board(wet_f32, SR)
                wet = np.nan_to_num(processed.flatten().astype(np.float64), nan=0.0)

            # Dry bypass: hum eléctrico, city, clock, cassette — sin chorus/reverb
            dry = np.nan_to_num(dry, nan=0.0)
            audio = wet + dry

            audio = np.tanh(audio * 0.88)
            peak  = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.84

            sys.stdout.buffer.write((audio * 32000).astype(np.int16).tobytes())
            sys.stdout.buffer.flush()

            chunk_count += 1
            if chunk_count % 15 == 0:
                prof = SCENE_PROFILES.get(state.scene, {})
                label = prof.get('label', state.scene)
                print(f"[INFO] {label} | bright={state.brightness:.2f} density={state.density:.2f} energy={state.energy:.2f}", file=sys.stderr)

    except BrokenPipeError:
        print("[INFO] Pipe cerrado.", file=sys.stderr)
    except KeyboardInterrupt:
        print("[INFO] Detenido.", file=sys.stderr)


if __name__ == "__main__":
    main()
