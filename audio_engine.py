#!/usr/bin/env python3
"""
audio_engine.py — Generador de audio ambiental para stream 24/7.

Escena activa: /tmp/star_scene.txt (orbital|nieve|bosque|submarina|montana|desierto)
Cada escena tiene un perfil de audio radicalmente distinto.
"""

import sys, os, time, json, math, random, sqlite3, urllib.request, threading
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
AUDIO_SNAPSHOT   = "/tmp/star_audio_snapshot.json"
SNAPSHOT_MAX_AGE = 6 * 3600
SNAPSHOT_INTERVAL = 30
STARTUP_FADE_SECONDS = 10.0

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
        [55.00, 110.00, 164.82, 220.00, 329.62, 440.00],
        [61.74, 123.48, 185.00, 261.62, 370.00, 493.88],
        [65.42, 130.82, 196.00, 261.62, 311.12, 466.16],
        [58.28, 116.54, 174.62, 233.08, 349.22, 466.16],
    ],
    # Montaña: amplio, viento, elevado
    'mountain': [
        [65.41, 164.81, 196.00, 261.63, 392.00, 523.25],
        [73.42, 146.83, 220.00, 261.63, 349.23, 440.00],
        [82.41, 164.81, 207.65, 261.63, 311.13, 415.30],
        [87.31, 174.61, 233.08, 261.63, 349.23, 466.16],
    ],
    # Reactor nuclear: zumbido industrial, cromatismo tenso
    'reactor': [
        [55.00, 110.00, 155.56, 207.65, 233.08],
        [61.74, 123.47, 174.61, 220.00, 246.94],
        [69.30, 138.59, 185.00, 246.94, 277.18],
        [58.27, 116.54, 164.81, 220.00, 261.63],
    ],
    # Volcánica: movimiento telúrico, tensión de magma
    'volcanic': [
        [49.00, 98.00, 146.84, 196.00, 277.18, 370.00],
        [55.00, 110.00, 155.56, 220.00, 311.12, 415.30],
        [46.24, 92.50, 138.60, 185.00, 261.62, 349.22],
        [51.92, 103.82, 146.84, 207.66, 293.66, 392.00],
    ],
}

BELL_SETS = {
    # Bajados una octava respecto al original — máximo 440 Hz para que no chirrién
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
    # Bajos y espaciados — sin notas > 300 Hz para evitar chirriante
    'dark_station': [146.83, 130.81, 110.00, 123.47, 164.81, 146.83, 130.81, 98.00,
                     155.56, 130.81, 123.47, 110.00, 174.61, 146.83],
    'arctic':       [196.00, 146.83, 130.81, 164.81, 146.83, 174.61, 155.56, 130.81,
                     207.65, 155.56, 138.59, 164.81, 196.00, 146.83],
    'forest':       [196.00, 164.81, 130.81, 164.81, 220.00, 196.00, 174.61, 146.83,
                     261.63, 196.00, 164.81, 130.81, 220.00, 185.00],
    'desert':       [220.00, 196.00, 164.81, 174.61, 146.83, 196.00, 220.00, 185.00,
                     246.94, 196.00, 174.61, 155.56, 207.65, 185.00],
    'deep_sea':     [55.00,  65.41,  82.41,  73.42,  61.74,  55.00,  49.00,  58.27,
                     69.30,  77.78,  87.31,  73.42,  65.41,  55.00],
    'mountain':     [196.00, 164.81, 146.83, 174.61, 196.00, 220.00, 164.81, 196.00,
                     246.94, 196.00, 174.61, 164.81, 220.00, 185.00],
    'reactor':      [103.83, 110.00, 116.54, 103.83,  92.50, 110.00, 123.47, 103.83,
                      87.31, 110.00, 130.81, 116.54,  98.00, 110.00],
    'volcanic':     [ 36.71,  41.20,  49.00,  43.65,  36.71,  41.20,  32.70,  41.20,
                      55.00,  49.00,  43.65,  36.71,  46.25,  41.20],
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
        # fan reducido (1.6→1.0) para menos ruido broadband de ventilación
        'fan_gain': 1.0, 'radio_gain': 0.42, 'water_gain': 0.0,
        'wind_gain': 0.5, 'whistle_gain': 0.22, 'bell_gain': 0.18, 'ding_gain': 0.7,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': True, 'rain_glass': True, 'blizzard': False,
        'creak_gain': 0.0, 'corona_gain': 0.0, 'city_gain': 0.12, 'clock_night': True, 'cassette': 0.003,
        'label': 'ORBITAL STATION',
        # Soporte vital + comunicaciones fragmentadas SCP
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.20, 'morse_gain': 0.06, 'anomaly_gain': 0.0,
        'rcs_gain': 0.85, 'telemetry_gain': 0.90,
    },
    'nieve':     {
        'chord': 'arctic', 'reverb_base': 0.90, 'wet_base': 0.48,
        'chorus_base': 0.34, 'sub_base': 0.012,
        'fan_gain': 0.2, 'radio_gain': 0.28, 'water_gain': 0.0,
        # wind_gain reducido (1.8→0.85): el blizzard ya aporta textura, el viento no debe saturar
        'wind_gain': 0.85, 'whistle_gain': 0.40, 'bell_gain': 0.20, 'ding_gain': 0.3,
        'rain_base': 0.22, 'rain_wx_mul': 0.35, 'snow': True,
        'elec_hum': False, 'rain_glass': True, 'blizzard': True,
        'creak_gain': 0.5, 'corona_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'ARCTIC OUTPOST',
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        'glacier_gain': 0.70, 'wire_gain': 0.75,
    },
    'bosque':    {
        'chord': 'forest', 'reverb_base': 0.78, 'wet_base': 0.38,
        'chorus_base': 0.28, 'sub_base': 0.010,
        'fan_gain': 0.0, 'radio_gain': 0.22, 'water_gain': 0.5,
        'wind_gain': 0.9, 'whistle_gain': 0.30, 'bell_gain': 0.22, 'ding_gain': 0.0,
        'rain_base': 0.55, 'rain_wx_mul': 0.9, 'snow': False,
        'elec_hum': False, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.0, 'corona_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'FOREST STATION',
        # Anomaly sutil — algo extraño en el bosque
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        'drops_gain': 0.90, 'sway_gain': 0.70,
    },
    'submarina': {
        'chord': 'deep_sea', 'reverb_base': 0.97, 'wet_base': 0.58,
        'chorus_base': 0.18, 'sub_base': 0.022,
        'fan_gain': 0.4, 'radio_gain': 0.28, 'water_gain': 1.0,
        'wind_gain': 0.2, 'whistle_gain': 0.08, 'bell_gain': 0.22, 'ding_gain': 0.5,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': True, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.35, 'corona_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'DEEP SEA BASE',
        # Soporte vital + anomaly profundo
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.15, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        'subcreak_gain': 0.80, 'sonar_gain': 1.0,
    },
    'montana':   {
        'chord': 'mountain', 'reverb_base': 0.92, 'wet_base': 0.55,
        'chorus_base': 0.20, 'sub_base': 0.014,
        'fan_gain': 0.0, 'radio_gain': 0.0, 'water_gain': 0.0,
        # wind_gain aumentado (0.30→0.55) para más carácter de montaña
        'wind_gain': 0.55, 'whistle_gain': 0.40, 'bell_gain': 0.14, 'ding_gain': 0.0,
        'rain_base': 0.60, 'rain_wx_mul': 0.5, 'snow': False,
        'elec_hum': False, 'rain_glass': False, 'blizzard': True,
        'creak_gain': 0.25, 'corona_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'MOUNTAIN BASE',
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        'avalanche_gain': 0.70, 'harp_gain': 0.80, 'wire_gain': 0.55,
    },
    'desierto':  {
        'chord': 'desert', 'reverb_base': 0.68, 'wet_base': 0.28,
        'chorus_base': 0.18, 'sub_base': 0.008,
        'fan_gain': 0.0, 'radio_gain': 0.0, 'water_gain': 0.0,
        'wind_gain': 0.7, 'whistle_gain': 0.22, 'bell_gain': 0.12, 'ding_gain': 0.0,
        'rain_base': 0.0, 'rain_wx_mul': 0.08, 'snow': False,
        'elec_hum': False, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.0, 'corona_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'DESERT HEAT',
        # Anomaly leve — mirages del desierto
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        'sand_gain': 0.90,
    },
    'electrica': {
        'chord': 'dark_station', 'reverb_base': 0.96, 'wet_base': 0.62,
        'chorus_base': 0.30, 'sub_base': 0.016,
        'fan_gain': 0.0, 'radio_gain': 0.0, 'water_gain': 0.0,
        'wind_gain': 0.4, 'whistle_gain': 0.55, 'bell_gain': 0.10, 'ding_gain': 0.0,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': 'dual', 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.3, 'corona_gain': 0.85, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.002,
        'label': 'ELECTRIC FIELD',
        # Anomaly fuerte + alarma + morse fragmentado = máximo SCP
        'emf_gain': 1.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.20, 'anomaly_gain': 0.0,
    },
    # ── Nuevas escenas ─────────────────────────────────────────
    'volcanica': {
        'chord': 'volcanic', 'reverb_base': 0.97, 'wet_base': 0.62,
        'chorus_base': 0.20, 'sub_base': 0.028,
        # fan removido (géiseres, no ventilador), agua de géiser añadida
        'fan_gain': 0.0, 'radio_gain': 0.18, 'water_gain': 0.25,
        'wind_gain': 0.15, 'whistle_gain': 0.10, 'bell_gain': 0.0, 'ding_gain': 0.0,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': False, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.70, 'corona_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.001,
        'label': 'GEOTHERMAL OUTPOST',
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        'magma_gain': 1.0, 'steam_gain': 0.90,
    },
    'reactor': {
        'chord': 'reactor', 'reverb_base': 0.88, 'wet_base': 0.44,
        'chorus_base': 0.18, 'sub_base': 0.015,
        'fan_gain': 1.3, 'radio_gain': 0.22, 'water_gain': 0.30,
        'wind_gain': 0.0, 'whistle_gain': 0.0, 'bell_gain': 0.0, 'ding_gain': 0.0,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': True, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.12, 'corona_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.003,
        'label': 'NUCLEAR REACTOR',
        # Alarma de fondo de reactor + anomaly sutil de radiación
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        'geiger_gain': 0.95, 'coolant_gain': 0.80, 'valve_gain': 0.72,
    },
    'tormenta': {
        'chord': 'mountain', 'reverb_base': 0.96, 'wet_base': 0.55,
        'chorus_base': 0.28, 'sub_base': 0.018,
        'fan_gain': 0.0, 'radio_gain': 0.28, 'water_gain': 0.0,
        # wind_gain reducido (1.65→0.90): la lluvia + trueno ya dan la energía, el viento queda de fondo
        'wind_gain': 0.90, 'whistle_gain': 0.55, 'bell_gain': 0.06, 'ding_gain': 0.0,
        'rain_base': 0.75, 'rain_wx_mul': 0.12, 'snow': False,
        'elec_hum': False, 'rain_glass': True, 'blizzard': False,
        'creak_gain': 0.22, 'corona_gain': 0.45, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.001,
        'label': 'STORM BUNKER',
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        'thunder_gain': 0.80,
    },
    # ── Escenas nuevas Jun 2026 ────────────────────────────────
    'array_suiza': {
        'chord': 'arctic', 'reverb_base': 0.92, 'wet_base': 0.52,
        'chorus_base': 0.22, 'sub_base': 0.010,
        'fan_gain': 0.0, 'radio_gain': 0.55, 'water_gain': 0.0,
        'wind_gain': 0.70, 'whistle_gain': 0.45, 'bell_gain': 0.12, 'ding_gain': 0.0,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': True, 'rain_glass': False, 'blizzard': True,   # viento alpino activo
        'creak_gain': 0.45, 'corona_gain': 0.20, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.003,
        'label': 'SIGNAL ARRAY — ALPS',
        'emf_gain': 0.80, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.25, 'anomaly_gain': 0.0,
        'rcs_gain': 0.0, 'telemetry_gain': 0.90,
        'wire_gain': 0.65, 'glacier_gain': 0.25,  # cables tensados + crujido metálico del array
    },
    'latam_noche': {
        'chord': 'forest', 'reverb_base': 0.65, 'wet_base': 0.30,
        'chorus_base': 0.15, 'sub_base': 0.010,
        'fan_gain': 0.0, 'radio_gain': 0.10, 'water_gain': 0.0,
        'wind_gain': 0.20, 'whistle_gain': 0.0, 'bell_gain': 0.05, 'ding_gain': 0.0,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': False, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.0, 'corona_gain': 0.0, 'city_gain': 0.55, 'clock_night': False, 'cassette': 0.004,
        'label': 'LATAM NOCHE',
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        # Perros, tráfico y motos muy al fondo — melodía domina
        'dog_gain': 0.45, 'traffic_gain': 0.50, 'moto_gain': 0.35,
        'frog_gain': 0.0, 'leaves_gain': 0.0,
    },
    'amazonica': {
        'chord': 'forest', 'reverb_base': 0.88, 'wet_base': 0.45,
        'chorus_base': 0.30, 'sub_base': 0.014,
        'fan_gain': 0.0, 'radio_gain': 0.08, 'water_gain': 0.28,
        'wind_gain': 0.10, 'whistle_gain': 0.0, 'bell_gain': 0.0, 'ding_gain': 0.0,
        'rain_base': 0.40, 'rain_wx_mul': 0.20, 'snow': False,
        'elec_hum': False, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.0, 'corona_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.001,
        'label': 'AMAZONIA PROFUNDA',
        'emf_gain': 0.0, 'siren_gain': 0.0, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        # Coro de sapos + hojas — selva densa nocturna
        'frog_gain': 0.85, 'leaves_gain': 0.60,
        'dog_gain': 0.0, 'traffic_gain': 0.0, 'moto_gain': 0.0,
        'drops_gain': 0.70, 'sway_gain': 0.55,
    },
    'scp_exterior': {
        'chord': 'arctic', 'reverb_base': 0.82, 'wet_base': 0.40,
        'chorus_base': 0.18, 'sub_base': 0.010,
        'fan_gain': 0.0, 'radio_gain': 0.18, 'water_gain': 0.0,
        'wind_gain': 0.45, 'whistle_gain': 0.20, 'bell_gain': 0.0, 'ding_gain': 0.0,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': True, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.30, 'corona_gain': 0.30, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.003,
        'label': 'SCP FOUNDATION — EXTERIOR',
        'emf_gain': 0.40, 'siren_gain': 0.25, 'breath_gain': 0.0, 'morse_gain': 0.0, 'anomaly_gain': 0.0,
        # Grillos, hojas y perro lejano — instalación abandonada con luces funcionando
        'leaves_gain': 0.75, 'frog_gain': 0.0,
        'dog_gain': 0.25, 'traffic_gain': 0.0, 'moto_gain': 0.0,
        'sway_gain': 0.50,
    },
    'scp_contencion': {
        'chord': 'dark_station', 'reverb_base': 0.96, 'wet_base': 0.60,
        'chorus_base': 0.25, 'sub_base': 0.018,
        'fan_gain': 0.0, 'radio_gain': 0.12, 'water_gain': 0.0,
        'wind_gain': 0.30, 'whistle_gain': 0.15, 'bell_gain': 0.0, 'ding_gain': 0.0,
        'rain_base': 0.0, 'rain_wx_mul': 0.0, 'snow': False,
        'elec_hum': True, 'rain_glass': False, 'blizzard': False,
        'creak_gain': 0.55, 'corona_gain': 0.0, 'city_gain': 0.0, 'clock_night': False, 'cassette': 0.003,
        'label': 'SCP CONTAINMENT PLATFORM',
        'emf_gain': 0.60, 'siren_gain': 0.50, 'breath_gain': 0.0, 'morse_gain': 0.10, 'anomaly_gain': 0.0,
        # Bosque conífero nocturno + hojas + niebla (creak simula cadenas)
        'leaves_gain': 0.50, 'frog_gain': 0.0,
        'dog_gain': 0.0, 'traffic_gain': 0.0, 'moto_gain': 0.0,
        'sway_gain': 0.65, 'glacier_gain': 0.30,
    },
}

DEFAULT_SCENE = 'orbital'


def _soften_scene_profiles():
    """Mantiene las capas en rango ambient: presentes, pero nunca protagonistas agresivas."""
    caps = {
        'bell_gain': 0.25,
        'ding_gain': 0.40,
        'whistle_gain': 0.45,
        'corona_gain': 0.65,
        'morse_gain': 0.25,
        'siren_gain': 0.15,
        'fan_gain': 0.85,
        'radio_gain': 0.45,
        'leaves_gain': 0.65,
        'frog_gain': 0.75,
        'dog_gain': 0.45,
        'moto_gain': 0.40,
        'traffic_gain': 0.65,
        'geiger_gain': 0.75,
        'wire_gain': 0.65,
        'drops_gain': 0.75,
        'sonar_gain': 0.85,
        'avalanche_gain': 0.65,
        'harp_gain': 0.65,
        'sand_gain': 0.75,
        'steam_gain': 0.75,
        'thunder_gain': 0.85,
        'valve_gain': 0.65,
        'rcs_gain': 0.65,
        'telemetry_gain': 0.75,
        'subcreak_gain': 0.75,
        'glacier_gain': 0.65,
    }
    for prof in SCENE_PROFILES.values():
        for key, cap in caps.items():
            if key in prof:
                prof[key] = min(prof[key], cap)


_soften_scene_profiles()


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


def weather_sound_cause(w_code, wind_spd, humidity, hour):
    """Describe el factor climático que domina la textura audible del momento."""
    if 95 <= w_code <= 99:
        return "tormenta"
    if 80 <= w_code <= 82:
        return "chubascos"
    if 61 <= w_code <= 67:
        return "lluvia"
    if 71 <= w_code <= 77:
        return "nieve"
    if wind_spd >= 18:
        return "viento fuerte"
    if humidity >= 75:
        return "humedad"
    if hour >= 19 or hour < 6:
        return "noche"
    return "clima estable"


def fetch_scene():
    try:
        with open(SCENE_FILE) as f:
            s = f.read().strip()
            return s if s in SCENE_PROFILES else DEFAULT_SCENE
    except Exception:
        return DEFAULT_SCENE


def day_phase(hour):
    """Multiplicadores de brightness/density/energy según hora del día.
    Noche (19:00 a 06:00): calmado, súper suave, ambiente flotante.
    Mañana (06:00 a 10:00): transición gradual.
    Día (10:00 a 19:00): activo y brillante.
    """
    if 19 <= hour or hour < 6:
        return 0.10, 0.05, 0.10
    elif 6 <= hour < 10:
        return 0.40, 0.35, 0.35
    else:
        return 0.82, 0.85, 0.78


def biquad_bandpass(fc, Q, sr=44100):
    """Diseño analítico ultra-rápido de filtro biquad paso banda (EQ Cookbook)."""
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1.0 + alpha
    b = np.array([alpha, 0.0, -alpha]) / a0
    a = np.array([a0, -2.0 * cos_w0, 1.0 - alpha]) / a0
    return b, a



def biquad_highpass(fc, Q, sr=44100):
    """Diseño analítico ultra-rápido de filtro biquad paso alto (EQ Cookbook)."""
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1.0 + alpha
    b = np.array([(1.0 + cos_w0)/2.0, -(1.0 + cos_w0), (1.0 + cos_w0)/2.0]) / a0
    a = np.array([a0, -2.0 * cos_w0, 1.0 - alpha]) / a0
    return b, a


def biquad_lowpass(fc, Q, sr=44100):
    """Diseño analítico ultra-rápido de filtro biquad paso bajo (EQ Cookbook)."""
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1.0 + alpha
    b = np.array([(1.0 - cos_w0)/2.0, 1.0 - cos_w0, (1.0 - cos_w0)/2.0]) / a0
    a = np.array([a0, -2.0 * cos_w0, 1.0 - alpha]) / a0
    return b, a


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

        # Modulaciones por clima real (Santiago)
        temp_factor = (self.temp - 15.0) / 40.0 * 0.15          # -0.05 a +0.10 aprox
        hum_factor  = (self.humidity - 50.0) / 100.0 * 0.08     # -0.04 a +0.04 aprox
        wind_factor = (self.wind_spd / 30.0) * 0.25             # 0 a 0.25+

        self.t_brightness = float(np.clip(b + temp_factor - load_pen * 0.1, 0.05, 1.0))
        self.t_density    = float(np.clip(d + device_bonus + viewer_bonus - load_pen * 0.15, 0.05, 1.0))
        self.t_energy     = float(np.clip(e + wind_factor - load_pen * 0.1, 0.05, 1.0))
        self.t_sub        = prof['sub_base'] * (1.0 + self.t_energy * 0.5)
        self.rain_intensity = weather_rain_intensity(self.w_code)
        
        # Bono de reverb nocturna para un sonido más profundo/exquisito
        night_reverb_bonus = 0.05 if (hour >= 19 or hour < 6) else 0.0
        self.t_reverb     = float(np.clip(prof['reverb_base'] + hum_factor + night_reverb_bonus, 0.50, 0.98))
        self.t_reverb_wet = float(np.clip(prof['wet_base'] + hum_factor * 0.75 + night_reverb_bonus * 0.5, 0.10, 0.85))
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

    def export(self, peak=None, rms=None, chunk_count=None, chord_label=None,
               layer_label=None, master_gain=None):
        prof  = SCENE_PROFILES.get(self.scene, SCENE_PROFILES[DEFAULT_SCENE])
        label = prof['label']
        ram_pct = f"{int(self.load_ram_pct * 100)}%"
        hour = time.localtime().tm_hour
        cause = weather_sound_cause(self.w_code, self.wind_spd, self.humidity, hour)
        metrics = {
            "chord": chord_label or self.chord_set,
            "reverb": f"{self.reverb_room:.2f}",
            "wet": f"{self.reverb_wet:.2f}",
            "cause": cause,
            "peak": "n/a" if peak is None else f"{peak:.3f}",
            "rms": "n/a" if rms is None else f"{rms:.4f}",
            "chunk": "0" if chunk_count is None else str(chunk_count),
            "layer": layer_label or "n/a",
            "gain": "n/a" if master_gain is None else f"{master_gain:.2f}",
        }
        try:
            with open(STATE_FILE, 'w') as f:
                extra = "|".join(f"{key}={value}" for key, value in metrics.items())
                f.write(f"{label}|{self.load1:.2f}|{ram_pct}|{self.devices}|{extra}")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────
# Síntesis principal
# ──────────────────────────────────────────────────────────

def load_audio_snapshot():
    """Restaura el reloj musical tras reconexiones breves de FFmpeg."""
    try:
        if not os.path.exists(AUDIO_SNAPSHOT):
            return None
        with open(AUDIO_SNAPSHOT, 'r') as f:
            data = json.load(f)
        age = time.time() - float(data.get('wall_time', 0))
        if age < 0 or age > SNAPSHOT_MAX_AGE:
            return None
        return data
    except Exception:
        return None


def save_audio_snapshot(synth):
    """Guarda estado mínimo de forma atómica para no reiniciar el loop audible."""
    tmp = f"{AUDIO_SNAPSHOT}.tmp"
    try:
        with open(tmp, 'w') as f:
            json.dump(synth.snapshot(), f)
        os.replace(tmp, AUDIO_SNAPSHOT)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


class Synthesizer:
    def __init__(self):
        snap = load_audio_snapshot()
        self.session_seed = int(snap.get('session_seed', int(time.time()))) if snap else int(time.time())
        self.progression_rng = random.Random(self.session_seed + 404)

        self.t_abs     = float(snap.get('t_abs', 0.0)) if snap else 0.0
        self.chord_t   = float(snap.get('chord_t', 0.0)) if snap else 0.0
        self.chord_idx = int(snap.get('chord_idx', 0)) if snap else 0
        self.chord_dur = float(snap.get('chord_dur', 0.0)) if snap else 0.0
        if self.chord_dur <= 0:
            self.chord_dur = self._next_chord_duration()

        self.bell_rng  = random.Random(self.session_seed + 17)
        self.mel_rng   = random.Random(self.session_seed + 31)
        self.mel_scene = None
        self.wind_rng  = np.random.RandomState(self.session_seed + 101)
        self.last_bell = float(snap.get('last_bell', max(0.0, self.t_abs - 90.0))) if snap else 0.0
        self.last_note = float(snap.get('last_note', max(0.0, self.t_abs - 120.0))) if snap else 0.0
        self.active_phrase = snap.get('active_phrase', []) if snap else []
        self.bell_active = []
        self.lead_lp_state = 0.0
        self.wind_zi1  = None
        self.wind_zi2  = None
        self.rcs_zi_rumble = np.zeros(2)
        self.rcs_zi_hiss = np.zeros(2)
        self.ambient   = AmbientSounds(start_t=self.t_abs)
        self.startup_fade_left = STARTUP_FADE_SECONDS
        self.last_layer = "init"

        # ── Perlin-like LFO state (smooth random walk para modular el pad orgánicamente) ──
        # Técnica de chicomcastro/procedural-music-generator: random walk filtrado en vez de LFO sinético
        self._perlin_rng = np.random.RandomState(self.session_seed + 7777)
        self._perlin_val = 0.0    # valor actual del random walk
        self._perlin_vel = 0.0    # velocidad del random walk

        if snap:
            log(f"audio snapshot restaurado: t={self.t_abs:.0f}s chord={self.chord_idx} dur={self.chord_dur:.1f}s")

    def _next_chord_duration(self):
        """Duraciones largas e irregulares para evitar ciclos armónicos obvios."""
        return self.progression_rng.uniform(70.0, 135.0)

    def _ensure_scene_melody_rng(self, scene):
        """Cambia la secuencia melódica al cambiar de escena sin reiniciar el audio."""
        if scene == self.mel_scene:
            return
        scene_hash = sum((i + 1) * ord(c) for i, c in enumerate(scene))
        self.mel_rng = random.Random(self.session_seed + 31 + scene_hash)
        self.active_phrase = []
        self.last_note = max(0.0, self.t_abs - 120.0)
        self.mel_scene = scene

    def snapshot(self):
        return {
            'session_seed': self.session_seed,
            't_abs': self.t_abs,
            'chord_t': self.chord_t,
            'chord_idx': self.chord_idx,
            'chord_dur': self.chord_dur,
            'last_bell': self.last_bell,
            'last_note': self.last_note,
            'active_phrase': self.active_phrase,
            'wall_time': time.time(),
        }

    def generate_chunk(self, state):
        n = CHUNK
        self._ensure_scene_melody_rng(state.scene)
        t = np.linspace(self.t_abs, self.t_abs + n / SR, n, endpoint=False)
        chords  = CHORD_SETS.get(state.chord_set, CHORD_SETS['dark_station'])
        bells_n = BELL_SETS.get(state.chord_set, BELL_SETS['dark_station'])
        notes_n = MELODY_SETS.get(state.chord_set, MELODY_SETS['dark_station'])

        # ── Pad FM (2 operadores) — reemplaza triangle+sine ────────────────────────────
        # FM Synthesis: carrier = sin(2π fc t + I * sin(2π fm t))
        # Ratio fm:fc y índice I dan el timbre. I bajo (0.3-1.5) → pad suave, cálido.
        # Ratio 1:1 → suena a órgano; 3:2 → quinta harmónica sutil; 2:1 → más brillante.
        # Técnica documentada en Andy Farnell "Designing Sound" y en la investigación de info.txt.
        FM_RATIOS = {
            # chord_set : (fm_ratio, mod_index_base, mod_index_lfo_depth)
            'dark_station':  (0.5,  0.24, 0.08),
            'arctic':        (0.5,  0.16, 0.06),  # frío, sin brillo tonal
            'forest':        (0.75, 0.22, 0.08),  # cálido, verde
            'deep_sea':      (0.5,  0.26, 0.10),  # profundo, oscuro
            'orbital':       (0.5,  0.20, 0.07),  # soporte vital, no beep
            'volcanic':      (0.5,  0.30, 0.12),  # denso, rugoso
            'mountain':      (0.5,  0.18, 0.06),  # aire amplio sin silbido
            'reactor':       (0.5,  0.24, 0.08),  # industrial oscuro
            'desert':        (0.75, 0.20, 0.07),  # cálido, seco
            'jungle':        (0.75, 0.22, 0.08),  # orgánico
            'coastal':       (0.5,  0.18, 0.06),  # abierto, suave
            'tundra':        (0.5,  0.14, 0.05),  # frío, amplio
        }
        fm_ratio, fm_idx_base, fm_lfo_depth = FM_RATIOS.get(
            state.chord_set, (1.0, 0.50, 0.22))

        # Perlin-like LFO: random walk suavizado para modulación orgánica
        # (En vez del sin(0.07t) estático que se vuelve predecible y robótico)
        dt = n / SR
        noise_kick = self._perlin_rng.normal(0, 0.008)  # impulso aleatorio suave
        self._perlin_vel += noise_kick
        self._perlin_vel *= 0.992                        # amortiguamiento (spring)
        self._perlin_val += self._perlin_vel * dt
        self._perlin_val  = float(np.clip(self._perlin_val, -1.0, 1.0))
        # Suavizado: interpola linealmente sobre el chunk
        perlin_prev = self._perlin_val - self._perlin_vel * dt
        perlin_lfo  = np.linspace(perlin_prev, self._perlin_val, n)

        # Índice FM con LFO orgánico (el timbre "respira" suavemente)
        fm_index = fm_idx_base + fm_lfo_depth * perlin_lfo

        def _build_pad_fm(chord, t_vec, fm_ratio, fm_index):
            """2-operator FM pad con 4 voces ligeramente detuneadas por nota.
            Voz A: carrier fc, mod fm_ratio*fc
            Voz B: carrier fc*1.002 (detune sutil → chorus natural)
            """
            out = np.zeros(len(t_vec))
            for i, freq in enumerate(chord):
                g = 0.140 if i == 0 else 0.110 / (i ** 0.5)
                fm = freq * fm_ratio   # frecuencia del modulador
                # Voz A: entonación central
                mod_a = np.sin(2 * np.pi * fm * 0.9985 * t_vec)
                car_a = np.sin(2 * np.pi * freq * 0.9975 * t_vec + fm_index * mod_a)
                # Voz B: detune +2 cents para chorus natural sin coros digitales
                mod_b = np.sin(2 * np.pi * fm * 1.0015 * t_vec)
                car_b = np.sin(2 * np.pi * freq * 1.0025 * t_vec + fm_index * mod_b)
                # Voz C: sub-octava muy sutil para calidez en bajos
                sub_c = np.sin(2 * np.pi * (freq * 0.5) * t_vec) * 0.15
                out += (car_a * 0.52 + car_b * 0.48) * g + sub_c * g * 0.18
            return out

        ch_cur  = chords[self.chord_idx % len(chords)]
        ch_next = chords[(self.chord_idx + 1) % len(chords)]
        fade_dur = min(18.0, self.chord_dur * 0.22)
        pad       = _build_pad_fm(ch_cur,  t, fm_ratio, fm_index)
        blend_pad = _build_pad_fm(ch_next, t, fm_ratio, fm_index)

        ct_arr    = self.chord_t + (t - t[0])
        blend_lin = np.clip((ct_arr - (self.chord_dur - fade_dur)) / fade_dur, 0.0, 1.0)
        fade_out  = np.cos(blend_lin * np.pi * 0.5)
        fade_in   = np.sin(blend_lin * np.pi * 0.5)
        pad       = pad * fade_out + blend_pad * fade_in
        # Saturación suave tipo tape (pre-boost + tanh + renormalización)
        # Basado en la técnica documentada en info.txt: pre-filter boost + waveshaping
        pad_driven = pad * 1.35
        pad = np.tanh(pad_driven) / np.tanh(np.array(1.35)) * 0.80
        pad *= 0.85 + 0.15 * (0.5 + 0.5 * perlin_lfo)   # amplitud respira con Perlin

        self.chord_t += n / SR
        if self.chord_t >= self.chord_dur:
            self.chord_t -= self.chord_dur
            self.chord_idx = (self.chord_idx + 1) % len(chords)
            self.chord_dur = self._next_chord_duration()

        # ── Sub-drone mejorado ───────────────────────────────────────────
        # Drone con FM sutil: modulador a 0.5x la raíz (suboctava) — da calidez analógica
        # y el vibrato usa el mismo Perlin LFO para que "respire" con el pad.
        root_raw  = ch_cur[0] / 2.0
        root_freq = max(root_raw, 55.0)
        # Vibrato suave derivado del Perlin LFO (no sinético sino orgánico)
        vibrato = 1.0 + 0.004 * perlin_lfo + 0.002 * np.sin(2 * np.pi * 0.03 * t)
        sin_d   = np.sin(2 * np.pi * root_freq * vibrato * t)
        # Mod suave en la raíz (FM index 0.2, modulador a media frecuencia)
        mod_sub = np.sin(2 * np.pi * root_freq * 0.5 * t) * 0.20
        fm_d    = np.sin(2 * np.pi * root_freq * vibrato * t + mod_sub)
        drone   = sin_d * 0.72 + fm_d * 0.28
        drone  *= np.clip(t / 15.0, 0.0, 1.0) * state.sub_gain

        # ── Viento filtrado (biquad bandpass en cascada ultra-rápido) ──
        prof = SCENE_PROFILES.get(state.scene, SCENE_PROFILES[DEFAULT_SCENE])

        # ── [NEW] Extracción de ganancias de nuevos sintetizadores ──
        rcs_gain       = prof.get('rcs_gain', 0.0)
        telemetry_gain = prof.get('telemetry_gain', 0.0)
        glacier_gain   = prof.get('glacier_gain', 0.0)
        wire_gain      = prof.get('wire_gain', 0.0)
        drops_gain     = prof.get('drops_gain', 0.0)
        sway_gain      = prof.get('sway_gain', 0.0)
        subcreak_gain  = prof.get('subcreak_gain', 0.0)
        sonar_gain     = prof.get('sonar_gain', 0.0)
        avalanche_gain = prof.get('avalanche_gain', 0.0)
        harp_gain      = prof.get('harp_gain', 0.0)
        sand_gain      = prof.get('sand_gain', 0.0)
        magma_gain     = prof.get('magma_gain', 0.0)
        steam_gain     = prof.get('steam_gain', 0.0)
        geiger_gain    = prof.get('geiger_gain', 0.0)
        coolant_gain   = prof.get('coolant_gain', 0.0)
        valve_gain     = prof.get('valve_gain', 0.0)
        thunder_gain   = prof.get('thunder_gain', 0.0)
        wind_gain_mul = prof['wind_gain']
        fc_mean = 300 + 1800 * state.brightness
        
        noise = self.wind_rng.uniform(-1.0, 1.0, n)
        if self.wind_zi1 is None:
            self.wind_zi1 = np.zeros(2)
            self.wind_zi2 = np.zeros(2)
            
        b, a = biquad_bandpass(fc_mean, 0.65, SR)
        y1, self.wind_zi1 = sps.lfilter(b, a, noise, zi=self.wind_zi1)
        wind_out, self.wind_zi2 = sps.lfilter(b, a, y1, zi=self.wind_zi2)
        # Factor 2.8→1.1: el viento era demasiado dominante y enmascaraba pad/melodía
        wind = wind_out * (0.030 + state.energy * 0.055) * wind_gain_mul * 1.1

        # ── Melodías Ambientales Procedimentales ──
        # Genera frases de varias notas espaciadas para crear movimiento melódico real
        mel = np.zeros(n)
        t_samples = self.t_abs + np.arange(n) / SR

        melody_gap = self.mel_rng.uniform(30.0, 60.0)
        if not self.active_phrase and (self.t_abs - self.last_note) > melody_gap:
            num_notes = self.mel_rng.randint(3, 6)
            phrase = []
            t_offset = self.mel_rng.uniform(1.0, 3.0)
            for _ in range(num_notes):
                freq = self.mel_rng.choice(notes_n)
                # Ocasionalmente baja una octava
                if self.mel_rng.random() < 0.25:
                    freq *= 0.5
                dur = self.mel_rng.uniform(3.5, 7.0)
                atk = self.mel_rng.uniform(1.0, 2.5)
                rel = self.mel_rng.uniform(2.0, 4.5)
                # Guardamos: start_time, freq, dur, attack, release, lp_state
                phrase.append((self.t_abs + t_offset, freq, dur, atk, rel, 0.0))
                t_offset += dur * self.mel_rng.uniform(0.6, 1.0) # ligero solape o separación
            self.active_phrase = phrase
            self.last_note = self.t_abs

        still = []
        for item in self.active_phrase:
            start_n, freq_n, dur_n, ATTACK, RELEASE, lp_state = item
            total = dur_n + RELEASE
            if start_n >= self.t_abs + n / SR:
                still.append(item); continue
            if start_n + total <= self.t_abs:
                continue

            age = t_samples - start_n
            mask = (age > 0) & (age < total)
            if not np.any(mask):
                if start_n + total > self.t_abs + n / SR:
                    still.append(item)
                continue

            a = age[mask]
            # Envolvente: fade-in sinusoidal + sustain plano + fade-out sinusoidal
            env = np.where(
                a < ATTACK,
                np.sin(np.pi * 0.5 * a / ATTACK),           # sube suavemente
                np.where(
                    a < dur_n,
                    1.0,                                      # plano
                    np.cos(np.pi * 0.5 * (a - dur_n) / RELEASE)  # baja suavemente
                )
            )

            # Tres osciladores detuneados para dar calidez analógica
            phase_a = 2 * np.pi * freq_n * 0.998 * a
            phase_b = 2 * np.pi * freq_n * 1.002 * a
            phase_c = 2 * np.pi * freq_n * 0.5   * a   # suboctava
            wave = (np.sin(phase_a) * 0.45 +
                    np.sin(phase_b) * 0.45 +
                    np.sin(phase_c) * 0.10)

            # Filtro paso bajo suave para redondear el tono
            b_lp, a_lp = [0.06], [1.0, -0.94]
            zi = np.array([lp_state])
            filtered, zf = sps.lfilter(b_lp, a_lp, wave * env, zi=zi)
            lp_state = float(zf[0])

            # Multiplicador aumentado de 0.038 a 0.08 para darles mayor protagonismo
            mel[mask] += filtered * 0.450
            if start_n + total > self.t_abs + n / SR:
                still.append((start_n, freq_n, dur_n, ATTACK, RELEASE, lp_state))

        self.active_phrase = still

        # ── Campanas Karplus-Strong ──────────────────────────────────────────
        # Algoritmo de modelado físico: cuerda pulsada / metal resonante.
        bells = np.zeros(n)
        bell_gain = prof['bell_gain']
        if bell_gain > 0:
            bell_chance = 0.0035 * state.density
            min_gap = max(90.0, 1.5 / max(bell_chance, 0.001))
            if (self.t_abs - self.last_bell) > min_gap and self.bell_rng.random() < bell_chance:
                freq_b = self.bell_rng.choice(bells_n)
                amp_b  = self.bell_rng.uniform(0.004, 0.014) * state.density * bell_gain
                # ── Karplus-Strong: modelado físico ──
                # Técnica de info.txt (tao_synth, tones). Genera el espectro
                # armónico naturalmente, sin necesidad de sumar parciales.
                ks_n    = int(7.0 * SR)
                period  = max(4, int(round(SR / max(freq_b, 20.0))))
                rng_ks  = np.random.RandomState(int(self.t_abs * 1000) % 2**31)
                work    = rng_ks.uniform(-1.0, 1.0, period).astype(np.float64)
                decay_f = 0.9994  # Q ≈ 230 — muy resonante (metal/madera)
                out_ks  = np.empty(ks_n, dtype=np.float64)
                for _s in range(0, ks_n, period):
                    _e = min(_s + period, ks_n)
                    out_ks[_s:_e] = work[:_e-_s]
                    nw      = np.empty_like(work)
                    nw[0]   = decay_f * 0.5 * (work[-1] + work[0])
                    nw[1:]  = decay_f * 0.5 * (work[:-1] + work[1:])
                    work    = nw
                self.bell_active.append((self.t_abs, out_ks * amp_b))
                self.last_bell = self.t_abs

            t_samples = self.t_abs + np.arange(n) / SR
            still = []
            for (ts, ks_buf) in self.bell_active:
                start_i = max(0, int((self.t_abs - ts) * SR))
                end_i   = min(len(ks_buf), start_i + n)
                if start_i < len(ks_buf):
                    chunk_len = end_i - start_i
                    offset = n - chunk_len  # alineado al final del chunk si comienza tarde
                    start_chunk = max(0, int((ts - self.t_abs) * SR))
                    take = ks_buf[start_i:end_i]
                    bells[start_chunk:start_chunk + len(take)] += take
                if ts + len(ks_buf) / SR > self.t_abs + n / SR:
                    still.append((ts, ks_buf))
            self.bell_active = still

        # ── Sonar ──
        period = 12.0 + (1.0 - state.energy) * 18.0
        pt  = t % period
        mask = pt < 0.30
        sonar = np.zeros(n)
        sonar[mask] = np.sin(pt[mask]*2*np.pi*state.sonar_freq) * np.exp(-20.0*pt[mask]) * 0.008

        ambient = self.ambient.generate_chunk(self.t_abs, n, state)

        synth_layers = {
            "pad": float(np.sqrt(np.mean((pad * 0.07) ** 2))),
            "drone": float(np.sqrt(np.mean((drone * 0.10) ** 2))),
            "wind": float(np.sqrt(np.mean((wind * 0.24) ** 2))),
            "drone_note": float(np.sqrt(np.mean((mel * 0.0) ** 2))),
            "bells": float(np.sqrt(np.mean((bells * 0.06) ** 2))),
            "ambient": float(np.sqrt(np.mean((ambient * 1.00) ** 2))),
        }
        dominant = max(synth_layers, key=synth_layers.get)
        if dominant == "ambient":
            dominant = getattr(self.ambient, "last_layer", "ambient")
        self.last_layer = dominant

        # ── Mezcla rebalanceada ──────────────────────────────────────────
        # Filosofía: pad/melodía son el primer plano musical (textura harmónica).
        # Ambiente/viento son el fondo texturizado — no deben tapar lo tonal.
        mix = (
            pad     * 0.45 +   # pad warm
            drone   * 0.38 +   # sub-drone
            wind    * 0.22 +   # viento
            mel     * 0.75 +   # melodía
            bells   * 0.30 +   # campanas plucks
            sonar   * 0.12 +   # sonar
            ambient * 1.00     # cama ambiental
        )

        # El limitador e integrador se aplican al final del master bus para mantener dinámicas

        if self.startup_fade_left > 0:
            fade_elapsed = STARTUP_FADE_SECONDS - self.startup_fade_left
            env = np.clip((fade_elapsed + np.arange(n) / SR) / STARTUP_FADE_SECONDS, 0.0, 1.0)
            mix *= env
            self.startup_fade_left = max(0.0, self.startup_fade_left - n / SR)
        self.t_abs += n / SR
        return mix


# ──────────────────────────────────────────────────────────
# Capas de ambiente
# ──────────────────────────────────────────────────────────

# ── Importar módulos de generadores ─────────────────────────────────────────
from generators.gen_base import BaseGeneratorsMixin
from generators.gen_space import SpaceGeneratorsMixin
from generators.gen_polar import PolarGeneratorsMixin
from generators.gen_nature import NatureGeneratorsMixin
from generators.gen_submarine import SubmarineGeneratorsMixin
from generators.gen_industrial import IndustrialGeneratorsMixin
from generators.gen_urban import UrbanGeneratorsMixin
from generators.gen_diffusion import MonoDiffuser

class AmbientSounds(BaseGeneratorsMixin, SpaceGeneratorsMixin, PolarGeneratorsMixin, NatureGeneratorsMixin, SubmarineGeneratorsMixin, IndustrialGeneratorsMixin, UrbanGeneratorsMixin):
    # Fonemas para síntesis de voz del walkie-talkie
    PHONEMES = {
        'A':   {'buzz': 1.0, 'noise': 0.0, 'formants': [730, 1090, 2440], 'bandwidths': [80, 90, 150]},
        'E':   {'buzz': 1.0, 'noise': 0.0, 'formants': [530, 1840, 2480], 'bandwidths': [60, 90, 150]},
        'I':   {'buzz': 1.0, 'noise': 0.0, 'formants': [270, 2290, 3010], 'bandwidths': [50, 100, 200]},
        'O':   {'buzz': 1.0, 'noise': 0.0, 'formants': [570, 840, 2410],  'bandwidths': [70, 80, 150]},
        'U':   {'buzz': 1.0, 'noise': 0.0, 'formants': [300, 870, 2240],  'bandwidths': [50, 80, 100]},
        'M':   {'buzz': 1.0, 'noise': 0.0, 'formants': [280, 950, 2300],  'bandwidths': [40, 150, 300], 'amp_scale': 0.4},
        'N':   {'buzz': 1.0, 'noise': 0.0, 'formants': [280, 1300, 2300], 'bandwidths': [40, 150, 300], 'amp_scale': 0.4},
        'R':   {'buzz': 1.0, 'noise': 0.0, 'formants': [350, 1060, 1500], 'bandwidths': [60, 100, 120]},
        'L':   {'buzz': 1.0, 'noise': 0.0, 'formants': [380, 1200, 2500], 'bandwidths': [60, 100, 150]},
        'V':   {'buzz': 0.6, 'noise': 0.3, 'formants': [250, 900, 2000],  'bandwidths': [100, 200, 300], 'amp_scale': 0.3},
        'Z':   {'buzz': 0.5, 'noise': 0.5, 'formants': [250, 1800, 2800], 'bandwidths': [100, 200, 300], 'amp_scale': 0.4},
        'S':   {'buzz': 0.0, 'noise': 1.0, 'formants': [4000, 6000, 8000],'bandwidths': [1000, 1500, 2000], 'amp_scale': 0.6},
        'SH':  {'buzz': 0.0, 'noise': 1.0, 'formants': [2500, 3500, 5000],'bandwidths': [800, 1000, 1500], 'amp_scale': 0.7},
        'F':   {'buzz': 0.0, 'noise': 1.0, 'formants': [1500, 3000, 6000],'bandwidths': [1000, 2000, 3000], 'amp_scale': 0.4},
        'P':   {'buzz': 0.0, 'noise': 0.6, 'formants': [500, 1500, 3000], 'bandwidths': [200, 400, 800], 'amp_scale': 0.7},
        'T':   {'buzz': 0.0, 'noise': 0.7, 'formants': [1000, 2500, 4000],'bandwidths': [400, 800, 1200], 'amp_scale': 0.7},
        'K':   {'buzz': 0.0, 'noise': 0.8, 'formants': [1500, 2000, 3500],'bandwidths': [300, 600, 1000], 'amp_scale': 0.8},
        'SIL': {'buzz': 0.0, 'noise': 0.0, 'formants': [500, 1500, 2500], 'bandwidths': [500, 500, 500], 'amp_scale': 0.0},
    }

    # Frases de radio sintetizadas
    PHRASE_RECIPES = {
        "ROGER": [
            ('SIL', 0.08), ('R', 0.08), ('O', 0.14), ('Z', 0.08), ('E', 0.12), ('R', 0.14), ('SIL', 0.10)
        ],
        "COPY_THAT": [
            ('SIL', 0.08), ('K', 0.06), ('O', 0.14), ('P', 0.07), ('I', 0.12), ('SIL', 0.06),
            ('T', 0.05), ('A', 0.14), ('T', 0.08), ('SIL', 0.10)
        ],
        "TEN_FOUR": [
            ('SIL', 0.08), ('T', 0.06), ('E', 0.14), ('N', 0.14), ('SIL', 0.10),
            ('F', 0.09), ('O', 0.16), ('R', 0.16), ('SIL', 0.10)
        ],
        "BASE_COPY": [
            ('SIL', 0.08), ('M', 0.08), ('E', 0.14), ('I', 0.09), ('S', 0.12), ('SIL', 0.08),
            ('K', 0.06), ('O', 0.12), ('P', 0.06), ('I', 0.12), ('SIL', 0.10)
        ],
        "SECTOR_FOUR_CLEAR": [
            ('SIL', 0.08), ('S', 0.10), ('E', 0.12), ('K', 0.06), ('T', 0.05), ('O', 0.10), ('R', 0.09), ('SIL', 0.06),
            ('F', 0.09), ('O', 0.14), ('R', 0.12), ('SIL', 0.10),
            ('K', 0.06), ('L', 0.08), ('I', 0.16), ('R', 0.14), ('SIL', 0.10)
        ],
        "SYSTEM_NOMINAL": [
            ('SIL', 0.08), ('S', 0.10), ('I', 0.09), ('S', 0.09), ('T', 0.05), ('E', 0.10), ('M', 0.12), ('SIL', 0.08),
            ('N', 0.09), ('O', 0.12), ('M', 0.10), ('I', 0.08), ('N', 0.09), ('A', 0.10), ('L', 0.14), ('SIL', 0.10)
        ],
        "ATMOSPHERE_STABLE": [
            ('SIL', 0.08), ('A', 0.12), ('T', 0.05), ('M', 0.09), ('O', 0.12), ('S', 0.09),
            ('F', 0.09), ('I', 0.12), ('R', 0.10), ('SIL', 0.08),
            ('S', 0.10), ('T', 0.05), ('E', 0.12), ('I', 0.08), ('M', 0.08), ('L', 0.10), ('SIL', 0.10)
        ],
        "ANOMALY_DETECTED": [
            ('SIL', 0.08), ('A', 0.12), ('N', 0.09), ('O', 0.12), ('M', 0.09), ('A', 0.10),
            ('L', 0.09), ('I', 0.12), ('SIL', 0.08),
            ('T', 0.05), ('I', 0.09), ('T', 0.05), ('E', 0.12), ('K', 0.06), ('T', 0.05), ('E', 0.10), ('T', 0.06), ('SIL', 0.10)
        ],
    }

    def __init__(self, start_t=0.0):
        self.rng = random.Random(9999)
        self.last_layer = "ambient"

        # ── Radio walkie-talkie ──
        self.radio_next_start = start_t + 45.0
        self.radio_end        = 0.0
        self.radio_active     = False
        self.radio_trans_dur  = 0.0
        self.radio_buffer     = None

        # ── Ventiladores ──
        self.fan_zi1 = None; self.fan_zi2 = None
        self.fan_rng = np.random.RandomState(7)
        nyq = SR / 2.0
        def make_fan_sos(fc, bw=40):
            lo = np.clip((fc - bw/2) / nyq, 0.005, 0.490)
            hi = np.clip((fc + bw/2) / nyq, lo + 0.005, 0.495)
            return sps.butter(2, [lo, hi], btype='band', output='sos')
        self.fan_sos1 = make_fan_sos(112)
        self.fan_sos2 = make_fan_sos(187)

        # ── Pasos ──
        self.step_next  = start_t + 18.0
        self.step_queue = []

        # ── Dings metálicos ──
        self.ding_next   = start_t + 38.0
        self.ding_active = []

        # ── Carrito ──
        self.cart_next   = start_t + 180.0
        self.cart_active = None

        # ── Agua (bosque/submarino) ──
        self.water_rng = np.random.RandomState(42)
        self.water_zi  = None
        self.bubble_queue = []      # [(t_abs, freq, amp, decay)]
        self.bubble_next  = start_t + 6.0
        self.water_lfo_t  = 0.0
        lo_w = np.clip(250 / nyq, 0.005, 0.490)
        hi_w = np.clip(620 / nyq, lo_w + 0.01, 0.495)
        self.water_sos = sps.butter(2, [lo_w, hi_w], btype='band', output='sos')

        # ── Knock / golpe puerta ──
        self.knock_next   = start_t + 240.0
        self.knock_queue  = []
        self.knock_hit_i  = 0

        # ── Lluvia ──
        self.rain_rng   = np.random.RandomState(77)
        self.rain_zi1   = None
        self.rain_zi2   = None
        self.rain_lfo_t = 0.0
        self.drop_queue = []
        self.drop_next  = start_t + 3.0
        # Precalcular filtros de lluvia
        self.rain_sos1 = sps.butter(4, [np.clip(200/nyq, 0.005, 0.49), np.clip(900/nyq, 0.01, 0.49)], btype='band', output='sos')
        self.rain_sos2 = sps.butter(4, [np.clip(900/nyq, 0.01, 0.49), np.clip(4000/nyq, 0.02, 0.49)], btype='band', output='sos')

        # Precalcular filtros de nieve
        self.snow_sos1 = sps.butter(4, [np.clip(80/nyq, 0.005, 0.49), np.clip(800/nyq, 0.01, 0.49)], btype='band', output='sos')
        self.snow_sos2 = sps.butter(3, [np.clip(800/nyq, 0.01, 0.49), np.clip(3000/nyq, 0.02, 0.49)], btype='band', output='sos')
        self.snow_zi1  = None
        self.snow_zi2  = None

        # ── Grilllos nocturnos procedimentales (Múltiples grillos en coro espacial) ──
        self.cricket_schedulers = [
            {'next': start_t + 8.0,  'rng': random.Random(1001), 'fc': 1650, 'pulses': 3},
            {'next': start_t + 16.0, 'rng': random.Random(1002), 'fc': 1850, 'pulses': 4},
            {'next': start_t + 24.0, 'rng': random.Random(1003), 'fc': 1450, 'pulses': 2},
        ]
        self.cricket_active_chirps = []

        # ── Truenos procedimentales (Físicamente realistas) ──
        self.thunder_next   = start_t + 60.0
        self.thunder_active = None
        self.thunder_rumble_zi = None

        # ── Hum eléctrico (50 Hz + armónicos) ──
        self.hum_ph = [0.0, 0.0, 0.0, 0.0]   # fases para 50/100/150/200 Hz

        # ── Blizzard (viento en ráfagas) ──
        self.gust_rng   = np.random.RandomState(13)
        self.gust_zi1   = None
        self.gust_zi2   = None
        self.gust_lfo_t = 0.0
        self.whistle_rng = np.random.RandomState(1313)
        self.whistle_zi  = [np.zeros(2), np.zeros(2), np.zeros(2)]
        self.whistle_real_zi = np.zeros(2)
        self.chain_next  = start_t + 10.0
        self.chain_active = []
        # Precalcular filtros de viento blizzard
        self.wind_sos1 = sps.butter(4, [np.clip(40/nyq, 0.005, 0.49), np.clip(200/nyq, 0.005, 0.49)], btype='band', output='sos')
        self.wind_sos2 = sps.butter(4, [np.clip(150/nyq, 0.005, 0.49), np.clip(500/nyq, 0.01, 0.49)], btype='band', output='sos')

        # ── Crujido estructural ──
        self.creak_next  = start_t + 90.0
        self.creak_queue = []
        self.creak_rng   = random.Random(31337)
        self.corona_next  = start_t + 2.0
        self.corona_queue = []
        self.corona_rng   = random.Random(424242)
        self.corona_np_rng = np.random.RandomState(424242)
        self.corona_sos = sps.butter(3, [np.clip(3200 / nyq, 0.005, 0.490), np.clip(12000 / nyq, 0.005 + 0.01, 0.495)], btype='band', output='sos')
        self.corona_zi = None
        self.spark_zi = None
        self.spark_hp_sos = sps.butter(2, np.clip(3000 / nyq, 0.005, 0.490), btype='highpass', output='sos')
        self.spark_hp_zi = None

        # ── Murmullo distante ciudad/maquinaria ──
        self.city_rng = np.random.RandomState(55)
        # Precalcular filtro de ciudad
        lo_c = np.clip(55 / nyq, 0.005, 0.490)
        hi_c = np.clip(230 / nyq, lo_c + 0.01, 0.495)
        self.city_sos = sps.butter(2, [lo_c, hi_c], btype='band', output='sos')
        self.city_zi  = None

        # ── Cassette hiss nocturno ──
        self.cass_rng = np.random.RandomState(88)
        # Precalcular filtro de cassette
        lo_ca = np.clip(3800 / nyq, 0.005, 0.490)
        hi_ca = np.clip(11000 / nyq, lo_ca + 0.01, 0.495)
        self.cass_sos = sps.butter(2, [lo_ca, hi_ca], btype='band', output='sos')
        self.cass_zi  = None

        # ── EMF sweep ──
        self.emf_rng    = np.random.RandomState(17)
        self.emf_next   = 0.0
        self.emf_queue  = []

        # ── Sirena de contención ──
        self.siren_rng    = np.random.RandomState(23)
        self.siren_next   = 80.0
        self.siren_active = None
        # Precalcular filtro de sirena
        lo_s = np.clip(120 / nyq, 0.005, 0.490)
        hi_s = np.clip(480 / nyq, lo_s + 0.01, 0.495)
        self.siren_sos = sps.butter(4, [lo_s, hi_s], btype='band', output='sos')
        self.siren_zi  = None

        # ── Respiración mecánica ──
        self.breath_rng = np.random.RandomState(37)
        self.breath_ts  = 0.0
        # Precalcular filtro de respiración
        lo_b = np.clip(150 / nyq, 0.005, 0.490)
        hi_b = np.clip(600 / nyq, lo_b + 0.01, 0.495)
        self.breath_sos = sps.butter(3, [lo_b, hi_b], btype='band', output='sos')
        self.breath_zi  = None

        # ── Morse distante ──
        self.morse_rng  = np.random.RandomState(53)
        self.morse_seq  = []
        self.morse_next = 45.0

        # ── Anomalía de tono ──
        self.anomaly_rng    = np.random.RandomState(61)
        self.anomaly_ph     = 0.0
        self.anomaly_f      = 220.0
        self.anomaly_tf     = 220.0
        self.anomaly_amp    = 0.0
        self.anomaly_tamp   = 0.0
        self.anomaly_ts     = 0.0
        self.anomaly_dur    = 0.0
        self.anomaly_next   = 30.0
        self.anomaly_active = False

        # ── [NEW] Estado de nuevos sonidos inmersivos ──
        # RCS
        self.rcs_rng = np.random.RandomState(8081)
        self.rcs_next = start_t + 15.0
        self.rcs_active = None
        # Telemetry
        self.telemetry_rng = np.random.RandomState(8082)
        self.telemetry_next = start_t + 8.0
        self.telemetry_queue = []
        # Glacier
        self.glacier_rng = np.random.RandomState(8083)
        self.glacier_next = start_t + 40.0
        self.glacier_active = None
        # Wire Whistle
        self.wire_rng = np.random.RandomState(8084)
        self.wire_zi1 = None
        self.wire_zi2 = None
        # Metal Drops
        self.drops_rng = np.random.RandomState(8085)
        self.drops_next = start_t + 0.1
        self.drops_queue = []
        # Tree Sway (multicapa: zi1=hojas peque, zi2=hojas grandes, zi3=ramas)
        self.sway_rng = np.random.RandomState(8086)
        self.sway_zi = [np.zeros(2), np.zeros(2), np.zeros(2)]
        # Sub Creak (hull groan con glide de frecuencia)
        self.subcreak_rng = np.random.RandomState(8087)
        self.subcreak_next = start_t + 20.0
        self.subcreak_active = None
        # Sonar Ping
        self.sonar_ping_rng = np.random.RandomState(8088)
        self.sonar_ping_next = start_t + 4.0
        self.sonar_ping_active = []
        # Avalanche
        self.avalanche_rng = np.random.RandomState(8089)
        self.avalanche_next = start_t + 85.0
        self.avalanche_active = None
        # Tension Harp
        self.harp_rng = np.random.RandomState(8090)
        self.harp_zi = [None, None, None, None]
        # Sandstorm Gust (dust devils con eventos discretos + fondo)
        self.sand_rng = np.random.RandomState(8091)
        self.sand_zi = [np.zeros(2)]   # [zi_bg]
        self.sand_queue = []           # [(ts, dur, fc, amp)]
        self.sand_next  = start_t + 2.0
        self.sand_rock_next = start_t + 25.0
        # ── Volcánica ──────────────────────────────────────────────
        # Magma Rumble: sub-bass pulsante continuo (15-45Hz)
        self.magma_rng = np.random.RandomState(8092)
        self.magma_zi  = np.zeros(2)
        self.magma_bubble_next = start_t + 5.0
        self.magma_bubble_queue = []
        # Steam Vent: eventos de vapor/gas a presión
        self.steam_rng  = np.random.RandomState(8093)
        self.steam_next = start_t + 6.0
        self.steam_active = None   # (ts, dur, fc, amp)
        self.steam_zi   = np.zeros(2)
        # ── Reactor ────────────────────────────────────────────────
        # Geiger: clicks de contador por proceso de Poisson
        self.geiger_rng = np.random.RandomState(8094)
        self.geiger_clicks = []    # lista de (ts_click, amp)
        self.geiger_next = start_t + self._geiger_interarrival()
        # Coolant Flow: flujo de refrigerante bajo presión (continuo)
        self.coolant_rng = np.random.RandomState(8095)
        self.coolant_zi  = np.zeros(2)
        # Pressure Valve: ráfagas de vapor de válvulas de presión
        self.valve_rng  = np.random.RandomState(8096)
        self.valve_next = start_t + 12.0
        self.valve_active = None   # (ts, dur, fc, amp)
        self.valve_zi   = np.zeros(2)
        # ── Tormenta ───────────────────────────────────────────────
        # Thunder: truenos con cola de reverb larga
        self.thunder_rng  = np.random.RandomState(8097)
        self.thunder_next = start_t + 10.0
        self.thunder_active = None   # (ts, dur, amp)
        self.thunder_zi   = np.zeros(2)
        # ── Urban / Noche ──────────────────────────────────────────
        # Perros: 2 instancias independientes
        self.dog_rng   = [random.Random(9101), random.Random(9102)]
        self.dog_next  = [start_t + self.dog_rng[0].uniform(15.0, 40.0),
                          start_t + self.dog_rng[1].uniform(30.0, 80.0)]
        self.dog_queue = []
        # Tráfico: rumble continuo + pases de autos
        self.traffic_rng = np.random.RandomState(9103)
        self.traffic_zi  = np.zeros(2)
        self.car_rng     = random.Random(9104)
        self.car_next    = start_t + self.car_rng.uniform(8.0, 25.0)
        self.car_queue   = []
        # Motos
        self.moto_rng  = random.Random(9105)
        self.moto_next = start_t + self.moto_rng.uniform(30.0, 90.0)
        self.moto_queue = []
        # Sapos: coro grave y pausado, más fondo natural que chirrido protagonista.
        _frog_seed = random.Random(9106)
        self.frog_params = [
            (_frog_seed.uniform(420, 620),  _frog_seed.uniform(0.8, 1.6), _frog_seed.uniform(0, 6.28), _frog_seed.uniform(0.0020, 0.0060)),
            (_frog_seed.uniform(520, 760),  _frog_seed.uniform(1.0, 1.9), _frog_seed.uniform(0, 6.28), _frog_seed.uniform(0.0018, 0.0052)),
            (_frog_seed.uniform(700, 980),  _frog_seed.uniform(1.2, 2.2), _frog_seed.uniform(0, 6.28), _frog_seed.uniform(0.0014, 0.0046)),
            (_frog_seed.uniform(450, 680),  _frog_seed.uniform(0.9, 1.8), _frog_seed.uniform(0, 6.28), _frog_seed.uniform(0.0020, 0.0055)),
            (_frog_seed.uniform(620, 900),  _frog_seed.uniform(1.3, 2.4), _frog_seed.uniform(0, 6.28), _frog_seed.uniform(0.0012, 0.0040)),
            (_frog_seed.uniform(500, 740),  _frog_seed.uniform(1.0, 2.0), _frog_seed.uniform(0, 6.28), _frog_seed.uniform(0.0016, 0.0048)),
        ]
        self.frog_zi = [np.zeros(2) for _ in self.frog_params]
        self.frog_lp_zi = np.zeros(2)
        # Hojas: ruido de alta frecuencia
        self.leaves_rng  = np.random.RandomState(9107)
        self.leaves_zi1  = np.zeros(2)
        self.leaves_zi2  = np.zeros(2)

    def _thunder_chunk(self, t_abs, n, rain_intensity):
        """Truenos físicamente modelados por distancia con delay luz-sonido y ecos."""
        if rain_intensity < 0.35:
            return np.zeros(n)
            
        out = np.zeros(n)
        
        if self.thunder_active is None and t_abs >= self.thunder_next:
            dist = self.rng.uniform(0.6, 5.5)
            delay = dist / 0.343
            crack_amp = 0.05 * (1.0 / dist) * rain_intensity if dist < 3.0 else 0.0
            rumble_amp = self.rng.uniform(0.045, 0.095) * rain_intensity
            rumble_dur = self.rng.uniform(7.0, 14.0)
            fc = float(np.clip(160.0 - 22.0 * dist, 45.0, 150.0))
            
            self.thunder_active = {
                'ts': t_abs,
                'dist': dist,
                'delay': delay,
                'crack_amp': crack_amp,
                'rumble_amp': rumble_amp,
                'rumble_dur': rumble_dur,
                'fc': fc
            }
            self.thunder_next = t_abs + delay + rumble_dur + self.rng.uniform(40.0, 150.0)
            
        if self.thunder_active:
            ta_active = self.thunder_active
            ts = ta_active['ts']
            dist = ta_active['dist']
            delay = ta_active['delay']
            crack_amp = ta_active['crack_amp']
            rumble_amp = ta_active['rumble_amp']
            rumble_dur = ta_active['rumble_dur']
            fc = ta_active['fc']
            
            t_samples = np.linspace(t_abs, t_abs + n / SR, n, endpoint=False)
            
            # 1. Relámpago (Crack de alta frecuencia inicial)
            if crack_amp > 0:
                age_crack = t_samples - ts
                mask_crack = (age_crack > 0) & (age_crack < 0.12)
                if np.any(mask_crack):
                    age_c = age_crack[mask_crack]
                    noise_c = np.random.uniform(-1.0, 1.0, np.sum(mask_crack))
                    b_c, a_c = biquad_bandpass(1400.0, 0.5, SR)
                    crack_filtered = sps.lfilter(b_c, a_c, noise_c)
                    env_crack = np.exp(-35.0 * age_c)
                    out[mask_crack] += crack_filtered * env_crack * crack_amp
            
            # 2. Retumbar (Rumble de baja frecuencia retardado)
            rumble_start_t = ts + delay
            age_rumble = t_samples - rumble_start_t
            mask_rumble = (age_rumble > 0) & (age_rumble < rumble_dur)
            
            if np.any(mask_rumble):
                age_active = age_rumble[mask_rumble]
                env = np.exp(-0.30 * age_active)
                
                # Ecos y reflexiones secundarias
                reflections = (1.0 + 
                               0.40 * np.exp(-1.5 * np.abs(age_active - 1.2)) + 
                               0.25 * np.exp(-1.0 * np.abs(age_active - 2.5)) + 
                               0.15 * np.exp(-0.8 * np.abs(age_active - 4.0)))
                
                # Modulación de rugido lento
                rumble_mod = 1.0 + 0.35 * np.sin(2 * np.pi * 6.5 * age_active) * np.sin(2 * np.pi * 0.8 * age_active)
                full_env = env * reflections * rumble_mod
                
                # Ruido de baja frecuencia con paso bajo variable por ráfagas
                noise_r = np.random.uniform(-1.0, 1.0, np.sum(mask_rumble))
                rumble_filtered = np.zeros_like(noise_r)
                
                block_size = 512
                zi = self.thunder_rumble_zi if self.thunder_rumble_zi is not None else np.zeros(2)
                
                for start in range(0, len(noise_r), block_size):
                    end = min(start + block_size, len(noise_r))
                    if start == end:
                        break
                    t_avg = np.mean(age_active[start:end])
                    # Barrido de frecuencia: decrece exponencialmente con el tiempo de fc a 38 Hz
                    fc_sweep = float(np.clip(fc * np.exp(-0.16 * t_avg), 38.0, fc))
                    # Q decae suavemente para suavizar resonancias al final
                    Q_sweep = float(np.clip(1.2 - 0.04 * t_avg, 0.6, 1.2))
                    
                    b_th, a_th = biquad_lowpass(fc_sweep, Q_sweep, SR)
                    rumble_filtered[start:end], zi = sps.lfilter(b_th, a_th, noise_r[start:end], zi=zi)
                
                self.thunder_rumble_zi = zi
                out[mask_rumble] += rumble_filtered * full_env * rumble_amp
                
            if t_abs + n / SR >= ts + delay + rumble_dur:
                self.thunder_active = None
                self.thunder_rumble_zi = None
                
        return out

    def generate_chunk(self, t_abs, n, state):
        prof = SCENE_PROFILES.get(state.scene, SCENE_PROFILES[DEFAULT_SCENE])

        # ── [NEW] Extracción de ganancias de nuevos sintetizadores ──
        rcs_gain       = prof.get('rcs_gain', 0.0)
        telemetry_gain = prof.get('telemetry_gain', 0.0)
        glacier_gain   = prof.get('glacier_gain', 0.0)
        wire_gain      = prof.get('wire_gain', 0.0)
        drops_gain     = prof.get('drops_gain', 0.0)
        sway_gain      = prof.get('sway_gain', 0.0)
        subcreak_gain  = prof.get('subcreak_gain', 0.0)
        sonar_gain     = prof.get('sonar_gain', 0.0)
        avalanche_gain = prof.get('avalanche_gain', 0.0)
        harp_gain      = prof.get('harp_gain', 0.0)
        sand_gain      = prof.get('sand_gain', 0.0)
        magma_gain     = prof.get('magma_gain', 0.0)
        steam_gain     = prof.get('steam_gain', 0.0)
        geiger_gain    = prof.get('geiger_gain', 0.0)
        coolant_gain   = prof.get('coolant_gain', 0.0)
        valve_gain     = prof.get('valve_gain', 0.0)
        thunder_gain   = prof.get('thunder_gain', 0.0)
        dog_gain     = prof.get('dog_gain', 0.0)
        traffic_gain = prof.get('traffic_gain', 0.0)
        moto_gain    = prof.get('moto_gain', 0.0)
        frog_gain    = prof.get('frog_gain', 0.0)
        leaves_gain  = prof.get('leaves_gain', 0.0)

        radio    = self._radio_chunk(t_abs, n, prof['radio_gain'])
        fan      = self._fan_chunk(t_abs, n, prof['fan_gain'])
        steps    = self._step_chunk(t_abs, n, state.density, state.scene)
        dings    = self._ding_chunk(t_abs, n, state.density, prof['ding_gain'])
        cart     = self._cart_chunk(t_abs, n)
        water    = self._water_chunk(t_abs, n, prof['water_gain'])
        knock    = self._knock_chunk(t_abs, n, state.scene)
        blizzard = self._blizzard_chunk(t_abs, n, prof['wind_gain']) if prof['blizzard'] else np.zeros(n)
        whistle  = self._air_whistle_chunk(t_abs, n, prof.get('whistle_gain', 0.0), state.wind_spd, state.scene)
        creak    = self._creak_chunk(t_abs, n, prof['creak_gain'])
        corona   = self._corona_crackle_chunk(t_abs, n, prof.get('corona_gain', 0.0))
        chains   = self._chains_chunk(t_abs, n, prof.get('creak_gain', 0.0) if state.scene in ('scp_contencion', 'scp_exterior') else 0.0)

        # ── [NEW] Generación de nuevos chunks de sintetizadores ──
        rcs        = self._rcs_thruster_chunk(t_abs, n, rcs_gain)
        telemetry  = self._telemetry_chunk(t_abs, n, telemetry_gain)
        glacier    = self._glacier_creak_chunk(t_abs, n, glacier_gain)
        wire       = self._wire_whistle_chunk(t_abs, n, wire_gain, state.wind_spd)
        drops      = self._metal_drops_chunk(t_abs, n, drops_gain, state.rain_intensity)
        sway       = self._tree_sway_chunk(t_abs, n, sway_gain, state.wind_spd)
        subcreak   = self._sub_creak_chunk(t_abs, n, subcreak_gain)
        sonar_ping = self._sonar_ping_chunk(t_abs, n, sonar_gain)
        avalanche  = self._avalanche_chunk(t_abs, n, avalanche_gain)
        harp       = self._tension_harp_chunk(t_abs, n, harp_gain, state.wind_spd)
        sand       = self._sandstorm_gust_chunk(t_abs, n, sand_gain, state.wind_spd)
        magma      = self._magma_rumble_chunk(t_abs, n, magma_gain)
        steam      = self._steam_vent_chunk(t_abs, n, steam_gain)
        geiger     = self._geiger_click_chunk(t_abs, n, geiger_gain)
        coolant    = self._coolant_flow_chunk(t_abs, n, coolant_gain)
        valve      = self._valve_vent_chunk(t_abs, n, valve_gain)
        thunder    = self._thunder_chunk(t_abs, n, thunder_gain)
        dog_bark = self._dog_bark_chunk(t_abs, n, dog_gain)
        traffic  = self._traffic_chunk(t_abs, n, traffic_gain)
        moto     = self._moto_chunk(t_abs, n, moto_gain)
        frog     = self._frog_chunk(t_abs, n, frog_gain)
        leaves   = self._leaves_chunk(t_abs, n, leaves_gain)
        cricket  = self._cricket_chunk(t_abs, n, state)
        hum      = self._elec_hum_chunk(t_abs, n, prof['elec_hum'])
        city     = self._city_chunk(t_abs, n, prof['city_gain'])
        clock    = self._clock_chunk(t_abs, n, prof['clock_night'])
        cassette = self._cassette_chunk(n, prof['cassette'])
        emf      = self._emf_chunk(t_abs, n, prof['emf_gain'])
        siren    = self._siren_chunk(t_abs, n, prof['siren_gain'])
        breath   = self._breath_chunk(t_abs, n, prof['breath_gain'])
        morse    = self._morse_chunk(t_abs, n, prof['morse_gain'])
        anomaly  = self._anomaly_chunk(t_abs, n, prof['anomaly_gain'])

        rain_gain = prof['rain_base'] + state.rain_intensity * prof['rain_wx_mul']
        rain = self._rain_chunk(t_abs, n, rain_gain,
                                is_snow=prof['snow'], rain_glass=prof['rain_glass'])

        # Rich spatial ambient: reactivamos todas las capas con ganancias calibradas
        layer_levels = {
            "radio": float(np.sqrt(np.mean((radio * 0.85) ** 2))),
            "fan": float(np.sqrt(np.mean((fan * 0.35) ** 2))),
            "rain": float(np.sqrt(np.mean((rain + drops) ** 2))),
            "wind": float(np.sqrt(np.mean((blizzard * 0.28 + whistle * 0.65 + wire * 0.90 + sway * 0.70 + sand * 0.85) ** 2))),
            "water": float(np.sqrt(np.mean((water * 0.80) ** 2))),
            "hum": float(np.sqrt(np.mean(hum ** 2))),
            "emf": float(np.sqrt(np.mean((emf * 0.60 + telemetry * 0.85) ** 2))),
            "creak": float(np.sqrt(np.mean((creak * 0.25 + glacier * 0.75 + subcreak * 0.80) ** 2))),
            "corona": float(np.sqrt(np.mean((corona * 0.60) ** 2))),
            "rcs": float(np.sqrt(np.mean((rcs * 0.80) ** 2))),
            "sonar": float(np.sqrt(np.mean((sonar_ping * 0.85) ** 2))),
            "avalanche": float(np.sqrt(np.mean((avalanche * 0.80) ** 2))),
            "harp": float(np.sqrt(np.mean((harp * 0.75) ** 2))),
            "magma": float(np.sqrt(np.mean((magma * 0.92) ** 2))),
            "steam": float(np.sqrt(np.mean((steam * 0.80 + valve * 0.82) ** 2))),
            "geiger": float(np.sqrt(np.mean((geiger * 0.90) ** 2))),
            "coolant": float(np.sqrt(np.mean((coolant * 0.70) ** 2))),
            "thunder": float(np.sqrt(np.mean((thunder * 0.95) ** 2))),
            "cricket": float(np.sqrt(np.mean((cricket * 0.45) ** 2))),
            "pad": 0.0,
        }
        self.last_layer = max(layer_levels, key=layer_levels.get)

        mix = (radio * 0.46 + fan * 0.22 + steps * 0.22 + dings * 0.18 + cart * 0.22 +
               water * 0.72 + knock * 0.22 + rain * 0.52 + blizzard * 0.18 + whistle * 0.24 + creak * 0.16 + corona * 0.18 +
               hum * 0.90 + city * 0.34 + clock * 0.16 + cassette * 0.18 + emf * 0.28 +
               siren * 0.08 + breath * 0.34 + morse * 0.10 + anomaly * 0.22 +
               rcs * 0.34 + telemetry * 0.32 + glacier * 0.34 + wire * 0.28 + drops * 0.34 +
               sway * 0.40 + subcreak * 0.42 + sonar_ping * 0.24 + avalanche * 0.30 +
               harp * 0.22 + sand * 0.36 +
               magma * 0.58 + steam * 0.46 + geiger * 0.22 +
               coolant * 0.44 + valve * 0.24 + thunder * 0.44 +
               dog_bark * 0.18 + traffic * 0.36 + moto * 0.16 + frog * 0.30 + leaves * 0.26 +
               cricket * 0.18 + chains * 0.45)
        return mix



# ──────────────────────────────────────────────────────────
# Pedalboard — se actualiza in-place (no rebuild)
# ──────────────────────────────────────────────────────────

def build_board(state):
    if not HAS_PEDALBOARD:
        return None
    cutoff_hz = 950.0 if state.scene == 'submarina' else (4200 + state.brightness * 1200)
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=62.0),   # corta sub-bass inaudible
        Chorus(rate_hz=0.25 + state.brightness * 0.35,
               depth=0.18 + (1 - state.brightness) * 0.22,
               centre_delay_ms=7.0, feedback=0.10, mix=state.chorus_mix),
        Reverb(room_size=state.reverb_room,
               damping=0.3 + state.brightness * 0.35,
               wet_level=state.reverb_wet,
               dry_level=1.0 - state.reverb_wet * 0.65),
        # Compresor ambient: captura columnas/picos antes de que lleguen a AAC/YouTube.
        Compressor(threshold_db=-18, ratio=2.2, attack_ms=80, release_ms=1600),
        LowpassFilter(cutoff_frequency_hz=cutoff_hz),
        Gain(gain_db=5.5),
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
    if state.scene == 'submarina':
        board[4].cutoff_frequency_hz = 950.0
    else:
        board[4].cutoff_frequency_hz = 4200 + state.brightness * 1200


# ──────────────────────────────────────────────────────────
# Hilos en segundo plano para evitar congelar el bucle de audio
# ──────────────────────────────────────────────────────────

def weather_worker(state):
    while True:
        try:
            temp, humidity, wind_spd, w_code = fetch_weather()
            state.temp = temp
            state.humidity = humidity
            state.wind_spd = wind_spd
            state.w_code = w_code
            log(f"Clima actualizado: {temp}°C {humidity}% {wind_spd}km/h código={w_code}")
        except Exception as e:
            log(f"Clima background falló: {e}", "WARN")
        time.sleep(WEATHER_INTERVAL)


def network_worker(state):
    while True:
        try:
            devices = fetch_network_devices()
            state.devices = devices
            log(f"Dispositivos actualizados: {devices}")
        except Exception as e:
            log(f"Network background falló: {e}", "WARN")
        time.sleep(NETWORK_INTERVAL)


def system_worker(state):
    while True:
        try:
            load1, used_pct = fetch_system_load()
            state.load1 = load1
            state.load_ram_pct = used_pct
        except Exception as e:
            log(f"System background falló: {e}", "WARN")
        time.sleep(SYSTEM_INTERVAL)


def viewers_worker(state):
    while True:
        try:
            viewers = fetch_viewers()
            state.viewers = viewers
        except Exception as e:
            log(f"Viewers background falló: {e}", "WARN")
        time.sleep(30)


def log(msg, level="INFO"):
    print(f"[{level}] {msg}", file=sys.stderr, flush=True)


def main():
    state = AudioState()
    synth = Synthesizer()
    last_snapshot = 0
    board = None
    chunk_count = 0
    master_gain = 1.0
    diffuser = MonoDiffuser(sr=SR, rt60=2.6, mix=0.22, damp=0.5)  # cola difusa estilo Bitwig
    _stage = "init"   # rastrea dónde estamos si algo explota

    # Cargar valores iniciales síncronamente antes de arrancar los hilos
    log("Obteniendo clima e información de red iniciales...")
    try:
        state.temp, state.humidity, state.wind_spd, state.w_code = fetch_weather()
        log(f"Clima inicial: {state.temp}°C {state.humidity}% {state.wind_spd}km/h código={state.w_code}")
    except Exception as e:
        log(f"Fallo inicial de clima: {e}", "WARN")
    
    try:
        state.devices = fetch_network_devices()
        log(f"Dispositivos iniciales: {state.devices}")
    except Exception as e:
        log(f"Fallo inicial de red: {e}", "WARN")
        
    try:
        state.load1, state.load_ram_pct = fetch_system_load()
    except Exception as e:
        log(f"Fallo inicial de carga de sistema: {e}", "WARN")
        
    try:
        state.viewers = fetch_viewers()
    except Exception as e:
        log(f"Fallo inicial de espectadores: {e}", "WARN")

    # Iniciar hilos en segundo plano para recolección asíncrona de datos
    log("Iniciando hilos de segundo plano para recolección de datos...")
    t_weather = threading.Thread(target=weather_worker, args=(state,), daemon=True)
    t_network = threading.Thread(target=network_worker, args=(state,), daemon=True)
    t_system = threading.Thread(target=system_worker, args=(state,), daemon=True)
    t_viewers = threading.Thread(target=viewers_worker, args=(state,), daemon=True)
    
    t_weather.start()
    t_network.start()
    t_system.start()
    t_viewers.start()

    log("audio_engine iniciado. Escribiendo PCM s16le a stdout...")

    try:
        while True:
            now = time.time()

            _stage = "update_targets"
            state.update_targets()
            state.lerp_step(CHUNK)

            _stage = "build_board"
            if board is None:
                board = build_board(state)
                log(f"Pedalboard construido: HAS_PEDALBOARD={HAS_PEDALBOARD}")
            else:
                update_board_inplace(board, state)

            _stage = "generate_chunk"
            audio = synth.generate_chunk(state)

            _stage = "nan_check"
            nan_count = int(np.sum(np.isnan(audio)))
            inf_count = int(np.sum(np.isinf(audio)))
            if nan_count or inf_count:
                log(f"NaN={nan_count} Inf={inf_count} en chunk {chunk_count} — limpiando", "WARN")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.8, neginf=-0.8)
            audio = np.clip(audio, -1.0, 1.0)

            _stage = "pedalboard"
            if board and HAS_PEDALBOARD:
                audio_f32 = np.clip(audio.astype(np.float32), -1.0, 1.0).reshape(1, -1)
                processed = board(audio_f32, SR)
                audio = np.nan_to_num(processed.flatten().astype(np.float64), nan=0.0)

            _stage = "diffusion"
            audio = diffuser.process(audio)   # cola difusa (a prueba de fallos: devuelve seco si algo falla)

            _stage = "master_bus"
            # Ganancia adaptativa RMS calibrada para un volumen de transmisión óptimo
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms > 1e-6:
                raw_gain = 0.52 / rms
                master_gain += (float(np.clip(raw_gain, 0.40, 15.00)) - master_gain) * 0.04
            audio = np.tanh(audio * master_gain * 0.95)
            peak = float(np.max(np.abs(audio)))

            _stage = "write_stdout"
            sys.stdout.buffer.write((audio * 32760).astype(np.int16).tobytes())
            sys.stdout.buffer.flush()

            _stage = "save_snapshot"
            if now - last_snapshot > SNAPSHOT_INTERVAL:
                save_audio_snapshot(synth)
                last_snapshot = now

            chunk_count += 1
            chords = CHORD_SETS.get(state.chord_set, CHORD_SETS['dark_station'])
            chord_num = (synth.chord_idx % len(chords)) + 1
            chord_label = f"{state.chord_set}:{chord_num}/{len(chords)}"
            # Exportar estado extendido al final del chunk en un hilo secundario daemon
            # para evitar que la latencia de I/O en disco provoque micro-cortes
            threading.Thread(
                target=state.export,
                kwargs={
                    'peak': peak,
                    'rms': rms,
                    'chunk_count': chunk_count,
                    'chord_label': chord_label,
                    'layer_label': synth.last_layer,
                    'master_gain': master_gain
                },
                daemon=True
            ).start()
            if chunk_count % 15 == 0:
                prof  = SCENE_PROFILES.get(state.scene, {})
                label = prof.get('label', state.scene)
                log(f"{label} | bright={state.brightness:.2f} density={state.density:.2f} "
                    f"energy={state.energy:.2f} peak={peak:.3f} rms={rms:.4f} "
                    f"gain={master_gain:.2f} layer={synth.last_layer} "
                    f"t={synth.t_abs:.0f}s chunk={chunk_count}")

    except BrokenPipeError:
        log("Pipe cerrado.")
    except KeyboardInterrupt:
        log("Detenido.")
    except Exception as e:
        import traceback
        log(f"CRASH en etapa '{_stage}' chunk={chunk_count} t={synth.t_abs:.1f}s — {type(e).__name__}: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")


if __name__ == "__main__":
    main()
