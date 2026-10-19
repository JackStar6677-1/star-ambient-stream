#!/usr/bin/env python3
"""
audio_engine.py — Motor Autónomo de Streaming Lo-Fi con Alta Variación Rítmica, Armónica y Sonora.
Versión 4:
  - 6 Grooves rítmicos distintos por tema:
      * ambient_pure: sin batería, pura contemplación acústica/analógica.
      * half_time_lofi: bombo sutil y rimshot solo en el beat 3 (half-time relajado).
      * classic_boombap: bombo en 1 y 3-and, rim en 2 y 4 con swing MPC sutil.
      * dilla_drunken: micro-timing humano arrastrado (+25-35ms), ghost notes, kicks sincopados.
      * bossa_lofi: patrón cross-stick bossa syncopation sobre sub-bombo suave.
      * chill_waltz_triplets: subdivisión ternaria / shaker orgánico continuo.
  - Síntesis percusiva multivariable (4 kicks, 4 rims/snares, 4 tipos de hats/shakers).
  - Rasgueos de piano y Rhodes humanos (micro-offset de 15 a 35ms por cuerda/nota).
  - Patrones de acompañamiento variados (strum sostenido, stab sincopado en contratiempo, arpegio fluido).
  - 16 progresiones de acordes ricas en extensiones de 9na, 11na y 13ra (8 Cafe / Día, 8 Outpost / Noche).
  - Bajo orgánico con líneas de paso (notas raíz, quintas, octavas suaves y deslizamientos).
  - Generador de vinilo analógico procedural (crackle estocástico + hiss cálido + wow/flutter tape).
  - Foley orgánico reactivo según la escena (/tmp/star_scene.txt).
  - Salida PCM s16le estéreo continua y rotación WAV compatible con audio_supervisor.py.
"""

import os
import sys
import time
import math
import random
import pathlib
import numpy as np
import scipy.signal as sps
import scipy.io.wavfile as wavfile

try:
    from pedalboard import (
        Pedalboard, Reverb, Chorus, Compressor, Gain, LowpassFilter, HighpassFilter
    )
    HAVE_PEDALBOARD = True
except ImportError:
    HAVE_PEDALBOARD = False

SR = 44100
SCENE_FILE = "/tmp/star_scene.txt"
AUDIO_DIR = pathlib.Path("/tmp/star_lofi_buffer")
CHUNK_SECONDS = 4
CHUNK_SAMPLES = SR * CHUNK_SECONDS

def midi_to_freq(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))

# ─────────────────────────────────────────────────────────────
# 1. GENERADORES DE PERCUSIÓN MULTIMODAL CON ALTA DIVERSIDAD
# ─────────────────────────────────────────────────────────────

def synth_kick(variant: int = 0) -> np.ndarray:
    """4 tipos de bombos lo-fi:
       0: Sub 808 suave y profundo.
       1: Acústico lo-fi con golpe de maza de fieltro.
       2: Bombo saturado de cinta analógica (muffled tape).
       3: Punchy boombap seco y compacto.
    """
    if variant == 0:  # Sub 808 suave
        dur = 0.40
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        f_env = 48.0 + 35.0 * np.exp(-t * 26.0)
        phase = 2.0 * np.pi * np.cumsum(f_env) / SR
        body = np.sin(phase) * np.exp(-t * 7.5)
        sat = np.tanh(body * 1.2)
        b, a = sps.butter(2, 180 / (SR / 2), btype='low')
        return sps.lfilter(b, a, sat) * 0.90

    elif variant == 1:  # Fieltro acústico
        dur = 0.32
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        imp_len = int(0.010 * SR)
        impact = np.random.normal(0, 1, imp_len) * np.exp(-np.linspace(0, 12, imp_len))
        b_imp, a_imp = sps.butter(2, [180 / (SR / 2), 850 / (SR / 2)], btype='band')
        impact_f = sps.lfilter(b_imp, a_imp, impact) * 0.25
        f_env = 55.0 + 40.0 * np.exp(-t * 30.0)
        phase = 2.0 * np.pi * np.cumsum(f_env) / SR
        body = (np.sin(phase) + 0.12 * np.sin(2.0 * phase)) * np.exp(-t * 11.0)
        out = np.zeros(n)
        out[:imp_len] += impact_f
        out += body
        sat = np.tanh(out * 1.3)
        b_lp, a_lp = sps.butter(2, 3500 / (SR / 2), btype='low')
        return sps.lfilter(b_lp, a_lp, sat) * 0.92

    elif variant == 2:  # Muffled tape kick
        dur = 0.28
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        f_env = 42.0 + 25.0 * np.exp(-t * 20.0)
        phase = 2.0 * np.pi * np.cumsum(f_env) / SR
        body = np.sin(phase) * np.exp(-t * 9.0)
        tape_noise = np.random.normal(0, 0.05, n) * np.exp(-t * 15.0)
        sig = (body + tape_noise)
        sat = np.tanh(sig * 1.6)
        b, a = sps.butter(2, 1600 / (SR / 2), btype='low')
        return sps.lfilter(b, a, sat) * 0.88

    else:  # Punchy dry boombap
        dur = 0.25
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        f_env = 58.0 + 50.0 * np.exp(-t * 35.0)
        phase = 2.0 * np.pi * np.cumsum(f_env) / SR
        body = np.sin(phase) * np.exp(-t * 14.0)
        sat = np.tanh(body * 1.5)
        b, a = sps.butter(2, 4000 / (SR / 2), btype='low')
        return sps.lfilter(b, a, sat) * 0.95


def synth_rimshot(variant: int = 0) -> np.ndarray:
    """4 tipos de cajas / rims:
       0: Cross-stick de madera con resonancia natural.
       1: Brushed snare con barrido suave de escobilla.
       2: Tape clap / chasquido vintage multicapa.
       3: Rimshot apagado y seco (mellow click).
    """
    if variant == 0:  # Cross-stick de madera
        dur = 0.16
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        modes = [(430.0, 0.40, 32.0), (890.0, 0.35, 48.0), (1700.0, 0.22, 68.0), (2900.0, 0.12, 90.0)]
        wood = np.zeros(n)
        for freq, amp, decay in modes:
            wood += amp * np.sin(2.0 * np.pi * freq * t) * np.exp(-t * decay)
        wire_len = int(0.09 * SR)
        wire_noise = np.random.normal(0, 1, wire_len)
        b, a = sps.butter(2, [1400 / (SR / 2), 6000 / (SR / 2)], btype='band')
        wire = sps.lfilter(b, a, wire_noise) * np.exp(-np.linspace(0, 28, wire_len)) * 0.16
        out = wood.copy()
        out[:wire_len] += wire
        return out * 0.86

    elif variant == 1:  # Brushed snare
        dur = 0.24
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        noise = np.random.normal(0, 1, n)
        b, a = sps.butter(2, [900 / (SR / 2), 5200 / (SR / 2)], btype='band')
        brush = sps.lfilter(b, a, noise) * np.exp(-t * 16.0)
        body = np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-t * 36.0) * 0.28
        out = brush * 0.70 + body
        return out * 0.82

    elif variant == 2:  # Tape clap / Snap
        dur = 0.18
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        out = np.zeros(n)
        for offset_ms in [0, 8, 17]:
            pos = int((offset_ms / 1000.0) * SR)
            burst_len = int(0.045 * SR)
            burst_noise = np.random.normal(0, 1, burst_len) * np.exp(-np.linspace(0, 18, burst_len))
            b, a = sps.butter(2, [1200 / (SR / 2), 4800 / (SR / 2)], btype='band')
            filtered = sps.lfilter(b, a, burst_noise) * 0.35
            out[pos:pos + len(filtered)] += filtered
        tail = np.random.normal(0, 1, n) * np.exp(-t * 22.0) * 0.12
        b, a = sps.butter(2, [800 / (SR / 2), 3500 / (SR / 2)], btype='band')
        out += sps.lfilter(b, a, tail)
        return out * 0.84

    else:  # Mellow click
        dur = 0.12
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        modes = [(520.0, 0.50, 45.0), (1200.0, 0.30, 60.0)]
        out = np.zeros(n)
        for freq, amp, decay in modes:
            out += amp * np.sin(2.0 * np.pi * freq * t) * np.exp(-t * decay)
        b, a = sps.butter(2, 2200 / (SR / 2), btype='low')
        return sps.lfilter(b, a, out) * 0.78


def synth_hat(variant: int = 0, open_hat: bool = False) -> np.ndarray:
    """4 tipos de platillos/shakers lo-fi suaves y calidos (sin estridencia ni picos resonantes):
       0: Closed warm tape hat (ruido rosa filtrado suave, cuerpo redondo).
       1: Sizzle open hat (suspiro lo-fi amortiguado, sin agudos estridentes).
       2: Shaker organico de semillas / cabasa acustica.
       3: Mellow ride / ping sutil vintage.
    """
    if variant == 2:  # Shaker organico suave
        dur = 0.075
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        white = np.random.normal(0, 1, n)
        b, a = sps.butter(2, [2600 / (SR / 2), 6200 / (SR / 2)], btype='band')
        shaker = sps.lfilter(b, a, white)
        env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.6
        return shaker * env * 0.12

    elif variant == 3:  # Ride cymbal / Ping amortiguado
        dur = 0.28 if open_hat else 0.10
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        modes = [1850.0, 2420.0, 3350.0]
        bell = np.zeros(n)
        for m in modes:
            bell += np.sin(2.0 * np.pi * m * t) * 0.035
        noise = np.random.normal(0, 1, n) * 0.20
        b, a = sps.butter(2, [2400 / (SR / 2), 6500 / (SR / 2)], btype='band')
        sig = sps.lfilter(b, a, bell + noise)
        decay = 18.0 if open_hat else 55.0
        return sig * np.exp(-t * decay) * 0.11

    elif variant == 1 or open_hat:  # Sizzle open hat calido
        dur = 0.18
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        noise = np.random.normal(0, 1, n)
        b, a = sps.butter(2, [2500 / (SR / 2), 6000 / (SR / 2)], btype='band')
        sig = sps.lfilter(b, a, noise)
        decay = 19.0
        return sig * np.exp(-t * decay) * 0.12

    else:  # Closed tape hat (variant 0)
        dur = 0.045
        n = int(dur * SR)
        t = np.linspace(0, dur, n, endpoint=False)
        noise = np.random.normal(0, 1, n)
        b, a = sps.butter(2, [2400 / (SR / 2), 5600 / (SR / 2)], btype='band')
        filtered = sps.lfilter(b, a, noise)
        click_len = min(int(0.002 * SR), n)
        filtered[:click_len] += np.random.normal(0, 0.25, click_len) * np.exp(-np.linspace(0, 8, click_len))
        decay = 75.0
        return filtered * np.exp(-t * decay) * 0.10
def synth_vinyl_crackle(duration: float) -> np.ndarray:
    """Genera crujido de aguja de tocadiscos y polvo estocástico auténtico."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n, endpoint=False)
    # 1. Hiss de superficie suave
    surface = np.random.normal(0, 1, n)
    b_hiss, a_hiss = sps.butter(2, [800 / (SR / 2), 4500 / (SR / 2)], btype='band')
    hiss = sps.lfilter(b_hiss, a_hiss, surface) * 0.0055

    # 2. Clicks y pops estocásticos (partículas de polvo en el vinilo)
    clicks = np.zeros(n)
    num_clicks = int(duration * random.uniform(3.0, 7.0))
    for _ in range(num_clicks):
        pos = random.randint(0, max(0, n - 200))
        click_dur = random.randint(30, 120)
        pop_env = np.exp(-np.linspace(0, 12, click_dur))
        pop = np.random.normal(0, 1, click_dur) * pop_env * random.uniform(0.012, 0.035)
        clicks[pos:pos + click_dur] += pop

    return hiss + clicks


def apply_tape_flutter(audio: np.ndarray) -> np.ndarray:
    """Modulación sutil de tono (tape wow & flutter) con LFO ultra-lento de 0.35 Hz."""
    n = audio.shape[1]
    t = np.linspace(0, n / SR, n, endpoint=False)
    # Variación de retardo de ±0.8ms a 0.38Hz
    mod = 0.0008 * np.sin(2.0 * np.pi * 0.38 * t) + 0.0003 * np.sin(2.0 * np.pi * 1.15 * t)
    indices = np.arange(n) + (mod * SR)
    indices = np.clip(indices, 0, n - 1)

    out = np.zeros_like(audio)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.clip(idx_floor + 1, 0, n - 1)
    frac = indices - idx_floor

    out[0] = audio[0, idx_floor] * (1.0 - frac) + audio[0, idx_ceil] * frac
    out[1] = audio[1, idx_floor] * (1.0 - frac) + audio[1, idx_ceil] * frac
    return out


# ─────────────────────────────────────────────────────────────
# 3. FOLEY ORGÁNICO REACTIVO
# ─────────────────────────────────────────────────────────────

def synth_scene_foley(scene: str, duration: float) -> np.ndarray:
    """Generador procedural de paisajes sonoros y Foley de alta inmersion,
    especificamente calibrado para cada escena visual del streaming:
    nieve, tormenta/lluvia, servidores/ventiladores/reactor, base submarina, etc.
    """
    n = int(duration * SR)
    t = np.linspace(0, duration, n, endpoint=False)
    out = np.zeros(n)
    sc = (scene or "").lower()

    # 1. ESCENAS DE NIEVE / POLAR (Puesto Artico / Nieve 2h)
    if any(k in sc for k in ("nieve", "polar", "arctic", "glaciar")):
        # Viento sub-grave continuo del artico (muy suave y envolvente)
        b_sub, a_sub = sps.butter(2, [28 / (SR / 2), 160 / (SR / 2)], btype='band')
        sub_rumble = sps.lfilter(b_sub, a_sub, np.random.normal(0, 1, n)) * 0.024
        # Silbido y rafagas de ventisca organica atenuada
        gust_lfo = 0.60 + 0.40 * np.sin(2.0 * np.pi * 0.065 * t + 0.4 * np.sin(2.0 * np.pi * 0.018 * t))
        b_howl, a_howl = sps.butter(2, [260 / (SR / 2), 920 / (SR / 2)], btype='band')
        howl = sps.lfilter(b_howl, a_howl, np.random.normal(0, 1, n)) * gust_lfo * 0.020
        # Cristales de nieve fina / ventisca granular muy suave
        b_ice, a_ice = sps.butter(2, [2200 / (SR / 2), 5200 / (SR / 2)], btype='band')
        ice_drift = sps.lfilter(b_ice, a_ice, np.random.normal(0, 1, n)) * (gust_lfo * 0.75) * 0.010
        # Crujidos esporadicos de tension por frio extremo
        num_creaks = max(1, int(duration / 18))
        for _ in range(num_creaks):
            pos = random.randint(0, max(0, n - int(1.2 * SR)))
            dur_k = 1.0
            n_k = int(dur_k * SR)
            t_k = np.linspace(0, dur_k, n_k, endpoint=False)
            creak = np.sin(2.0 * np.pi * 125.0 * t_k) * np.exp(-t_k * 5.0) * 0.008
            out[pos:pos + n_k] += creak
        out += sub_rumble + howl + ice_drift

    # 2. ESCENAS DE LLUVIA Y TORMENTA (Tormenta 2h / Bunker)
    elif any(k in sc for k in ("tormenta", "lluvia", "storm", "rain")):
        # Capa continua de lluvia acogedora en segundo plano (amortiguada)
        b_rain, a_rain = sps.butter(2, [350 / (SR / 2), 3400 / (SR / 2)], btype='band')
        rain_wash = sps.lfilter(b_rain, a_rain, np.random.normal(0, 1, n)) * 0.022
        # Impacto estocastico de gotas de lluvia tenue
        droplets = np.zeros(n)
        num_drops = int(duration * 18)
        for _ in range(num_drops):
            pos = random.randint(0, max(0, n - 300))
            d_len = random.randint(50, 220)
            drop = np.random.normal(0, 0.009, d_len) * np.exp(-np.linspace(0, 7.5, d_len))
            droplets[pos:pos + d_len] += drop
        # Truenos lejanos y profundos con reverberacion natural
        thunders = np.zeros(n)
        num_thunders = max(1, int(duration / 25))
        for _ in range(num_thunders):
            pos = random.randint(0, max(0, n - int(4.5 * SR)))
            dur_th = 4.0
            n_th = int(dur_th * SR)
            t_th = np.linspace(0, dur_th, n_th, endpoint=False)
            th_noise = np.random.normal(0, 1, n_th)
            b_th, a_th = sps.butter(2, [22 / (SR / 2), 85 / (SR / 2)], btype='band')
            th_env = (np.sin(np.pi * np.clip(t_th / dur_th, 0, 1)) ** 2) * np.exp(-t_th * 0.45)
            thunders[pos:pos + n_th] += sps.lfilter(b_th, a_th, th_noise) * th_env * 0.028
        out += rain_wash + droplets + thunders

    # 3. ESCENAS DE SERVIDORES, REACTORES Y VENTILADORES (Reactor 2h / SCP Contencion / Servidores)
    elif any(k in sc for k in ("reactor", "scp_contencion", "contencion", "servidor", "server", "ventilador")):
        # Zumbido electromagnetico suave (60 Hz + 120 Hz)
        hum = np.sin(2.0 * np.pi * 60.0 * t) * 0.014 + np.sin(2.0 * np.pi * 120.0 * t) * 0.007
        # Aspas de ventiladores industriales difusas
        fans = (np.sin(2.0 * np.pi * 184.0 * t) * 0.010 + np.sin(2.0 * np.pi * 246.0 * t) * 0.008)
        # Flujo de aire de servidores filtrado en segundo plano
        air_noise = np.random.normal(0, 1, n)
        b_air, a_air = sps.butter(2, [320 / (SR / 2), 1750 / (SR / 2)], btype='band')
        airflow = sps.lfilter(b_air, a_air, air_noise) * 0.020
        # Actividad tenue de cabezales HDD
        hdd_clicks = np.zeros(n)
        for _ in range(int(duration * 4)):
            pos = random.randint(0, max(0, n - 250))
            dur_c = random.randint(25, 65)
            hdd_clicks[pos:pos + dur_c] += np.random.normal(0, 0.008, dur_c) * np.exp(-np.linspace(0, 6, dur_c))
        out += hum + fans + airflow + hdd_clicks

    # 4. BASE SUBMARINA (Submarina 2h / Oceano Profundo)
    elif any(k in sc for k in ("submarina", "ocean", "deep")):
        # Resonancia de casco bajo presion hidrostatica profunda (38 Hz)
        hull_hum = (np.sin(2.0 * np.pi * 38.0 * t) * 0.016 + np.sin(2.0 * np.pi * 76.0 * t) * 0.008)
        flow_noise = np.random.normal(0, 1, n)
        b_flow, a_flow = sps.butter(2, 260 / (SR / 2), btype='low')
        water_flow = sps.lfilter(b_flow, a_flow, flow_noise) * 0.018
        # Sonar distante muy tenue
        num_pings = max(1, int(duration / 20))
        for _ in range(num_pings):
            pos = random.randint(0, max(0, n - int(3.0 * SR)))
            dur_p = 2.5
            n_p = int(dur_p * SR)
            t_p = np.linspace(0, dur_p, n_p, endpoint=False)
            ping = (np.sin(2.0 * np.pi * 680.0 * t_p) + 0.25 * np.sin(2.0 * np.pi * 1360.0 * t_p)) * np.exp(-t_p * 1.8) * 0.012
            out[pos:pos + n_p] += ping
        out += hull_hum + water_flow

    # 5. ESTACION ORBITAL Y ESPACIO (Orbital 2h / Space)
    elif any(k in sc for k in ("orbital", "space", "satelite")):
        b_air, a_air = sps.butter(2, [140 / (SR / 2), 650 / (SR / 2)], btype='band')
        ecs_flow = sps.lfilter(b_air, a_air, np.random.normal(0, 1, n)) * 0.018
        gyro = np.sin(2.0 * np.pi * 48.0 * t) * 0.010 + np.sin(2.0 * np.pi * 96.0 * t) * 0.005
        num_bursts = max(1, int(duration / 15))
        for _ in range(num_bursts):
            pos = random.randint(0, max(0, n - int(1.2 * SR)))
            dur_b = 0.8
            n_b = int(dur_b * SR)
            t_b = np.linspace(0, dur_b, n_b, endpoint=False)
            telemetry = np.sin(2.0 * np.pi * 1450.0 * t_b) * np.sin(2.0 * np.pi * 10.0 * t_b) * np.exp(-t_b * 3.8) * 0.008
            out[pos:pos + n_b] += telemetry
        out += ecs_flow + gyro

    # 6. BOSQUE PROFUNDO (Bosque 2h)
    elif any(k in sc for k in ("bosque", "forest", "nature")):
        b_wind, a_wind = sps.butter(2, [320 / (SR / 2), 1800 / (SR / 2)], btype='band')
        wind_lfo = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.12 * t)
        canopy = sps.lfilter(b_wind, a_wind, np.random.normal(0, 1, n)) * wind_lfo * 0.016
        num_leaves = int(duration / 6)
        for _ in range(num_leaves):
            pos = random.randint(0, max(0, n - int(1.2 * SR)))
            dur_l = 1.0
            n_l = int(dur_l * SR)
            t_l = np.linspace(0, dur_l, n_l, endpoint=False)
            b_r, a_r = sps.butter(2, [1400 / (SR / 2), 3800 / (SR / 2)], btype='band')
            rustle = sps.lfilter(b_r, a_r, np.random.normal(0, 1, n_l)) * np.sin(np.pi * np.clip(t_l / dur_l, 0, 1)) * 0.008
            out[pos:pos + n_l] += rustle
        out += canopy

    # 7. DESIERTO (Desierto 2h)
    elif any(k in sc for k in ("desierto", "desert", "arid")):
        b_d, a_d = sps.butter(2, [180 / (SR / 2), 1100 / (SR / 2)], btype='band')
        desert_lfo = 0.50 + 0.50 * np.sin(2.0 * np.pi * 0.08 * t)
        d_wind = sps.lfilter(b_d, a_d, np.random.normal(0, 1, n)) * desert_lfo * 0.020
        out += d_wind

    # 8. CAMPO ELECTRICO / OBSERVATORIO (Electrica 2h / Array Suiza / SCP Exterior)
    elif any(k in sc for k in ("electrica", "array", "suiza", "scp_exterior")):
        b_e, a_e = sps.butter(2, [100 / (SR / 2), 950 / (SR / 2)], btype='band')
        tower_hum = (np.sin(2.0 * np.pi * 100.0 * t) * 0.010 + np.sin(2.0 * np.pi * 200.0 * t) * 0.005)
        cable_wind = sps.lfilter(b_e, a_e, np.random.normal(0, 1, n)) * 0.016
        out += tower_hum + cable_wind

    # 9. CALDERA VOLCANICA (Volcanica 2h)
    elif any(k in sc for k in ("volcanica", "volcano", "magma")):
        b_v, a_v = sps.butter(2, [24 / (SR / 2), 120 / (SR / 2)], btype='band')
        magma_rumble = sps.lfilter(b_v, a_v, np.random.normal(0, 1, n)) * (0.7 + 0.3 * np.sin(2.0 * np.pi * 0.04 * t)) * 0.022
        out += magma_rumble

    # 10. CAFETERIA / CIUDAD NOCTURNA (Veo3 Manana/Tarde/Noche, Latam Noche)
    else:
        rain_soft = np.random.normal(0, 1, n)
        b_rain, a_rain = sps.butter(2, [280 / (SR / 2), 2400 / (SR / 2)], btype='band')
        out += sps.lfilter(b_rain, a_rain, rain_soft) * 0.016
        num_clinks = max(1, int(duration / 20))
        for _ in range(num_clinks):
            pos = random.randint(0, max(0, n - int(0.8 * SR)))
            dur_c = 0.6
            n_c = int(dur_c * SR)
            t_c = np.linspace(0, dur_c, n_c, endpoint=False)
            clink = (np.sin(2.0 * np.pi * 2200.0 * t_c) * 0.4 + np.sin(2.0 * np.pi * 3800.0 * t_c) * 0.2) * np.exp(-t_c * 24.0) * 0.006
            out[pos:pos + n_c] += clink

    return out
# 4. PALETA DE 16 PROGRESIONES ARMÓNICAS RICAS (CAFE & SCI-FI)
# ─────────────────────────────────────────────────────────────

CAFE_THEMES = [
    {
        "name": "D minor Gloom Station (57 BPM)",
        "bpm": 57,
        "scale": [50, 53, 55, 57, 60, 62, 65, 69],
        "chords": [
            [50, 53, 57, 60, 64],  # Dm9
            [46, 50, 53, 57, 62],  # Bbmaj9
            [43, 46, 50, 53, 58],  # Gm9
            [48, 52, 55, 62, 64],  # Cadd9
            [45, 48, 52, 55, 60],  # Am7
        ],
        "bass": [38, 34, 31, 36, 33],
    },
    {
        "name": "C Major Sunday Afternoon (78 BPM)",
        "bpm": 78,
        "scale": [60, 62, 64, 67, 69, 72, 74, 76],
        "chords": [
            [60, 64, 67, 71, 74],  # Cmaj9
            [57, 60, 64, 67, 71],  # Am9
            [53, 57, 60, 64, 69],  # Fmaj9
            [55, 59, 62, 65, 71],  # G13
        ],
        "bass": [48, 45, 41, 43],
    },
    {
        "name": "D Dorian Midnight Espresso (74 BPM)",
        "bpm": 74,
        "scale": [57, 60, 62, 64, 67, 69, 72, 74],
        "chords": [
            [50, 53, 57, 60, 64],  # Dm9
            [55, 58, 62, 65, 69],  # Gm9
            [53, 57, 60, 64, 67],  # Fmaj7
            [52, 55, 59, 62, 67],  # Em7
        ],
        "bass": [38, 43, 41, 40],
    },
    {
        "name": "F Lydian Golden Sunset (80 BPM)",
        "bpm": 80,
        "scale": [53, 57, 60, 62, 65, 67, 69, 72],
        "chords": [
            [53, 57, 60, 64, 69],  # Fmaj9
            [50, 53, 57, 60, 65],  # Dm11
            [48, 52, 55, 59, 64],  # Cmaj9
            [55, 58, 62, 65, 70],  # Gm11
        ],
        "bass": [41, 38, 36, 43],
    },
    {
        "name": "Bb Neo-Soul Velvet (72 BPM)",
        "bpm": 72,
        "scale": [58, 61, 63, 65, 68, 70, 73, 75],
        "chords": [
            [58, 61, 65, 68, 72],  # Bbm9
            [54, 58, 61, 65, 70],  # Gbmaj9
            [49, 53, 56, 60, 65],  # Dbmaj9
            [56, 60, 63, 66, 71],  # Ab13
        ],
        "bass": [46, 42, 37, 44],
    },
    {
        "name": "Eb Major Warm Rain (75 BPM)",
        "bpm": 75,
        "scale": [51, 55, 58, 60, 63, 65, 67, 70],
        "chords": [
            [51, 55, 58, 62, 65],  # Ebmaj9
            [48, 51, 55, 58, 62],  # Cm9
            [53, 56, 60, 63, 67],  # Fm9
            [46, 50, 53, 56, 62],  # Bb13
        ],
        "bass": [39, 36, 41, 34],
    },
    {
        "name": "G Major Chill Bossa (82 BPM)",
        "bpm": 82,
        "scale": [55, 57, 59, 62, 64, 67, 69, 71],
        "chords": [
            [55, 59, 62, 66, 69],  # Gmaj9
            [52, 55, 59, 62, 66],  # Em9
            [48, 52, 55, 59, 64],  # Cmaj9
            [50, 53, 57, 60, 66],  # D13
        ],
        "bass": [43, 40, 36, 38],
    },
    {
        "name": "A Major Sakura Coffee (76 BPM)",
        "bpm": 76,
        "scale": [57, 59, 61, 64, 66, 69, 71, 73],
        "chords": [
            [57, 61, 64, 68, 71],  # Amaj9
            [54, 57, 61, 64, 68],  # F#m9
            [47, 50, 54, 57, 61],  # Bm9
            [52, 56, 59, 62, 68],  # E13
        ],
        "bass": [45, 42, 35, 40],
    }
]

OUTPOST_THEMES = [
    {
        "name": "Deep Space D Minor (56 BPM)",
        "bpm": 56,
        "scale": [50, 53, 55, 57, 60, 62, 65, 69],
        "chords": [
            [50, 53, 57, 60, 64],  # Dm9
            [46, 50, 53, 57, 62],  # Bbmaj9
            [43, 46, 50, 53, 58],  # Gm9
            [45, 48, 52, 55, 60],  # Am7
        ],
        "bass": [38, 34, 31, 33],
    },
    {
        "name": "F# Lydian Orbital Sanctuary (68 BPM)",
        "bpm": 68,
        "scale": [54, 57, 59, 61, 64, 66, 69, 71],
        "chords": [
            [54, 61, 64, 68, 71],  # F#m9
            [50, 57, 61, 66, 68],  # Dmaj7#11
            [57, 61, 64, 68, 71],  # Aadd9
            [49, 56, 61, 64, 68],  # C#m7
        ],
        "bass": [42, 38, 45, 37],
    },
    {
        "name": "A Minor Cosmic Drift (66 BPM)",
        "bpm": 66,
        "scale": [57, 60, 62, 64, 67, 69, 72, 74],
        "chords": [
            [57, 60, 64, 67, 71],  # Am9
            [53, 57, 60, 64, 69],  # Fmaj7
            [50, 53, 57, 60, 64],  # Dm9
            [52, 55, 59, 62, 67],  # Em7
        ],
        "bass": [45, 41, 38, 40],
    },
    {
        "name": "C Minor SCP Quarantine (64 BPM)",
        "bpm": 64,
        "scale": [48, 51, 53, 55, 58, 60, 63, 65],
        "chords": [
            [48, 55, 60, 63, 67],  # Cm9
            [46, 53, 58, 62, 65],  # Bbmaj7
            [44, 51, 56, 60, 63],  # Abmaj7
            [43, 50, 55, 58, 62],  # Gm7
        ],
        "bass": [36, 34, 32, 31],
    },
    {
        "name": "E Minor Solar Array (67 BPM)",
        "bpm": 67,
        "scale": [52, 55, 57, 59, 62, 64, 67, 69],
        "chords": [
            [52, 59, 62, 66, 69],  # Em9
            [48, 55, 60, 64, 67],  # Cmaj9
            [55, 62, 65, 69, 72],  # Gmaj7
            [50, 57, 60, 64, 67],  # D7sus4
        ],
        "bass": [40, 36, 43, 38],
    },
    {
        "name": "B Minor Ion Horizon (65 BPM)",
        "bpm": 65,
        "scale": [47, 50, 52, 54, 57, 59, 62, 64],
        "chords": [
            [47, 54, 57, 61, 64],  # Bm9
            [43, 50, 55, 59, 62],  # Gmaj7
            [40, 47, 52, 55, 59],  # Em9
            [42, 49, 54, 57, 61],  # F#m7
        ],
        "bass": [35, 31, 28, 30],
    },
    {
        "name": "G Minor Nebula Station (63 BPM)",
        "bpm": 63,
        "scale": [55, 58, 60, 62, 65, 67, 70, 72],
        "chords": [
            [55, 58, 62, 65, 69],  # Gm9
            [51, 55, 58, 62, 65],  # Ebmaj7
            [48, 51, 55, 58, 62],  # Cm9
            [50, 53, 57, 60, 64],  # Dm7
        ],
        "bass": [43, 39, 36, 38],
    },
    {
        "name": "D# Minor Sub-Zero Bunker (60 BPM)",
        "bpm": 60,
        "scale": [51, 54, 56, 58, 61, 63, 66, 68],
        "chords": [
            [51, 54, 58, 61, 65],  # D#m9
            [47, 51, 54, 58, 63],  # Bmaj9
            [44, 47, 51, 54, 59],  # G#m9
            [46, 49, 53, 56, 61],  # A#m7
        ],
        "bass": [39, 35, 32, 34],
    }
]

# ─────────────────────────────────────────────────────────────
# 5. GENERADOR MELÓDICO HUMANO CON FRASAS DINÁMICAS
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 5. GENERADOR MELÓDICO HUMANO CON FRASAS DINÁMICAS
# ─────────────────────────────────────────────────────────────

def synth_lofi_lead_melody(scale: list, num_bars: int, bar_dur: float, beat_dur: float, total_samples: int, dark: bool) -> np.ndarray:
    bus = np.zeros(total_samples)
    curr_idx = len(scale) // 2

    for b in range(num_bars):
        bar_start = b * bar_dur
        # Silencios estratégicos (respiro musical orgánico en compases alternados)
        if b % 4 == 3 or b % 8 in (2, 5):
            continue

        rhythms = [
            [0.5, 1.5, 2.5],
            [0.0, 1.25, 2.5, 3.25],
            [1.0, 2.0, 3.0],
            [0.75, 2.25],
            [1.5, 2.5, 3.5]
        ]
        rhythm = random.choice(rhythms)
        for beat_pos in rhythm:
            note_time = bar_start + beat_pos * beat_dur
            idx = int(note_time * SR)
            if idx >= total_samples - int(0.7 * SR):
                continue

            # Movimiento melódico por grados conjuntos (pasos suaves con saltos ocasionales)
            step = random.choices([-2, -1, 0, 1, 2, -3, 3], weights=[0.18, 0.28, 0.12, 0.26, 0.10, 0.03, 0.03])[0]
            curr_idx = max(0, min(len(scale) - 1, curr_idx + step))
            note = scale[curr_idx]

            f = midi_to_freq(note)
            dur_n = random.uniform(0.35, 0.70)
            n_len = int(dur_n * SR)
            n_len = min(n_len, total_samples - idx)
            tn = np.linspace(0, dur_n, n_len, endpoint=False)

            # Vibrato expresivo retardado estilo jazz/lo-fi
            vib_onset = np.clip((tn - 0.12) / 0.15, 0, 1)
            vib = np.sin(2.0 * np.pi * 5.0 * tn) * (0.012 * f * vib_onset)
            instant_phase = 2.0 * np.pi * f * tn + (2.0 * np.pi * vib * tn)

            tone = (np.sin(instant_phase) + 
                    0.20 * np.sin(2.0 * instant_phase) + 
                    0.05 * np.sin(3.0 * instant_phase))
            env = np.clip(tn / 0.018, 0, 1) * np.exp(-tn * 2.8)
            vel = random.uniform(0.30, 0.44)
            bus[idx:idx + n_len] += tone * env * vel

    b_lp, a_lp = sps.butter(2, (1800 if dark else 2600) / (SR / 2), btype='low')
    return sps.lfilter(b_lp, a_lp, bus)


def synth_rhodes_voice(f: float, t_arr: np.ndarray, decay_rate: float = 0.58) -> np.ndarray:
    """Sintetiza una nota de piano Rhodes Mark I auténtico con ataque de campana (tine strike),
    cuerpo armónico rico, vibrato suitcase y suave saturación no-lineal analógica."""
    tine_f = f * 3.82
    tine = 0.16 * np.sin(2.0 * np.pi * tine_f * t_arr) * np.exp(-t_arr * 42.0)
    body = (
        np.sin(2.0 * np.pi * f * t_arr) +
        0.30 * np.sin(4.0 * np.pi * f * t_arr) +
        0.12 * np.sin(6.0 * np.pi * f * t_arr) +
        0.04 * np.sin(8.0 * np.pi * f * t_arr)
    )
    env = np.clip(t_arr / 0.016, 0, 1) * np.exp(-t_arr * decay_rate)
    trem = 1.0 + 0.08 * np.sin(2.0 * np.pi * 4.2 * t_arr)
    raw = (body + tine) * env * trem
    return np.tanh(1.25 * raw) * 0.82


def select_theme_and_rhythm_for_scene(scene_name: str, mode: str):
    """Mapea cada escena visual a su paleta armónica y groove más coherente e inmersivo."""
    sc = (scene_name or "").lower()
    dark = (mode == "outpost")

    # 1. Nieve / Polar: Glacial ambient espacioso, sin batería o con shakers mínimos
    if any(k in sc for k in ("nieve", "polar", "arctic", "glaciar")):
        matching = [t for t in OUTPOST_THEMES if any(w in t["name"].lower() for w in ("sub-zero", "deep space", "gloom"))]
        theme = random.choice(matching) if matching else random.choice(OUTPOST_THEMES)
        rhythm_style = random.choice([0, 0, 1, 5])

    # 2. Tormenta / Lluvia: Neo-soul melancólico, lluvia y grooves relajados
    elif any(k in sc for k in ("tormenta", "lluvia", "storm", "rain")):
        matching = [t for t in (CAFE_THEMES + OUTPOST_THEMES) if any(w in t["name"].lower() for w in ("rain", "velvet", "gloom", "espresso"))]
        theme = random.choice(matching) if matching else random.choice(CAFE_THEMES)
        rhythm_style = random.choice([1, 3, 5])

    # 3. Orbital / Espacio: Armónicos Lydian suspendidos, ingravidez cósmica
    elif any(k in sc for k in ("orbital", "space", "satelite")):
        matching = [t for t in OUTPOST_THEMES if any(w in t["name"].lower() for w in ("orbital", "cosmic", "nebula"))]
        theme = random.choice(matching) if matching else random.choice(OUTPOST_THEMES)
        rhythm_style = random.choice([0, 1, 5])

    # 4. Base Submarina: Graves abisales, acordes oscuros de baja frecuencia
    elif any(k in sc for k in ("submarina", "ocean", "deep")):
        matching = [t for t in OUTPOST_THEMES if any(w in t["name"].lower() for w in ("quarantine", "gloom", "sub-zero"))]
        theme = random.choice(matching) if matching else random.choice(OUTPOST_THEMES)
        rhythm_style = random.choice([0, 1, 5])

    # 5. Reactor / SCP Contención: Tensión cyberpunk industrial, stabs y sub-graves
    elif any(k in sc for k in ("reactor", "scp_contencion", "contencion", "servidor")):
        matching = [t for t in OUTPOST_THEMES if any(w in t["name"].lower() for w in ("quarantine", "ion", "nebula"))]
        theme = random.choice(matching) if matching else random.choice(OUTPOST_THEMES)
        rhythm_style = random.choice([1, 2, 3])

    # 6. Bosque: Acordes acústicos cálidos, arpegios orgánicos
    elif any(k in sc for k in ("bosque", "forest", "nature")):
        matching = [t for t in CAFE_THEMES if any(w in t["name"].lower() for w in ("sunset", "sakura", "bossa"))]
        theme = random.choice(matching) if matching else random.choice(CAFE_THEMES)
        rhythm_style = random.choice([4, 5, 1])

    # 7. Desierto: Escalas exóticas, groove lento e hipnótico
    elif any(k in sc for k in ("desierto", "desert", "arid")):
        matching = [t for t in OUTPOST_THEMES if any(w in t["name"].lower() for w in ("solar", "ion", "deep space"))]
        theme = random.choice(matching) if matching else random.choice(OUTPOST_THEMES)
        rhythm_style = random.choice([1, 3])

    # 8. Array Suiza / Eléctrica: Resonancia de telescopios alpinos y campo EMF
    elif any(k in sc for k in ("electrica", "array", "suiza", "scp_exterior")):
        matching = [t for t in OUTPOST_THEMES if any(w in t["name"].lower() for w in ("solar", "ion", "orbital"))]
        theme = random.choice(matching) if matching else random.choice(OUTPOST_THEMES)
        rhythm_style = random.choice([0, 1, 5])

    # 9. Caldera Volcánica: Magma subterráneo
    elif any(k in sc for k in ("volcanica", "volcano", "magma")):
        matching = [t for t in OUTPOST_THEMES if any(w in t["name"].lower() for w in ("sub-zero", "quarantine", "gloom"))]
        theme = random.choice(matching) if matching else random.choice(OUTPOST_THEMES)
        rhythm_style = random.choice([0, 1])

    # 10. Cafetería / Mañana / Tarde / Noche (Veo3 & Latam)
    elif "manana" in sc:
        matching = [t for t in CAFE_THEMES if any(w in t["name"].lower() for w in ("sunday", "sunset"))]
        theme = random.choice(matching) if matching else random.choice(CAFE_THEMES)
        rhythm_style = random.choice([2, 4, 1])
    elif "tarde" in sc:
        matching = [t for t in CAFE_THEMES if any(w in t["name"].lower() for w in ("bossa", "sakura", "rain"))]
        theme = random.choice(matching) if matching else random.choice(CAFE_THEMES)
        rhythm_style = random.choice([4, 5, 2])
    elif "noche" in sc or "latam" in sc:
        matching = [t for t in CAFE_THEMES if any(w in t["name"].lower() for w in ("espresso", "velvet", "gloom"))]
        theme = random.choice(matching) if matching else random.choice(CAFE_THEMES)
        rhythm_style = random.choice([3, 1, 2])
    else:
        pool = OUTPOST_THEMES if dark else CAFE_THEMES
        theme = random.choice(pool)
        rhythm_style = random.choice([0, 1, 2, 3, 4, 5])

    return theme, rhythm_style


# ─────────────────────────────────────────────────────────────
# 6. COMPOSICIÓN INTEGRAL CON 6 ESTILOS RÍTMICOS VARIADOS
# ─────────────────────────────────────────────────────────────

def compose_track(scene_name: str, mode: str, num_bars: int = 16) -> np.ndarray:
    dark = (mode == "outpost")
    theme, rhythm_style = select_theme_and_rhythm_for_scene(scene_name, mode)

    bpm = theme["bpm"]
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4.0
    total_dur = num_bars * bar_dur
    total_samples = int(total_dur * SR)

    kick_var = random.choice([0, 1, 2, 3])
    rim_var = random.choice([0, 1, 2, 3])
    hat_var = random.choice([0, 1, 2, 3])

    kick_sample = synth_kick(variant=kick_var)
    rim_sample = synth_rimshot(variant=rim_var)
    hat_sample = synth_hat(variant=hat_var, open_hat=False)
    open_hat_sample = synth_hat(variant=hat_var, open_hat=True)

    rhodes_bus = np.zeros(total_samples)
    bass_bus = np.zeros(total_samples)
    drum_bus = np.zeros(total_samples)
    melody_bus = synth_lofi_lead_melody(theme["scale"], num_bars, bar_dur, beat_dur, total_samples, dark)
    foley_bus = synth_scene_foley(scene_name, total_dur)
    vinyl_bus = synth_vinyl_crackle(total_dur)

    chords = theme["chords"]
    bass_notes = theme["bass"]

    for bar in range(num_bars):
        bar_start = bar * bar_dur
        chord_idx = bar % len(chords)
        current_chord = chords[chord_idx]
        current_bass = bass_notes[chord_idx]

        # ─── A. ACOMPAÑAMIENTO ARMÓNICO CON RASGUEO HUMANO (STRUM) ───
        strum_direction = random.choice([1, -1])
        sorted_notes = sorted(current_chord, reverse=(strum_direction == -1))
        comp_pattern = random.choice(["sustained", "syncopated_stab", "arpeggiated"])

        if comp_pattern == "sustained":
            for n_idx, note in enumerate(sorted_notes):
                strum_offset = n_idx * random.uniform(0.012, 0.028)
                idx = int((bar_start + strum_offset) * SR)
                if idx >= total_samples:
                    continue
                n_len = min(int(bar_dur * 0.95 * SR), total_samples - idx)
                t_c = np.linspace(0, n_len / SR, n_len, endpoint=False)
                f = midi_to_freq(note)
                rhodes_bus[idx:idx + n_len] += synth_rhodes_voice(f, t_c, decay_rate=0.58) * (0.54 / len(current_chord))

        elif comp_pattern == "syncopated_stab":
            for hit_beat in [0.0, 2.5]:
                for n_idx, note in enumerate(sorted_notes):
                    strum_offset = n_idx * random.uniform(0.010, 0.022)
                    idx = int((bar_start + hit_beat * beat_dur + strum_offset) * SR)
                    if idx >= total_samples:
                        continue
                    dur_hit = (bar_dur - hit_beat * beat_dur) * 0.85
                    n_len = min(int(dur_hit * SR), total_samples - idx)
                    t_c = np.linspace(0, n_len / SR, n_len, endpoint=False)
                    f = midi_to_freq(note)
                    decay_rate = 1.2 if hit_beat > 0 else 0.70
                    rhodes_bus[idx:idx + n_len] += synth_rhodes_voice(f, t_c, decay_rate=decay_rate) * (0.52 / len(current_chord))

        else:  # arpeggiated
            for a_idx, note in enumerate(current_chord):
                arp_beat = (a_idx * 0.75) % 4.0
                idx = int((bar_start + arp_beat * beat_dur) * SR)
                if idx >= total_samples:
                    continue
                n_len = min(int(1.2 * SR), total_samples - idx)
                t_c = np.linspace(0, n_len / SR, n_len, endpoint=False)
                f = midi_to_freq(note)
                rhodes_bus[idx:idx + n_len] += synth_rhodes_voice(f, t_c, decay_rate=1.8) * 0.18

        # ─── B. LÍNEA DE BAJO DINÁMICA CON NOTAS DE PASO Y 2DO ARMÓNICO ───
        idx_b0 = int(bar_start * SR)
        fb0 = midi_to_freq(current_bass)
        n_len_b = min(int(bar_dur * 0.90 * SR), total_samples - idx_b0)
        t_b0 = np.linspace(0, n_len_b / SR, n_len_b, endpoint=False)
        bass_sig0 = (np.sin(2.0 * np.pi * fb0 * t_b0) + 
                     0.22 * np.sin(4.0 * np.pi * fb0 * t_b0)) * np.exp(-t_b0 * 0.38)
        bass_sig0 = np.tanh(1.25 * bass_sig0) * 0.75
        bass_bus[idx_b0:idx_b0 + n_len_b] += bass_sig0 * 0.70

        if bar % 2 == 1:
            idx_bpass = int((bar_start + 2.5 * beat_dur) * SR)
            if idx_bpass < total_samples:
                fb_pass = midi_to_freq(current_bass + random.choice([7, 12, -5]))
                n_len_bp = min(int(1.2 * SR), total_samples - idx_bpass)
                t_bp = np.linspace(0, n_len_bp / SR, n_len_bp, endpoint=False)
                bass_pass = (np.sin(2.0 * np.pi * fb_pass * t_bp) + 
                             0.18 * np.sin(4.0 * np.pi * fb_pass * t_bp)) * np.exp(-t_bp * 1.0)
                bass_pass = np.tanh(1.2 * bass_pass) * 0.65
                bass_bus[idx_bpass:idx_bpass + n_len_bp] += bass_pass * 0.48

        # ─── C. BATERÍAS CON LOS 6 GROOVES DIVERSOS ───
        is_breakdown = (bar % 8 == 7)  # Compás 7: respiro total de drums

        if not is_breakdown and rhythm_style != 0:  # Style 0 = Sin batería
            if rhythm_style == 1:  # Half-time lo-fi (minimalista y espacioso)
                # Bombo en beat 0
                idx_k = int(bar_start * SR)
                if idx_k < total_samples:
                    k_len = min(len(kick_sample), total_samples - idx_k)
                    drum_bus[idx_k:idx_k + k_len] += kick_sample[:k_len] * 0.85
                # Rimshot solo en beat 2 (half-time)
                idx_r = int((bar_start + 2.0 * beat_dur) * SR)
                if idx_r < total_samples:
                    r_len = min(len(rim_sample), total_samples - idx_r)
                    drum_bus[idx_r:idx_r + r_len] += rim_sample[:r_len] * 0.90
                # Hats en contratiempo (offbeats 1.0, 3.0)
                for h_pos in [1.0, 3.0]:
                    idx_h = int((bar_start + h_pos * beat_dur) * SR)
                    if idx_h < total_samples:
                        h_len = min(len(hat_sample), total_samples - idx_h)
                        drum_bus[idx_h:idx_h + h_len] += hat_sample[:h_len] * 0.18

            elif rhythm_style == 2:  # Classic Boom-bap con swing suave
                for b_pos in [0, 1, 2, 3]:
                    swing = 0.020 if (b_pos % 2 == 1) else 0.0
                    idx_b = int((bar_start + b_pos * beat_dur + swing) * SR)
                    # Hats en cada negra y corchea
                    if idx_b < total_samples:
                        h_len = min(len(hat_sample), total_samples - idx_b)
                        drum_bus[idx_b:idx_b + h_len] += hat_sample[:h_len] * 0.20
                    # Bombo en 0 y 2.5
                    if b_pos == 0 and idx_b < total_samples:
                        k_len = min(len(kick_sample), total_samples - idx_b)
                        drum_bus[idx_b:idx_b + k_len] += kick_sample[:k_len]
                    elif b_pos == 2:
                        idx_k2 = int((bar_start + 2.5 * beat_dur + 0.012) * SR)
                        if idx_k2 < total_samples:
                            k_len = min(len(kick_sample), total_samples - idx_k2)
                            drum_bus[idx_k2:idx_k2 + k_len] += kick_sample[:k_len] * 0.78
                    # Rim en 1 y 3
                    if b_pos in (1, 3) and idx_b < total_samples:
                        r_len = min(len(rim_sample), total_samples - idx_b)
                        drum_bus[idx_b:idx_b + r_len] += rim_sample[:r_len]

            elif rhythm_style == 3:  # Dilla Drunken MPC swing
                human_lag = random.uniform(0.018, 0.034)
                # Bombo arrastrado en 0 y 1.75
                idx_k1 = max(0, int(bar_start * SR))
                if 0 <= idx_k1 < total_samples:
                    k_len = min(len(kick_sample), total_samples - idx_k1)
                    drum_bus[idx_k1:idx_k1 + k_len] += kick_sample[:k_len] * 0.95
                idx_k2 = max(0, int((bar_start + 1.75 * beat_dur + human_lag) * SR))
                if 0 <= idx_k2 < total_samples:
                    k_len = min(len(kick_sample), total_samples - idx_k2)
                    drum_bus[idx_k2:idx_k2 + k_len] += kick_sample[:k_len] * 0.70
                # Rimshot rezagado intencional en 1 y 3 (+lag)
                for r_pos in [1.0, 3.0]:
                    idx_r = max(0, int((bar_start + r_pos * beat_dur + human_lag) * SR))
                    if 0 <= idx_r < total_samples:
                        r_len = min(len(rim_sample), total_samples - idx_r)
                        drum_bus[idx_r:idx_r + r_len] += rim_sample[:r_len] * 0.92
                # Ghost rim tap en 2.75
                idx_ghost = max(0, int((bar_start + 2.75 * beat_dur) * SR))
                if 0 <= idx_ghost < total_samples:
                    r_len = min(len(rim_sample), total_samples - idx_ghost)
                    drum_bus[idx_ghost:idx_ghost + r_len] += rim_sample[:r_len] * 0.28
                # Shaker o hats con micro-variación
                for step in range(8):
                    pos_h = max(0.0, bar_start + (step * 0.5) * beat_dur + random.uniform(-0.008, 0.015))
                    idx_h = max(0, int(pos_h * SR))
                    if 0 <= idx_h < total_samples:
                        h_len = min(len(hat_sample), total_samples - idx_h)
                        drum_bus[idx_h:idx_h + h_len] += hat_sample[:h_len] * random.uniform(0.12, 0.22)

            elif rhythm_style == 4:  # Bossa Nova cross-stick
                # Kicks continuos suaves en 0 y 2
                for k_pos in [0.0, 2.0]:
                    idx_k = int((bar_start + k_pos * beat_dur) * SR)
                    if idx_k < total_samples:
                        k_len = min(len(kick_sample), total_samples - idx_k)
                        drum_bus[idx_k:idx_k + k_len] += kick_sample[:k_len] * 0.75
                # Sincopación clásica de cross-stick [0.0, 1.5, 2.75, 3.5]
                for r_pos in [0.0, 1.5, 2.75, 3.5]:
                    idx_r = int((bar_start + r_pos * beat_dur) * SR)
                    if idx_r < total_samples:
                        r_len = min(len(rim_sample), total_samples - idx_r)
                        drum_bus[idx_r:idx_r + r_len] += rim_sample[:r_len] * 0.80
                # Shakers o hats en corcheas constantes
                for step in range(8):
                    idx_h = int((bar_start + (step * 0.5) * beat_dur) * SR)
                    if idx_h < total_samples:
                        h_len = min(len(hat_sample), total_samples - idx_h)
                        drum_bus[idx_h:idx_h + h_len] += hat_sample[:h_len] * 0.16

            elif rhythm_style == 5:  # Laidback Shaker & Ride con platillo abierto
                # Kick suave solo en beat 0
                idx_k = int(bar_start * SR)
                if idx_k < total_samples:
                    k_len = min(len(kick_sample), total_samples - idx_k)
                    drum_bus[idx_k:idx_k + k_len] += kick_sample[:k_len] * 0.80
                # Rim en 2.0
                idx_r = int((bar_start + 2.0 * beat_dur) * SR)
                if idx_r < total_samples:
                    r_len = min(len(rim_sample), total_samples - idx_r)
                    drum_bus[idx_r:idx_r + r_len] += rim_sample[:r_len] * 0.85
                # Open hat en 3.5 como suspiro rítmico
                idx_open = int((bar_start + 3.5 * beat_dur) * SR)
                if idx_open < total_samples:
                    o_len = min(len(open_hat_sample), total_samples - idx_open)
                    drum_bus[idx_open:idx_open + o_len] += open_hat_sample[:o_len] * 0.20
                # Shakers orgánicos continuos
                for step in range(8):
                    idx_h = int((bar_start + (step * 0.5) * beat_dur) * SR)
                    if idx_h < total_samples:
                        h_len = min(len(hat_sample), total_samples - idx_h)
                        drum_bus[idx_h:idx_h + h_len] += hat_sample[:h_len] * 0.16
        else:
            # En el breakdown solo platillos o shakers muy tenues
            for step in range(4):
                idx_h = int((bar_start + step * beat_dur) * SR)
                if idx_h < total_samples:
                    h_len = min(len(hat_sample), total_samples - idx_h)
                    drum_bus[idx_h:idx_h + h_len] += hat_sample[:h_len] * 0.08

    # ─── MASTER STEREO BUS & PROCESAMIENTO ANALÓGICO ───
    stereo = np.zeros((2, total_samples))
    mix_l = (rhodes_bus * 0.68 + bass_bus * 0.74 + drum_bus * 0.60 + 
             foley_bus * 0.30 + melody_bus * 0.48 + vinyl_bus * 0.32)
    mix_r = (rhodes_bus * 0.68 + bass_bus * 0.74 + drum_bus * 0.60 + 
             foley_bus * 0.30 + melody_bus * 0.44 + vinyl_bus * 0.32)
    stereo[0] = mix_l
    stereo[1] = mix_r

    if HAVE_PEDALBOARD:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=30.0),
            LowpassFilter(cutoff_frequency_hz=3000.0 if dark else 3400.0),
            Chorus(rate_hz=0.65, depth=0.20, mix=0.15),
            Reverb(room_size=0.85 if dark else 0.70, damping=0.45, wet_level=0.28, dry_level=0.88),
            Compressor(threshold_db=-14.0, ratio=2.6),
            Gain(gain_db=1.2)
        ])
        processed = board(stereo, SR)
    else:
        # Fallback SciPy si no hay pedalboard
        b, a = sps.butter(2, [32 / (SR / 2), 3200 / (SR / 2)], btype='band')
        processed = np.zeros_like(stereo)
        processed[0] = sps.lfilter(b, a, stereo[0])
        processed[1] = sps.lfilter(b, a, stereo[1])

    # Wow & Flutter de cinta analógica
    fluttered = apply_tape_flutter(processed)

    # Normalización de pico a -1.2 dBFS
    max_val = np.max(np.abs(fluttered))
    if max_val > 0.001:
        fluttered = (fluttered / max_val) * (10.0 ** (-1.2 / 20.0))

    return fluttered


def read_active_scene_and_mode():
    scene = "veo3_manana"
    try:
        if os.path.exists(SCENE_FILE):
            with open(SCENE_FILE, "r") as f:
                scene = f.read().strip()
    except Exception:
        pass
    if any(x in scene for x in ("veo3", "cafe", "latam", "manana", "tarde", "noche")):
        mode = "cafe"
    else:
        mode = "outpost"
    return scene, mode


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    buffer_files = []

    sys.stderr.write("[INFO] audio_engine v4 iniciado: Multi-Groove Lo-Fi Dinámico con 16 Paletas Armónicas y Tape Wow/Flutter\\n")
    sys.stderr.flush()

    track_idx = 0
    pcm_queue = bytearray()
    prev_tail = None
    n_fade = int(2.0 * SR)
    t_fade = np.linspace(0, np.pi / 2.0, n_fade)
    fade_out = np.cos(t_fade)
    fade_in  = np.sin(t_fade)

    while True:
        scene_name, mode = read_active_scene_and_mode()
        track_idx += 1
        filename = AUDIO_DIR / f"lofi_track_{track_idx:04d}_{mode}.wav"

        audio = compose_track(scene_name, mode, num_bars=16)

        # Equal-power crossfade continuo entre bloques de pistas
        if prev_tail is None:
            # Primera pista: salida directa salvo la cola que se reserva
            to_output = audio[:, :-n_fade]
            prev_tail = audio[:, -n_fade:]
        else:
            # Pistas consecutivas: mezcla suave de la cola anterior con la nueva cabeza
            head = audio[:, :n_fade]
            blended = prev_tail * fade_out + head * fade_in
            body = audio[:, n_fade:-n_fade]
            to_output = np.hstack([blended, body])
            prev_tail = audio[:, -n_fade:]

        audio_int16 = (to_output.T * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(str(filename), SR, audio_int16)
        buffer_files.append(str(filename))

        while len(buffer_files) > 2:
            removed = buffer_files.pop(0)
            try:
                os.remove(removed)
            except Exception:
                pass

        track_bytes = audio_int16.tobytes()
        pcm_queue.extend(track_bytes)

        chunk_bytes_len = CHUNK_SAMPLES * 2 * 2
        while len(pcm_queue) >= chunk_bytes_len:
            chunk = pcm_queue[:chunk_bytes_len]
            pcm_queue = pcm_queue[chunk_bytes_len:]
            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            except BrokenPipeError:
                return


if __name__ == "__main__":
    main()
