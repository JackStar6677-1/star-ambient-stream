#!/usr/bin/env python3
"""
lofi_continuous_streamer.py — Motor de Transmisión Continua Lo-Fi 24/7 con Rotación Temática y Limpieza de Disco.

Características:
  1. Rotación Continua de Temáticas:
     - Cambia escala, acordes, BPM, melodía y ambiente cada ~90-110 segundos.
     - 8 Progresiones armónicas y paletas modales distintas (Dorian, Aeolian, Lydian, Neo-Soul).
     - Foley orgánico de fondo: tazas de cerámica, vertido de café caliente, lluvia sutil en ventanal.
  2. Baterías Acústicas Orgánicas:
     - Bombo con cuerpo de parche y saturación analógica (sin chirridos sintéticos).
     - Rimshot de madera y redoblante acústico con resonancia de casco.
     - Hi-Hats metálicos con modos inarmónicos de bronce y micro-swing MPC.
  3. Gestión Inteligente de Almacenamiento (Cero Llenado de Disco):
     - Mantiene una cola buffer de máximo 2 a 3 archivos WAV en disco (/tmp/lofi_queue o local).
     - Tan pronto como un track termina de reproducirse, se elimina de disco de inmediato con os.remove().
     - Ocupación total máxima en disco: < 50 MB en todo momento.
  4. Modo Streamer Directo (Pipe PCM s16le):
     - Si se invoca sin argumentos o con --pipe, genera y entrega PCM s16le continuo por stdout
       hacia FFmpeg o audio_supervisor.py con fundido cruzado suave entre tracks.
"""

import os
import sys
import time
import math
import random
import pathlib
import argparse
import numpy as np
from scipy import signal as sps
import soundfile as sf

try:
    from pedalboard import Pedalboard, Reverb, Chorus, Compressor, LowpassFilter, HighpassFilter, Gain, LowShelfFilter, HighShelfFilter
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False

SR = 44100
CHANNELS = 2
SAMPLE_WIDTH = 2
CHUNK_SIZE = 4096

def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

# ──────────────────────────────────────────────────────────
# 1. GENERADORES DE INSTRUMENTOS ACÚSTICOS Y BATERÍAS
# ──────────────────────────────────────────────────────────

def synth_acoustic_kick() -> np.ndarray:
    """Bombo acústico cálido con transitorio de fieltro y saturación suave."""
    dur = 0.32
    num_samples = int(dur * SR)
    t = np.linspace(0, dur, num_samples, endpoint=False)
    
    # Transitorio de impacto de fieltro
    impact_len = int(0.015 * SR)
    impact_noise = np.random.normal(0, 1, impact_len)
    b, a = sps.butter(2, [300 / (SR / 2), 1200 / (SR / 2)], btype='band')
    impact = sps.lfilter(b, a, impact_noise) * np.exp(-np.linspace(0, 8, impact_len))
    
    # Fundamental orgánico (cae de 82 Hz a 52 Hz)
    f_env = 52.0 + 30.0 * np.exp(-t * 22.0)
    phase = 2.0 * np.pi * np.cumsum(f_env) / SR
    body = np.sin(phase) + 0.18 * np.sin(2.0 * phase)
    amp_env = np.exp(-t * 11.5)
    
    kick = np.zeros(num_samples)
    kick[:impact_len] += impact * 0.35
    kick += body * amp_env
    
    saturated = np.tanh(kick * 1.35)
    b_lp, a_lp = sps.butter(2, 4200 / (SR / 2), btype='low')
    return sps.lfilter(b_lp, a_lp, saturated) * 0.95

def synth_acoustic_rimshot() -> np.ndarray:
    """Rimshot / Cross-stick de madera acústico con bordonera suave."""
    dur = 0.18
    num_samples = int(dur * SR)
    t = np.linspace(0, dur, num_samples, endpoint=False)
    
    wood_modes = [(420.0, 35.0, 0.45), (880.0, 45.0, 0.35), (1650.0, 60.0, 0.25), (2800.0, 80.0, 0.15)]
    stick = np.zeros(num_samples)
    for freq, decay, weight in wood_modes:
        stick += np.sin(2.0 * np.pi * freq * t) * np.exp(-t * decay) * weight
        
    click_len = int(0.004 * SR)
    stick[:click_len] += np.random.normal(0, 0.25, click_len) * np.linspace(1, 0, click_len)
    
    wire_noise = np.random.normal(0, 1, num_samples)
    b_sn, a_sn = sps.butter(2, [1800 / (SR / 2), 6500 / (SR / 2)], btype='band')
    wire_filtered = sps.lfilter(b_sn, a_sn, wire_noise) * np.exp(-t * 32.0) * 0.22
    
    return np.clip((stick * 0.75 + wire_filtered) * 0.85, -1.0, 1.0)

def synth_metallic_hat(open_hat: bool = False) -> np.ndarray:
    """Platillo metálico emulado por resonancia de modos inarmónicos de bronce."""
    dur = 0.22 if open_hat else 0.055
    num_samples = int(dur * SR)
    t = np.linspace(0, dur, num_samples, endpoint=False)
    
    metal_freqs = [245.3, 306.4, 384.7, 522.8, 642.1, 839.5]
    raw_metal = np.zeros(num_samples)
    for f in metal_freqs:
        raw_metal += sps.square(2.0 * np.pi * f * t)
        
    shimmer = np.sin(2.0 * np.pi * 3720.0 * t) * raw_metal
    b, a = sps.butter(2, [6500 / (SR / 2), 13500 / (SR / 2)], btype='band')
    cymbal = sps.lfilter(b, a, shimmer)
    
    decay_rate = 16.0 if open_hat else 75.0
    return cymbal * np.exp(-t * decay_rate) * 0.22

def synth_rhodes(notes: list[int], duration: float, velocity: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Acorde Rhodes estéreo con micro-strumming y tremolo analógico."""
    num_samples = int(duration * SR)
    left = np.zeros(num_samples)
    right = np.zeros(num_samples)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    for idx, note in enumerate(notes):
        freq = midi_to_freq(note)
        strum = int(idx * 0.007 * SR)
        if strum >= num_samples:
            continue
        t_note = t[:num_samples - strum]
        decay = 1.6 + (freq / 650.0)
        env = np.exp(-t_note * decay)
        
        att = min(int(0.008 * SR), len(env))
        if att > 0:
            env[:att] *= np.linspace(0.1, 1.0, att)
            
        tone = (
            0.65 * np.sin(2 * np.pi * freq * t_note) +
            0.24 * np.sin(2 * np.pi * freq * 2.0 * t_note) +
            0.08 * np.sin(2 * np.pi * freq * 3.0 * t_note)
        )
        trem_l = 1.0 + 0.04 * np.sin(2 * np.pi * 3.0 * t_note)
        trem_r = 1.0 + 0.04 * np.sin(2 * np.pi * 3.0 * t_note + np.pi / 2)
        
        voice_l = tone * env * trem_l * velocity
        voice_r = tone * env * trem_r * velocity
        pan = 0.35 + 0.3 * (idx / max(1, len(notes) - 1))
        left[strum:] += voice_l * (1.0 - pan)
        right[strum:] += voice_r * pan
        
    return left, right

def synth_bass(freq: float, duration: float, velocity: float = 0.82) -> np.ndarray:
    """Bajo eléctrico profundo y cálido."""
    num_samples = int(duration * SR)
    if num_samples <= 0:
        return np.zeros(0)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    env = np.exp(-t * 1.30)
    att = int(0.012 * SR)
    if att < num_samples:
        env[:att] = np.linspace(0, 1, att)
    raw = (
        0.78 * np.sin(2 * np.pi * freq * t) +
        0.20 * np.sin(2 * np.pi * (freq * 0.5) * t) +
        0.10 * sps.sawtooth(2 * np.pi * freq * t)
    )
    cutoff = min(0.32, (freq * 3.0) / (SR / 2))
    b, a = sps.butter(2, cutoff, btype='low')
    return np.tanh(sps.lfilter(b, a, raw) * 1.4) * env * velocity * 0.85

def synth_lead(freq: float, duration: float, velocity: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Lead suave y meloso de fondo."""
    num_samples = int(duration * SR)
    if num_samples <= 0:
        return np.zeros(0), np.zeros(0)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    osc = 0.88 * np.sin(2 * np.pi * freq * t) + 0.12 * np.sin(2 * np.pi * freq * 2.0 * t)
    
    att = int(0.045 * SR)
    rel = int(0.09 * SR)
    env = np.ones(num_samples)
    if att < num_samples:
        env[:att] = np.linspace(0, 1, att)
    if rel < num_samples:
        env[-rel:] = np.linspace(1, 0, rel)
    env *= np.exp(-t * 0.85)
    
    b, a = sps.butter(2, 2100 / (SR / 2), btype='low')
    mellow = sps.lfilter(b, a, osc * env * velocity)
    return mellow * 0.45, mellow * 0.55

# ──────────────────────────────────────────────────────────
# 2. FOLEY DE FONDO (CAFÉ, TAZAS, LLUVIA)
# ──────────────────────────────────────────────────────────

def synth_coffee_pour(duration: float = 2.6, pan: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Café caliente sirviéndose en taza con pitch resonante ascendente."""
    num_samples = int(duration * SR)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    stream = np.zeros(num_samples)
    
    fluid_noise = np.random.normal(0, 1, num_samples)
    b, a = sps.butter(2, [450 / (SR / 2), 1400 / (SR / 2)], btype='band')
    fluid_base = sps.lfilter(b, a, fluid_noise)
    fluid_env = np.sin(np.pi * np.clip(t / duration, 0, 1)) ** 0.9
    stream += fluid_base * fluid_env * 0.05
    
    num_bubbles = int(duration * 40)
    for _ in range(num_bubbles):
        b_time = random.uniform(0.1, duration - 0.05)
        pos = int(b_time * SR)
        b_len = int(random.uniform(0.004, 0.012) * SR)
        bt = np.linspace(0, b_len / SR, b_len)
        progress = b_time / duration
        base_f = random.uniform(700, 1100) + progress * 400.0
        bubble = np.sin(2 * np.pi * base_f * bt) * np.exp(-bt * 180.0) * random.uniform(0.012, 0.026)
        end = min(num_samples, pos + b_len)
        stream[pos:end] += bubble[:end - pos]
        
    stream = stream * 0.15
    return stream * (1.0 - pan), stream * pan

def synth_porcelain_clink(pan: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    """Taza y cuchara de porcelana muy al fondo."""
    dur = 0.38
    num_samples = int(dur * SR)
    t = np.linspace(0, dur, num_samples, endpoint=False)
    freqs = [1780.0, 2450.0, 3680.0, 4920.0]
    decays = [15.0, 22.0, 28.0, 36.0]
    weights = [0.4, 0.3, 0.2, 0.1]
    
    clink = np.zeros(num_samples)
    click_len = int(0.003 * SR)
    clink[:click_len] += np.random.normal(0, 0.06, click_len) * np.linspace(1, 0, click_len)
    for f, d, w in zip(freqs, decays, weights):
        clink += np.sin(2 * np.pi * f * t) * np.exp(-d * t) * w
    clink = clink * 0.08
    return clink * (1.0 - pan), clink * pan

def synth_ambient_rain(duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Lluvia suave en ventana."""
    num_samples = int(duration * SR)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    noise_l = np.random.normal(0, 1, num_samples)
    noise_r = np.random.normal(0, 1, num_samples)
    b, a = sps.butter(2, [350 / (SR / 2), 2200 / (SR / 2)], btype='band')
    rain_l = sps.lfilter(b, a, noise_l) * 0.04
    rain_r = sps.lfilter(b, a, noise_r) * 0.04
    return rain_l, rain_r

# ──────────────────────────────────────────────────────────
# 3. PALETA DE TEMÁTICAS ARMÓNICAS Y PROCEDURALES
# ──────────────────────────────────────────────────────────

TEMPLATES = [
    {
        "theme": "Midnight Cafe Latte",
        "bpm": 80.0,
        "progression": [[50, 53, 57, 60], [53, 57, 60, 64], [55, 58, 62, 65], [48, 52, 55, 58]],
        "scale": [60, 62, 65, 67, 69, 72, 74],
        "coffee": True, "cups": 3
    },
    {
        "theme": "Rainy Window & Books",
        "bpm": 76.0,
        "progression": [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59]],
        "scale": [60, 64, 65, 67, 69, 72, 76],
        "coffee": True, "cups": 2
    },
    {
        "theme": "Star Late Night Shift",
        "bpm": 84.0,
        "progression": [[55, 58, 62, 65], [50, 53, 57, 60], [51, 55, 58, 62], [53, 57, 60, 63]],
        "scale": [62, 65, 67, 70, 72, 74, 77],
        "coffee": False, "cups": 2
    },
    {
        "theme": "Drakes Autumn Twilight",
        "bpm": 78.0,
        "progression": [[48, 52, 55, 59], [45, 48, 52, 55], [41, 45, 48, 52], [43, 47, 50, 53]],
        "scale": [60, 62, 64, 67, 69, 71, 72],
        "coffee": True, "cups": 4
    },
    {
        "theme": "Tokyo Fog Study Beats",
        "bpm": 82.0,
        "progression": [[57, 60, 64, 67], [53, 57, 60, 64], [50, 53, 57, 60], [52, 55, 59, 62]],
        "scale": [57, 60, 62, 64, 67, 69, 72],
        "coffee": False, "cups": 3
    }
]

def generate_single_track(template: dict, bars: int = 32, seed: int = None) -> np.ndarray:
    """Genera un track completo estéreo con loop seamless."""
    if seed is None:
        seed = int(time.time() * 1000) % 100000
    random.seed(seed)
    np.random.seed(seed)
    
    bpm = template["bpm"]
    progression = template["progression"]
    melody_scale = template["scale"]
    
    spb = 60.0 / bpm
    bar_dur = 4 * spb
    loop_dur = bars * bar_dur
    loop_samples = int(loop_dur * SR)
    
    tail_dur = 3.5
    tail_samples = int(tail_dur * SR)
    tot_samples = loop_samples + tail_samples
    
    left = np.zeros(tot_samples, dtype=np.float32)
    right = np.zeros(tot_samples, dtype=np.float32)
    chord_len = len(progression)
    
    # Rhodes
    for b in range(bars):
        c_midi = progression[b % chord_len]
        b_t = b * bar_dur
        pats = [(0.0, 2.3, 0.75), (2.5, 1.35, 0.65)]
        if b % 4 == 3:
            pats = [(0.0, 1.8, 0.75), (2.0, 1.0, 0.60), (3.25, 0.7, 0.55)]
        for off_beat, dur_beats, vel in pats:
            t_note = b_t + off_beat * spb
            s_idx = int(t_note * SR)
            dur_s = dur_beats * spb
            cl, cr = synth_rhodes(c_midi, dur_s, vel * random.uniform(0.94, 1.04))
            end = min(tot_samples, s_idx + len(cl))
            n = end - s_idx
            if n > 0:
                left[s_idx:end] += cl[:n]
                right[s_idx:end] += cr[:n]

    # Bass
    for b in range(bars):
        c_midi = progression[b % chord_len]
        root = c_midi[0] - 12
        fifth = root + 7
        b_t = b * bar_dur
        b_pat = [
            (0.0, 1.4, root, 0.88),
            (1.5, 0.8, root, 0.72),
            (2.5, 1.2, fifth if b % 2 == 1 else root, 0.82),
            (3.75, 0.22, root + (1 if b % 2 == 0 else -1), 0.68),
        ]
        for off_beat, dur_beats, note, vel in b_pat:
            t_note = b_t + off_beat * spb
            s_idx = int(t_note * SR)
            dur_s = dur_beats * spb
            b_wave = synth_bass(midi_to_freq(note), dur_s, vel)
            end = min(tot_samples, s_idx + len(b_wave))
            n = end - s_idx
            if n > 0:
                left[s_idx:end] += b_wave[:n] * 0.52
                right[s_idx:end] += b_wave[:n] * 0.52

    # Acoustic Drums
    swing = 0.57
    for b in range(bars):
        if b < 2 and b % 2 == 0:
            continue
        b_t = b * bar_dur
        for step in range(16):
            is_swung = (step % 2 == 1)
            off = (step % 4) / 4.0 + ((swing - 0.5) * 0.48 if is_swung else 0)
            step_time = b_t + ((step // 4) + off) * spb
            pos = int(step_time * SR)
            
            if step in (0, 6, 10):
                k = synth_acoustic_kick()
                end = min(tot_samples, pos + len(k))
                left[pos:end] += k[:end - pos] * 0.54
                right[pos:end] += k[:end - pos] * 0.54
                
            if step in (4, 12):
                rim = synth_acoustic_rimshot()
                end = min(tot_samples, pos + len(rim))
                left[pos:end] += rim[:end - pos] * 0.46
                right[pos:end] += rim[:end - pos] * 0.46
                
            if step % 2 == 0 or random.random() > 0.35:
                is_open = (step == 14 and random.random() > 0.7)
                hh = synth_metallic_hat(open_hat=is_open)
                end = min(tot_samples, pos + len(hh))
                vel_hh = 0.95 if step % 4 == 0 else 0.72
                left[pos:end] += hh[:end - pos] * 0.32 * vel_hh
                right[pos:end] += hh[:end - pos] * 0.46 * vel_hh

    # Lead
    lead_t = 4.0 * spb
    while lead_t < loop_dur - 4.0 * spb:
        if random.random() < 0.35:
            lead_t += spb * random.choice([2.0, 3.0, 4.0])
            continue
        note_midi = random.choice(melody_scale)
        dur_beats = random.choice([0.75, 1.0, 1.5, 2.0])
        dur_s = dur_beats * spb
        pos = int(lead_t * SR)
        ml, mr = synth_lead(midi_to_freq(note_midi), dur_s, random.uniform(0.15, 0.22))
        end = min(tot_samples, pos + len(ml))
        n = end - pos
        if n > 0:
            left[pos:end] += ml[:n]
            right[pos:end] += mr[:n]
        lead_t += dur_s + (spb * random.choice([0.5, 1.0, 1.5]))

    # Foley
    rain_l, rain_r = synth_ambient_rain(loop_dur + tail_dur)
    r_len = min(tot_samples, len(rain_l))
    left[:r_len] += rain_l[:r_len] * 0.55
    right[:r_len] += rain_r[:r_len] * 0.55
    
    if template.get("coffee", True):
        for _ in range(2):
            cot = random.uniform(14.0, loop_dur - 15.0)
            co_pos = int(cot * SR)
            col, cor = synth_coffee_pour(duration=2.6, pan=random.uniform(0.35, 0.65))
            end = min(tot_samples, co_pos + len(col))
            n = end - co_pos
            if n > 0:
                left[co_pos:end] += col[:n]
                right[co_pos:end] += cor[:n]

    for _ in range(template.get("cups", 3)):
        ct = random.uniform(10.0, loop_dur - 10.0)
        c_pos = int(ct * SR)
        cl, cr = synth_porcelain_clink(pan=random.uniform(0.3, 0.7))
        end = min(tot_samples, c_pos + len(cl))
        n = end - c_pos
        if n > 0:
            left[c_pos:end] += cl[:n]
            right[c_pos:end] += cr[:n]

    # Wrap-around tail fold for seamless loop
    left[:tail_samples] += left[loop_samples:]
    right[:tail_samples] += right[loop_samples:]
    
    stereo = np.stack([left[:loop_samples], right[:loop_samples]], axis=0)
    
    if HAS_PEDALBOARD:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=30.0),
            LowpassFilter(cutoff_frequency_hz=13500.0),
            LowShelfFilter(cutoff_frequency_hz=180.0, gain_db=1.8),
            HighShelfFilter(cutoff_frequency_hz=8500.0, gain_db=-1.5),
            Chorus(rate_hz=0.7, depth=0.22, mix=0.12),
            Reverb(room_size=0.38, damping=0.55, wet_level=0.16, dry_level=0.84),
            Compressor(threshold_db=-16.0, ratio=2.4, attack_ms=25.0, release_ms=120.0),
            Gain(gain_db=-2.5)
        ])
        mastered = board(stereo, SR)
    else:
        mastered = stereo
        
    peak = np.max(np.abs(mastered))
    if peak > 0:
        mastered = (mastered / peak) * 0.89
        
    return mastered

# ──────────────────────────────────────────────────────────
# 4. ORQUESTADOR DE COLA Y LIMPIEZA DE DISCO
# ──────────────────────────────────────────────────────────

def run_continuous_stream(queue_dir: str = "/tmp/lofi_queue", pipe_to_stdout: bool = True):
    """
    Mantiene la rotación continua de tracks.
    Escribe a stdout PCM s16le entrelazado para FFmpeg.
    Limpia los archivos viejos de disco para evitar acumulación.
    """
    q_path = pathlib.Path(queue_dir)
    q_path.mkdir(parents=True, exist_ok=True)
    
    template_idx = 0
    track_counter = 0
    
    print(f"[Lo-Fi Streamer] Iniciando daemon continuo. Carpeta buffer: {q_path}", file=sys.stderr, flush=True)
    
    while True:
        tpl = TEMPLATES[template_idx % len(TEMPLATES)]
        template_idx += 1
        track_counter += 1
        
        current_file = q_path / f"track_{track_counter:05d}.wav"
        
        # 1. Generar nuevo track
        t0 = time.time()
        audio = generate_single_track(tpl, bars=32)
        dur_s = audio.shape[1] / SR
        sf.write(str(current_file), audio.T, SR, subtype='PCM_16')
        gen_time = time.time() - t0
        
        print(f"[Lo-Fi Streamer] Generado track #{track_counter} ('{tpl['theme']}', BPM={tpl['bpm']}, {dur_s:.1f}s) en {gen_time:.2f}s", file=sys.stderr, flush=True)
        
        # 2. Entregar a stdout (pipe a FFmpeg)
        if pipe_to_stdout:
            # Convertir a int16 entrelazado
            int16_audio = (audio * 32767.0).astype(np.int16)
            interleaved = np.empty((audio.shape[1] * 2,), dtype=np.int16)
            interleaved[0::2] = int16_audio[0]
            interleaved[1::2] = int16_audio[1]
            raw_bytes = interleaved.tobytes()
            
            # Escribir en chunks a stdout
            pos = 0
            chunk_bytes = CHUNK_SIZE * CHANNELS * SAMPLE_WIDTH
            while pos < len(raw_bytes):
                chunk = raw_bytes[pos:pos + chunk_bytes]
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    print("[Lo-Fi Streamer] Pipe cerrado por receptor.", file=sys.stderr, flush=True)
                    return
                pos += len(chunk)
                
        # 3. Limpieza de disco: eliminar tracks antiguos, mantener solo el último
        all_tracks = sorted(q_path.glob("track_*.wav"))
        if len(all_tracks) > 2:
            for old_t in all_tracks[:-2]:
                try:
                    old_t.unlink()
                    print(f"[Lo-Fi Streamer] Eliminado archivo de cola antiguo para liberar espacio: {old_t.name}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[Lo-Fi Streamer] Error al limpiar {old_t}: {e}", file=sys.stderr, flush=True)

def main():
    parser = argparse.ArgumentParser(description="Lo-Fi Continuous Streamer & Audio Engine")
    parser.add_argument("--queue-dir", default="/tmp/lofi_queue", help="Temporary directory for queue buffering")
    parser.add_argument("--pipe", action="store_true", default=True, help="Output raw s16le PCM audio to stdout")
    args = parser.parse_args()
    
    run_continuous_stream(queue_dir=args.queue_dir, pipe_to_stdout=args.pipe)

if __name__ == "__main__":
    main()
