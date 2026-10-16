#!/usr/bin/env python3
"""
hybrid_streamer.py — Motor Autónomo de Streaming Continuo con Doble Modo (Cafe & Outpost).

Capacidades:
  1. Lee /tmp/star_scene.txt para determinar el modo:
     - Si la escena pertenece a Cafe ('veo3_manana', 'veo3_tarde', 'veo3_noche'):
       Genera pistas Cozy Cafe Lo-Fi (acordes maj7/9, foley de cafetería, tazas, lluvia).
     - Si la escena pertenece a Outpost ('orbital', 'nieve', 'scp_contencion', etc.):
       Genera pistas Dark Outpost Sci-Fi Lo-Fi (acordes menores etéreos, RCS thrusters, telemetría, racks de servidores).
  2. Baterías acústicas orgánicas en ambos modos (mazo de fieltro, rimshot de madera, hi-hats bronce B20 con micro-swing MPC).
  3. Auto-limpieza en disco: purga archivos de audio anteriores para mantener almacenamiento < 40 MB.
  4. Salida en streaming continuo PCM s16le directo a stdout para alimentar FFmpeg.
"""

import os
import sys
import time
import math
import random
import pathlib
import subprocess
import numpy as np
from scipy import signal as sps
import soundfile as sf
from pedalboard import (
    Pedalboard, Reverb, Chorus, Compressor, LowpassFilter, HighpassFilter, Gain, LowShelfFilter, HighShelfFilter
)

SR = 44100
SCENE_FILE = "/tmp/star_scene.txt"
AUDIO_DIR = pathlib.Path("/tmp/star_lofi_buffer")

def midi_to_freq(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))

# ──────────────────────────────────────────────────────────
# 1. BATERÍAS ACÚSTICAS REALISTAS COMPARTIDAS
# ──────────────────────────────────────────────────────────

def synth_lofi_kick(dark: bool = False) -> np.ndarray:
    dur = 0.35 if dark else 0.30
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    
    imp_len = int(0.012 * SR)
    noise = np.random.normal(0, 1, imp_len)
    b, a = sps.butter(2, [200 / (SR / 2), 1000 / (SR / 2)], btype='band')
    impact = sps.lfilter(b, a, noise) * np.exp(-np.linspace(0, 10, imp_len))
    
    base_f = 48.0 if dark else 54.0
    sweep_f = 24.0 if dark else 30.0
    f_env = base_f + sweep_f * np.exp(-t * 22.0)
    phase = 2.0 * np.pi * np.cumsum(f_env) / SR
    body = np.sin(phase) + 0.15 * np.sin(2.0 * phase)
    amp_env = np.exp(-t * (10.0 if dark else 12.0))
    
    kick = np.zeros(n)
    kick[:imp_len] += impact * 0.30
    kick += body * amp_env
    
    sat = np.tanh(kick * 1.35)
    lp_cutoff = 3000 if dark else 4200
    b_lp, a_lp = sps.butter(2, lp_cutoff / (SR / 2), btype='low')
    return sps.lfilter(b_lp, a_lp, sat) * 0.94

def synth_lofi_rimshot() -> np.ndarray:
    dur = 0.18
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    
    stick = np.random.normal(0, 1, int(0.005 * SR)) * np.exp(-np.linspace(0, 8, int(0.005 * SR)))
    modes = [(420.0, 0.40, 30.0), (880.0, 0.35, 45.0), (1650.0, 0.25, 65.0), (2800.0, 0.15, 85.0)]
    wood = np.zeros(n)
    for freq, amp, decay in modes:
        wood += amp * np.sin(2.0 * np.pi * freq * t) * np.exp(-t * decay)
        
    wire_len = int(0.12 * SR)
    wire_noise = np.random.normal(0, 1, wire_len)
    b, a = sps.butter(2, [1200 / (SR / 2), 6500 / (SR / 2)], btype='band')
    wire = sps.lfilter(b, a, wire_noise) * np.exp(-np.linspace(0, 24, wire_len)) * 0.18
    
    out = np.zeros(n)
    out[:len(stick)] += stick * 0.4
    out += wood
    out[:wire_len] += wire
    return out * 0.88

def synth_lofi_hat(closed: bool = True) -> np.ndarray:
    dur = 0.055 if closed else 0.22
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    
    modes = [285.3, 397.6, 521.8, 642.1, 789.4, 912.7]
    cymb = np.zeros(n)
    for m in modes:
        f = m * 5.8
        cymb += np.sin(2.0 * np.pi * f * t) * (0.8 + 0.2 * np.sin(2.0 * np.pi * (m * 0.25) * t))
    
    noise = np.random.normal(0, 1, n)
    cymb = cymb * 0.5 + noise * 0.5
    b, a = sps.butter(2, [6000 / (SR / 2), 12000 / (SR / 2)], btype='band')
    filtered = sps.lfilter(b, a, cymb)
    decay = 65.0 if closed else 18.0
    return filtered * np.exp(-t * decay) * 0.45

# ──────────────────────────────────────────────────────────
# 2. FOLEY AMBIENTAL (CAFE vs OUTPOST)
# ──────────────────────────────────────────────────────────

def synth_cafe_foley(duration: float) -> np.ndarray:
    """Foley de cafetería: lluvia suave, tazas y café sirviéndose."""
    n = int(duration * SR)
    out = np.zeros(n)
    
    # 1. Lluvia suave
    rain = np.random.normal(0, 1, n)
    b, a = sps.butter(2, [400 / (SR / 2), 3200 / (SR / 2)], btype='band')
    rain_f = sps.lfilter(b, a, rain) * 0.018
    out += rain_f
    
    # 2. Tintineo sutil de vajilla
    num_clinks = int(duration / 12) + 1
    for _ in range(num_clinks):
        pos = random.randint(0, max(0, n - int(1.0 * SR)))
        dur_c = 0.8
        n_c = int(dur_c * SR)
        t_c = np.linspace(0, dur_c, n_c, endpoint=False)
        clink = (np.sin(2.0 * np.pi * 3200.0 * t_c) * 0.5 + 
                 np.sin(2.0 * np.pi * 5400.0 * t_c) * 0.3) * np.exp(-t_c * 28.0) * 0.008
        out[pos:pos + n_c] += clink
        
    return out

def synth_outpost_foley(duration: float) -> np.ndarray:
    """Foley de puesto espacial: soporte vital, propulsores RCS y telemetría."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n, endpoint=False)
    
    # Zumbido de servidores y flujo de aire
    hum = (np.sin(2.0 * np.pi * 50.0 * t) * 0.010 + np.sin(2.0 * np.pi * 100.0 * t) * 0.005)
    noise = np.random.normal(0, 1, n)
    b, a = sps.butter(2, [140 / (SR / 2), 480 / (SR / 2)], btype='band')
    air = sps.lfilter(b, a, noise) * 0.012
    lfo = 0.85 + 0.15 * np.sin(2.0 * np.pi * 0.08 * t)
    out = (hum + air) * lfo
    
    # Propulsor RCS sutil
    if duration > 15:
        rcs_pos = int(n * 0.4)
        rcs_dur = 3.5
        n_rcs = int(rcs_dur * SR)
        t_rcs = np.linspace(0, rcs_dur, n_rcs, endpoint=False)
        rcs_env = np.clip(t_rcs / 0.5, 0.0, 1.0) * np.clip((rcs_dur - t_rcs) / 0.8, 0.0, 1.0)
        rcs_noise = np.random.uniform(-1.0, 1.0, n_rcs)
        b_rcs, a_rcs = sps.butter(2, 85.0 / (SR / 2), btype='low')
        rcs = sps.lfilter(b_rcs, a_rcs, rcs_noise) * rcs_env * 0.04
        out[rcs_pos:rcs_pos + n_rcs] += rcs
        
    # Telemetría suave
    if duration > 10:
        tel_pos = int(n * 0.7)
        for i, f in enumerate([440.0, 554.0, 659.0]):
            sub_pos = tel_pos + int(i * 0.28 * SR)
            dur_b = 0.14
            n_b = int(dur_b * SR)
            t_b = np.linspace(0, dur_b, n_b, endpoint=False)
            beep = np.sin(2.0 * np.pi * f * t_b) * np.sin(np.pi * np.clip(t_b / dur_b, 0, 1)) * 0.006
            if sub_pos + n_b < n:
                out[sub_pos:sub_pos + n_b] += beep
                
    return out

# ──────────────────────────────────────────────────────────
# 3. COMPOSITOR POLIMÓRFICO
# ──────────────────────────────────────────────────────────

CAFE_THEMES = [
    {
        "bpm": 80,
        "chords": [[60, 64, 67, 71], [53, 57, 60, 64], [55, 58, 62, 65], [48, 52, 55, 59]], # Cmaj7, Fmaj7, Gm7, Cmaj7
        "bass": [48, 41, 43, 36],
        "melody": [71, 72, 71, 67, 64, 67, 69, 71],
        "dark": False,
    },
    {
        "bpm": 76,
        "chords": [[53, 57, 60, 64], [50, 53, 57, 60], [55, 58, 62, 65], [52, 55, 59, 62]], # Fmaj7, Dm7, Gm7, Em7
        "bass": [41, 38, 43, 40],
        "melody": [69, 67, 64, 62, 64, 67, 69, None],
        "dark": False,
    }
]

OUTPOST_THEMES = [
    {
        "bpm": 70,
        "chords": [[54, 61, 64, 68, 71], [50, 57, 61, 66, 68], [57, 61, 64, 68, 71], [49, 56, 61, 64, 68]], # F#m9, Dmaj7#11, Aadd9, C#m7
        "bass": [42, 38, 45, 37],
        "melody": [68, None, 71, 73, 71, 68, 64, 66],
        "dark": True,
    },
    {
        "bpm": 68,
        "chords": [[57, 60, 64, 67, 71], [53, 57, 60, 64, 69], [50, 53, 57, 60, 64], [52, 55, 59, 62, 67]], # Am9, Fmaj7, Dm9, Em7
        "bass": [45, 41, 38, 40],
        "melody": [71, 67, None, 64, 67, 69, 64, None],
        "dark": True,
    }
]

def compose_track(mode: str, num_bars: int = 16) -> np.ndarray:
    dark = (mode == "outpost")
    themes = OUTPOST_THEMES if dark else CAFE_THEMES
    theme = random.choice(themes)
    
    bpm = theme["bpm"]
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4.0
    total_dur = num_bars * bar_dur
    total_samples = int(total_dur * SR)
    
    rhodes_bus = np.zeros(total_samples)
    bass_bus = np.zeros(total_samples)
    drum_bus = np.zeros(total_samples)
    foley_bus = synth_outpost_foley(total_dur) if dark else synth_cafe_foley(total_dur)
    
    chords = theme["chords"]
    bass_notes = theme["bass"]
    kick = synth_lofi_kick(dark=dark)
    rim = synth_lofi_rimshot()
    hat = synth_lofi_hat(closed=True)
    
    for bar in range(num_bars):
        bar_start = bar * bar_dur
        chord_idx = bar % len(chords)
        current_chord = chords[chord_idx]
        current_bass = bass_notes[chord_idx]
        
        # Rhodes
        bar_idx = int(bar_start * SR)
        n_chord = int(bar_dur * 0.96 * SR)
        t_c = np.linspace(0, bar_dur * 0.96, n_chord, endpoint=False)
        chord_sig = np.zeros(n_chord)
        for note in current_chord:
            f = midi_to_freq(note)
            tone = np.sin(2.0 * np.pi * f * t_c) + 0.22 * np.sin(4.0 * np.pi * f * t_c)
            env = np.clip(t_c / 0.02, 0, 1) * np.exp(-t_c * 0.6)
            chord_sig += tone * env
        chord_sig /= len(current_chord)
        cutoff = 1400 if dark else 2600
        b_c, a_c = sps.butter(2, cutoff / (SR / 2), btype='low')
        chord_sig = sps.lfilter(b_c, a_c, chord_sig)
        c_len = min(len(chord_sig), total_samples - bar_idx)
        rhodes_bus[bar_idx:bar_idx + c_len] += chord_sig[:c_len] * 0.60
        
        # Bass
        n_bass = int(bar_dur * 0.90 * SR)
        t_b = np.linspace(0, bar_dur * 0.90, n_bass, endpoint=False)
        fb = midi_to_freq(current_bass)
        bass_sig = np.sin(2.0 * np.pi * fb * t_b) * np.exp(-t_b * 0.45)
        b_len = min(len(bass_sig), total_samples - bar_idx)
        bass_bus[bar_idx:bar_idx + b_len] += bass_sig[:b_len] * 0.70
        
        # Drums
        for beat in range(4):
            swing = 0.022 if (beat % 2 == 1) else 0.0
            idx_beat = int((bar_start + beat * beat_dur + swing) * SR)
            if idx_beat < total_samples:
                h_len = min(len(hat), total_samples - idx_beat)
                drum_bus[idx_beat:idx_beat + h_len] += hat[:h_len] * 0.60
            if beat == 0 and idx_beat < total_samples:
                k_len = min(len(kick), total_samples - idx_beat)
                drum_bus[idx_beat:idx_beat + k_len] += kick[:k_len]
            elif beat == 2:
                idx_sync = int((bar_start + 2.75 * beat_dur + 0.015) * SR)
                if idx_sync < total_samples:
                    k_len = min(len(kick), total_samples - idx_sync)
                    drum_bus[idx_sync:idx_sync + k_len] += kick[:k_len] * 0.75
            if beat in (1, 3) and idx_beat < total_samples:
                r_len = min(len(rim), total_samples - idx_beat)
                drum_bus[idx_beat:idx_beat + r_len] += rim[:r_len]
                
    # Master Bus
    stereo = np.zeros((2, total_samples))
    mix = rhodes_bus * 0.65 + bass_bus * 0.75 + drum_bus * 0.85 + foley_bus * 0.55
    stereo[0] = mix
    stereo[1] = mix
    
    board = Pedalboard([
        Chorus(rate_hz=0.8, depth=0.25, mix=0.20),
        Reverb(room_size=0.85 if dark else 0.75, damping=0.45, wet_level=0.30, dry_level=0.85),
        Compressor(threshold_db=-14.0, ratio=2.8),
        Gain(gain_db=1.0)
    ])
    processed = board(stereo, SR)
    max_val = np.max(np.abs(processed))
    if max_val > 0.001:
        processed = (processed / max_val) * (10.0 ** (-1.2 / 20.0))
        
    return processed

def read_active_mode() -> str:
    try:
        if os.path.exists(SCENE_FILE):
            with open(SCENE_FILE, "r") as f:
                scene = f.read().strip()
                if scene in ("veo3_manana", "veo3_tarde", "veo3_noche"):
                    return "cafe"
                return "outpost"
    except Exception:
        pass
    return "outpost"

def cleanup_old_files(keep_files: list):
    try:
        for f in AUDIO_DIR.glob("*.wav"):
            if str(f) not in keep_files:
                try:
                    f.unlink()
                except Exception:
                    pass
    except Exception:
        pass

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    buffer_files = []
    
    sys.stderr.write("[INFO] hybrid_streamer iniciado: Dual-Engine Lo-Fi (Cafe & Outpost)\n")
    sys.stderr.flush()
    
    track_idx = 0
    while True:
        mode = read_active_mode()
        track_idx += 1
        filename = AUDIO_DIR / f"hybrid_track_{track_idx:04d}_{mode}.wav"
        
        audio = compose_track(mode, num_bars=16)
        sf.write(str(filename), audio.T, SR)
        buffer_files.append(str(filename))
        
        # Mantener sólo 2 archivos en disco
        if len(buffer_files) > 2:
            removed = buffer_files.pop(0)
            try:
                os.remove(removed)
            except Exception:
                pass
                
        # Stream raw PCM s16le chunks to stdout
        pcm_bytes = (audio[0] * 32767).astype(np.int16).tobytes()
        try:
            sys.stdout.buffer.write(pcm_bytes)
            sys.stdout.buffer.flush()
        except BrokenPipeError:
            break

if __name__ == "__main__":
    main()
