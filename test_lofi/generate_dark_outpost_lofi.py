#!/usr/bin/env python3
"""
generate_dark_outpost_lofi.py — Generador Procedural: Dark Outpost Sci-Fi Lo-Fi (v1)

Combina:
  1. Armonías menores y acordes suspendidos de ciencia ficción (F#m9, Dmaj7#11, Am9, Cm9, Bm9).
  2. Baterías acústicas orgánicas Lo-Fi (mazo de fieltro, rimshot de madera, hi-hats bronce B20).
  3. Foley de puesto de avanzada remoto:
     - Propulsores RCS / descompresión de válvulas a la lejanía.
     - Telemetría y barridos sinusoidales de comunicaciones espaciales.
     - Zumbido y flujo de aire de racks de servidores / soporte vital (45-80 Hz).
     - Radioestática cósmica y scanner espacial sutil (-26 dB).
  4. Muted Rhodes / Dark Synth Pads con wow & flutter de cinta analógica.
"""

import os
import math
import random
import pathlib
import numpy as np
from scipy import signal as sps
import soundfile as sf
import matplotlib.pyplot as plt
from pedalboard import (
    Pedalboard, Reverb, Chorus, Compressor, LowpassFilter, HighpassFilter, Gain, LowShelfFilter, HighShelfFilter
)

SR = 44100

def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

# ──────────────────────────────────────────────────────────
# 1. BATERÍAS ACÚSTICAS LO-FI (SUBDUED & WOODEN)
# ──────────────────────────────────────────────────────────

def synth_dark_kick() -> np.ndarray:
    """Bombo profundo y apagado, estilo estación solitaria."""
    dur = 0.35
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    
    # Impacto inicial sordo
    imp_len = int(0.012 * SR)
    noise = np.random.normal(0, 1, imp_len)
    b, a = sps.butter(2, [180 / (SR / 2), 800 / (SR / 2)], btype='band')
    impact = sps.lfilter(b, a, noise) * np.exp(-np.linspace(0, 10, imp_len))
    
    # Tono fundamental descendente (72 Hz -> 48 Hz)
    f_env = 48.0 + 24.0 * np.exp(-t * 24.0)
    phase = 2.0 * np.pi * np.cumsum(f_env) / SR
    body = np.sin(phase) + 0.12 * np.sin(2.0 * phase)
    amp_env = np.exp(-t * 9.5)
    
    kick = np.zeros(n)
    kick[:imp_len] += impact * 0.28
    kick += body * amp_env
    
    # Saturación cálida
    saturated = np.tanh(kick * 1.4)
    b_lp, a_lp = sps.butter(2, 3200 / (SR / 2), btype='low')
    return sps.lfilter(b_lp, a_lp, saturated) * 0.92

def synth_dark_rimshot() -> np.ndarray:
    """Rimshot seco de madera con resonancia metálica contenida."""
    dur = 0.20
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    
    impact_len = int(0.005 * SR)
    stick = np.random.normal(0, 1, impact_len) * np.exp(-np.linspace(0, 8, impact_len))
    
    # Resonancias del aro
    modes = [
        (420.0, 0.40, 28.0),
        (880.0, 0.35, 42.0),
        (1650.0, 0.25, 60.0),
        (2800.0, 0.15, 80.0),
    ]
    wood = np.zeros(n)
    for freq, amp, decay in modes:
        wood += amp * np.sin(2.0 * np.pi * freq * t) * np.exp(-t * decay)
    
    snare_len = int(0.12 * SR)
    wire_noise = np.random.normal(0, 1, snare_len)
    b, a = sps.butter(2, [1200 / (SR / 2), 6500 / (SR / 2)], btype='band')
    wire = sps.lfilter(b, a, wire_noise) * np.exp(-np.linspace(0, 22, snare_len)) * 0.20
    
    out = np.zeros(n)
    out[:impact_len] += stick * 0.4
    out += wood
    out[:snare_len] += wire
    return out * 0.85

def synth_dark_hat(closed: bool = True) -> np.ndarray:
    """Hi-hat de aleación oscura con armónicos inarmónicos."""
    dur = 0.055 if closed else 0.22
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    
    modes = [285.3, 397.6, 521.8, 642.1, 789.4, 912.7]
    cymb = np.zeros(n)
    for i, m in enumerate(modes):
        f = m * 5.8
        phase = 2.0 * np.pi * f * t
        cymb += np.sin(phase) * (0.8 + 0.2 * np.sin(2.0 * np.pi * (m * 0.25) * t))
    
    # Ruido modulado
    noise = np.random.normal(0, 1, n)
    cymb = cymb * 0.5 + noise * 0.5
    
    # Filtro paso banda cerrado
    b, a = sps.butter(2, [5500 / (SR / 2), 11500 / (SR / 2)], btype='band')
    filtered = sps.lfilter(b, a, cymb)
    
    decay = 65.0 if closed else 16.0
    env = np.exp(-t * decay)
    return filtered * env * 0.45

# ──────────────────────────────────────────────────────────
# 2. FOLEY DE PUESTO DE AVANZADA (SCI-FI OUTPOST FOLEY)
# ──────────────────────────────────────────────────────────

def synth_rcs_thruster(dur: float = 3.2) -> np.ndarray:
    """
    Ráfaga lejana de propulsor de maniobra RCS o purga de válvula.
    Sub-rumble + siseo difuso de gas comprimido.
    """
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    
    # Envolvente suave de activación y cierre
    env = np.clip(t / 0.5, 0.0, 1.0) * np.clip((dur - t) / 0.8, 0.0, 1.0)
    noise = np.random.uniform(-1.0, 1.0, n)
    
    # Componente grave (estruendo sordo sub)
    b_low, a_low = sps.butter(2, 85.0 / (SR / 2), btype='low')
    rumble = sps.lfilter(b_low, a_low, noise)
    
    # Componente de siseo de gas
    b_hiss, a_hiss = sps.butter(2, [800 / (SR / 2), 2200 / (SR / 2)], btype='band')
    hiss = sps.lfilter(b_hiss, a_hiss, noise)
    
    thruster = (rumble * 0.85 + hiss * 0.15) * env
    return thruster * 0.08  # nivel muy sutil al fondo

def synth_space_telemetry(dur: float = 2.5) -> np.ndarray:
    """
    Pitidos y barridos suaves de telemetría y terminal orbital.
    Tonos sinusoidales puros con trémolo y decaimiento suave.
    """
    n = int(dur * SR)
    out = np.zeros(n)
    
    beeps = [
        (0.1, 0.14, 440.0),
        (0.35, 0.12, 554.37),
        (0.60, 0.16, 659.25),
        (0.95, 0.22, 880.0),
    ]
    
    for start_t, b_dur, freq in beeps:
        start_idx = int(start_t * SR)
        b_len = int(b_dur * SR)
        if start_idx + b_len > n:
            continue
        t_b = np.linspace(0, b_dur, b_len, endpoint=False)
        # Tono con ligero barrido
        sweep = freq + 40.0 * np.exp(-t_b * 12.0)
        phase = 2.0 * np.pi * np.cumsum(sweep) / SR
        tone = np.sin(phase) + 0.08 * np.sin(phase * 1.5)
        env = np.sin(np.pi * np.clip(t_b / b_dur, 0.0, 1.0)) ** 1.8
        out[start_idx:start_idx + b_len] += tone * env * 0.025
        
    return out

def synth_server_airflow(duration: float) -> np.ndarray:
    """
    Zumbido de ventiladores de racks de servidores y flujo de aire de la estación.
    Zumbido eléctrico de 50 Hz/100 Hz + ruido de baja frecuencia modelado.
    """
    n = int(duration * SR)
    t = np.linspace(0, duration, n, endpoint=False)
    
    # 50 Hz fundamental + 100 Hz armónico (zumbido de fuentes de alimentación)
    hum = (np.sin(2.0 * np.pi * 50.0 * t) * 0.012 + 
           np.sin(2.0 * np.pi * 100.0 * t) * 0.006)
    
    # Flujo de aire continuo filtrado
    pink_noise = np.random.normal(0, 1, n)
    b_air, a_air = sps.butter(2, [120 / (SR / 2), 480 / (SR / 2)], btype='band')
    airflow = sps.lfilter(b_air, a_air, pink_noise) * 0.015
    
    # Modulación lenta LFO (respiración del soporte vital)
    lfo = 0.85 + 0.15 * np.sin(2.0 * np.pi * 0.08 * t)
    return (hum + airflow) * lfo

# ──────────────────────────────────────────────────────────
# 3. INSTRUMENTACIÓN DARK SCI-FI LO-FI
# ──────────────────────────────────────────────────────────

def synth_dark_rhodes_chord(notes: list, duration: float, velocity: float = 0.6) -> np.ndarray:
    """
    Piano eléctrico tipo Rhodes oscuro y etéreo.
    Filtro paso bajo envolvente, armónicos de púa metálica y vibrato sutil.
    """
    n = int(duration * SR)
    chord_sig = np.zeros(n)
    t = np.linspace(0, duration, n, endpoint=False)
    
    for note in notes:
        freq = midi_to_freq(note)
        # Detune estéreo/micro-chorus
        detune = (random.random() - 0.5) * 0.8
        f = freq + detune
        
        # Fundamental pura + armónicos apagados (timbre sedoso)
        tone = (
            np.sin(2.0 * np.pi * f * t) * 1.0 +
            np.sin(2.0 * np.pi * 2 * f * t) * 0.28 +
            np.sin(2.0 * np.pi * 3 * f * t) * 0.08 +
            np.sin(2.0 * np.pi * 4 * f * t) * 0.02
        )
        
        # Envolvente: ataque 10ms, caída exponencial suave
        att_len = int(0.015 * SR)
        env = np.ones(n)
        env[:att_len] = np.linspace(0, 1, att_len)
        env[att_len:] = np.exp(-t[att_len:] * 0.65)
        
        # LFO de vibrato lento (tape wow)
        vibrato = np.sin(2.0 * np.pi * 3.8 * t) * 0.003
        f_vib = f * (1.0 + vibrato)
        phase_vib = 2.0 * np.pi * np.cumsum(f_vib) / SR
        tone_vib = np.sin(phase_vib) + 0.22 * np.sin(2.0 * phase_vib)
        
        chord_sig += tone_vib * env * velocity
        
    chord_sig /= max(len(notes), 1)
    
    # Filtrado paso bajo oscuro (1400 Hz) para dar esa calidez solitaria
    b_lp, a_lp = sps.butter(2, 1400 / (SR / 2), btype='low')
    return sps.lfilter(b_lp, a_lp, chord_sig)

def synth_sub_bass_note(note: float, duration: float) -> np.ndarray:
    """Bajo sub-grave cálido y profundo (40 - 90 Hz)."""
    n = int(duration * SR)
    t = np.linspace(0, duration, n, endpoint=False)
    f = midi_to_freq(note)
    
    # Fundamental con 2º armónico muy sutil
    tone = np.sin(2.0 * np.pi * f * t) + 0.15 * np.sin(2.0 * np.pi * 2 * f * t)
    env = np.clip(t / 0.02, 0.0, 1.0) * np.exp(-t * 0.45)
    bass = tone * env
    
    # Saturación analógica
    sat_bass = np.tanh(bass * 1.25)
    b_lp, a_lp = sps.butter(2, 350 / (SR / 2), btype='low')
    return sps.lfilter(b_lp, a_lp, sat_bass) * 0.85

def synth_lead_synth_phrase(melody_notes: list, step_dur: float) -> np.ndarray:
    """Melodía principal flotante, con filtro modulado y delay espacial."""
    total_dur = len(melody_notes) * step_dur
    n = int(total_dur * SR)
    lead = np.zeros(n)
    
    for i, note in enumerate(melody_notes):
        if note is None:
            continue
        start_idx = int(i * step_dur * SR)
        dur = step_dur * 0.92
        n_note = int(dur * SR)
        if start_idx + n_note > n:
            continue
        t_note = np.linspace(0, dur, n_note, endpoint=False)
        f = midi_to_freq(note)
        
        # Onda sinusoidal con leve armónico y portamento
        sig = np.sin(2.0 * np.pi * f * t_note) + 0.18 * np.sin(2.0 * np.pi * 2 * f * t_note)
        env = np.sin(np.pi * np.clip(t_note / dur, 0.0, 1.0)) ** 1.3
        
        lead[start_idx:start_idx + n_note] += sig * env * 0.35
        
    return lead

# ──────────────────────────────────────────────────────────
# 4. COMPOSICIÓN PROCEDURAL DARK OUTPOST
# ──────────────────────────────────────────────────────────

THEMES = [
    {
        "name": "Orbital Observation Deck",
        "bpm": 70,
        "root_chords": [
            [54, 61, 64, 68, 71],   # F#m9 (F#, C#, E, G#, B)
            [50, 57, 61, 66, 68],   # Dmaj7#11 (D, A, C#, F#, G#)
            [57, 61, 64, 68, 71],   # Aadd9 (A, C#, E, G#, B)
            [49, 56, 61, 64, 68],   # C#m7 (C#, G#, C#, E, G#)
        ],
        "bass_notes": [42, 38, 45, 37],
        "melody": [68, None, 71, 73, 71, 68, 64, 66],
        "foley_type": "orbital",  # RCS + telemetría
    },
    {
        "name": "Arctic Outpost Permafrost",
        "bpm": 68,
        "root_chords": [
            [57, 60, 64, 67, 71],   # Am9 (A, C, E, G, B)
            [53, 57, 60, 64, 69],   # Fmaj7 (F, A, C, E, A)
            [50, 53, 57, 60, 64],   # Dm9 (D, F, A, C, E)
            [52, 55, 59, 62, 67],   # Em7 (E, G, B, D, G)
        ],
        "bass_notes": [45, 41, 38, 40],
        "melody": [71, 67, None, 64, 67, 69, 64, None],
        "foley_type": "polar",    # Viento gélido + telemetría
    },
    {
        "name": "Deep Isolation Facility (SCP)",
        "bpm": 72,
        "root_chords": [
            [52, 55, 59, 62, 66],   # Em9 (E, G, B, D, F#)
            [48, 55, 59, 64, 66],   # Cmaj7#11 (C, G, B, E, F#)
            [57, 60, 64, 67, 71],   # Am9 (A, C, E, G, B)
            [47, 54, 57, 62, 66],   # B7alt (B, F#, A, D, F#)
        ],
        "bass_notes": [40, 36, 45, 35],
        "melody": [66, 67, 66, 62, 59, None, 62, 64],
        "foley_type": "facility", # Racks de servidores + telemetría lejana
    }
]

def compose_dark_outpost_track(theme: dict, num_bars: int = 8) -> np.ndarray:
    """Compone una pista completa de Dark Outpost Lo-Fi con loop perfecto."""
    bpm = theme["bpm"]
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4.0
    total_dur = num_bars * bar_dur
    total_samples = int(total_dur * SR)
    
    # 1. Pistas de Instrumentos
    rhodes_bus = np.zeros(total_samples)
    bass_bus   = np.zeros(total_samples)
    drum_bus   = np.zeros(total_samples)
    lead_bus   = np.zeros(total_samples)
    foley_bus  = synth_server_airflow(total_dur)
    
    chords = theme["root_chords"]
    bass_notes = theme["bass_notes"]
    
    # Cargar baterías
    kick = synth_dark_kick()
    rim  = synth_dark_rimshot()
    hat_cl = synth_dark_hat(closed=True)
    hat_op = synth_dark_hat(closed=False)
    
    # Patrón de batería Lo-Fi oscuro (con micro-swing)
    for bar in range(num_bars):
        bar_start = bar * bar_dur
        chord_idx = bar % len(chords)
        current_chord = chords[chord_idx]
        current_bass  = bass_notes[chord_idx]
        
        # Rhodes pad
        bar_idx = int(bar_start * SR)
        rhodes_chunk = synth_dark_rhodes_chord(current_chord, bar_dur * 0.98, velocity=0.58)
        chunk_len = min(len(rhodes_chunk), total_samples - bar_idx)
        rhodes_bus[bar_idx:bar_idx + chunk_len] += rhodes_chunk[:chunk_len]
        
        # Sub Bass
        bass_chunk = synth_sub_bass_note(current_bass, bar_dur * 0.88)
        chunk_len = min(len(bass_chunk), total_samples - bar_idx)
        bass_bus[bar_idx:bar_idx + chunk_len] += bass_chunk[:chunk_len]
        
        # Beats (4 tiempos por compás)
        # Bombo en tiempo 1 y síncopa en 3.5
        for beat in range(4):
            # Swing MPC en corcheas
            swing = 0.022 if (beat % 2 == 1) else 0.0
            t_beat = bar_start + beat * beat_dur + swing
            idx_beat = int(t_beat * SR)
            
            # Hi-hat continuo
            if idx_beat < total_samples:
                cl_len = min(len(hat_cl), total_samples - idx_beat)
                drum_bus[idx_beat:idx_beat + cl_len] += hat_cl[:cl_len] * 0.65
                
            # Bombo en tiempo 0 y 2.5
            if beat == 0 and idx_beat < total_samples:
                k_len = min(len(kick), total_samples - idx_beat)
                drum_bus[idx_beat:idx_beat + k_len] += kick[:k_len]
            elif beat == 2:
                # Síncopa en 2.75
                t_sync = bar_start + (2.75 * beat_dur) + 0.015
                idx_sync = int(t_sync * SR)
                if idx_sync < total_samples:
                    k_len = min(len(kick), total_samples - idx_sync)
                    drum_bus[idx_sync:idx_sync + k_len] += kick[:k_len] * 0.78
                    
            # Rimshot en tiempos 1 y 3 (el 2 y 4 estándar)
            if beat in (1, 3) and idx_beat < total_samples:
                r_len = min(len(rim), total_samples - idx_beat)
                drum_bus[idx_beat:idx_beat + r_len] += rim[:r_len]
                
    # 2. Melodía flotante (entra en compases 3-7)
    melody = theme["melody"]
    lead_start = 2 * bar_dur
    lead_dur = 4 * bar_dur
    lead_step = lead_dur / len(melody)
    lead_audio = synth_lead_synth_phrase(melody, lead_step)
    lead_idx = int(lead_start * SR)
    l_len = min(len(lead_audio), total_samples - lead_idx)
    lead_bus[lead_idx:lead_idx + l_len] += lead_audio[:l_len]
    
    # 3. Foley temático (RCS y telemetría espacial)
    rcs_sample = synth_rcs_thruster(dur=3.5)
    telemetry_sample = synth_space_telemetry(dur=2.8)
    
    # Insertar RCS a la mitad
    rcs_pos = int(total_samples * 0.35)
    r_len = min(len(rcs_sample), total_samples - rcs_pos)
    foley_bus[rcs_pos:rcs_pos + r_len] += rcs_sample[:r_len]
    
    # Insertar telemetría
    tel_pos = int(total_samples * 0.68)
    t_len = min(len(telemetry_sample), total_samples - tel_pos)
    foley_bus[tel_pos:tel_pos + t_len] += telemetry_sample[:t_len]
    
    # 4. Master Bus & Pedalboard
    stereo_mix = np.zeros((2, total_samples))
    stereo_mix[0] = rhodes_bus * 0.65 + bass_bus * 0.75 + drum_bus * 0.85 + lead_bus * 0.55 + foley_bus * 0.60
    stereo_mix[1] = rhodes_bus * 0.65 + bass_bus * 0.75 + drum_bus * 0.85 + lead_bus * 0.55 + foley_bus * 0.60
    
    # Efectos espaciales: Reverb amplia, chorus analógico, compresión y limitación
    board = Pedalboard([
        Chorus(rate_hz=0.8, depth=0.25, centre_delay_ms=12.0, feedback=0.15, mix=0.20),
        Reverb(room_size=0.82, damping=0.45, wet_level=0.30, dry_level=0.85, width=1.0),
        LowShelfFilter(cutoff_frequency_hz=160.0, gain_db=2.5),
        HighShelfFilter(cutoff_frequency_hz=6500.0, gain_db=-2.0),
        Compressor(threshold_db=-14.0, ratio=2.8, attack_ms=25.0, release_ms=140.0),
        Gain(gain_db=1.0)
    ])
    
    processed = board(stereo_mix, SR)
    
    # Normalización suave a -1.2 dBFS
    max_val = np.max(np.abs(processed))
    if max_val > 0.001:
        target = 10.0 ** (-1.2 / 20.0)
        processed = (processed / max_val) * target
        
    return processed

def main():
    out_dir = pathlib.Path("C:/Users/Jack/Documents/Github/JackStar6677-1/star-ambient-stream/test_lofi")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GENERANDO 3 PISTAS DARK OUTPOST SCI-FI LO-FI...")
    print("=" * 60)
    
    for i, theme in enumerate(THEMES, 1):
        filename = f"{i:02d}_{theme['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.wav"
        filepath = out_dir / filename
        print(f"\n[+] Componiendo {theme['name']} ({theme['bpm']} BPM, 8 compases)...")
        audio = compose_dark_outpost_track(theme, num_bars=8)
        
        # Guardar audio WAV
        sf.write(str(filepath), audio.T, SR)
        dur = audio.shape[1] / SR
        print(f"    Guardado en: {filepath} ({dur:.1f}s)")
        
        # Generar espectrograma
        spec_path = out_dir / f"{filepath.stem}_spectrogram.png"
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.specgram(audio[0], Fs=SR, NFFT=1024, noverlap=512, cmap='inferno')
        ax.set_title(f"FFT Spectrogram: {theme['name']} (Dark Sci-Fi Outpost Lo-Fi)")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")
        ax.set_ylim(0, 14000)
        plt.tight_layout()
        plt.savefig(str(spec_path), dpi=150)
        plt.close(fig)
        print(f"    Espectrograma generado: {spec_path}")

    print("\n[OK] Generación completada con éxito.")

if __name__ == "__main__":
    main()
