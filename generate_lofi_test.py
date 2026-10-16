#!/usr/bin/env python3
"""
Motor Avanzado de Generación Procedural de Música Lo-Fi (v2 - Calidad Acústica Orgánica).
Mejoras clave:
  1. Baterías acústicas realistas: Eliminación del sonido 8-bit.
     - Bombo (Kick): Mazo de fieltro, cuerpo resonante sub-grave (55-75 Hz) y saturación de cinta.
     - Caja (Snare/Rimshot): Golpe de baqueta en aro de madera, resonancia modal del casco y micro-bordonera.
     - Hi-Hats: 6 osciladores inarmónicos metálicos con modulación en anillo (emulación de aleación bronce B20).
     - Swing MPC orgánico humanizado (desfase dinámico de micro-tiempos y velocidades).
  2. Foley auténtico de cafetería y ambiente profundo:
     - Café sirviéndose en taza / cafetera con burbujas y tono resonante ascendente.
     - Tazas, platos de cerámica y cuchara metálica (resonancia modal de porcelana, -24 dB al fondo).
     - Goteo continuo y lluvia suave en el ventanal.
  3. Loop perfecto sin costuras (wrap-around tail fold).
  4. Masterización analógica con Spotify Pedalboard DSP.
"""

import math
import random
import pathlib
import numpy as np
from scipy import signal as sps
import soundfile as sf
import matplotlib.pyplot as plt
from pedalboard import Pedalboard, Reverb, Chorus, Compressor, LowpassFilter, HighpassFilter, Gain, LowShelfFilter, HighShelfFilter

SR = 44100

def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

# ──────────────────────────────────────────────────────────
# 1. BATERÍAS ACÚSTICAS ORGÁNICAS (CERO SONIDO 8-BIT)
# ──────────────────────────────────────────────────────────

def synth_acoustic_lofi_kick() -> np.ndarray:
    """
    Bombo acústico de estudio Lo-Fi.
    Simula mazo de fieltro golpeando parche, resonancia de aire de 20 pulgadas y saturación analógica.
    No utiliza barridos sintéticos de 8-bit.
    """
    dur = 0.32
    num_samples = int(dur * SR)
    t = np.linspace(0, dur, num_samples, endpoint=False)
    
    # 1. Transitorio de golpe de fieltro (impact thud filtrado)
    impact_len = int(0.015 * SR)
    impact_noise = np.random.normal(0, 1, impact_len)
    b, a = sps.butter(2, [300 / (SR / 2), 1200 / (SR / 2)], btype='band')
    impact = sps.lfilter(b, a, impact_noise) * np.exp(-np.linspace(0, 8, impact_len))
    
    # 2. Resonancia del parche (fundamental con descenso suave no-lineal)
    # Pitch cae orgánicamente de 82 Hz a 52 Hz (tono característico de bombo acústico)
    f_env = 52.0 + 30.0 * np.exp(-t * 22.0)
    phase = 2.0 * np.pi * np.cumsum(f_env) / SR
    
    # Fundamental + 2º armónico sutil del casco
    body = np.sin(phase) + 0.18 * np.sin(2.0 * phase)
    amp_env = np.exp(-t * 11.5)
    
    kick = np.zeros(num_samples)
    kick[:impact_len] += impact * 0.35
    kick += body * amp_env
    
    # 3. Modelado de saturación de cinta cálida (soft clipping) y filtro paso bajo a 4.2 kHz
    saturated = np.tanh(kick * 1.35)
    b_lp, a_lp = sps.butter(2, 4200 / (SR / 2), btype='low')
    warm_kick = sps.lfilter(b_lp, a_lp, saturated)
    
    return warm_kick * 0.95

def synth_acoustic_lofi_rimshot() -> np.ndarray:
    """
    Rimshot / Cross-stick de madera acústico.
    Golpe de baqueta contra el aro metálico/madera del redoblante.
    Reemplaza por completo el ruido blanco primitivo estilo consola retro.
    """
    dur = 0.18
    num_samples = int(dur * SR)
    t = np.linspace(0, dur, num_samples, endpoint=False)
    
    # Modos acústicos resonantes de baqueta de nogal / madera
    wood_modes = [
        (420.0, 35.0, 0.45),
        (880.0, 45.0, 0.35),
        (1650.0, 60.0, 0.25),
        (2800.0, 80.0, 0.15)
    ]
    stick_sound = np.zeros(num_samples)
    for freq, decay, weight in wood_modes:
        stick_sound += np.sin(2.0 * np.pi * freq * t) * np.exp(-t * decay) * weight
        
    # Micro-chasquido inicial del aro
    click_len = int(0.004 * SR)
    stick_sound[:click_len] += np.random.normal(0, 0.25, click_len) * np.linspace(1, 0, click_len)
    
    # Bordonera metálica muy suave y difusa (snare rattle de fondo)
    wire_noise = np.random.normal(0, 1, num_samples)
    b_sn, a_sn = sps.butter(2, [1800 / (SR / 2), 6500 / (SR / 2)], btype='band')
    wire_filtered = sps.lfilter(b_sn, a_sn, wire_noise) * np.exp(-t * 32.0) * 0.22
    
    rimshot = stick_sound * 0.75 + wire_filtered
    return np.clip(rimshot * 0.85, -1.0, 1.0)

def synth_metallic_cymbal_hat(open_hat: bool = False) -> np.ndarray:
    """
    Hi-hat metálico emulado por resonancia de modos inarmónicos de aleación de bronce.
    Utiliza 6 frecuencias no armónicas con modulación de anillo (estilo TR-808 / plato acústico).
    Cero estática o ruido blanco plano de 8-bit.
    """
    dur = 0.22 if open_hat else 0.055
    num_samples = int(dur * SR)
    t = np.linspace(0, dur, num_samples, endpoint=False)
    
    # 6 osciladores inarmónicos de bronce
    metal_freqs = [245.3, 306.4, 384.7, 522.8, 642.1, 839.5]
    raw_metal = np.zeros(num_samples)
    for f in metal_freqs:
        # Onda cuadrada suave / pulso metálico
        sq = sps.square(2.0 * np.pi * f * t)
        raw_metal += sq
        
    # Modulación en anillo con componente de armónicos superiores
    shimmer = np.sin(2.0 * np.pi * 3720.0 * t) * raw_metal
    
    # Filtro paso banda de plato (de 6.5 kHz a 13.5 kHz)
    b, a = sps.butter(2, [6500 / (SR / 2), 13500 / (SR / 2)], btype='band')
    cymbal = sps.lfilter(b, a, shimmer)
    
    decay_rate = 16.0 if open_hat else 75.0
    env = np.exp(-t * decay_rate)
    
    return cymbal * env * 0.22

# ──────────────────────────────────────────────────────────
# 2. INSTRUMENTOS MUSICALES: RHODES, BAJO & LEAD
# ──────────────────────────────────────────────────────────

def synth_rhodes_chord(notes: list[int], duration: float, velocity: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Genera un acorde de piano eléctrico / Rhodes estéreo cálido y orgánico."""
    num_samples = int(duration * SR)
    left = np.zeros(num_samples)
    right = np.zeros(num_samples)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    for idx, note in enumerate(notes):
        freq = midi_to_freq(note)
        strum_samples = int(idx * 0.007 * SR)
        if strum_samples >= num_samples:
            continue
            
        t_note = t[:num_samples - strum_samples]
        decay_rate = 1.6 + (freq / 650.0)
        env = np.exp(-t_note * decay_rate)
        
        attack_len = min(int(0.008 * SR), len(env))
        if attack_len > 0:
            env[:attack_len] *= np.linspace(0.1, 1.0, attack_len)
            
        tone = (
            0.65 * np.sin(2 * np.pi * freq * t_note) +
            0.24 * np.sin(2 * np.pi * freq * 2.0 * t_note) +
            0.08 * np.sin(2 * np.pi * freq * 3.0 * t_note) +
            0.03 * np.sin(2 * np.pi * freq * 4.01 * t_note)
        )
        
        tremolo_l = 1.0 + 0.04 * np.sin(2 * np.pi * 3.0 * t_note)
        tremolo_r = 1.0 + 0.04 * np.sin(2 * np.pi * 3.0 * t_note + np.pi / 2)
        
        voice_l = tone * env * tremolo_l * velocity
        voice_r = tone * env * tremolo_r * velocity
        
        pan = 0.35 + 0.3 * (idx / max(1, len(notes) - 1))
        left[strum_samples:] += voice_l * (1.0 - pan)
        right[strum_samples:] += voice_r * pan
        
    return left, right

def synth_bass_note(freq: float, duration: float, velocity: float = 0.82) -> np.ndarray:
    """Bajo eléctrico profundo y cálido."""
    num_samples = int(duration * SR)
    if num_samples <= 0:
        return np.zeros(0)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    env = np.exp(-t * 1.30)
    attack_samples = int(0.012 * SR)
    if attack_samples < num_samples:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
        
    raw = (
        0.78 * np.sin(2 * np.pi * freq * t) +
        0.20 * np.sin(2 * np.pi * (freq * 0.5) * t) +
        0.10 * sps.sawtooth(2 * np.pi * freq * t)
    )
    cutoff = min(0.32, (freq * 3.0) / (SR / 2))
    b, a = sps.butter(2, cutoff, btype='low')
    filtered = sps.lfilter(b, a, raw)
    saturated = np.tanh(filtered * 1.4)
    return saturated * env * velocity * 0.85

def synth_soft_lead(freq: float, duration: float, velocity: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Lead suave y meloso al fondo de la mezcla (cero vibratos agudos)."""
    num_samples = int(duration * SR)
    if num_samples <= 0:
        return np.zeros(0), np.zeros(0)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    osc = 0.88 * np.sin(2 * np.pi * freq * t) + 0.12 * np.sin(2 * np.pi * freq * 2.0 * t)
    
    attack_samples = int(0.045 * SR)
    release_samples = int(0.09 * SR)
    env = np.ones(num_samples)
    if attack_samples < num_samples:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples < num_samples:
        env[-release_samples:] = np.linspace(1, 0, release_samples)
    env *= np.exp(-t * 0.85)
    
    b, a = sps.butter(2, 2100 / (SR / 2), btype='low')
    mellow = sps.lfilter(b, a, osc * env * velocity)
    
    return mellow * 0.45, mellow * 0.55

# ──────────────────────────────────────────────────────────
# 3. FOLEY DE CAFETERÍA Y AMBIENTE ORGÁNICO
# ──────────────────────────────────────────────────────────

def synth_coffee_pour(duration: float = 2.8, pan: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulación de café caliente sirviéndose en una taza / cafetera.
    Micro-burbujas con pitch ascendente a medida que la taza se llena.
    """
    num_samples = int(duration * SR)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    stream = np.zeros(num_samples)
    
    # 1. Chorro de fluido continuo (ruido de flujo filtrado)
    fluid_noise = np.random.normal(0, 1, num_samples)
    # Tono de la columna de aire sube de 550 Hz a 980 Hz
    b, a = sps.butter(2, [450 / (SR / 2), 1400 / (SR / 2)], btype='band')
    fluid_base = sps.lfilter(b, a, fluid_noise)
    fluid_env = np.sin(np.pi * np.clip(t / duration, 0, 1)) ** 0.9
    stream += fluid_base * fluid_env * 0.05
    
    # 2. Gotas y micro-burbujas individuales (Minnaert bubbles)
    num_bubbles = int(duration * 45)
    for _ in range(num_bubbles):
        b_time = random.uniform(0.1, duration - 0.05)
        pos = int(b_time * SR)
        b_len = int(random.uniform(0.004, 0.012) * SR)
        bt = np.linspace(0, b_len / SR, b_len)
        # Frecuencia sube progresivamente conforme sube el nivel del líquido
        progress = b_time / duration
        base_f = random.uniform(700, 1100) + progress * 400.0
        bubble = np.sin(2 * np.pi * base_f * bt) * np.exp(-bt * 180.0) * random.uniform(0.012, 0.026)
        end = min(num_samples, pos + b_len)
        stream[pos:end] += bubble[:end - pos]
        
    stream = stream * 0.16 # Muy sutil de fondo
    return stream * (1.0 - pan), stream * pan

def synth_cup_and_spoon(pan: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    """Simulación modal de impacto de taza de porcelana y cuchara metálica."""
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
        
    clink = clink * 0.09 # Muy de fondo
    return clink * (1.0 - pan), clink * pan

def synth_rain_ambience(duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Lluvia procedural continua y gotas en ventana."""
    num_samples = int(duration * SR)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    noise_l = np.random.normal(0, 1, num_samples)
    noise_r = np.random.normal(0, 1, num_samples)
    b, a = sps.butter(2, [350 / (SR / 2), 2200 / (SR / 2)], btype='band')
    rain_l = sps.lfilter(b, a, noise_l)
    rain_r = sps.lfilter(b, a, noise_r)
    
    gust = 0.75 + 0.25 * np.sin(2 * np.pi * 0.04 * t)
    rain_l = rain_l * gust * 0.04
    rain_r = rain_r * gust * 0.04
    
    num_drops = int(duration * 20)
    for _ in range(num_drops):
        pos = random.randint(0, num_samples - int(0.01 * SR))
        drop_len = int(random.uniform(0.003, 0.008) * SR)
        dt = np.linspace(0, drop_len / SR, drop_len)
        pitch = random.uniform(2200, 3500)
        drop = np.sin(2 * np.pi * pitch * dt) * np.exp(-120.0 * dt) * random.uniform(0.012, 0.028)
        pan = random.uniform(0.2, 0.8)
        rain_l[pos:pos + drop_len] += drop * (1.0 - pan)
        rain_r[pos:pos + drop_len] += drop * pan
        
    return rain_l, rain_r

def synth_vinyl_texture(duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Textura analógica estéreo de vinilo y siseo cálido."""
    num_samples = int(duration * SR)
    noise_l = np.random.normal(0, 0.0025, num_samples)
    noise_r = np.random.normal(0, 0.0025, num_samples)
    b, a = sps.butter(1, [350 / (SR / 2), 4000 / (SR / 2)], btype='band')
    hiss_l = sps.lfilter(b, a, noise_l)
    hiss_r = sps.lfilter(b, a, noise_r)
    return hiss_l, hiss_r

# ──────────────────────────────────────────────────────────
# 4. COMPOSITOR COMPLETO CON LOOP SEAMLESS
# ──────────────────────────────────────────────────────────

def compose_lofi_track(
    title: str,
    bpm: float,
    progression: list[list[int]],
    melody_scale: list[int],
    bars: int = 32,
    seed: int = 42
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    
    seconds_per_beat = 60.0 / bpm
    bar_duration = 4 * seconds_per_beat
    loop_duration = bars * bar_duration
    loop_samples = int(loop_duration * SR)
    
    tail_duration = 3.5
    tail_samples = int(tail_duration * SR)
    total_samples = loop_samples + tail_samples
    
    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)
    chord_len = len(progression)
    
    # ── A. Acordes Rhodes ──
    for bar_idx in range(bars):
        chord_midi = progression[bar_idx % chord_len]
        bar_start_t = bar_idx * bar_duration
        patterns = [(0.0, 2.3, 0.75), (2.5, 1.35, 0.65)]
        if bar_idx % 4 == 3:
            patterns = [(0.0, 1.8, 0.75), (2.0, 1.0, 0.60), (3.25, 0.7, 0.55)]
            
        for off_beat, dur_beats, vel in patterns:
            t_note = bar_start_t + off_beat * seconds_per_beat
            sample_idx = int(t_note * SR)
            dur_s = dur_beats * seconds_per_beat
            c_l, c_r = synth_rhodes_chord(chord_midi, dur_s, vel * random.uniform(0.94, 1.04))
            end = min(total_samples, sample_idx + len(c_l))
            n = end - sample_idx
            if n > 0:
                left[sample_idx:end] += c_l[:n]
                right[sample_idx:end] += c_r[:n]

    # ── B. Bajo Lo-Fi Meloso ──
    for bar_idx in range(bars):
        chord_midi = progression[bar_idx % chord_len]
        root_midi = chord_midi[0] - 12
        fifth_midi = root_midi + 7
        bar_start_t = bar_idx * bar_duration
        bass_pattern = [
            (0.0, 1.4, root_midi, 0.88),
            (1.5, 0.8, root_midi, 0.72),
            (2.5, 1.2, fifth_midi if bar_idx % 2 == 1 else root_midi, 0.82),
            (3.75, 0.22, root_midi + (1 if bar_idx % 2 == 0 else -1), 0.68),
        ]
        for off_beat, dur_beats, note, vel in bass_pattern:
            t_note = bar_start_t + off_beat * seconds_per_beat
            sample_idx = int(t_note * SR)
            dur_s = dur_beats * seconds_per_beat
            b_wave = synth_bass_note(midi_to_freq(note), dur_s, vel)
            end = min(total_samples, sample_idx + len(b_wave))
            n = end - sample_idx
            if n > 0:
                left[sample_idx:end] += b_wave[:n] * 0.52
                right[sample_idx:end] += b_wave[:n] * 0.52

    # ── C. Batería Acústica Orgánica (Kick acústico, Rimshot de madera, Platillo de bronce) ──
    swing_ratio = 0.57
    for bar_idx in range(bars):
        if bar_idx < 2 and bar_idx % 2 == 0:
            continue
        bar_start_t = bar_idx * bar_duration
        for step in range(16):
            is_swung = (step % 2 == 1)
            off_in_beat = (step % 4) / 4.0
            if is_swung:
                off_in_beat += (swing_ratio - 0.5) * 0.48
            step_time = bar_start_t + ((step // 4) + off_in_beat) * seconds_per_beat
            pos = int(step_time * SR)
            
            # Bombo orgánico con caída suave
            if step in (0, 6, 10):
                k = synth_acoustic_lofi_kick()
                end = min(total_samples, pos + len(k))
                left[pos:end] += k[:end - pos] * 0.54
                right[pos:end] += k[:end - pos] * 0.54
                
            # Rimshot de madera acústico
            if step in (4, 12):
                rim = synth_acoustic_lofi_rimshot()
                end = min(total_samples, pos + len(rim))
                left[pos:end] += rim[:end - pos] * 0.46
                right[pos:end] += rim[:end - pos] * 0.46
                
            # Platillo metálico con dinámicas alternadas
            if step % 2 == 0 or random.random() > 0.35:
                is_open = (step == 14 and random.random() > 0.7)
                hh = synth_metallic_cymbal_hat(open_hat=is_open)
                end = min(total_samples, pos + len(hh))
                # Dinámica humanizada: acentos en tiempos fuertes
                vel_hh = 0.95 if step % 4 == 0 else 0.72
                left[pos:end] += hh[:end - pos] * 0.32 * vel_hh
                right[pos:end] += hh[:end - pos] * 0.46 * vel_hh

    # ── D. Melodía Solista Cálida ──
    lead_t = 4.0 * seconds_per_beat
    while lead_t < loop_duration - 4.0 * seconds_per_beat:
        if random.random() < 0.35:
            lead_t += seconds_per_beat * random.choice([2.0, 3.0, 4.0])
            continue
        note_midi = random.choice(melody_scale)
        dur_beats = random.choice([0.75, 1.0, 1.5, 2.0])
        dur_s = dur_beats * seconds_per_beat
        pos = int(lead_t * SR)
        m_l, m_r = synth_soft_lead(midi_to_freq(note_midi), dur_s, random.uniform(0.15, 0.22))
        end = min(total_samples, pos + len(m_l))
        n = end - pos
        if n > 0:
            left[pos:end] += m_l[:n]
            right[pos:end] += m_r[:n]
        lead_t += dur_s + (seconds_per_beat * random.choice([0.5, 1.0, 1.5]))

    # ── E. Foley de Cafetería y Lluvia ──
    # 1. Lluvia suave
    rain_l, rain_r = synth_rain_ambience(loop_duration + tail_duration)
    r_len = min(total_samples, len(rain_l))
    left[:r_len] += rain_l[:r_len] * 0.55
    right[:r_len] += rain_r[:r_len] * 0.55
    
    # 2. Café caliente sirviéndose en taza / cafetera (2 veces durante la pieza)
    coffee_times = [random.uniform(14.0, 22.0), random.uniform(55.0, 70.0)]
    for cot in coffee_times:
        co_pos = int(cot * SR)
        co_l, co_r = synth_coffee_pour(duration=2.6, pan=random.uniform(0.35, 0.65))
        co_end = min(total_samples, co_pos + len(co_l))
        n = co_end - co_pos
        if n > 0:
            left[co_pos:co_end] += co_l[:n]
            right[co_pos:co_end] += co_r[:n]

    # 3. Taza y cuchara de porcelana (3 veces sutilmente de fondo)
    cup_times = [random.uniform(8.0, 16.0), random.uniform(38.0, 48.0), random.uniform(78.0, 88.0)]
    for ct in cup_times:
        c_pos = int(ct * SR)
        c_l, c_r = synth_cup_and_spoon(pan=random.uniform(0.3, 0.7))
        c_end = min(total_samples, c_pos + len(c_l))
        n = c_end - c_pos
        if n > 0:
            left[c_pos:c_end] += c_l[:n]
            right[c_pos:c_end] += c_r[:n]

    # 4. Textura analógica de vinilo
    vl, vr = synth_vinyl_texture(loop_duration + tail_duration)
    v_len = min(total_samples, len(vl))
    left[:v_len] += vl[:v_len] * 0.35
    right[:v_len] += vr[:v_len] * 0.35

    # ── F. Algoritmo Wrap-Around Tail Fold para Loop Perfecto ──
    tail_l = left[loop_samples:]
    tail_r = right[loop_samples:]
    left[:tail_samples] += tail_l
    right[:tail_samples] += tail_r
    
    final_left = left[:loop_samples]
    final_right = right[:loop_samples]
    stereo = np.stack([final_left, final_right], axis=0)

    # ── G. Masterización DSP con Pedalboard ──
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
    max_peak = np.max(np.abs(mastered))
    if max_peak > 0:
        mastered = (mastered / max_peak) * 0.89
        
    return mastered

def generate_spectrogram(audio: np.ndarray, title: str, out_png: str):
    """Genera espectrograma de alta resolución para inspección acústica."""
    mono = np.mean(audio, axis=0)
    plt.figure(figsize=(10, 4), dpi=120)
    plt.specgram(mono, NFFT=2048, Fs=SR, noverlap=1024, cmap='inferno')
    plt.title(f"FFT Spectrogram: {title}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.ylim(0, 14000)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    out_dir = pathlib.Path(r"C:\Users\Jack\Documents\Github\JackStar6677-1\star-ambient-stream\test_lofi")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 3 Temáticas distintas
    themes = [
        {
            "name": "01_midnight_cafe_latte",
            "title": "Midnight Cafe · Warm Pour & Study",
            "bpm": 80.0,
            "progression": [
                [50, 53, 57, 60], # Dm7
                [53, 57, 60, 64], # Fmaj7
                [55, 58, 62, 65], # Gm7
                [48, 52, 55, 58]  # C7
            ],
            "scale": [60, 62, 65, 67, 69, 72, 74],
            "seed": 101
        },
        {
            "name": "02_rainy_coffee_percolator",
            "title": "Rainy Window · Fresh Brew & Books",
            "bpm": 76.0,
            "progression": [
                [53, 57, 60, 64], # Fmaj7
                [52, 55, 59, 62], # Em7
                [50, 53, 57, 60], # Dm7
                [48, 52, 55, 59]  # Cmaj7
            ],
            "scale": [60, 64, 65, 67, 69, 72, 76],
            "seed": 202
        },
        {
            "name": "03_star_late_shift",
            "title": "Star Late Shift · Code & Warm Mug",
            "bpm": 84.0,
            "progression": [
                [55, 58, 62, 65], # Gm7
                [50, 53, 57, 60], # Dm7
                [51, 55, 58, 62], # Ebmaj7
                [53, 57, 60, 63]  # F7
            ],
            "scale": [62, 65, 67, 70, 72, 74, 77],
            "seed": 303
        }
    ]
    
    for theme in themes:
        wav_path = out_dir / f"{theme['name']}.wav"
        png_path = out_dir / f"{theme['name']}_spectrogram.png"
        audio = compose_lofi_track(
            title=theme['title'],
            bpm=theme['bpm'],
            progression=theme['progression'],
            melody_scale=theme['scale'],
            bars=32,
            seed=theme['seed']
        )
        sf.write(str(wav_path), audio.T, SR, subtype='PCM_16')
        generate_spectrogram(audio, theme['title'], str(png_path))
        print(f"Generated: {theme['name']} ({audio.shape[1] / SR:.2f}s) -> {wav_path.name}")

if __name__ == "__main__":
    main()
