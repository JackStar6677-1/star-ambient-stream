#!/usr/bin/env python3
"""
Weather-adaptive ambient audio generator.
Uses numpy for fast vectorized synthesis + pedalboard (Spotify) for pro effects.

Usage:
  python generate_weather_ambient.py                 # weather-based preset
  python generate_weather_ambient.py --dark-station  # force Dark Station: Polar Rift
"""

import wave, os, sys, json, urllib.request, random
import numpy as np
from scipy import signal as sps

try:
    from pedalboard import Pedalboard, Reverb, Chorus, Compressor, LowpassFilter, Gain
    HAS_PEDALBOARD = True
    print("[INFO] pedalboard loaded — pro effects active")
except ImportError:
    HAS_PEDALBOARD = False
    print("[WARN] pedalboard not found — using basic fallback reverb")

SR = 44100
DURATION = 60
N = SR * DURATION
output_path = "sci_fi_ambient_sample.wav"


# ──────────────────────────────────────────────────────────
# Weather API
# ──────────────────────────────────────────────────────────

def get_weather():
    print("[INFO] Detecting location and weather...")
    try:
        req = urllib.request.Request("http://ip-api.com/json/", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
            lat = data.get('lat', -33.45)
            lon = data.get('lon', -70.66)
            city = data.get('city', 'Santiago')
            country = data.get('country', 'Chile')
    except Exception as e:
        print(f"[WARN] Location failed: {e}. Defaulting to Santiago, Chile.")
        lat, lon, city, country = -33.45, -70.66, 'Santiago', 'Chile'

    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            w = json.loads(r.read().decode())['current']
            return w['temperature_2m'], w['relative_humidity_2m'], w['wind_speed_10m'], w['weather_code'], city, country
    except Exception as e:
        print(f"[WARN] Weather failed: {e}. Using defaults.")
        return 20.0, 50.0, 10.0, 0, city, country


# ──────────────────────────────────────────────────────────
# DSP Primitives
# ──────────────────────────────────────────────────────────

def make_pad(chords, t, chord_dur=15.0, fade_dur=5.0):
    """Crossfading sine-wave chord pads with tremolo."""
    n = len(t)
    pad = np.zeros(n)

    for idx, chord in enumerate(chords):
        s = int(idx * chord_dur * SR)
        e = min(int((idx + 1) * chord_dur * SR), n)
        if s >= n:
            break
        seg_t = t[s:e]
        seg_len = len(seg_t)

        fade_n = min(int(fade_dur * SR), seg_len)
        env = np.ones(seg_len)
        env[-fade_n:] = np.linspace(1.0, 0.0, fade_n)
        next_chord = chords[(idx + 1) % len(chords)]

        cur = np.zeros(seg_len)
        nxt = np.zeros(seg_len)
        for i, freq in enumerate(chord):
            g = 0.17 if i == 0 else 0.11 / (i ** 0.5)
            cur += np.sin(2 * np.pi * freq * 0.998 * seg_t) * g
            cur += np.sin(2 * np.pi * freq * 1.002 * seg_t) * g * 0.65
        for i, freq in enumerate(next_chord):
            g = 0.17 if i == 0 else 0.11 / (i ** 0.5)
            nxt += np.sin(2 * np.pi * freq * 0.998 * seg_t) * g
            nxt += np.sin(2 * np.pi * freq * 1.002 * seg_t) * g * 0.65

        pad[s:e] += cur * env + nxt * (1.0 - env)

    pad = np.tanh(pad * 0.85) * 0.36
    pad *= 0.88 + 0.12 * np.sin(2 * np.pi * 0.08 * t)  # breathing tremolo
    return pad


def make_wind(t, min_cut, max_cut, gain, n_seg=80):
    """Swept bandpass noise — space wind / blizzard."""
    rng = np.random.RandomState(seed=99)
    noise = rng.uniform(-1.0, 1.0, len(t))
    output = np.zeros_like(noise)
    seg = max(len(t) // n_seg, 1)
    zi = None

    for i in range(n_seg):
        s = i * seg
        e = min(s + seg, len(t))
        t_mid = t[s + (e - s) // 2]
        sweep = 0.5 + 0.5 * np.sin(2 * np.pi * 0.04 * t_mid)
        fc = min_cut + (max_cut - min_cut) * sweep

        nyq = SR / 2.0
        lo = np.clip(fc * 0.60 / nyq, 0.001, 0.490)
        hi = np.clip(fc * 1.40 / nyq, lo + 0.010, 0.498)

        b, a = sps.butter(3, [lo, hi], btype='band')
        if zi is None:
            zi = sps.lfilter_zi(b, a) * noise[s]
        seg_out, zi = sps.lfilter(b, a, noise[s:e], zi=zi)
        output[s:e] = seg_out

    return output * gain


def make_sub_drone(freq, t, gain=0.05):
    """Low sub-bass drone with slow vibrato."""
    vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 0.05 * t)
    drone = np.sin(2 * np.pi * freq * vibrato * t)
    env = np.clip(t / 8.0, 0.0, 1.0)
    return drone * env * gain


def make_melody(notes, t, gain=0.13):
    """Melodic lead with ADSR envelopes — sine-based, warm."""
    n = len(t)
    mel = np.zeros(n)
    ATTACK, DECAY, SUS_LVL, RELEASE = 0.4, 0.5, 0.45, 1.2
    NOTE_DUR = 2.5

    for start_t, freq in notes:
        s = int(start_t * SR)
        total_n = int((NOTE_DUR + RELEASE) * SR)
        e = min(s + total_n, n)
        if s >= n:
            break
        seg_len = e - s
        local_t = np.arange(seg_len) / SR

        a_n = int(ATTACK * SR)
        d_n = int(DECAY * SR)
        sus_n = int(max(0.0, NOTE_DUR - ATTACK - DECAY) * SR)
        r_n = int(RELEASE * SR)

        env = np.zeros(seg_len)
        p = 0
        end = min(p + a_n, seg_len)
        env[p:end] = np.linspace(0, 1, a_n)[:end - p]; p = end
        if p < seg_len:
            end = min(p + d_n, seg_len)
            env[p:end] = np.linspace(1, SUS_LVL, d_n)[:end - p]; p = end
        if p < seg_len:
            end = min(p + sus_n, seg_len)
            env[p:end] = SUS_LVL; p = end
        if p < seg_len:
            end = min(p + r_n, seg_len)
            env[p:end] = np.linspace(SUS_LVL, 0, r_n)[:end - p]

        wave = (np.sin(2 * np.pi * freq * 0.997 * local_t) +
                np.sin(2 * np.pi * freq * 1.003 * local_t)) * 0.5
        mel[s:e] += wave * env * gain

    return mel


def make_bells(bell_notes, chance_per_sec, t, seed=7):
    """Sparse crystalline bell hits with natural exponential decay."""
    n = len(t)
    bells = np.zeros(n)
    rng = random.Random(seed)
    i = 0
    while i < n:
        if rng.random() < chance_per_sec / SR:
            freq = rng.choice(bell_notes)
            amp = rng.uniform(0.06, 0.18)
            decay = 0.35 + freq / 750.0
            dur = min(int(5.0 * SR), n - i)
            lt = np.arange(dur) / SR
            bells[i:i + dur] += np.sin(2 * np.pi * freq * lt) * amp * np.exp(-decay * lt)
            i += int(SR * 0.25)
        i += 1
    return bells


def make_sonar(t, freq=1200, period=4.0):
    """Repeating sonar / telemetry ping."""
    pt = t % period
    mask = pt < 0.35
    ping = np.zeros(len(t))
    ping[mask] = np.sin(pt[mask] * 2 * np.pi * freq) * np.exp(-18.0 * pt[mask]) * 0.012
    return ping


def make_rain(humidity, code, t, seed=13):
    """Rain/mist layer — only when weather warrants it."""
    if not (code >= 51 or humidity > 75):
        return np.zeros(len(t))
    n = len(t)
    rain = np.zeros(n)
    rng = random.Random(seed)
    rate = 0.0003 + max(0, humidity - 50) * 0.00002
    i = 0
    while i < n:
        if rng.random() < rate:
            freq = rng.uniform(1500, 3500)
            amp = rng.uniform(0.03, 0.09)
            decay = rng.uniform(8, 15)
            dur = min(int(0.25 * SR), n - i)
            lt = np.arange(dur) / SR
            rain[i:i + dur] += np.sin(2 * np.pi * freq * lt) * amp * np.exp(-decay * lt)
        i += 1
    return rain


def apply_effects(audio, preset='default'):
    """Apply effects chain. Uses pedalboard if available, otherwise basic fallback."""
    if not HAS_PEDALBOARD:
        # Fallback: manual comb reverb
        out = audio.copy()
        for delay_ms, fb in [(113, 0.48), (173, 0.43), (293, 0.38), (359, 0.33)]:
            d = int(delay_ms * SR / 1000)
            delayed = np.zeros_like(audio)
            delayed[d:] = audio[:-d] * fb
            out += delayed * 0.14
        peak = np.max(np.abs(out))
        if peak > 0:
            out = out / peak * 0.85
        return out

    audio_f32 = audio.astype(np.float32).reshape(1, -1)

    cfgs = {
        'dark_station': Pedalboard([
            Chorus(rate_hz=0.25, depth=0.40, centre_delay_ms=8.0, feedback=0.15, mix=0.45),
            Reverb(room_size=0.95, damping=0.25, wet_level=0.60, dry_level=0.40),
            Compressor(threshold_db=-22, ratio=4.0, attack_ms=80, release_ms=600),
            LowpassFilter(cutoff_frequency_hz=6500),
            Gain(gain_db=3.0),
        ]),
        'arctic': Pedalboard([
            Chorus(rate_hz=0.40, depth=0.30, centre_delay_ms=6.0, feedback=0.10, mix=0.35),
            Reverb(room_size=0.88, damping=0.35, wet_level=0.52, dry_level=0.48),
            Compressor(threshold_db=-20, ratio=3.5, attack_ms=60, release_ms=500),
            LowpassFilter(cutoff_frequency_hz=7000),
            Gain(gain_db=2.5),
        ]),
        'desert': Pedalboard([
            Chorus(rate_hz=0.60, depth=0.20, centre_delay_ms=5.0, feedback=0.08, mix=0.25),
            Reverb(room_size=0.70, damping=0.65, wet_level=0.35, dry_level=0.60),
            Compressor(threshold_db=-16, ratio=3.0, attack_ms=25, release_ms=250),
            LowpassFilter(cutoff_frequency_hz=9000),
            Gain(gain_db=2.0),
        ]),
        'default': Pedalboard([
            Chorus(rate_hz=0.50, depth=0.25, centre_delay_ms=7.0, feedback=0.10, mix=0.30),
            Reverb(room_size=0.82, damping=0.45, wet_level=0.42, dry_level=0.55),
            Compressor(threshold_db=-18, ratio=3.5, attack_ms=40, release_ms=350),
            LowpassFilter(cutoff_frequency_hz=8000),
            Gain(gain_db=2.0),
        ]),
    }

    board = cfgs.get(preset, cfgs['default'])
    processed = board(audio_f32, SR)
    return processed.flatten().astype(np.float64)


def write_wav(audio):
    """Normalize, soft-clip, write WAV."""
    audio = np.tanh(audio * 0.95)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.88
    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32000).astype(np.int16).tobytes())
    print(f"[SUCCESS] Written {output_path} ({DURATION}s, {SR}Hz)")


# ──────────────────────────────────────────────────────────
# Presets
# ──────────────────────────────────────────────────────────

def preset_dark_station(t):
    """Dark Station: Polar Rift — cold, ominous, crystalline."""
    print("[INFO] Preset → DARK STATION: POLAR RIFT")
    chords = [
        [73.42, 146.83, 220.00, 261.63, 311.13],   # Dm9
        [98.00, 196.00, 233.08, 261.63, 392.00],   # Gm9
        [116.54, 174.61, 233.08, 293.66, 466.16],  # BbMaj7
        [110.00, 164.81, 220.00, 261.63, 329.63],  # Am7
    ]
    bell_notes = [587.33, 659.25, 739.99, 932.33, 1046.50, 1244.51]
    melody_notes = [
        (3.0, 293.66), (8.0, 233.08), (14.0, 220.00),
        (21.0, 261.63), (27.0, 349.23), (33.0, 293.66),
        (39.0, 233.08), (45.0, 196.00), (51.0, 261.63), (57.0, 220.00),
    ]
    pad    = make_pad(chords, t, chord_dur=15.0, fade_dur=5.0)
    drone  = make_sub_drone(36.71, t, gain=0.07)      # D1 sub-bass rumble
    wind   = make_wind(t, 1200, 4000, gain=0.12)      # blizzard
    melody = make_melody(melody_notes, t, gain=0.13)
    bells  = make_bells(bell_notes, chance_per_sec=0.04, t=t, seed=7)
    sonar  = make_sonar(t, freq=650, period=6.0)

    mix = pad*0.36 + drone*1.0 + wind*0.25 + melody*0.22 + bells*0.13 + sonar*0.04
    mix = apply_effects(mix, 'dark_station')
    write_wav(mix)
    return "DARK STATION: POLAR RIFT"


def preset_arctic(t, humidity, wind_speed, code):
    print("[INFO] Preset → ARCTIC FROST")
    chords = [
        [65.41, 196.00, 233.08, 293.66, 392.00],
        [103.83, 207.65, 261.63, 311.13, 392.00],
        [87.31, 174.61, 233.08, 261.63, 349.23],
        [73.42, 146.83, 207.65, 261.63, 311.13],
    ]
    bell_notes = [622.25, 698.46, 783.99, 932.33, 1046.50, 1244.51]
    melody_notes = [
        (2.0, 392.00), (6.0, 293.66), (10.0, 233.08), (15.0, 392.00),
        (21.0, 392.00), (25.0, 261.63), (29.0, 311.13), (35.0, 392.00),
        (41.0, 349.23), (45.0, 293.66), (49.0, 261.63), (54.0, 349.23),
    ]
    wg = 0.09 * max(0.5, min(2.5, wind_speed / 15.0))

    pad    = make_pad(chords, t)
    wind   = make_wind(t, 600, 2100, gain=wg)
    melody = make_melody(melody_notes, t, gain=0.14)
    bells  = make_bells(bell_notes, chance_per_sec=0.18, t=t)
    sonar  = make_sonar(t, freq=1000, period=4.0)
    rain   = make_rain(humidity, code, t)

    mix = pad*0.36 + wind*0.25 + melody*0.20 + bells*0.12 + sonar*0.03 + rain*0.04
    mix = apply_effects(mix, 'arctic')
    write_wav(mix)
    return "ARCTIC FROST"


def preset_desert(t, wind_speed):
    print("[INFO] Preset → DESERT HEAT WAVE")
    chords = [
        [73.42, 220.00, 277.18, 329.63, 440.00],
        [77.78, 155.56, 233.08, 293.66, 440.00],
        [98.00, 196.00, 233.08, 293.66, 392.00],
        [110.00, 220.00, 293.66, 329.63, 440.00],
    ]
    bell_notes = [440.00, 493.88, 554.37, 659.25, 739.99, 880.00]
    melody_notes = [
        (2.0, 440.00), (5.0, 554.37), (8.0, 659.25), (11.0, 739.99),
        (17.0, 554.37), (20.0, 659.25), (23.0, 739.99), (26.0, 880.00),
        (32.0, 739.99), (35.0, 659.25), (38.0, 554.37), (41.0, 440.00),
        (47.0, 554.37), (50.0, 440.00), (53.0, 493.88), (56.0, 440.00),
    ]
    wg = 0.04 * max(0.5, min(2.5, wind_speed / 15.0))

    pad    = make_pad(chords, t)
    wind   = make_wind(t, 150, 550, gain=wg)
    melody = make_melody(melody_notes, t, gain=0.16)
    bells  = make_bells(bell_notes, chance_per_sec=0.04, t=t)
    sonar  = make_sonar(t, freq=1400, period=4.0)

    mix = pad*0.36 + wind*0.20 + melody*0.22 + bells*0.12 + sonar*0.03
    mix = apply_effects(mix, 'desert')
    write_wav(mix)
    return "DESERT HEAT WAVE"


def preset_temperate(t, humidity, wind_speed, code):
    print("[INFO] Preset → LUSH TEMPERATE")
    chords = [
        [65.41, 196.00, 233.08, 293.66, 392.00],
        [103.83, 155.56, 196.00, 261.63, 392.00],
        [116.54, 174.61, 261.63, 293.66, 349.23],
        [98.00, 146.83, 174.61, 233.08, 293.66],
    ]
    bell_notes = [523.25, 587.33, 659.25, 783.99, 880.00, 1046.50]
    melody_notes = [
        (2.0, 392.00), (5.0, 293.66), (8.0, 233.08), (11.0, 392.00),
        (17.0, 392.00), (20.0, 261.63), (23.0, 311.13), (26.0, 392.00),
        (32.0, 349.23), (35.0, 293.66), (38.0, 261.63), (41.0, 349.23),
        (47.0, 293.66), (50.0, 233.08), (53.0, 349.23), (56.0, 293.66),
    ]
    wg = 0.06 * max(0.5, min(2.5, wind_speed / 15.0))

    pad    = make_pad(chords, t)
    wind   = make_wind(t, 350, 1300, gain=wg)
    melody = make_melody(melody_notes, t, gain=0.14)
    bells  = make_bells(bell_notes, chance_per_sec=0.10, t=t)
    sonar  = make_sonar(t, freq=1200, period=4.0)
    rain   = make_rain(humidity, code, t)

    mix = pad*0.35 + wind*0.22 + melody*0.20 + bells*0.12 + sonar*0.03 + rain*0.04
    mix = apply_effects(mix, 'default')
    write_wav(mix)
    return "LUSH TEMPERATE"


# ──────────────────────────────────────────────────────────
# Public API (called by stream_manager.py)
# ──────────────────────────────────────────────────────────

def generate_audio(temp, humidity, wind, code, city, country):
    t = np.linspace(0, DURATION, N, endpoint=False)
    if temp > 30.0:
        return preset_desert(t, wind)
    elif temp < 15.0:
        return preset_arctic(t, humidity, wind, code)
    else:
        return preset_temperate(t, humidity, wind, code)


def main():
    if '--dark-station' in sys.argv:
        t = np.linspace(0, DURATION, N, endpoint=False)
        preset_dark_station(t)
        return
    temp, humidity, wind, code, city, country = get_weather()
    print(f"[INFO] {temp}°C, {humidity}%, {wind}km/h — {city}, {country}")
    env = generate_audio(temp, humidity, wind, code, city, country)
    print(f"[SUCCESS] Environment: {env}")


if __name__ == "__main__":
    main()
