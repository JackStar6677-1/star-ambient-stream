"""
demo_diffusion.py — Genera un WAV A/B para comparar el sonido SECO (como ahora,
mono y plano) vs el sonido con DIFUSIÓN ESTÉREO (estilo Bitwig Poly Grid).

Salida: demo_diffusion.wav  ->  [6s seco mono] [1s silencio] [10s difuso estéreo]

Ejecutar:
    python proto/demo_diffusion.py
"""
import os
import wave
import numpy as np

from diffusion import StereoDiffuser

SR = 44100


def drone(seconds, seed=7):
    """Drone ambiental en Re# menor (D# F# A#) con microvariación de pitch
    y envolventes lentas — material parecido al que produce el motor real."""
    rng = np.random.RandomState(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    # Re# menor: D#3=155.56, F#3=185.00, A#3=233.08, + octava grave y aguda
    freqs = [77.78, 155.56, 185.00, 233.08, 311.13, 466.16]
    out = np.zeros(n)
    for i, f in enumerate(freqs):
        # Microvariación de pitch (jitter lento) = "vida", clave anti-sintético
        vib = 1.0 + 0.0025 * np.sin(2 * np.pi * (0.07 + 0.03 * i) * t + rng.uniform(0, 6.28))
        # Detune sutil por capa
        det = 1.0 + (i - 2.5) * 0.0006
        # Envolvente lenta independiente por capa (respiración)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * (0.03 + 0.02 * i) * t + rng.uniform(0, 6.28))
        amp = (0.9 / (i + 1.5)) * (0.4 + 0.6 * env)
        out += np.sin(2 * np.pi * f * det * np.cumsum(vib) / SR) * amp
    # Fade in/out global
    fade = int(SR * 1.5)
    win = np.ones(n)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    out *= win
    # Normalizar
    out /= (np.max(np.abs(out)) + 1e-9)
    return out * 0.7


def to_int16_stereo(sig):
    """sig (N,) mono o (N,2) estéreo -> int16 interleaved."""
    if sig.ndim == 1:
        sig = np.column_stack([sig, sig])
    sig = np.clip(sig, -1.0, 1.0)
    return (sig * 32000).astype(np.int16)


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    dry = drone(6.0)                       # seco, mono (como ahora)
    gap = np.zeros(int(SR * 1.0))

    rv = StereoDiffuser(sr=SR, rt60=3.5, mix=0.42, width=1.1, damp=0.45)
    wet = rv.process(drone(10.0, seed=11))  # difuso, estéreo

    dry_st = to_int16_stereo(dry)
    gap_st = to_int16_stereo(gap)
    wet_st = to_int16_stereo(wet)
    full = np.concatenate([dry_st, gap_st, wet_st], axis=0)

    path = os.path.join(here, "demo_diffusion.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(full.tobytes())

    dur = len(full) / SR
    print(f"[SUCCESS] WAV escrito: {path}")
    print(f"[INFO] Duracion {dur:.1f}s -> 6s SECO mono | 1s silencio | 10s DIFUSO estereo")


if __name__ == "__main__":
    main()
