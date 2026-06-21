"""
Generadores Polares — crujidos de glaciar y silbido de antena
Mixin de métodos generadores para la clase Synth de audio_engine.py.
"""
import numpy as np
import scipy.signal as sps
from .gen_utils import biquad_bandpass, biquad_lowpass, biquad_highpass, SR

class PolarGeneratorsMixin:
    def _glacier_creak_chunk(self, t_abs, n, glacier_gain):
        if glacier_gain <= 0:
            return np.zeros(n)
        if self.glacier_active is None and t_abs >= self.glacier_next:
            dur = self.glacier_rng.uniform(5.0, 11.0)
            amp = self.glacier_rng.uniform(0.009, 0.024) * glacier_gain
            # Generar momentos de crujidos rápidos individuales
            cracks = []
            for _ in range(self.glacier_rng.randint(1, 4)):
                cracks.append(self.glacier_rng.uniform(0.2, dur - 1.0))
            self.glacier_active = (t_abs, dur, amp, cracks)
            self.glacier_next = t_abs + dur + self.glacier_rng.uniform(55.0, 170.0)
        out = np.zeros(n)
        if self.glacier_active:
            ts, dur, amp, cracks = self.glacier_active
            t_samples = t_abs + np.arange(n) / SR
            age = t_samples - ts
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                age_act = age[mask]
                # Gemido de baja frecuencia (viento helado + tension estructural del hielo)
                lfo = np.sin(2.0 * np.pi * 5.5 * age_act) * np.sin(2.0 * np.pi * 0.6 * age_act)
                env = np.clip(age_act / 1.5, 0.0, 1.0) * np.clip((dur - age_act) / 1.5, 0.0, 1.0)
                noise = self.glacier_rng.uniform(-1.0, 1.0, np.sum(mask))
                b_gl, a_gl = biquad_lowpass(70.0, 2.0, SR)
                rumble, _ = sps.lfilter(b_gl, a_gl, noise, zi=np.zeros(2))
                out[mask] = rumble * (1.0 + 0.45 * lfo) * env * amp
                # Añadir los chasquidos rápidos (cracks)
                for c_t in cracks:
                    c_age = age - c_t
                    c_mask = (c_age >= 0.0) & (c_age < 0.12)
                    if np.any(c_mask):
                        c_age_act = c_age[c_mask]
                        c_noise = self.glacier_rng.uniform(-1.0, 1.0, np.sum(c_mask))
                        b_bp, a_bp = biquad_bandpass(950.0, 1.8, SR)
                        c_filt, _ = sps.lfilter(b_bp, a_bp, c_noise, zi=np.zeros(2))
                        c_env = np.clip(c_age_act / 0.012, 0.0, 1.0) * np.exp(-22.0 * c_age_act)
                        out[c_mask] += c_filt * c_env * amp * 0.26
            if t_abs + n / SR >= ts + dur:
                self.glacier_active = None
        return out

    def _wire_whistle_chunk(self, t_abs, n, wire_gain, wind_spd):
        if wire_gain <= 0 or wind_spd < 3.0:
            return np.zeros(n)
        t = t_abs + np.arange(n) / SR
        # Dos resonadores graves que representan cables de distinto grosor y tension.
        fc1 = 260.0 + wind_spd * 7.0 + 16.0 * np.sin(2.0 * np.pi * 0.07 * t)
        fc2 = 420.0 + wind_spd * 9.0 + 24.0 * np.sin(2.0 * np.pi * 0.11 * t + 1.2)
        # Amplitud modulada por el viento
        amp_mod = np.clip((wind_spd - 3.0) / 20.0, 0.05, 1.0) * wire_gain * 0.009
        noise = self.wire_rng.uniform(-1.0, 1.0, n)
        # Filtrado paso banda con Q muy alto
        # Para fc1
        b1, a1 = biquad_bandpass(np.mean(fc1), 36.0, SR)
        zi1 = self.wire_zi1 if self.wire_zi1 is not None else np.zeros(2)
        res1, self.wire_zi1 = sps.lfilter(b1, a1, noise, zi=zi1)
        # Para fc2
        b2, a2 = biquad_bandpass(np.mean(fc2), 32.0, SR)
        zi2 = self.wire_zi2 if self.wire_zi2 is not None else np.zeros(2)
        res2, self.wire_zi2 = sps.lfilter(b2, a2, noise, zi=zi2)
        # Fluctuación de amplitud
        lfo_amp = 0.6 + 0.4 * np.sin(2.0 * np.pi * 0.25 * t)
        return (res1 * 0.55 + res2 * 0.45) * amp_mod * lfo_amp

