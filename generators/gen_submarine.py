"""
Generadores Submarinos — crujido del casco y ping sonar
Mixin de métodos generadores para la clase Synth de audio_engine.py.
"""
import numpy as np
import scipy.signal as sps
from .gen_utils import biquad_bandpass, biquad_lowpass, SR

class SubmarineGeneratorsMixin:
    def _sub_creak_chunk(self, t_abs, n, subcreak_gain):
        """Hull groan: presión hidrostática produce gemido de frecuencia descendente
        (stick-slip friction en plancha de acero, 150→70 Hz con glide lento) +
        armónicos 2f y 3f + choque de bulkhead metálico."""
        if subcreak_gain <= 0:
            return np.zeros(n)
        if self.subcreak_active is None and t_abs >= self.subcreak_next:
            dur  = self.subcreak_rng.uniform(6.0, 14.0)
            amp  = self.subcreak_rng.uniform(0.010, 0.026) * subcreak_gain
            f0   = self.subcreak_rng.uniform(130.0, 165.0)   # frecuencia inicial
            f1   = self.subcreak_rng.uniform(55.0,  80.0)    # frecuencia final (glide)
            self.subcreak_active = (t_abs, dur, amp, f0, f1)
            self.subcreak_next   = t_abs + dur + self.subcreak_rng.uniform(55.0, 140.0)
        out = np.zeros(n)
        if self.subcreak_active:
            ts, dur, amp, f0, f1 = self.subcreak_active
            t_samples = t_abs + np.arange(n) / SR
            age = t_samples - ts
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                age_act = age[mask]
                # Glide exponencial de frecuencia (descenso de presión)
                alpha   = np.log(f1 / f0) / dur
                f_t     = f0 * np.exp(alpha * age_act)
                # Gemido principal: fundamental + 2° y 3° armónico (metal resonante)
                # Integración analítica para fase continua y click-free en los límites de chunk
                # Nota: age_act ya está en segundos, por lo que no se divide por SR.
                phase1 = 2.0 * np.pi * f0 * (np.exp(alpha * age_act) - 1.0) / alpha
                phase2 = 2.0 * np.pi * f0 * 2.01 * (np.exp(alpha * age_act) - 1.0) / alpha
                phase3 = 2.0 * np.pi * f0 * 3.03 * (np.exp(alpha * age_act) - 1.0) / alpha
                groan   = (np.sin(phase1) * 0.55 +
                           np.sin(phase2) * 0.28 +
                           np.sin(phase3) * 0.17)
                # Vibrato lento (fluctuación de presión)
                vib_lfo = 1.0 + 0.025 * np.sin(2.0 * np.pi * 1.4 * age_act)
                groan  *= vib_lfo
                # Envolvente: onset rápido, sustain, decaimiento lento
                env = (np.clip(age_act / 1.1, 0.0, 1.0) *
                       np.clip((dur - age_act) / 2.0, 0.0, 1.0))
                # Bulkhead hit al inicio (transiente impulsivo ~180 Hz)
                hit_env = np.exp(-28.0 * age_act)
                hit     = np.sin(2.0 * np.pi * 150.0 * age_act) * hit_env * 0.16
                out[mask] = (groan * env + hit) * amp
            if t_abs + n / SR >= ts + dur:
                self.subcreak_active = None
        return out

    def _sonar_ping_chunk(self, t_abs, n, sonar_gain):
        if sonar_gain <= 0:
            return np.zeros(n)
        # Ping activo cada 45s (más espaciado, más realista)
        period = 45.0
        # Programar pings
        if t_abs >= self.sonar_ping_next:
            self.sonar_ping_active.append((t_abs, 520.0, 0.015 * sonar_gain, "main"))       # Main ping lejano
            self.sonar_ping_active.append((t_abs + 1.8, 470.0, 0.004 * sonar_gain, "echo1")) # Echo 1
            self.sonar_ping_active.append((t_abs + 3.4, 410.0, 0.0015 * sonar_gain, "echo2")) # Echo 2
            self.sonar_ping_next = t_abs + period
        out = np.zeros(n)
        t_samples = t_abs + np.arange(n) / SR
        still = []
        for (ts, freq, amp, ptype) in self.sonar_ping_active:
            age = t_samples - ts
            dur = 2.5 if ptype == "main" else 1.8
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                age_act = age[mask]
                # Barrido de frecuencia descendente rápido al inicio (Chirp)
                f_sweep = freq - 80.0 * np.minimum(age_act, 0.4)
                env_decay = 4.2 if ptype == "main" else 2.6
                env = np.clip(age_act / 0.035, 0.0, 1.0) * np.exp(-env_decay * age_act)
                # Modulación de onda (ping sonar clasico)
                ping_wave = np.sin(2.0 * np.pi * f_sweep * age_act) * env * amp
                out[mask] += ping_wave
            if ts + dur > t_abs + n / SR:
                still.append((ts, freq, amp, ptype))
        self.sonar_ping_active = still
        return out

