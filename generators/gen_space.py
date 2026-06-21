"""
Generadores de Orbital — propulsores RCS y telemetría
Mixin de métodos generadores para la clase Synth de audio_engine.py.
"""
import numpy as np
import scipy.signal as sps
from .gen_utils import biquad_bandpass, biquad_highpass, biquad_lowpass, SR

class SpaceGeneratorsMixin:
    def _rcs_thruster_chunk(self, t_abs, n, rcs_gain):
        if rcs_gain <= 0:
            return np.zeros(n)
        if self.rcs_active is None and t_abs >= self.rcs_next:
            dur = self.rcs_rng.uniform(1.8, 4.6)
            amp = self.rcs_rng.uniform(0.010, 0.024) * rcs_gain
            self.rcs_active = (t_abs, dur, amp)
            self.rcs_next = t_abs + dur + self.rcs_rng.uniform(40.0, 120.0)
            self.rcs_zi_rumble = np.zeros(2)
            self.rcs_zi_hiss = np.zeros(2)
        out = np.zeros(n)
        if self.rcs_active:
            ts, dur, amp = self.rcs_active
            t_samples = t_abs + np.arange(n) / SR
            age = t_samples - ts
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                age_act = age[mask]
                rumble_env = np.clip(age_act / 0.45, 0.0, 1.0) * np.clip((dur - age_act) / 0.70, 0.0, 1.0)
                noise = self.rcs_rng.uniform(-1.0, 1.0, np.sum(mask))
                b_rcs, a_rcs = biquad_lowpass(85.0, 1.5, SR)
                rumble, self.rcs_zi_rumble = sps.lfilter(b_rcs, a_rcs, noise, zi=self.rcs_zi_rumble)
                b_hiss, a_hiss = biquad_bandpass(1150.0, 0.8, SR)
                hiss, self.rcs_zi_hiss = sps.lfilter(b_hiss, a_hiss, noise, zi=self.rcs_zi_hiss)
                click = np.zeros_like(age_act)
                click += np.exp(-420.0 * age_act) * 0.20
                click += np.exp(-420.0 * np.clip(dur - age_act, 0.0, 99.0)) * 0.12
                out[mask] = (rumble * 0.82 + hiss * 0.13 + click * 0.05) * amp * rumble_env
            if t_abs + n / SR >= ts + dur:
                self.rcs_active = None
        return out

    def _telemetry_chunk(self, t_abs, n, telemetry_gain):
        if telemetry_gain <= 0:
            return np.zeros(n)
        if t_abs >= self.telemetry_next:
            num_beeps = 1
            spacing = self.telemetry_rng.uniform(0.55, 0.95)
            freq = self.telemetry_rng.uniform(360, 680)
            amp = self.telemetry_rng.uniform(0.0012, 0.0035) * telemetry_gain
            t_cur = t_abs
            for _ in range(num_beeps):
                self.telemetry_queue.append((t_cur, 0.24, freq, amp))
                t_cur += spacing
                freq += self.telemetry_rng.uniform(-35, 35)
            self.telemetry_next = t_cur + self.telemetry_rng.uniform(120.0, 260.0)
        out = np.zeros(n)
        t_samples = t_abs + np.arange(n) / SR
        still = []
        for (ts, dur, freq, amp) in self.telemetry_queue:
            age = t_samples - ts
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                age_act = age[mask]
                f_sweep = freq - 35.0 * age_act
                env = np.clip(age_act / 0.055, 0.0, 1.0) * np.clip((dur - age_act) / 0.10, 0.0, 1.0)
                val = np.sin(2.0 * np.pi * f_sweep * age_act) * amp * env * np.exp(-5.0 * age_act)
                out[mask] += val
            if ts + dur > t_abs + n / SR:
                still.append((ts, dur, freq, amp))
        self.telemetry_queue = still
        return out
