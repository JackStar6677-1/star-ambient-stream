"""
Generadores Naturales — gotas, árboles, avalancha, arena, truenos
Mixin de métodos generadores para la clase Synth de audio_engine.py.
"""
import numpy as np
import scipy.signal as sps
from .gen_utils import biquad_bandpass, biquad_lowpass, SR

class NatureGeneratorsMixin:
    def _metal_drops_chunk(self, t_abs, n, drops_gain, rain_intensity):
        if drops_gain <= 0 or rain_intensity <= 0.05:
            return np.zeros(n)
        # Poisson drops
        rate = 18.0 * rain_intensity * drops_gain
        num_drops = int(self.drops_rng.poisson(rate * n / SR))
        num_drops = np.clip(num_drops, 0, 120)
        # Generar micro-resonadores
        frequencies = [520.0, 680.0, 840.0, 1030.0, 1280.0]
        t_samples = t_abs + np.arange(n) / SR
        out = np.zeros(n)
        for _ in range(num_drops):
            dt = self.drops_rng.uniform(0, n / SR)
            ts = t_abs + dt
            freq = self.drops_rng.choice(frequencies) + self.drops_rng.uniform(-40, 40)
            amp = self.drops_rng.uniform(0.0025, 0.010) * drops_gain
            decay = self.drops_rng.uniform(38.0, 82.0)
            age = t_samples - ts
            mask = (age >= 0.0) & (age < 0.06)
            if np.any(mask):
                age_act = age[mask]
                # Transitorio de impacto inicial (noise click)
                click_noise = self.drops_rng.uniform(-0.6, 0.6, np.sum(mask))
                # Resonador sinusoidal puro amortiguado
                sine = np.sin(2.0 * np.pi * freq * age_act)
                env = np.exp(-decay * age_act)
                # Combinacion click y cuerpo
                click_env = np.clip(age_act / 0.004, 0.0, 1.0) * np.exp(-120.0 * age_act)
                out[mask] += (sine * 0.88 + click_noise * click_env * 0.12) * amp * env
        return out

    def _tree_sway_chunk(self, t_abs, n, sway_gain, wind_spd):
        """Canopy rustle: 3 capas de ruido filtrado por banda que imitan
        hojas pequeñas (1.8-3.2 kHz), hojas grandes (0.4-1.0 kHz) y
        ramas flexionándose (80-250 Hz). El panning LFO simula la dirección
        del viento atravesando el dosel."""
        if sway_gain <= 0 or wind_spd < 1.5:
            return np.zeros(n)
        t = t_abs + np.arange(n) / SR
        # Intensidad proporcional al cuadrado del viento (aerodinámica)
        intensity = np.clip((wind_spd / 12.0) ** 2, 0.04, 1.2) * sway_gain
        noise = self.sway_rng.uniform(-1.0, 1.0, n).astype(np.float64)

        # Capa 1: hojas pequeñas oscuras, sin brillo de hiss de primer plano.
        b1, a1 = biquad_bandpass(1550.0, 1.0, SR)
        zi1 = self.sway_zi[0] if self.sway_zi is not None and len(self.sway_zi) > 0 else np.zeros(2)
        leaf_small, zi1_new = sps.lfilter(b1, a1, noise, zi=zi1)

        # Capa 2: hojas grandes / helechos (300-900 Hz)
        b2, a2 = biquad_bandpass(580.0, 1.1, SR)
        zi2 = self.sway_zi[1] if self.sway_zi is not None and len(self.sway_zi) > 1 else np.zeros(2)
        noise2 = self.sway_rng.uniform(-1.0, 1.0, n).astype(np.float64)
        leaf_large, zi2_new = sps.lfilter(b2, a2, noise2, zi=zi2)

        # Capa 3: ramas que se doblan (flexión estructural 60-180 Hz)
        b3, a3 = biquad_bandpass(115.0, 0.9, SR)
        zi3 = self.sway_zi[2] if self.sway_zi is not None and len(self.sway_zi) > 2 else np.zeros(2)
        noise3 = self.sway_rng.uniform(-1.0, 1.0, n).astype(np.float64)
        branch, zi3_new = sps.lfilter(b3, a3, noise3, zi=zi3)

        self.sway_zi = [zi1_new, zi2_new, zi3_new]

        # Panning LFO lento (viento cambia de dirección ~0.04 Hz)
        pan_lfo = 0.5 + 0.40 * np.sin(2.0 * np.pi * 0.04 * t + 0.8)
        # Ráfaga de viento (micro-variaciones ~0.3 Hz)
        gust_lfo = 0.55 + 0.45 * np.abs(np.sin(2.0 * np.pi * 0.28 * t))

        canopy = (leaf_small * 0.32 + leaf_large * 0.48 + branch * 0.20)
        return canopy * pan_lfo * gust_lfo * intensity * 0.135

    def _avalanche_chunk(self, t_abs, n, avalanche_gain):
        if avalanche_gain <= 0:
            return np.zeros(n)
        if self.avalanche_active is None and t_abs >= self.avalanche_next:
            dur = self.avalanche_rng.uniform(12.0, 22.0)
            amp = self.avalanche_rng.uniform(0.025, 0.055) * avalanche_gain
            self.avalanche_active = (t_abs, dur, amp)
            self.avalanche_next = t_abs + dur + self.avalanche_rng.uniform(90.0, 260.0)
        out = np.zeros(n)
        if self.avalanche_active:
            ts, dur, amp = self.avalanche_active
            t_samples = t_abs + np.arange(n) / SR
            age = t_samples - ts
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                age_act = age[mask]
                # Envolvente lenta: subida progresiva y descenso muy lento
                env = np.zeros_like(age_act)
                # 3.0s de subida
                env[age_act < 3.0] = age_act[age_act < 3.0] / 3.0
                # El resto es decaimiento exponencial
                env[age_act >= 3.0] = np.exp(-0.18 * (age_act[age_act >= 3.0] - 3.0))
                # Ruido de base marron (generado filtrando paso bajo a 40Hz)
                noise = self.avalanche_rng.uniform(-1.0, 1.0, np.sum(mask))
                # Filtro paso bajo a 45Hz
                b_av, a_av = biquad_lowpass(45.0, 0.85, SR)
                rumble, _ = sps.lfilter(b_av, a_av, noise, zi=np.zeros(2))
                # Modulación errática
                mod = 0.7 + 0.3 * self.avalanche_rng.uniform(-1.0, 1.0, np.sum(mask))
                out[mask] = rumble * mod * env * amp
            if t_abs + n / SR >= ts + dur:
                self.avalanche_active = None
        return out

    def _tension_harp_chunk(self, t_abs, n, harp_gain, wind_spd):
        if harp_gain <= 0 or wind_spd < 4.0:
            return np.zeros(n)
        t = t_abs + np.arange(n) / SR
        # 4 Frecuencias de tensión bajas: se sienten como cables, no como silbato.
        freqs = [95.0, 142.5, 190.0, 285.0]
        # Moduladas en afinación por el viento
        freqs_mod = [f + wind_spd * 1.5 + 4.0 * np.sin(2.0 * np.pi * (0.05 + 0.02*i) * t) for i, f in enumerate(freqs)]
        # Excitación proporcional al viento
        excit_level = np.clip((wind_spd - 4.0) / 18.0, 0.05, 1.0) * harp_gain * 0.012
        noise = self.harp_rng.uniform(-1.0, 1.0, n)
        out = np.zeros(n)
        # Filtrar individualmente cada resonancia de cable con un Q muy alto (240.0)
        for i, fc in enumerate(freqs_mod):
            b, a = biquad_bandpass(np.mean(fc), 42.0, SR)
            zi = self.harp_zi[i] if self.harp_zi[i] is not None else np.zeros(2)
            res, self.harp_zi[i] = sps.lfilter(b, a, noise, zi=zi)
            out += res * excit_level * (0.4 + 0.6 * np.sin(2.0 * np.pi * 0.11 * t + i * 1.5))
        return out * 0.72

    def _sandstorm_gust_chunk(self, t_abs, n, sand_gain, wind_spd):
        """Dust devils: los torbellinos de desierto reales no son continuos.
        Son ráfagas breves de arena que golpean la estructura en eventos
        discretos (hiss band 800-3500 Hz), intercalados con tramos de viento
        base más suave. Ocasionalmente un piedra golpea el metal."""
        if sand_gain <= 0 or wind_spd < 1.5:
            return np.zeros(n)

        # ── Fondo de viento seco (siempre activo, muy tenue) ──
        noise_bg = self.sand_rng.uniform(-1.0, 1.0, n).astype(np.float64)
        fc_bg = 620.0 + wind_spd * 14.0
        b_bg, a_bg = biquad_bandpass(fc_bg, 1.4, SR)
        zi_bg = self.sand_zi[0] if self.sand_zi is not None and len(self.sand_zi) > 0 else np.zeros(2)
        bg_filt, zi_bg_new = sps.lfilter(b_bg, a_bg, noise_bg, zi=zi_bg)
        base_amp = np.clip((wind_spd - 1.5) / 20.0, 0.0, 1.0) * sand_gain * 0.030


        # ── Ráfagas discretas de arena ──
        t_samples = t_abs + np.arange(n) / SR
        out = bg_filt * base_amp

        # Programar ráfagas: tasa proporcional al viento
        gust_rate = 0.08 + wind_spd * 0.018   # ráfagas/segundo
        while self.sand_next < t_abs + n / SR:
            dur    = self.sand_rng.uniform(0.4, 2.2)
            fc     = self.sand_rng.uniform(650.0, 1700.0)
            amp    = self.sand_rng.uniform(0.025, 0.085) * sand_gain
            self.sand_queue.append((self.sand_next, dur, fc, amp, np.zeros(2)))
            gap    = self.sand_rng.exponential(1.0 / max(gust_rate, 0.01))
            self.sand_next += dur + gap

        still = []
        for (ts, dur, fc, amp, zi) in self.sand_queue:
            age  = t_samples - ts
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                age_act = age[mask]
                # Envolvente del torbellino: ataque y decaimiento rápidos
                env = (np.clip(age_act / 0.30, 0.0, 1.0) *
                       np.clip((dur - age_act) / 0.55, 0.0, 1.0))
                noise_g = self.sand_rng.uniform(-1.0, 1.0, np.sum(mask)).astype(np.float64)
                b_g, a_g = biquad_bandpass(fc, 2.2, SR)
                gust_filt, zi = sps.lfilter(b_g, a_g, noise_g, zi=zi)
                out[mask] += gust_filt * env * amp
            if ts + dur > t_abs + n / SR:
                still.append((ts, dur, fc, amp, zi))
        self.sand_queue = still

        # ── Impacto de piedra ocasional en metal ──
        if t_abs >= self.sand_rock_next:
            rock_idx = int((self.sand_rock_next - t_abs) * SR)
            if 0 <= rock_idx < n:
                rock_amp = self.sand_rng.uniform(0.012, 0.040) * sand_gain
                rock_f   = self.sand_rng.uniform(450.0, 1100.0)
                for k in range(min(n - rock_idx, 800)):
                    out[rock_idx + k] += (np.sin(2.0 * np.pi * rock_f * k / SR) *
                                          np.exp(-45.0 * k / SR) * rock_amp)
            self.sand_rock_next = t_abs + self.sand_rng.uniform(18.0, 55.0)

        self.sand_zi = [zi_bg_new]
        return out

