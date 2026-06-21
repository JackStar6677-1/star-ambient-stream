"""
Generadores Industriales — magma, vapor, Geiger, refrigerante, válvulas
Mixin de métodos generadores para la clase Synth de audio_engine.py.
"""
import numpy as np
import scipy.signal as sps
from .gen_utils import biquad_bandpass, biquad_lowpass, SR

class IndustrialGeneratorsMixin:
    def _geiger_interarrival(self):
        """Tiempo hasta el siguiente click (distribución exponencial, ~55 clicks/min)."""
        rate = 55.0 / 60.0   # clicks por segundo en reposo
        return float(self.geiger_rng.exponential(1.0 / rate))

    def _magma_rumble_chunk(self, t_abs, n, magma_gain):
        """
        Sub-bass pulsante que simula el movimiento lento del magma.
        Física: columnas de lava generan vibraciones de muy baja frecuencia
        (8-45Hz) con modulación AM lenta — como un motor enorme a ralentí.
        3 componentes: tono principal 18Hz, armónico 36Hz y ruido sub-bass filtrado.
        """
        if magma_gain <= 0:
            return np.zeros(n)
        t_vec = t_abs + np.arange(n) / SR
        # Tono fundamental (38Hz) con LFO de pitch muy lento (cada ~20s)
        # Integración analítica para fase continua y click-free en los límites de chunk
        omega = 2.0 * np.pi * 0.05
        phase_base = 2.0 * np.pi * (t_vec - (0.08 / omega) * np.cos(omega * t_vec))
        f1 = np.sin(phase_base * 38.0) * 0.55
        # Segundo armónico (75.5Hz) detuneado
        f2 = np.sin(phase_base * 75.5) * 0.28
        # Tercer armónico (113Hz)
        f3 = np.sin(phase_base * 113.0) * 0.12
        tone = f1 + f2 + f3
        # Ruido rosa sub-bass (30-60Hz) — textura rugosa del magma
        noise = self.magma_rng.uniform(-1.0, 1.0, n)
        b_sub, a_sub = biquad_bandpass(45.0, 1.2, SR)
        noise_filt, self.magma_zi = sps.lfilter(b_sub, a_sub, noise, zi=self.magma_zi)
        # Modulación de amplitud lenta — pulso de presión (~0.018Hz = ~56s)
        amp_lfo = 0.60 + 0.40 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.018 * t_vec))
        # Surge ocasional: "empuje" más fuerte cada 70-150s (LFO secundario)
        surge = 0.85 + 0.15 * np.sin(2.0 * np.pi * 0.008 * t_vec) ** 2
        mix = (tone * 0.70 + noise_filt * 0.30) * amp_lfo * surge * 0.065 * magma_gain

        # Generar burbujas de lava viscosa
        while self.magma_bubble_next < t_abs + n / SR:
            gap = self.rng.uniform(0.8, 2.5) / max(magma_gain, 0.1)
            b_ts = self.magma_bubble_next
            b_dur = self.rng.uniform(0.18, 0.35)
            b_f0 = self.rng.uniform(90.0, 140.0) # frecuencia inicial
            b_f1 = b_f0 * self.rng.uniform(0.45, 0.65) # frecuencia final (pitch slide down)
            b_amp = self.rng.uniform(0.015, 0.035) * magma_gain
            b_dec = self.rng.uniform(14.0, 26.0)
            self.magma_bubble_queue.append((b_ts, b_dur, b_f0, b_f1, b_amp, b_dec))
            self.magma_bubble_next += gap

        bubble_sig = np.zeros(n)
        t_samples = t_abs + np.arange(n) / SR
        still = []
        for (ts, dur, f0, f1, amp, dec) in self.magma_bubble_queue:
            lt = t_samples - ts
            mask = (lt >= 0.0) & (lt < dur)
            if np.any(mask):
                lt_act = lt[mask]
                phases = 2.0 * np.pi * (f0 * lt_act + 0.5 * (f1 - f0) * lt_act**2 / dur)
                env = (1.0 - np.exp(-120.0 * lt_act)) * np.exp(-dec * lt_act)
                bubble_sig[mask] += np.sin(phases) * amp * env
            if ts + dur > t_abs + n / SR:
                still.append((ts, dur, f0, f1, amp, dec))
        self.magma_bubble_queue = still

        return mix + bubble_sig * 0.55

    def _steam_event_chunk(self, t_abs, n, rng, next_ref, active_ref, zi_ref,
                           fc_lo, fc_hi, dur_lo, dur_hi, wait_lo, wait_hi,
                           amp_lo, amp_hi, gain, label):
        """
        Generador genérico de evento de ráfaga de gas/vapor a presión.
        Modela: acumulación de presión → liberación brusca → disipación.
        Ataque: 140ms  |  Sostenido: dur*0.55  |  Release: dur*0.45
        """
        if gain <= 0:
            return np.zeros(n), next_ref, active_ref, zi_ref
        if active_ref is None and t_abs >= next_ref:
            dur  = rng.uniform(dur_lo, dur_hi)
            fc   = rng.uniform(fc_lo, fc_hi)
            amp  = rng.uniform(amp_lo, amp_hi) * gain
            active_ref = (t_abs, dur, fc, amp)
            next_ref = t_abs + dur + rng.uniform(wait_lo, wait_hi)
        if active_ref is None:
            return np.zeros(n), next_ref, active_ref, zi_ref
        ts, dur, fc, amp = active_ref
        t_samples = t_abs + np.arange(n) / SR
        age = t_samples - ts
        mask = (age >= 0.0) & (age < dur)
        out = np.zeros(n)
        if np.any(mask):
            a = age[mask]
            noise = rng.uniform(-1.0, 1.0, int(np.sum(mask)))
            b, a_coef = biquad_bandpass(fc, 1.4, SR)
            filt, zi_ref = sps.lfilter(b, a_coef, noise, zi=zi_ref)
            atk  = 0.14
            rel  = dur * 0.45
            env  = np.where(a < atk,
                            a / atk,
                            np.clip((dur - a) / rel, 0.0, 1.0))
            # Variación de amplitud rápida (chisporroteo de presión)
            sputter = 0.82 + 0.18 * np.abs(np.sin(2.0 * np.pi * 14.0 * a))
            out[mask] = filt * env * sputter * amp
        if t_abs + n / SR >= ts + dur:
            active_ref = None
            zi_ref = np.zeros(2)
        return out, next_ref, active_ref, zi_ref

    def _steam_vent_chunk(self, t_abs, n, steam_gain):
        """Venteo de vapor volcánico oscuro y lejano."""
        out, self.steam_next, self.steam_active, self.steam_zi = self._steam_event_chunk(
            t_abs, n, self.steam_rng, self.steam_next, self.steam_active, self.steam_zi,
            fc_lo=240, fc_hi=620, dur_lo=1.4, dur_hi=4.2,
            wait_lo=18.0, wait_hi=55.0, amp_lo=0.012, amp_hi=0.032, gain=steam_gain,
            label='steam')
        return out

    def _valve_vent_chunk(self, t_abs, n, valve_gain):
        """Válvula de presión del reactor, más cuerpo que siseo."""
        out, self.valve_next, self.valve_active, self.valve_zi = self._steam_event_chunk(
            t_abs, n, self.valve_rng, self.valve_next, self.valve_active, self.valve_zi,
            fc_lo=280, fc_hi=760, dur_lo=0.7, dur_hi=1.8,
            wait_lo=35.0, wait_hi=110.0, amp_lo=0.006, amp_hi=0.018, gain=valve_gain,
            label='valve')
        return out

    def _geiger_click_chunk(self, t_abs, n, geiger_gain):
        """
        Contador Geiger: proceso de Poisson. Cada click = impulso corto (5ms).
        Física: cada decaimiento nuclear produce un ionización → pulso de corriente.
        Tasa base: ~55 clicks/min. La tasa aumenta levemente con state.energy
        pero no accedemos a state aquí — usamos variación interna lenta.
        """
        if geiger_gain <= 0:
            return np.zeros(n)
        t_end = t_abs + n / SR
        out = np.zeros(n)
        # Generar clicks que caen en este chunk
        while self.geiger_next < t_end:
            if self.geiger_next >= t_abs:
                idx = int((self.geiger_next - t_abs) * SR)
                if 0 <= idx < n:
                    # Click: impulso de 5ms con decay exponencial
                    click_len = min(int(0.004 * SR), n - idx)
                    t_click = np.arange(click_len) / SR
                    decay = np.exp(-t_click / 0.003)
                    amp = self.geiger_rng.uniform(0.006, 0.018) * geiger_gain
                    out[idx:idx + click_len] += decay * amp
            self.geiger_next += self._geiger_interarrival()
        return np.clip(out, -1.0, 1.0)

    def _coolant_flow_chunk(self, t_abs, n, coolant_gain):
        """
        Flujo de refrigerante (agua/deuterio) a alta presión en tuberías.
        Física: flujo turbulento genera ruido broadband con pico 150-400Hz.
        Amplitud modulada por micro-variaciones de presión (~0.04Hz).
        """
        if coolant_gain <= 0:
            return np.zeros(n)
        t_vec = t_abs + np.arange(n) / SR
        noise = self.coolant_rng.uniform(-1.0, 1.0, n)
        b, a = biquad_bandpass(280.0, 1.8, SR)
        filt, self.coolant_zi = sps.lfilter(b, a, noise, zi=self.coolant_zi)
        # LFO de presión: variación suave del caudal
        pressure_lfo = 0.72 + 0.28 * np.sin(2.0 * np.pi * 0.038 * t_vec + 1.1)
        return filt * pressure_lfo * 0.048 * coolant_gain

    def _thunder_chunk(self, t_abs, n, thunder_gain):
        """
        Trueno con cola de reverberación larga — el sonido más impactante de la tormenta.
        Física: el canal de plasma del rayo colapsa generando una onda de presión
        omnidireccional (20-200Hz). La estructura del edificio filtra los agudos
        y refleja el sonido varias veces → cola larga (3-8s).
        Síntesis: impulso de banda ancha (20-150Hz) con envolvente de arma de fuego,
        seguido de cola de eco decreciente.
        """
        if thunder_gain <= 0:
            return np.zeros(n)
        if self.thunder_active is None and t_abs >= self.thunder_next:
            dur = self.thunder_rng.uniform(4.0, 9.0)
            amp = self.thunder_rng.uniform(0.055, 0.110) * thunder_gain
            self.thunder_active = (t_abs, dur, amp)
            self.thunder_next = t_abs + dur + self.thunder_rng.uniform(20.0, 110.0)
        if self.thunder_active is None:
            return np.zeros(n)
        ts, dur, amp = self.thunder_active
        t_samples = t_abs + np.arange(n) / SR
        age = t_samples - ts
        mask = (age >= 0.0) & (age < dur)
        out = np.zeros(n)
        if np.any(mask):
            a = age[mask]
            noise = self.thunder_rng.uniform(-1.0, 1.0, int(np.sum(mask)))
            # Filtro paso bajo resonante: frecuencias graves del trueno
            b, a_coef = biquad_bandpass(65.0, 0.9, SR)
            filt, self.thunder_zi = sps.lfilter(b, a_coef, noise, zi=self.thunder_zi)
            # Envolvente: impacto instantáneo (2ms) + cola exponencial larga
            env = np.where(a < 0.002,
                           a / 0.002,
                           np.exp(-(a - 0.002) / (dur * 0.28)))
            # Modulación: retumbo irregular (rattle de la estructura)
            rattle = 0.65 + 0.35 * np.abs(self.thunder_rng.uniform(-1, 1, len(a)))
            out[mask] = filt * env * rattle * amp
        if t_abs + n / SR >= ts + dur:
            self.thunder_active = None
            self.thunder_zi = np.zeros(2)
        return out

