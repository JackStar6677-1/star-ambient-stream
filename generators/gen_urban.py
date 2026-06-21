"""
Generadores Urbanos/Nocturnos — perros, tráfico, motos, sapos, hojas
Mixin de métodos generadores para la clase AmbientSounds de audio_engine.py.

Síntesis basada en modelos físicos simplificados:
- Perros: impulso armónico filtrado con envolvente ladrido + vibrato gutural
- Tráfico: ruido rosa de baja frecuencia + pases Doppler
- Motos: fundamental armónica + sweep Doppler pronunciado
- Sapos: AM tonal con coro desfasado (síntesis de coro de anfibios)
- Hojas: ruido de alta frecuencia modulado por LFO de ráfaga
"""
import random
import numpy as np
import scipy.signal as sps
from .gen_utils import biquad_bandpass, biquad_lowpass, biquad_highpass, SR


class UrbanGeneratorsMixin:

    # ── Perros ladrando ────────────────────────────────────────────────────────
    def _dog_bark_chunk(self, t_abs, n, dog_gain):
        """Perros lejanos ocasionales. 2 instancias con diferente timing y pitch.
        Síntesis: impulso de ruido bandpass (150-600 Hz) + envolvente ladrido (ataque 3ms, decay 200ms)
        + armónico secundario suave. Filtrado pasa-bajos final para simular distancia."""
        if dog_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        t_end = t_abs + n / SR

        for i in range(2):
            rng  = self.dog_rng[i]
            # Programar siguiente secuencia de ladridos
            while self.dog_next[i] < t_end:
                if self.dog_next[i] >= t_abs:
                    num_barks = rng.randint(1, 5)
                    bark_gap  = rng.uniform(0.18, 0.45)
                    fund_freq = rng.uniform(200 if i == 0 else 320, 350 if i == 0 else 520)
                    amp       = rng.uniform(0.005, 0.014) * dog_gain
                    t_bark    = self.dog_next[i]
                    for b in range(num_barks):
                        self.dog_queue.append((t_bark + b * bark_gap, fund_freq, amp, rng.uniform(0.18, 0.32), np.zeros(2), np.zeros(2), np.zeros(2)))
                self.dog_next[i] = self.dog_next[i] + rng.uniform(60.0, 240.0)

        t_samples = t_abs + np.arange(n) / SR
        still = []
        for (ts, freq, amp, dur, zi_bp, zi_bp2, zi_lp) in self.dog_queue:
            age = t_samples - ts
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                a = age[mask]
                # Ruido bandpass centrado en fundamental
                noise = np.random.uniform(-1.0, 1.0, int(np.sum(mask)))
                b_bp, a_bp = biquad_bandpass(freq, 2.5, SR)
                filt, zi_bp = sps.lfilter(b_bp, a_bp, noise, zi=zi_bp)
                # Segundo armonico (voz gutural del perro)
                b_bp2, a_bp2 = biquad_bandpass(freq * 1.85, 3.5, SR)
                harm, zi_bp2 = sps.lfilter(b_bp2, a_bp2, noise, zi=zi_bp2)
                # Envolvente: ataque muy rapido, decay exponencial
                atk = 0.018
                env = np.where(a < atk,
                               a / atk,
                               np.exp(-(a - atk) / (dur * 0.55)))
                # Vibrato gutural (oscilacion de 8 Hz)
                vib = 1.0 + 0.06 * np.sin(2 * np.pi * 8.2 * a)
                raw = (filt * 0.76 + harm * 0.24) * env * vib * amp
                # LP para simular distancia (corta agudos > 800 Hz)
                b_lp, a_lp = biquad_lowpass(820.0, 0.7, SR)
                raw_lp, zi_lp = sps.lfilter(b_lp, a_lp, raw, zi=zi_lp)
                out[np.where(mask)[0]] += raw_lp
            if ts + dur > t_abs + n / SR:
                still.append((ts, freq, amp, dur, zi_bp, zi_bp2, zi_lp))
        self.dog_queue = still
        return np.tanh(out * 1.4) * 0.36

    # ── Tráfico urbano lejano ─────────────────────────────────────────────────
    def _traffic_chunk(self, t_abs, n, traffic_gain):
        """Rumble continuo de ciudad lejana + pases ocasionales de autos.
        Continuo: ruido rosa filtrado 35-110 Hz con LFO lento de densidad.
        Pase de auto: sweep Doppler suave 60→120→55 Hz, envolvente de paso."""
        if traffic_gain <= 0:
            return np.zeros(n)

        t_vec = t_abs + np.arange(n) / SR

        # ── Rumble continuo de fondo ──
        noise = self.traffic_rng.uniform(-1.0, 1.0, n).astype(np.float64)
        b1, a1 = biquad_bandpass(65.0, 1.2, SR)
        rumble, self.traffic_zi = sps.lfilter(b1, a1, noise, zi=self.traffic_zi)
        # LFO de densidad de tráfico (0.008 Hz = ciclo de ~2 minutos)
        density_lfo = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.008 * t_vec + 1.3))
        continuous = rumble * density_lfo * 0.016 * traffic_gain

        # ── Pases de autos ──
        out = continuous.copy()
        t_end = t_abs + n / SR
        while self.car_next < t_end:
            if self.car_next >= t_abs:
                dur  = self.car_rng.uniform(2.5, 5.0)
                amp  = self.car_rng.uniform(0.008, 0.022) * traffic_gain
                self.car_queue.append((self.car_next, dur, amp, np.zeros(2)))
            self.car_next += self.car_rng.uniform(18.0, 90.0)

        still = []
        for (ts, dur, amp, zi) in self.car_queue:
            age = t_vec - ts
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                a = age[mask]
                # Doppler analítico continuo
                a_mid = 0.4 * dur
                phase = np.where(
                    a < a_mid,
                    80.0 * a + 50.0 * a**2 / dur,
                    40.0 * dur + 120.0 * (a - a_mid) - 50.0 * (a - a_mid)**2 / dur
                )
                phase_rad = 2.0 * np.pi * phase
                # Envolvente: sube y baja suavemente
                env = np.sin(np.pi * a / dur) ** 1.5
                noise_c = self.traffic_rng.uniform(-1.0, 1.0, int(np.sum(mask)))
                b_c, a_c = biquad_bandpass(85.0, 1.8, SR)
                filt_c, zi = sps.lfilter(b_c, a_c, noise_c, zi=zi)
                out[np.where(mask)[0]] += (np.sin(phase_rad) * 0.35 + filt_c * 0.65) * env * amp
            if ts + dur > t_abs + n / SR:
                still.append((ts, dur, amp, zi))
        self.car_queue = still
        return out * 0.82

    # ── Motos ─────────────────────────────────────────────────────────────────
    def _moto_chunk(self, t_abs, n, moto_gain):
        """Motos pasando lejanas.
        Síntesis: serie armónica de motor (fundamental + 2f + 3f + 4f) con
        sweep Doppler pronunciado y más rápido que un auto."""
        if moto_gain <= 0:
            return np.zeros(n)

        out = np.zeros(n)
        t_vec = t_abs + np.arange(n) / SR
        t_end = t_abs + n / SR

        while self.moto_next < t_end:
            if self.moto_next >= t_abs:
                dur = self.moto_rng.uniform(1.5, 3.5)
                amp = self.moto_rng.uniform(0.006, 0.016) * moto_gain
                f0  = self.moto_rng.uniform(90, 160)   # RPM del motor en crucero
                self.moto_queue.append((self.moto_next, dur, amp, f0, np.zeros(2)))
            self.moto_next += self.moto_rng.uniform(30.0, 180.0)

        still = []
        for (ts, dur, amp, f0, zi) in self.moto_queue:
            age = t_vec - ts
            mask = (age >= 0.0) & (age < dur)
            if np.any(mask):
                a = age[mask]
                # Doppler moto analítico continuo
                a_mid = 0.35 * dur
                phase = np.where(
                    a < a_mid,
                    f0 * (0.85 * a + (5.0 / 7.0) * a**2 / dur),
                    f0 * (0.385 * dur + 1.35 * (a - a_mid) - 0.5 * (a - a_mid)**2 / dur)
                )
                # Serie armonica: motor 2 tiempos
                ph1 = 2 * np.pi * phase
                ph2 = ph1 * 2.0
                ph3 = ph1 * 3.0
                ph4 = ph1 * 4.0
                wave = (np.sin(ph1) * 0.50 +
                        np.sin(ph2) * 0.28 +
                        np.sin(ph3) * 0.14 +
                        np.sin(ph4) * 0.08)
                # Distorsion suave de escape
                wave = np.tanh(wave * 1.8) * 0.6
                env = np.sin(np.pi * a / dur) ** 1.2
                # LP para simular distancia
                b_lp, a_lp = biquad_lowpass(760.0, 0.7, SR)
                wave_lp, zi = sps.lfilter(b_lp, a_lp, wave, zi=zi)
                out[np.where(mask)[0]] += wave_lp * env * amp
            if ts + dur > t_abs + n / SR:
                still.append((ts, dur, amp, f0, zi))
        self.moto_queue = still
        return out * 0.72

    # ── Sapos / ranas nocturnas ───────────────────────────────────────────────
    def _frog_chunk(self, t_abs, n, frog_gain):
        """Coro de sapos/ranas tropicales nocturnas.
        Síntesis: tono AM pulsante por individuo. Cada sapo tiene su propia
        frecuencia (600-1400 Hz) y tempo de croar (1.5-4 Hz). Desfasados entre
        sí para crear el efecto de coro natural."""
        if frog_gain <= 0:
            return np.zeros(n)

        t_vec = t_abs + np.arange(n) / SR
        out = np.zeros(n)

        for i, (freq, rate, phase_off, amp_base) in enumerate(self.frog_params):
            # Envolvente AM: el sapo croa a 'rate' Hz con duty cycle 30%
            lfo = np.sin(2 * np.pi * rate * t_vec + phase_off)
            # Convertir seno a pulso suave: solo la parte positiva (croar)
            pulse = np.clip(lfo, 0.0, None) ** 2.5
            # Tono carrier con vibrato suave
            vib = 1.0 + 0.025 * np.sin(2 * np.pi * 5.5 * t_vec + phase_off * 0.7)
            carrier = np.sin(2 * np.pi * freq * vib * t_vec + phase_off * 3.1)
            # Segundo armonico oscuro para textura sin chirrido.
            carrier2 = np.sin(2 * np.pi * freq * 1.95 * vib * t_vec + phase_off * 2.0)
            wave = (carrier * 0.82 + carrier2 * 0.18) * pulse * amp_base * frog_gain
            # Bandpass por individuo para separacion espectral
            b_bp, a_bp = biquad_bandpass(freq, 4.0, SR)
            filt, self.frog_zi[i] = sps.lfilter(b_bp, a_bp, wave, zi=self.frog_zi[i])
            out += filt

        b_lp, a_lp = biquad_lowpass(1800.0, 0.8, SR)
        out, self.frog_lp_zi = sps.lfilter(b_lp, a_lp, out, zi=self.frog_lp_zi)
        return np.tanh(out * 1.1) * 0.42

    # ── Hojas con viento ──────────────────────────────────────────────────────
    def _leaves_chunk(self, t_abs, n, leaves_gain):
        """Hojas lejanas: ruido oscuro y filtrado, sin hiss brillante de primer plano."""
        if leaves_gain <= 0:
            return np.zeros(n)

        t_vec = t_abs + np.arange(n) / SR
        noise = self.leaves_rng.uniform(-1.0, 1.0, n).astype(np.float64)

        b_hp, a_hp = biquad_highpass(700.0, 0.6, SR)
        b_lp, a_lp = biquad_lowpass(3200.0, 0.6, SR)
        filt, self.leaves_zi1 = sps.lfilter(b_hp, a_hp, noise, zi=self.leaves_zi1)
        filt, self.leaves_zi2 = sps.lfilter(b_lp, a_lp, filt,  zi=self.leaves_zi2)

        # Ráfagas: LFO rápido modulado por envolvente de ráfaga lenta
        gust_slow = 0.35 + 0.65 * np.abs(np.sin(2 * np.pi * 0.028 * t_vec + 0.9)) ** 1.8
        gust_fast = 0.60 + 0.40 * np.abs(np.sin(2 * np.pi * 0.21 * t_vec + 2.1))
        lfo = gust_slow * gust_fast

        return filt * lfo * 0.0065 * leaves_gain
