"""
Generadores base — radio, lluvia, viento, campanas, hum, EMF, etc.
Mixin de métodos generadores para la clase Synth de audio_engine.py.
"""
import time
import math
import numpy as np
import scipy.signal as sps
from .gen_utils import biquad_bandpass, biquad_lowpass, SR

class BaseGeneratorsMixin:
    def _synthesize_mdc1200(self):
        """Genera un chirp digital estilo MDC1200 (1200 baud AFSK con tonos de 1200 y 1800 Hz)."""
        dur = 0.18
        n_samples = int(dur * SR)
        
        # 1200 baudios -> 1200 bits/seg. Duración 180ms -> 216 bits
        baud = 1200
        spb = int(SR / baud)
        freqs = np.zeros(n_samples)
        
        for bit_idx in range(int(dur * baud)):
            if bit_idx < 16:
                # Preámbulo alternante
                bit = bit_idx % 2
            else:
                bit = self.rng.choice([0, 1])
            freq = 1200 if bit == 0 else 1800
            freqs[bit_idx*spb : (bit_idx+1)*spb] = freq
            
        phase = 2 * np.pi * np.cumsum(freqs) / SR
        wave = np.sin(phase)
        
        # Filtrado de canal — mantiene el carácter radio pero con amplitud reducida
        b_wt, a_wt = biquad_bandpass(1500.0, 0.5, SR)
        filtered = sps.lfilter(b_wt, a_wt, wave)
        # El MDC queda casi subliminal: su patrón FSK era una fuente clara de pitidos.
        distorted = np.tanh(filtered * 6.0) * 0.018
        
        # Envolvente
        env = np.ones(n_samples)
        fade = int(0.01 * SR)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
        
        return distorted * env

    def _synthesize_data_packet(self):
        """Genera una ráfaga de datos FSK Bell 202 típica de telemetría (1200/2200 Hz)."""
        dur = self.rng.uniform(0.35, 0.75)
        n_samples = int(dur * SR)
        
        baud_rate = self.rng.choice([150, 300, 600])
        samples_per_bit = int(SR / baud_rate)
        freqs = np.zeros(n_samples)
        
        for i in range(0, n_samples, samples_per_bit):
            bit = self.rng.choice([0, 1])
            target_freq = 1200 if bit == 0 else 2200
            freqs[i:i+samples_per_bit] = target_freq
            
        phase = 2 * np.pi * np.cumsum(freqs) / SR
        wave = np.sin(phase)
        
        b_wt, a_wt = biquad_bandpass(1500.0, 0.45, SR)
        filtered = sps.lfilter(b_wt, a_wt, wave)
        
        distorted = np.tanh(filtered * 4.0) * 0.025
        noise = np.random.uniform(-1.0, 1.0, n_samples) * 0.006
        
        env = np.ones(n_samples)
        fade = int(0.02 * SR)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
        
        return (distorted + noise) * env

    def _synthesize_full_transmission(self, duration=0.0):
        # Elegir tipo de transmisión
        r_t = self.rng.random(); trans_type = "speech" if r_t < 0.70 else ("data" if r_t < 0.92 else "interference")
        
        if trans_type == 'speech':
            phrase_name = self.rng.choice(list(self.PHRASE_RECIPES.keys()))
            recipe = self.PHRASE_RECIPES[phrase_name]
            
            speech_len_sec = sum(dur for name, dur in recipe)
            speech_len = int(speech_len_sec * SR)
            
            has_mdc = self.rng.random() < 0.10
            mdc_len_sec = 0.20 if has_mdc else 0.0
            
            has_roger = self.rng.random() < 0.35
            roger_len_sec = 0.30 if has_roger else 0.0
            
            total_dur = 0.20 + mdc_len_sec + speech_len_sec + roger_len_sec + 0.20
            total_samples = int(total_dur * SR)
            buffer = np.zeros(total_samples)
            
            # Squelch de entrada con fade para evitar clicks
            sq_len = int(0.15 * SR)
            noise_start = np.random.uniform(-1.0, 1.0, sq_len)
            b_sq, a_sq = biquad_bandpass(1600.0, 0.6, SR)
            sq_filtered = sps.lfilter(b_sq, a_sq, noise_start) * 0.07
            fade_in = int(0.015 * SR)
            sq_filtered[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
            sq_filtered[-fade_in:] *= np.linspace(1.0, 0.0, fade_in)
            buffer[:sq_len] += sq_filtered
            
            curr_pos = sq_len
            if has_mdc:
                mdc_signal = self._synthesize_mdc1200()
                mdc_len = len(mdc_signal)
                buffer[curr_pos : curr_pos + mdc_len] += mdc_signal
                curr_pos += mdc_len
                
            speech_start = curr_pos
            if speech_len > 0:
                r_v = self.rng.random(); voice_type = "male" if r_v < 0.45 else ("female" if r_v < 0.90 else "whisper")
                
                key_times = []
                f1_keys, f2_keys, f3_keys = [], [], []
                bw1_keys, bw2_keys, bw3_keys = [], [], []
                buzz_keys, noise_keys, amp_keys = [], [], []
                
                first_p = self.PHONEMES[recipe[0][0]]
                key_times.append(0.0)
                f1_keys.append(first_p['formants'][0])
                f2_keys.append(first_p['formants'][1])
                f3_keys.append(first_p['formants'][2])
                bw1_keys.append(first_p['bandwidths'][0])
                bw2_keys.append(first_p['bandwidths'][1])
                bw3_keys.append(first_p['bandwidths'][2])
                buzz_keys.append(first_p['buzz'])
                noise_keys.append(first_p['noise'])
                amp_keys.append(first_p.get('amp_scale', 1.0))
                
                t_curr = 0.0
                for name, dur in recipe:
                    p = self.PHONEMES[name]
                    t1 = t_curr + 0.15 * dur
                    key_times.append(t1)
                    f1_keys.append(p['formants'][0])
                    f2_keys.append(p['formants'][1])
                    f3_keys.append(p['formants'][2])
                    bw1_keys.append(p['bandwidths'][0])
                    bw2_keys.append(p['bandwidths'][1])
                    bw3_keys.append(p['bandwidths'][2])
                    buzz_keys.append(p['buzz'])
                    noise_keys.append(p['noise'])
                    amp_keys.append(p.get('amp_scale', 1.0))
                    
                    t2 = t_curr + 0.85 * dur
                    key_times.append(t2)
                    f1_keys.append(p['formants'][0])
                    f2_keys.append(p['formants'][1])
                    f3_keys.append(p['formants'][2])
                    bw1_keys.append(p['bandwidths'][0])
                    bw2_keys.append(p['bandwidths'][1])
                    bw3_keys.append(p['bandwidths'][2])
                    buzz_keys.append(p['buzz'])
                    noise_keys.append(p['noise'])
                    amp_keys.append(p.get('amp_scale', 1.0))
                    
                    t_curr += dur
                    
                key_times.append(speech_len_sec)
                last_p = self.PHONEMES[recipe[-1][0]]
                f1_keys.append(last_p['formants'][0])
                f2_keys.append(last_p['formants'][1])
                f3_keys.append(last_p['formants'][2])
                bw1_keys.append(last_p['bandwidths'][0])
                bw2_keys.append(last_p['bandwidths'][1])
                bw3_keys.append(last_p['bandwidths'][2])
                buzz_keys.append(last_p['buzz'])
                noise_keys.append(last_p['noise'])
                amp_keys.append(last_p.get('amp_scale', 1.0))
                
                t_speech = np.arange(speech_len) / SR
                f1_t = np.interp(t_speech, key_times, f1_keys)
                f2_t = np.interp(t_speech, key_times, f2_keys)
                f3_t = np.interp(t_speech, key_times, f3_keys)
                bw1_t = np.interp(t_speech, key_times, bw1_keys)
                bw2_t = np.interp(t_speech, key_times, bw2_keys)
                bw3_t = np.interp(t_speech, key_times, bw3_keys)
                buzz_t = np.interp(t_speech, key_times, buzz_keys)
                noise_t = np.interp(t_speech, key_times, noise_keys)
                amp_t = np.interp(t_speech, key_times, amp_keys)
                
                if voice_type == 'female':
                    f0 = self.rng.uniform(185, 230)
                    f1_t *= 1.15
                    f2_t *= 1.15
                    f3_t *= 1.15
                elif voice_type == 'male':
                    f0 = self.rng.uniform(95, 125)
                else: # whisper
                    f0 = 100
                    buzz_t *= 0.0
                    noise_t *= 1.4
                    
                declination = np.linspace(1.0, 0.82, speech_len)
                intonation = 1.0 - 0.08 * (t_speech / max(speech_len_sec, 0.01)) + 0.04 * np.sin(2 * np.pi * 1.5 * t_speech)
                pitch_t = f0 * declination * intonation * (1.0 + 0.02 * np.sin(2 * np.pi * 5.8 * t_speech))
                phase_t = 2 * np.pi * np.cumsum(pitch_t) / SR
                
                buzz = np.zeros(speech_len)
                n_harmonics = 12 if voice_type == 'male' else 8
                for h in range(1, n_harmonics + 1):
                    buzz += (1.0 / (h ** 1.1)) * np.sin(h * phase_t)
                max_b = np.max(np.abs(buzz))
                if max_b > 0:
                    buzz /= max_b
                    
                noise_sig = np.random.uniform(-1.0, 1.0, speech_len)
                src = buzz_t * buzz * 0.75 + noise_t * noise_sig * 0.25
                
                speech_sig = np.zeros(speech_len)
                block_size = 256
                zi1, zi2, zi3 = np.zeros(2), np.zeros(2), np.zeros(2)
                
                for start in range(0, speech_len, block_size):
                    end = min(start + block_size, speech_len)
                    if start == end:
                        break
                    
                    fc1 = np.mean(f1_t[start:end])
                    fc2 = np.mean(f2_t[start:end])
                    fc3 = np.mean(f3_t[start:end])
                    bw1 = np.mean(bw1_t[start:end])
                    bw2 = np.mean(bw2_t[start:end])
                    bw3 = np.mean(bw3_t[start:end])
                    
                    Q1 = max(0.5, fc1 / bw1)
                    Q2 = max(0.5, fc2 / bw2)
                    Q3 = max(0.5, fc3 / bw3)
                    
                    b1, a1 = biquad_bandpass(fc1, Q1, SR)
                    b2, a2 = biquad_bandpass(fc2, Q2, SR)
                    b3, a3 = biquad_bandpass(fc3, Q3, SR)
                    
                    y1, zi1 = sps.lfilter(b1, a1, src[start:end], zi=zi1)
                    y2, zi2 = sps.lfilter(b2, a2, src[start:end], zi=zi2)
                    y3, zi3 = sps.lfilter(b3, a3, src[start:end], zi=zi3)
                    
                    block_out = y1 * 0.45 + y2 * 0.32 + y3 * 0.23
                    block_out *= amp_t[start:end]
                    speech_sig[start:end] = block_out
                
                static_background = np.random.uniform(-1.0, 1.0, speech_len) * self.rng.uniform(0.006, 0.012)
                speech_sig += static_background
                
                b_wt, a_wt = biquad_bandpass(1100.0, 0.42, SR)
                speech_filtered = sps.lfilter(b_wt, a_wt, speech_sig)
                speech_distorted = np.tanh(speech_filtered * 11.0) * 0.16
                
                # Signal dropouts
                if self.rng.random() < 0.25:
                    dropout_start = self.rng.randint(int(0.2 * speech_len), int(0.7 * speech_len))
                    dropout_len = self.rng.randint(int(0.03 * SR), int(0.09 * SR))
                    if dropout_start + dropout_len < speech_len:
                        fade_do = int(0.005 * SR)
                        do_env = np.ones(dropout_len)
                        do_env[:fade_do] = np.linspace(1.0, 0.0, fade_do)
                        do_env[-fade_do:] = np.linspace(0.0, 1.0, fade_do)
                        do_env[fade_do:-fade_do] = 0.0
                        speech_distorted[dropout_start:dropout_start+dropout_len] *= do_env
                
                interf_freq = self.rng.choice([580, 800, 1100])
                interference = np.sin(2 * np.pi * interf_freq * t_speech) * 0.002
                
                # Fade in/out para la voz
                voice_fade = int(0.01 * SR)
                speech_distorted[:voice_fade] *= np.linspace(0.0, 1.0, voice_fade)
                speech_distorted[-voice_fade:] *= np.linspace(1.0, 0.0, voice_fade)
                
                buffer[speech_start:speech_start + speech_len] += speech_distorted + interference
                curr_pos += speech_len
                
            if has_roger:
                beep1_dur = int(0.08 * SR)
                t_b1 = np.arange(beep1_dur) / SR
                buffer[curr_pos : curr_pos + beep1_dur] += np.sin(2 * np.pi * 1150 * t_b1) * 0.022 * np.sin(np.pi * (t_b1/0.08))
                
                beep2_start = curr_pos + int(0.09 * SR)
                beep2_dur = int(0.08 * SR)
                t_b2 = np.arange(beep2_dur) / SR
                buffer[beep2_start : beep2_start + beep2_dur] += np.sin(2 * np.pi * 880 * t_b2) * 0.022 * np.sin(np.pi * (t_b2/0.08))
                curr_pos += int(0.18 * SR)
                
            sq_end_start = curr_pos
            sq_end_len = len(buffer) - sq_end_start
            if sq_end_len > 0:
                t_sq = np.arange(sq_end_len) / SR
                noise_end = np.random.uniform(-1.0, 1.0, sq_end_len)
                b_sq_e, a_sq_e = biquad_bandpass(1400.0, 0.5, SR)
                sq_tail = sps.lfilter(b_sq_e, a_sq_e, noise_end) * 0.08 * np.exp(-25 * t_sq)
                sq_fade_in = int(0.005 * SR)
                if len(sq_tail) > sq_fade_in:
                    sq_tail[:sq_fade_in] *= np.linspace(0.0, 1.0, sq_fade_in)
                buffer[sq_end_start:] += sq_tail
                
        elif trans_type == 'data':
            data_sig = self._synthesize_mdc1200() if self.rng.random() < 0.5 else self._synthesize_data_packet()
            total_samples = len(data_sig) + int(0.15 * SR)
            buffer = np.zeros(total_samples)
            buffer[int(0.05 * SR) : int(0.05 * SR) + len(data_sig)] += data_sig
            
            sq_start = int(0.05 * SR) + len(data_sig)
            sq_len = len(buffer) - sq_start
            if sq_len > 0:
                t_sq = np.arange(sq_len) / SR
                noise_end = np.random.uniform(-1.0, 1.0, sq_len)
                b_sq_e, a_sq_e = biquad_bandpass(1400.0, 0.5, SR)
                sq_tail = sps.lfilter(b_sq_e, a_sq_e, noise_end) * 0.06 * np.exp(-35 * t_sq)
                sq_fade_in = int(0.005 * SR)
                if len(sq_tail) > sq_fade_in:
                    sq_tail[:sq_fade_in] *= np.linspace(0.0, 1.0, sq_fade_in)
                buffer[sq_start:] += sq_tail
                
        else: # interference — ruido de banda estrecha, sin tono sweep (el sweep sonaba a sirena)
            dur = self.rng.uniform(1.2, 3.0)
            total_samples = int(dur * SR)
            buffer = np.zeros(total_samples)

            # Estática de radio: ruido filtrado en banda media, sin tono puro
            noise = np.random.uniform(-1.0, 1.0, total_samples)
            fc_interf = self.rng.uniform(1000, 2500)
            b_n, a_n = biquad_bandpass(fc_interf, 0.35, SR)
            noise_filtered = sps.lfilter(b_n, a_n, noise) * 0.05

            # AM lenta (modulación de amplitud a 3-8 Hz) — suena a "dropout" de señal
            t_interf = np.arange(total_samples) / SR
            am_rate = self.rng.uniform(3.0, 8.0)
            am = 0.5 + 0.5 * np.abs(np.sin(np.pi * am_rate * t_interf))

            signal = noise_filtered * am

            env = np.ones(total_samples)
            fade = int(0.08 * SR)
            env[:fade] = np.linspace(0.0, 1.0, fade)
            env[-fade:] = np.linspace(1.0, 0.0, fade)

            buffer += signal * env
            
        return buffer

    def _radio_chunk(self, t_abs, n, radio_gain):
        if radio_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        
        if not self.radio_active and t_abs >= self.radio_next_start:
            self.radio_active = True
            self.radio_buffer = self._synthesize_full_transmission()
            self.radio_trans_dur = len(self.radio_buffer) / SR
            self.radio_end = t_abs + self.radio_trans_dur
            self.radio_next_start = self.radio_end + self.rng.uniform(40.0, 90.0)

        if self.radio_active:
            trans_start_t = self.radio_end - self.radio_trans_dur
            start_sample = int((t_abs - trans_start_t) * SR)
            
            if start_sample < 0:
                pad_before = -start_sample
                buf_start = 0
            else:
                pad_before = 0
                buf_start = start_sample
                
            samples_needed = n - pad_before
            buf_len = len(self.radio_buffer)
            
            if buf_start >= buf_len:
                self.radio_active = False
                return out
                
            buf_end = min(buf_start + samples_needed, buf_len)
            samples_copied = buf_end - buf_start
            
            out[pad_before:pad_before + samples_copied] = self.radio_buffer[buf_start:buf_end] * radio_gain
            
            if buf_end >= buf_len:
                self.radio_active = False
                
        return out

    def _fan_chunk(self, t_abs, n, fan_gain):
        if fan_gain <= 0:
            return np.zeros(n)
        noise = self.fan_rng.uniform(-1, 1, n).astype(np.float64)
        if self.fan_zi1 is None:
            self.fan_zi1 = sps.sosfilt_zi(self.fan_sos1) * noise[0]
            self.fan_zi2 = sps.sosfilt_zi(self.fan_sos2) * noise[0]
        f1, self.fan_zi1 = sps.sosfilt(self.fan_sos1, noise, zi=self.fan_zi1)
        f2, self.fan_zi2 = sps.sosfilt(self.fan_sos2, noise, zi=self.fan_zi2)
        t = t_abs + np.arange(n) / SR
        lfo = 0.9 + 0.10 * np.sin(2*np.pi*0.025*t)
        return (f1 * 0.55 + f2 * 0.45) * lfo * 0.018 * fan_gain

    def _step_chunk(self, t_abs, n, density, scene=None):
        out = np.zeros(n)
        while self.step_next < t_abs + n / SR:
            gap  = self.rng.uniform(5, 14) / max(density, 0.1)
            amp  = self.rng.uniform(0.020, 0.048)
            freq = self.rng.uniform(52, 88)
            dec  = self.rng.uniform(28, 45)
            self.step_queue.append((self.step_next, amp, freq, dec))
            self.step_queue.append((self.step_next + self.rng.uniform(0.18, 0.28),
                                    amp * 0.70, freq * self.rng.uniform(0.88, 1.12), dec))
            self.step_next += gap
        
        t_samples = t_abs + np.arange(n) / SR
        still = []
        dur = 1.5
        for (ts, amp, freq, dec) in self.step_queue:
            lt = t_samples - ts
            mask = (lt >= 0.0) & (lt < dur)
            if np.any(mask):
                lt_act = lt[mask]
                if scene == 'nieve':
                    # Crunchy snow footsteps
                    noise = np.random.uniform(-1.0, 1.0, len(lt_act))
                    b_bp, a_bp = biquad_bandpass(950.0, 0.75, SR)
                    crunch = sps.lfilter(b_bp, a_bp, noise)
                    wave = crunch * amp * 3.5 * np.exp(-12.0 * lt_act)
                else:
                    wave = np.sin(2 * np.pi * freq * lt_act) * amp * np.exp(-dec * lt_act)
                out[mask] += wave
            if ts + dur > t_abs + n / SR:
                still.append((ts, amp, freq, dec))
        self.step_queue = still
        return out

    def _ding_chunk(self, t_abs, n, density, ding_gain):
        if ding_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        while self.ding_next < t_abs + n / SR:
            freq  = self.rng.uniform(900, 3200)
            amp   = self.rng.uniform(0.007, 0.022) * ding_gain
            decay = self.rng.uniform(4, 12)
            self.ding_active.append((self.ding_next, freq, amp, decay))
            self.ding_next += self.rng.uniform(20, 70) / max(density, 0.1)
        
        t_samples = t_abs + np.arange(n) / SR
        still = []
        dur = 3.0
        for (ts, freq, amp, decay) in self.ding_active:
            lt = t_samples - ts
            mask = (lt >= 0.0) & (lt < dur)
            if np.any(mask):
                lt_act = lt[mask]
                wave = (np.sin(2 * np.pi * freq * lt_act) * 0.60 +
                        np.sin(2 * np.pi * freq * 2.76 * lt_act) * 0.40) * amp * np.exp(-decay * lt_act)
                out[mask] += wave
            if ts + dur > t_abs + n / SR:
                still.append((ts, freq, amp, decay))
        self.ding_active = still
        return out

    def _cart_chunk(self, t_abs, n):
        out = np.zeros(n)
        if self.cart_active is None and t_abs >= self.cart_next:
            dur  = self.rng.uniform(3, 7)
            fb   = self.rng.uniform(38, 68)
            self.cart_active = (t_abs, dur, fb)
            self.cart_next   = t_abs + dur + self.rng.uniform(100, 280)
        if self.cart_active:
            ts, dur, fb = self.cart_active
            for i in range(n):
                ta  = t_abs + i / SR
                age = ta - ts
                if age < 0 or age > dur: continue
                if   age < 0.5:      env = age / 0.5
                elif age > dur - 0.5: env = (dur - age) / 0.5
                else:                 env = 1.0
                rumble = self.rng.uniform(-1, 1) * 0.030
                wheel  = math.sin(2*math.pi*(fb + 4*math.sin(age*0.6))*age) * 0.022
                out[i] += (rumble + wheel) * env * 0.45
            if t_abs > ts + dur:
                self.cart_active = None
        return out

    def _water_chunk(self, t_abs, n, water_gain):
        if water_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)

        # Flujo continuo: ruido bandpass modulado
        noise = self.water_rng.uniform(-1, 1, n).astype(np.float64)
        if self.water_zi is None:
            self.water_zi = sps.sosfilt_zi(self.water_sos) * 0.0

        flow_raw, self.water_zi = sps.sosfilt(self.water_sos, noise, zi=self.water_zi)

        # LFO lento para flujo de agua que varía
        t = np.arange(n) / SR
        lfo_flow = 0.6 + 0.4 * np.abs(np.sin(2*np.pi*0.04*(t + self.water_lfo_t)))
        self.water_lfo_t += n / SR
        out += flow_raw * lfo_flow * 0.015 * water_gain

        # Burbujas: click cortos con frecuencia aleatoria
        while self.bubble_next < t_abs + n / SR:
            freq  = self.rng.uniform(180, 900)
            amp   = self.rng.uniform(0.020, 0.055) * water_gain
            decay = self.rng.uniform(35, 100)
            self.bubble_queue.append((self.bubble_next, freq, amp, decay))
            self.bubble_next += self.rng.uniform(0.08, 0.5) / max(water_gain, 0.1)

        t_samples = t_abs + np.arange(n) / SR
        still = []
        dur = 0.08
        for (ts, freq, amp, decay) in self.bubble_queue:
            lt = t_samples - ts
            mask = (lt >= 0.0) & (lt < dur)
            if np.any(mask):
                lt_act = lt[mask]
                wave = np.sin(2 * np.pi * freq * lt_act) * amp * np.exp(-decay * lt_act)
                out[mask] += wave
            if ts + dur > t_abs + n / SR:
                still.append((ts, freq, amp, decay))
        self.bubble_queue = still
        return np.clip(out, -0.3, 0.3)

    def _knock_chunk(self, t_abs, n, scene):
        out = np.zeros(n)
        if t_abs < self.knock_next:
            return out
        if not self.knock_queue:
            # Programar 2-4 golpes
            n_hits = self.rng.randint(2, 5)
            is_metal = scene in ('orbital', 'submarina')
            freq_base = self.rng.uniform(180, 320) if is_metal else self.rng.uniform(80, 180)
            amp_base  = self.rng.uniform(0.04, 0.09)
            for i in range(n_hits):
                delay = i * self.rng.uniform(0.28, 0.45)
                amp   = amp_base * self.rng.uniform(0.7, 1.0)
                decay = 22 if is_metal else 14
                self.knock_queue.append((t_abs + delay, freq_base, amp, decay))
            # Próximo knock: 2-8 minutos
            self.knock_next = t_abs + self.rng.uniform(55, 160)

        t_samples = t_abs + np.arange(n) / SR
        still = []
        dur = 0.35
        for (ts, freq, amp, decay) in self.knock_queue:
            lt = t_samples - ts
            mask = (lt >= 0.0) & (lt < dur)
            if np.any(mask):
                lt_act = lt[mask]
                wave = (np.sin(2 * np.pi * freq * lt_act) * 0.65 +
                        np.sin(2 * np.pi * freq * 1.41 * lt_act) * 0.35) * amp * np.exp(-decay * lt_act)
                out[mask] += wave
            if ts + dur > t_abs + n / SR:
                still.append((ts, freq, amp, decay))
        self.knock_queue = still
        return out

    def _rain_chunk(self, t_abs, n, rain_gain, is_snow=False, rain_glass=False):
        """Lluvia multicapa: rumble bajo + hiss medio + gotas."""
        if rain_gain <= 0:
            return np.zeros(n)
        t   = np.arange(n) / SR

        if is_snow:
            # Nieve: hiss muy suave de alta frecuencia + susurro bajo
            w1 = self.rain_rng.uniform(-1, 1, n).astype(np.float64)
            if self.snow_zi1 is None:
                self.snow_zi1 = sps.sosfilt_zi(self.snow_sos1) * 0.0
            layer1, self.snow_zi1 = sps.sosfilt(self.snow_sos1, w1, zi=self.snow_zi1)

            w2 = self.rain_rng.uniform(-1, 1, n).astype(np.float64)
            if self.snow_zi2 is None:
                self.snow_zi2 = sps.sosfilt_zi(self.snow_sos2) * 0.0
            layer2, self.snow_zi2 = sps.sosfilt(self.snow_sos2, w2, zi=self.snow_zi2)

            lfo = 0.70 + 0.30 * np.abs(np.sin(2*np.pi*0.018*(t + self.rain_lfo_t)))
            self.rain_lfo_t += n / SR
            out = (layer1 * 0.55 + layer2 * 0.45) * lfo * 0.020 * rain_gain
            return np.clip(out, -0.25, 0.25)

        # ── LLUVIA: 3 sub-capas ────────────────────────────────────────────
        # Capa 1: rumble bajo (200-900 Hz) — la percusión de gotas en superficie
        w1 = self.rain_rng.uniform(-1, 1, n).astype(np.float64)
        if self.rain_zi1 is None:
            self.rain_zi1 = sps.sosfilt_zi(self.rain_sos1) * 0.0
        layer1, self.rain_zi1 = sps.sosfilt(self.rain_sos1, w1, zi=self.rain_zi1)

        # Capa 2: hiss medio (900-4000 Hz) — el "ssh" de la lluvia constante
        w2 = self.rain_rng.uniform(-1, 1, n).astype(np.float64)
        if self.rain_zi2 is None:
            self.rain_zi2 = sps.sosfilt_zi(self.rain_sos2) * 0.0
        layer2, self.rain_zi2 = sps.sosfilt(self.rain_sos2, w2, zi=self.rain_zi2)

        # LFO de intensidad lenta (0.022 Hz ~ 45s)
        lfo = 0.68 + 0.32 * np.abs(np.sin(2*np.pi*0.022*(t + self.rain_lfo_t)))
        self.rain_lfo_t += n / SR

        # Mezcla: más bajo que antes, más rumble, menos agudo
        hiss = (layer1 * 0.65 + layer2 * 0.35) * lfo
        out  = hiss * 0.028 * rain_gain

        # Gotas individuales — solo para lluvia intensa
        if rain_gain > 0.3:
            effective_rate = 1.0 / max(rain_gain * 0.8, 0.05)
            while self.drop_next < t_abs + n / SR:
                freq = self.rng.uniform(600, 2800)
                amp  = self.rng.uniform(0.004, 0.014) * rain_gain
                dec  = self.rng.uniform(60, 200)
                self.drop_queue.append((self.drop_next, freq, amp, dec))
                self.drop_next += self.rng.uniform(0.015, 0.08) * effective_rate

            t_samples = t_abs + np.arange(n) / SR
            still = []
            dur = 0.12
            for (ts, freq, amp, dec) in self.drop_queue:
                lt = t_samples - ts
                
                # Componente principal
                mask_main = (lt >= 0.0) & (lt < 0.05)
                if np.any(mask_main):
                    lt_m = lt[mask_main]
                    out[mask_main] += (np.sin(2 * np.pi * freq * lt_m) * 0.7 +
                                       np.sin(2 * np.pi * freq * 0.618 * lt_m) * 0.3) * amp * np.exp(-dec * lt_m)
                                       
                # Resonancia de vidrio
                if rain_glass:
                    mask_glass = (lt >= 0.0) & (lt < 0.12)
                    if np.any(mask_glass):
                        lt_g = lt[mask_glass]
                        g_freq = 1400.0 + freq * 0.18
                        out[mask_glass] += np.sin(2 * np.pi * g_freq * lt_g) * amp * 0.35 * np.exp(-40.0 * lt_g)
                        
                if ts + dur > t_abs + n / SR:
                    still.append((ts, freq, amp, dec))
            self.drop_queue = still

        return np.clip(out, -0.30, 0.30)

    def _elec_hum_chunk(self, t_abs, n, elec_hum):
        """Zumbido de transformador; 'dual' mezcla familias de 50 y 60 Hz."""
        if not elec_hum:
            return np.zeros(n)
        t     = np.arange(n) / SR
        t_abs_samples = t_abs + t
        freqs = [50.0, 100.0, 150.0, 200.0]
        amps  = [0.0022, 0.0012, 0.0006, 0.0003]
        out   = np.zeros(n)
        for i, (f, a) in enumerate(zip(freqs, amps)):
            out += np.sin(self.hum_ph[i] + 2*np.pi*f*t) * a
            self.hum_ph[i] = (self.hum_ph[i] + 2*np.pi*f*n/SR) % (2*np.pi)
        if elec_hum == 'dual':
            for f, a in zip([60.0, 120.0, 180.0, 240.0], [0.0017, 0.0009, 0.00045, 0.00022]):
                out += np.sin(2*np.pi*f*t_abs_samples) * a
        return out * (0.95 + 0.05 * np.sin(2*np.pi*0.05*t_abs_samples))

    def _blizzard_chunk(self, t_abs, n, wind_gain):
        """Viento multicapa: roar grave (40-200Hz) + whoosh medio (150-500Hz)."""
        if wind_gain <= 0:
            return np.zeros(n)
        t   = np.arange(n) / SR

        # Capa 1: roar profundo del viento (40-200 Hz)
        w1 = self.gust_rng.uniform(-1, 1, n).astype(np.float64)
        if self.gust_zi1 is None:
            self.gust_zi1 = sps.sosfilt_zi(self.wind_sos1) * 0.0
        roar, self.gust_zi1 = sps.sosfilt(self.wind_sos1, w1, zi=self.gust_zi1)

        # Capa 2: whoosh (150-500 Hz) — el silbido / ráfaga
        w2 = self.gust_rng.uniform(-1, 1, n).astype(np.float64)
        if self.gust_zi2 is None:
            self.gust_zi2 = sps.sosfilt_zi(self.wind_sos2) * 0.0
        whoosh, self.gust_zi2 = sps.sosfilt(self.wind_sos2, w2, zi=self.gust_zi2)

        # LFO lento de ráfaga
        lfo_slow = 0.50 + 0.50 * np.abs(np.sin(2*np.pi*0.055*(t + self.gust_lfo_t)))
        lfo_fast = 0.80 + 0.20 * np.sin(2*np.pi*0.31*(t + self.gust_lfo_t))
        self.gust_lfo_t += n / SR

        lfo = lfo_slow * lfo_fast
        mix = (roar * 0.70 + whoosh * 0.30) * lfo
        return mix * 0.032 * wind_gain

    def _air_whistle_chunk(self, t_abs, n, whistle_gain, wind_spd, scene=None):
        """Aire fuerte sin silbido tonal: whoosh oscuro, ancho y sin resonadores finos."""
        if whistle_gain <= 0:
            return np.zeros(n)
        t = t_abs + np.arange(n) / SR
        wind_push = 0.45 + min(max(wind_spd, 0.0) / 28.0, 1.0) * 0.75
        breath_lfo = (
            0.42
            + 0.38 * np.sin(2*np.pi*0.027*t + 0.7) ** 2
            + 0.20 * np.sin(2*np.pi*0.071*t + 2.1) ** 2
        )
        noise = self.whistle_rng.uniform(-1.0, 1.0, n).astype(np.float64)

        # Dos bandas anchas: cuerpo de aire y presión de estructura. Q bajo = sin tono puro.
        b_body, a_body = biquad_bandpass(330.0, 0.48, SR)
        body, self.whistle_zi[0] = sps.lfilter(b_body, a_body, noise, zi=self.whistle_zi[0])

        noise2 = self.whistle_rng.uniform(-1.0, 1.0, n).astype(np.float64)
        b_air, a_air = biquad_bandpass(760.0, 0.42, SR)
        air, self.whistle_zi[1] = sps.lfilter(b_air, a_air, noise2, zi=self.whistle_zi[1])

        # Whistle real con Q alto (silbido del viento muy real en la montaña)
        whistle_sig = np.zeros(n)
        block_size = 256
        zi_whistle = self.whistle_real_zi
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            if start == end:
                break
            t_block = t[start:end]
            # Frecuencia fluctuante del silbido
            fc = 900.0 + 350.0 * np.sin(2.0 * np.pi * 0.08 * t_block[0]) + 150.0 * np.sin(2.0 * np.pi * 0.45 * t_block[0])
            fc += self.rng.uniform(-30.0, 30.0)
            
            b_w, a_w = biquad_bandpass(fc, 22.0, SR)
            block_filt, zi_whistle = sps.lfilter(b_w, a_w, noise[start:end], zi=zi_whistle)
            whistle_sig[start:end] = block_filt
        self.whistle_real_zi = zi_whistle

        whistle_real = whistle_sig * (0.65 if scene == 'montana' else 0.20)
        out = body * 0.78 + air * 0.22 + whistle_real
        return np.tanh(out * 1.2) * breath_lfo * whistle_gain * wind_push * 0.014

    def _creak_chunk(self, t_abs, n, creak_gain):
        """El edificio cruje con el viento — nieve / submarino."""
        if creak_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        while self.creak_next < t_abs + n / SR:
            freq = self.creak_rng.uniform(170, 380)
            amp  = self.creak_rng.uniform(0.025, 0.065) * creak_gain
            dur  = self.creak_rng.uniform(0.25, 1.0)
            self.creak_queue.append((self.creak_next, freq, amp, dur))
            self.creak_next += self.creak_rng.uniform(12, 40)
        
        t_samples = t_abs + np.arange(n) / SR
        still = []
        for (ts, freq, amp, dur) in self.creak_queue:
            lt = t_samples - ts
            mask = (lt >= 0.0) & (lt < dur)
            if np.any(mask):
                lt_act = lt[mask]
                # Integración analítica de la fase del glide lineal
                phase = 2.0 * np.pi * freq * lt_act * (1.0 - 0.14 * lt_act / max(dur, 0.001))
                env = np.sin(np.pi * lt_act / max(dur, 0.001)) ** 0.7
                out[mask] += np.sin(phase) * amp * env
            if ts + dur > t_abs + n / SR:
                still.append((ts, freq, amp, dur))
        self.creak_queue = still
        return out

    def _corona_crackle_chunk(self, t_abs, n, corona_gain):
        """Descargas corona de alta tensión: chichirreo eléctrico de fondo, no muy fuerte."""
        if corona_gain <= 0:
            return np.zeros(n)
        
        # 1. Ruido base de alta frecuencia (el "hiss" de la ionización)
        noise = self.corona_np_rng.uniform(-1.0, 1.0, n).astype(np.float64)
        if self.corona_zi is None:
            self.corona_zi = sps.sosfilt_zi(self.corona_sos) * 0.0
        filtered_noise, self.corona_zi = sps.sosfilt(self.corona_sos, noise, zi=self.corona_zi)
        
        t_samples = t_abs + np.arange(n) / SR
        # Modulación de amplitud a 100 Hz (doble de la frecuencia de la red de 50 Hz en Chile)
        mod = np.abs(np.sin(2 * np.pi * 50.0 * t_samples)) ** 12
        sizzle = filtered_noise * (0.15 + 0.85 * mod)
        
        # 2. Chispas o clicks aleatorios (transitorios) alineados con el pico de tensión AC
        spark_prob = 45.0 / SR  # chispas raras, no fritura constante
        spark_prob_modulated = spark_prob * (0.05 + 0.95 * mod)
        sparks = (self.corona_np_rng.uniform(0.0, 1.0, n) < spark_prob_modulated).astype(np.float64)
        spark_amps = self.corona_np_rng.exponential(scale=0.012, size=n) * sparks
        
        # Decaimiento exponencial persistente (para evitar clicks entre chunks)
        if self.spark_zi is None:
            self.spark_zi = np.zeros(1)
        spark_decayed, self.spark_zi = sps.lfilter([1.0], [1.0, -0.93], spark_amps, zi=self.spark_zi)
        
        # Filtrado paso alto de las chispas para hacerlas agudas y secas
        if self.spark_hp_zi is None:
            self.spark_hp_zi = sps.sosfilt_zi(self.spark_hp_sos) * 0.0
        spark_decayed_hp, self.spark_hp_zi = sps.sosfilt(self.spark_hp_sos, spark_decayed, zi=self.spark_hp_zi)
        
        # 3. Mezclar sizzle base con chispas transitorias — más suave, sin picos agudos
        out = (sizzle * 0.18 + spark_decayed_hp * 0.10) * corona_gain * 0.48
        return np.tanh(out)

    def _city_chunk(self, t_abs, n, city_gain):
        """Rumble distante de sala de máquinas — orbital."""
        if city_gain <= 0:
            return np.zeros(n)
        noise = self.city_rng.uniform(-1, 1, n).astype(np.float64)
        if self.city_zi is None:
            self.city_zi = sps.sosfilt_zi(self.city_sos) * 0.0
        rumble, self.city_zi = sps.sosfilt(self.city_sos, noise, zi=self.city_zi)
        t = np.arange(n) / SR
        lfo = 0.70 + 0.30 * np.sin(2*np.pi*0.031*(t + t_abs))
        return rumble * lfo * 0.016 * city_gain

    def _clock_chunk(self, t_abs, n, clock_night):
        """Tick casi inaudible — solo entre las 22h y las 5AM."""
        if not clock_night:
            return np.zeros(n)
        if 5 <= time.localtime().tm_hour < 22:
            return np.zeros(n)
        out = np.zeros(n)
        for beat in range(int(t_abs), int(t_abs + n / SR) + 2):
            s = int((float(beat) - t_abs) * SR)
            if 0 <= s < n:
                dur = min(int(0.014 * SR), n - s)
                lt  = np.arange(dur) / SR
                out[s:s+dur] += np.sin(2*np.pi*3400*lt) * 0.007 * np.exp(-420*lt)
        return out

    def _cassette_chunk(self, n, cass_level):
        """Textura de cinta de cassette — solo de noche (18h-5AM)."""
        if cass_level <= 0:
            return np.zeros(n)
        noise = self.cass_rng.uniform(-1, 1, n).astype(np.float64)
        if self.cass_zi is None:
            self.cass_zi = sps.sosfilt_zi(self.cass_sos) * noise[0]
        hiss, self.cass_zi = sps.sosfilt(self.cass_sos, noise, zi=self.cass_zi)
        if 5 <= time.localtime().tm_hour < 18:
            return np.zeros(n)
        return hiss * cass_level

    def _emf_chunk(self, t_abs, n, emf_gain):
        """Perturbación electromagnética subliminal — rumble sub-sónico que se siente, no se escucha como nota.
        Rango 12-80 Hz: debajo del umbral melódico para no sonar a instrumento musical."""
        if emf_gain <= 0:
            return np.zeros(n)
        while self.emf_next < t_abs + n / SR:
            f_start = self.emf_rng.uniform(12, 55)
            f_end   = self.emf_rng.uniform(18, 80)
            dur     = self.emf_rng.uniform(8.0, 25.0)
            amp     = self.emf_rng.uniform(0.012, 0.030) * emf_gain
            self.emf_queue.append((self.emf_next, f_start, f_end, dur, amp))
            self.emf_next += dur + self.emf_rng.uniform(30, 120)
        out = np.zeros(n)
        t_samples = t_abs + np.arange(n) / SR
        still = []
        for (ts, f0, f1, dur, amp) in self.emf_queue:
            lt = t_samples - ts
            mask = (lt >= 0.0) & (lt < dur)
            if np.any(mask):
                max_idx = np.where(mask)[0][-1]
                total_samples_from_start = int((t_samples[max_idx] - ts) * SR) + 1
                lt_full = np.arange(total_samples_from_start) / SR
                freq_full = f0 + (f1 - f0) * (lt_full / dur)
                phase_full = 2.0 * np.pi * np.cumsum(freq_full) / SR
                active_len = np.sum(mask)
                phase_act = phase_full[-active_len:]
                lt_act = lt[mask]
                env = np.clip(lt_act / 2.5, 0.0, 1.0) * np.clip((dur - lt_act) / 2.5, 0.0, 1.0)
                out[mask] += np.sin(phase_act) * amp * env
            if ts + dur > t_abs + n / SR:
                still.append((ts, f0, f1, dur, amp))
        self.emf_queue = still
        return out

    def _siren_chunk(self, t_abs, n, siren_gain):
        """Alarma SCP distante -- casi subliminal, muy filtrada."""
        if siren_gain <= 0:
            return np.zeros(n)
        out = np.zeros(n)
        if self.siren_active is None and t_abs >= self.siren_next:
            dur = self.siren_rng.uniform(8, 22)
            amp = self.siren_rng.uniform(0.008, 0.018) * siren_gain
            self.siren_active = (t_abs, dur, amp)
            self.siren_next   = t_abs + dur + self.siren_rng.uniform(90, 360)
        if self.siren_active:
            ts, dur, amp = self.siren_active
            t  = np.arange(n) / SR
            ta = t_abs + t - ts
            mask = (ta >= 0) & (ta < dur)
            if mask.any():
                ta_act = ta[mask]
                k = np.floor(ta_act / 2.5)
                # Fase analítica periódica exacta
                ph = 1600.0 * np.pi * ta_act + 3000.0 * k + 1500.0 * (1.0 - np.cos(np.pi * (ta_act % 2.5) / 2.5))
                env = np.clip(ta_act / 1.2, 0.0, 1.0) * np.clip((dur - ta_act) / 1.2, 0.0, 1.0)
                raw = np.sin(ph) * amp * env
                if self.siren_zi is None:
                    self.siren_zi = sps.sosfilt_zi(self.siren_sos) * 0.0
                filt, self.siren_zi = sps.sosfilt(self.siren_sos, raw, zi=self.siren_zi)
                out[np.where(mask)[0]] += filt * 0.08
            if t_abs > ts + dur:
                self.siren_active = None
        return out

    def _breath_chunk(self, t_abs, n, breath_gain):
        """Ciclo respiratorio artificial -- soporte vital de anomalia."""
        if breath_gain <= 0:
            return np.zeros(n)
        cycle = 6.0
        t   = np.arange(n) / SR
        tc  = (t_abs + t) % cycle
        env = np.where(tc < 2.5, tc / 2.5,
              np.where(tc < 3.0, 1.0,
              np.where(tc < 5.5, 1.0 - (tc - 3.0) / 2.5, 0.0)))
        noise = self.breath_rng.uniform(-1, 1, n).astype(np.float64)
        if self.breath_zi is None:
            self.breath_zi = sps.sosfilt_zi(self.breath_sos) * 0.0
        filt, self.breath_zi = sps.sosfilt(self.breath_sos, noise, zi=self.breath_zi)
        pitch_lfo = 1.0 + 0.08 * np.sin(2*np.pi * tc / cycle)
        return filt * env * pitch_lfo * 0.022 * breath_gain

    def _morse_chunk(self, t_abs, n, morse_gain):
        """Senial morse erratica -- fragmentos SCP que nunca se completan."""
        if morse_gain <= 0:
            return np.zeros(n)
        DOT = 0.08; DASH = 0.22; GAP = 0.07
        SEQS = [
            [DOT,GAP,DOT,GAP,DOT,0.24, DASH,GAP,DASH,GAP,DASH,0.24, DOT,GAP,DOT,GAP,DOT,0.80],
            [DASH,GAP,DOT,GAP,DASH,0.30, DOT,0.50],
            [DOT,GAP,DASH,GAP,DOT,GAP,DOT,0.60],
            [DASH,GAP,DASH,0.40, DOT,GAP,DOT,0.80],
            [DOT,0.10, DASH,0.10, DOT,GAP,DOT,0.50],
        ]
        while self.morse_next < t_abs + n / SR:
            seq  = SEQS[self.morse_rng.randint(len(SEQS))]
            freq = self.morse_rng.uniform(380, 820)
            amp  = self.morse_rng.uniform(0.005, 0.014) * morse_gain
            cur  = self.morse_next
            i = 0
            while i < len(seq):
                val = seq[i]
                if i + 1 < len(seq) and seq[i+1] in (GAP, 0.24, 0.30, 0.40, 0.50, 0.60, 0.80):
                    if self.morse_rng.random() < 0.12:
                        cur += val + seq[i+1]; i += 2; continue
                    self.morse_seq.append((cur, val * 0.88, freq, amp))
                    cur += val + seq[i+1]; i += 2
                else:
                    self.morse_seq.append((cur, val * 0.88, freq, amp))
                    cur += val; i += 1
            self.morse_next += (cur - self.morse_next) + self.morse_rng.uniform(30, 150)
        out = np.zeros(n)
        t_samples = t_abs + np.arange(n) / SR
        still = []
        for (ts, dur, freq, amp) in self.morse_seq:
            lt = t_samples - ts
            mask = (lt >= 0.0) & (lt < dur)
            if np.any(mask):
                lt_act = lt[mask]
                env = np.clip(lt_act / 0.008, 0.0, 1.0) * np.clip((dur - lt_act) / 0.008, 0.0, 1.0)
                out[mask] += np.sin(2.0 * np.pi * freq * lt_act) * amp * env
            if ts + dur > t_abs + n / SR:
                still.append((ts, dur, freq, amp))
        self.morse_seq = still
        return out

    def _anomaly_chunk(self, t_abs, n, anomaly_gain):
        """Nota pura que aparece sola, deriva de pitch, desaparece -- SCP puro."""
        if anomaly_gain <= 0:
            return np.zeros(n)
        if not self.anomaly_active and t_abs >= self.anomaly_next:
            self.anomaly_active = True
            self.anomaly_tf   = self.anomaly_rng.uniform(90, 600)
            self.anomaly_tamp = self.anomaly_rng.uniform(0.008, 0.020) * anomaly_gain
            self.anomaly_dur  = self.anomaly_rng.uniform(8, 45)
            self.anomaly_ts   = t_abs
            self.anomaly_next = t_abs + self.anomaly_dur + self.anomaly_rng.uniform(60, 300)
        out = np.zeros(n)
        if self.anomaly_active:
            t  = np.arange(n) / SR
            ta = t_abs + t
            age = ta - self.anomaly_ts
            mask = (age >= 0) & (age < self.anomaly_dur)
            if mask.any():
                k = 0.00005
                self.anomaly_f   += (self.anomaly_tf  - self.anomaly_f)   * k * n
                self.anomaly_amp += (self.anomaly_tamp - self.anomaly_amp) * 0.0002 * n
                self.anomaly_f    = float(np.clip(self.anomaly_f, 60, 800))
                env = (np.clip(age/3.0, 0, 1) * np.clip((self.anomaly_dur - age)/3.0, 0, 1))
                ph  = self.anomaly_ph + 2*np.pi*self.anomaly_f/SR * np.arange(n)
                self.anomaly_ph = float(ph[-1] % (2*np.pi))
                vib = 1.0 + 0.004 * np.sin(2*np.pi*0.3*ta)
                out[mask] += np.sin(ph[mask]*vib[mask]) * float(self.anomaly_amp) * env[mask]
                if age[-1] >= self.anomaly_dur:
                    self.anomaly_active = False
                    self.anomaly_tf  = self.anomaly_rng.uniform(90, 600)
                    self.anomaly_tamp = 0.0
        return out

    def _cricket_chunk(self, t_abs, n, state):
        """Grillos realistas que cantan en coro de noche."""
        night_scenes = ('bosque', 'montana', 'desierto', 'amazonica', 'scp_exterior', 'scp_contencion')
        if state.brightness >= 0.34 or state.scene not in night_scenes:
            return np.zeros(n)
            
        out = np.zeros(n)
        
        # Procesar los planificadores de cada grillo
        for idx, sched in enumerate(self.cricket_schedulers):
            while sched['next'] < t_abs + n / SR:
                # Cada chirp tiene (tiempo_inicio, frecuencia_base, pulsos, amplitud)
                fc = sched['fc'] + sched['rng'].uniform(-70, 70)
                pulses = sched['rng'].randint(2, 5)
                amp = sched['rng'].uniform(0.0010, 0.0028)
                self.cricket_active_chirps.append((sched['next'], fc, pulses, amp))
                sched['next'] += sched['rng'].uniform(1.7, 4.2)
                
        still = []
        for ts, fc, pulses, amp in self.cricket_active_chirps:
            chirp_total_dur = pulses * 0.022 + 0.012
            if ts + chirp_total_dur <= t_abs:
                continue
            if ts >= t_abs + n / SR:
                still.append((ts, fc, pulses, amp))
                continue
                
            for k in range(pulses):
                p_start = ts + k * 0.022
                p_s = int((p_start - t_abs) * SR)
                p_dur = int(0.010 * SR)
                
                start = max(p_s, 0)
                end = min(p_s + p_dur, n)
                if start >= end:
                    continue
                    
                lt = (t_abs + np.arange(start, end) / SR) - p_start
                # Modulación FM analítica correcta
                phase = 2 * np.pi * fc * lt - (45.0 / 42.0) * np.cos(2 * np.pi * 42 * lt)
                env = np.sin(np.pi * (lt / 0.010)) ** 2
                out[start:end] += np.sin(phase) * env * amp
                
            if ts + chirp_total_dur > t_abs + n / SR:
                still.append((ts, fc, pulses, amp))
                
        self.cricket_active_chirps = still
        return out * 0.45

    def _chains_chunk(self, t_abs, n, chains_gain):
        if chains_gain <= 0:
            return np.zeros(n)
        
        while self.chain_next < t_abs + n / SR:
            dur = self.rng.uniform(1.8, 3.2)
            gap = self.rng.uniform(12.0, 24.0)
            self.chain_active.append((self.chain_next, dur, self.rng.uniform(0.015, 0.035) * chains_gain))
            self.chain_next += gap
            
        t_samples = t_abs + np.arange(n) / SR
        out = np.zeros(n)
        still = []
        
        for (ts, dur, amp) in self.chain_active:
            lt = t_samples - ts
            mask = (lt >= 0.0) & (lt < dur)
            if np.any(mask):
                lt_act = lt[mask]
                # 1. Ruido de arrastre (metal contra el suelo)
                noise = np.random.uniform(-1.0, 1.0, len(lt_act))
                b_bp, a_bp = biquad_bandpass(450.0, 1.2, SR)
                drag = sps.lfilter(b_bp, a_bp, noise)
                env = np.sin(np.pi * lt_act / dur) * (0.6 + 0.4 * np.sin(2 * np.pi * 8.0 * lt_act))
                drag_sig = drag * env * amp
                
                # 2. Clinks de eslabones
                clinks = np.zeros(len(lt_act))
                clink_step = 0.22
                for offset in np.arange(0.1, dur - 0.2, clink_step):
                    c_ts = offset + self.rng.uniform(-0.04, 0.04)
                    c_lt = lt_act - c_ts
                    c_mask = (c_lt >= 0.0) & (c_lt < 0.3)
                    if np.any(c_mask):
                        c_lt_act = c_lt[c_mask]
                        f_clink = self.rng.uniform(1200, 2800)
                        dec_clink = self.rng.uniform(35, 60)
                        wave = (np.sin(2*np.pi*f_clink*c_lt_act) * 0.7 + np.sin(2*np.pi*f_clink*2.2*c_lt_act) * 0.3)
                        wave *= np.exp(-dec_clink * c_lt_act) * amp * 1.5
                        clinks[c_mask] += wave
                
                out[mask] += drag_sig + clinks
                
            if ts + dur > t_abs + n / SR:
                still.append((ts, dur, amp))
                
        self.chain_active = still
        return out
