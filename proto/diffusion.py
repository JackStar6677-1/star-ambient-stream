"""
diffusion.py — Reverb por difusión estéreo (estilo cadena All-pass de Bitwig Poly Grid).

Replica el bloque que da el sonido "espacioso/pro" de la referencia:
  - Combs feedback en paralelo  -> cola larga (RT60 ajustable)
  - All-pass en serie           -> difusión densa (elimina eco metálico)
  - Delays distintos L/R + chorus modulado -> imagen estéreo amplia

Todo se expresa como filtros IIR (scipy.lfilter) con estado `zi` persistente,
así procesa por bloques sin glitches en los límites — igual que el motor real.

Uso:
    rv = StereoDiffuser(sr=44100, rt60=3.2, mix=0.38, width=1.0)
    wet_stereo = rv.process(mono_block)   # (N,) -> (N, 2)
"""
import numpy as np

try:
    from scipy.signal import lfilter
except ImportError as e:
    raise ImportError("scipy es necesario para diffusion.py") from e


def _comb_coeffs(delay, g):
    """Comb feedback IIR:  H(z) = 1 / (1 - g·z^-D)."""
    a = np.zeros(delay + 1)
    a[0] = 1.0
    a[-1] = -g
    return np.array([1.0]), a


def _allpass_coeffs(delay, g):
    """All-pass Schroeder:  H(z) = (-g + z^-D) / (1 - g·z^-D)."""
    b = np.zeros(delay + 1)
    b[0] = -g
    b[-1] = 1.0
    a = np.zeros(delay + 1)
    a[0] = 1.0
    a[-1] = -g
    return b, a


class _Filter:
    """Envoltorio de un IIR con estado persistente entre bloques."""
    def __init__(self, b, a):
        self.b, self.a = b, a
        self.zi = np.zeros(max(len(a), len(b)) - 1)

    def run(self, x):
        try:
            y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
            return y
        except Exception:
            # Si algo se desestabiliza, no rompemos el stream: pasamos seco.
            return x


class StereoDiffuser:
    """Reverb por difusión estéreo procesable por bloques."""

    # Delays en ms — primos entre sí para evitar resonancias correlacionadas.
    _COMB_L = [29.7, 37.1, 41.1, 43.7]
    _COMB_R = [31.3, 35.3, 39.7, 45.1]
    _AP_L   = [5.0, 1.7, 12.7, 9.5]
    _AP_R   = [5.3, 1.9, 13.1, 8.9]

    def __init__(self, sr=44100, rt60=3.2, mix=0.38, width=1.0, damp=0.35):
        self.sr = sr
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.width = float(np.clip(width, 0.0, 1.5))
        self.damp = float(np.clip(damp, 0.0, 0.95))

        def comb_g(delay_ms):
            # g para alcanzar el RT60 deseado:  g = 10^(-3·D / RT60)
            d = delay_ms / 1000.0
            return float(np.clip(10.0 ** (-3.0 * d / max(rt60, 0.1)), 0.0, 0.97))

        def build(comb_ms, ap_ms):
            combs = [_Filter(*_comb_coeffs(int(sr * m / 1000), comb_g(m))) for m in comb_ms]
            aps   = [_Filter(*_allpass_coeffs(int(sr * m / 1000), 0.7)) for m in ap_ms]
            return combs, aps

        self.combs_L, self.aps_L = build(self._COMB_L, self._AP_L)
        self.combs_R, self.aps_R = build(self._COMB_R, self._AP_R)

        # Damping = low-pass 1-polo:  y[n] = (1-a)·x[n] + a·y[n-1]  → IIR vectorizado.
        self._damp_L = _Filter(np.array([1.0 - self.damp]), np.array([1.0, -self.damp]))
        self._damp_R = _Filter(np.array([1.0 - self.damp]), np.array([1.0, -self.damp]))

        # Chorus de ensanchado: LFOs lentos desfasados por canal.
        self._lfo_phase = 0.0

    def _damp_lp(self, x, which):
        """Low-pass suave para que la cola pierda agudos (reverb más natural)."""
        return (self._damp_L if which == 'L' else self._damp_R).run(x)

    def _wet_channel(self, x, combs, aps, which):
        # Combs en paralelo (suma) -> damping -> all-pass en serie
        acc = np.zeros_like(x)
        for c in combs:
            acc += c.run(x)
        acc /= len(combs)
        acc = self._damp_lp(acc, which)
        for ap in aps:
            acc = ap.run(acc)
        return acc

    def process(self, mono):
        """mono (N,) -> estéreo húmedo+seco mezclado (N, 2)."""
        mono = np.asarray(mono, dtype=np.float64)
        n = len(mono)

        wet_L = self._wet_channel(mono, self.combs_L, self.aps_L, 'L')
        wet_R = self._wet_channel(mono, self.combs_R, self.aps_R, 'R')

        # Chorus de ensanchado: pequeña modulación de amplitud desfasada L/R
        t = (self._lfo_phase + np.arange(n)) / self.sr
        self._lfo_phase += n
        modL = 1.0 + 0.06 * np.sin(2 * np.pi * 0.11 * t)
        modR = 1.0 + 0.06 * np.sin(2 * np.pi * 0.11 * t + np.pi * 0.5)
        wet_L *= modL
        wet_R *= modR

        # Control de width: mid/side
        mid = 0.5 * (wet_L + wet_R)
        side = 0.5 * (wet_L - wet_R) * self.width
        wet_L = mid + side
        wet_R = mid - side

        # Mezcla seco (centrado) + húmedo
        dry = mono * (1.0 - self.mix)
        out = np.empty((n, 2))
        out[:, 0] = dry + wet_L * self.mix
        out[:, 1] = dry + wet_R * self.mix
        return out
