"""
gen_diffusion.py — Difusión mono (reverb por cadena all-pass, estilo Bitwig Poly Grid).

Versión MONO segura para el pipeline actual (PCM s16le mono). Añade una cola
difusa/espaciosa sobre la señal del master bus SIN cambiar a estéreo (eso exigiría
tocar ffmpeg). Cuando se migre el pipeline a estéreo, se sustituye por StereoDiffuser.

Diseño: combs feedback en paralelo (cola) -> damping low-pass -> all-pass en serie
(difusión). Todo IIR con scipy.lfilter + estado `zi` persistente -> procesa por
bloques sin glitches en los límites. A prueba de fallos: ante cualquier error
devuelve la señal seca, nunca rompe el stream.
"""
import numpy as np

try:
    from scipy.signal import lfilter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

SR = 44100


def _comb_coeffs(delay, g):
    a = np.zeros(delay + 1); a[0] = 1.0; a[-1] = -g
    return np.array([1.0]), a


def _allpass_coeffs(delay, g):
    b = np.zeros(delay + 1); b[0] = -g; b[-1] = 1.0
    a = np.zeros(delay + 1); a[0] = 1.0; a[-1] = -g
    return b, a


class _Filter:
    """IIR con estado persistente entre bloques."""
    def __init__(self, b, a):
        self.b, self.a = b, a
        self.zi = np.zeros(max(len(a), len(b)) - 1)

    def run(self, x):
        try:
            y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
            return y
        except Exception:
            return x


class MonoDiffuser:
    """Reverb por difusión mono, mezcla sutil sobre el master bus."""

    _COMB_MS = [29.7, 37.1, 41.1, 43.7]   # delays primos -> sin resonancias correlacionadas
    _AP_MS   = [5.0, 1.7, 12.7]           # difusión densa

    def __init__(self, sr=SR, rt60=2.6, mix=0.22, damp=0.5):
        self.ok = _HAS_SCIPY
        self.mix = float(np.clip(mix, 0.0, 1.0))
        if not self.ok:
            return
        try:
            def comb_g(ms):
                d = ms / 1000.0
                return float(np.clip(10.0 ** (-3.0 * d / max(rt60, 0.1)), 0.0, 0.96))
            self.combs = [_Filter(*_comb_coeffs(int(sr * m / 1000), comb_g(m))) for m in self._COMB_MS]
            self.aps   = [_Filter(*_allpass_coeffs(int(sr * m / 1000), 0.7)) for m in self._AP_MS]
            # Damping = low-pass 1-polo IIR
            self.damp_f = _Filter(np.array([1.0 - damp]), np.array([1.0, -damp]))
        except Exception:
            self.ok = False

    def process(self, mono):
        """mono (N,) -> mono con cola difusa mezclada. Nunca lanza excepción."""
        if not self.ok or self.mix <= 0.0:
            return mono
        try:
            x = np.asarray(mono, dtype=np.float64)
            wet = np.zeros_like(x)
            for c in self.combs:
                wet += c.run(x)
            wet /= len(self.combs)
            wet = self.damp_f.run(wet)
            for ap in self.aps:
                wet = ap.run(wet)
            out = x * (1.0 - self.mix) + wet * self.mix
            # Seguridad: si algo se desestabiliza, descartar el húmedo.
            if not np.all(np.isfinite(out)):
                return x
            return out
        except Exception:
            return mono
