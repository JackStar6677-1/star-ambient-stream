"""Utilidades compartidas por todos los módulos de generadores."""
import numpy as np
import scipy.signal as sps

SR = 44100

def biquad_bandpass(fc, Q, sr=44100):
    """Diseño analítico ultra-rápido de filtro biquad paso banda (EQ Cookbook)."""
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1.0 + alpha
    b = np.array([alpha, 0.0, -alpha]) / a0
    a = np.array([a0, -2.0 * cos_w0, 1.0 - alpha]) / a0
    return b, a

def biquad_highpass(fc, Q, sr=44100):
    """Diseño analítico ultra-rápido de filtro biquad paso alto (EQ Cookbook)."""
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1.0 + alpha
    b = np.array([(1.0 + cos_w0)/2.0, -(1.0 + cos_w0), (1.0 + cos_w0)/2.0]) / a0
    a = np.array([a0, -2.0 * cos_w0, 1.0 - alpha]) / a0
    return b, a

def biquad_lowpass(fc, Q, sr=44100):
    """Diseño analítico ultra-rápido de filtro biquad paso bajo (EQ Cookbook)."""
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1.0 + alpha
    b = np.array([(1.0 - cos_w0)/2.0, 1.0 - cos_w0, (1.0 - cos_w0)/2.0]) / a0
    a = np.array([a0, -2.0 * cos_w0, 1.0 - alpha]) / a0
    return b, a

