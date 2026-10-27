"""Signal-processing primitives: POS, band-pass, spectrum, SNR, peak refinement."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt


def pos(rgb: np.ndarray, fs: float, win_s: float = 1.6) -> Optional[np.ndarray]:
    """POS (Wang et al. 2017) on one RGB trace [T, 3] -> pulse signal [T] (unfiltered)."""
    T = len(rgb)
    l = int(round(win_s * fs))
    if l < 4 or T <= l:
        return None
    win = sliding_window_view(np.asarray(rgb, float).T, l, axis=1)      # [3, N, l]
    N = win.shape[1]
    cn = win / np.maximum(win.mean(axis=2, keepdims=True), 1e-6)
    s1 = cn[1] - cn[2]                       # G - B
    s2 = -2.0 * cn[0] + cn[1] + cn[2]        # -2R + G + B
    alpha = s1.std(axis=1) / np.maximum(s2.std(axis=1), 1e-9)
    h = s1 + alpha[:, None] * s2
    h -= h.mean(axis=1, keepdims=True)
    out = np.zeros(T)
    for k in range(l):                       # overlap-add
        out[k:k + N] += h[:, k]
    return out


def bandpass(x: np.ndarray, fs: float, band: Tuple[float, float], order: int = 3) -> Optional[np.ndarray]:
    lo, hi = band
    hi = min(hi, 0.95 * fs / 2.0)
    if hi <= lo:
        return None
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    if len(x) <= 3 * (2 * len(sos) + 1):
        return None
    return sosfiltfilt(sos, x)


def spectrum(x: np.ndarray, fs: float, band: Tuple[float, float], oversample: int = 8):
    """Hann-windowed, zero-padded periodogram restricted to `band`. Returns (freqs_hz, power)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    nfft = int(2 ** np.ceil(np.log2(max(len(x) * oversample, 256))))
    X = np.fft.rfft(x * np.hanning(len(x)), n=nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    m = (f >= band[0]) & (f <= band[1])
    return f[m], np.abs(X[m]) ** 2


def refine_peak(freqs: np.ndarray, psd: np.ndarray, i: int) -> float:
    """Parabolic interpolation around bin i -> frequency in Hz."""
    if 0 < i < len(psd) - 1:
        a, b, c = psd[i - 1], psd[i], psd[i + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-18:
            return float(freqs[i] + 0.5 * (a - c) / denom * (freqs[1] - freqs[0]))
    return float(freqs[i])


def peak_snr_db(freqs: np.ndarray, psd: np.ndarray, half_bw: float = 0.15) -> Tuple[float, float]:
    """(peak_freq_hz, SNR_dB): power near the peak + 2nd harmonic vs the rest of the band."""
    i = int(np.argmax(psd))
    f0 = float(freqs[i])
    sig = (np.abs(freqs - f0) <= half_bw) | (np.abs(freqs - 2 * f0) <= half_bw)
    p_sig = float(psd[sig].sum())
    p_noise = float(psd[~sig].sum())
    if p_sig <= 0:
        return f0, -np.inf
    return f0, 10.0 * np.log10(p_sig / max(p_noise, 1e-18))
