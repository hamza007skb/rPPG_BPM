"""BPM tracking: picks the right spectral peak and smooths it over time.

Plain "argmax of the spectrum every window" jumps around whenever noise beats the pulse.
This tracker
  * looks for peaks near the previous heart rate (a pulse cannot jump 30 BPM in a second),
  * rejects windows whose SNR is too low or that contained too much motion,
  * holds the last good value for a while instead of showing garbage,
  * smooths accepted measurements with a scalar Kalman filter weighted by confidence.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from .config import Config
from .dsp import refine_peak


class BPMTracker:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.reset()

    def reset(self):
        self.x: Optional[float] = None       # smoothed BPM
        self.P: float = 0.0                  # its variance
        self.t_last: Optional[float] = None
        self.t_accept: Optional[float] = None
        self._jump_cand: Optional[float] = None   # far-away candidate awaiting confirmation
        self._jump_n: int = 0

    # ------------------------------------------------------------------
    def _raw_peak(self, bpm_axis, freqs, psd, prior, sigma) -> float:
        pk, _ = find_peaks(psd, prominence=0.1 * psd.max())
        idx = pk if len(pk) else np.array([int(np.argmax(psd))])
        if prior is None:
            i = int(idx[np.argmax(psd[idx])])
            # fundamental weaker than its sub-harmonic-like twin at half the rate? prefer the lower one
            half = bpm_axis[i] / 2.0
            if half >= bpm_axis[0]:
                win = np.abs(bpm_axis - half) <= 3.0
                if win.any() and psd[win].max() >= 0.5 * psd[i]:
                    i = int(np.flatnonzero(win)[np.argmax(psd[win])])
        else:
            score = psd[idx] / psd.max() * np.exp(-0.5 * ((bpm_axis[idx] - prior) / sigma) ** 2)
            i = int(idx[np.argmax(score)])
        return refine_peak(freqs, psd, i) * 60.0

    def confidence(self, snr_db: float, motion_frac: float) -> float:
        z = (snr_db - self.cfg.accept_snr_db) / 2.0
        c = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return float(c * (1.0 - min(max(motion_frac, 0.0), 1.0)))

    # ------------------------------------------------------------------
    def update(self, t: float, freqs, psd, snr_db: float, motion_frac: float) -> dict:
        cfg = self.cfg
        bpm_axis = freqs * 60.0
        conf = self.confidence(snr_db, motion_frac)

        if not cfg.use_tracker:              # baseline: argmax every window, no gating
            raw = self._raw_argmax(freqs, psd)
            return dict(bpm=raw, raw_bpm=raw, confidence=conf, status="tracking", reason="")

        if self.x is not None and t - self.t_accept > cfg.hold_s:
            self.x = None                    # lost the pulse: next good window re-locks
        prior = self.x
        sigma = None
        if prior is not None:
            sigma = cfg.track_sigma_bpm + cfg.track_growth_bpm_per_s * (t - self.t_accept)
        meas = self._raw_peak(bpm_axis, freqs, psd, prior, sigma)

        reason = ""
        accept = True
        if snr_db < cfg.accept_snr_db:
            accept, reason = False, "low SNR"
        elif motion_frac > cfg.max_bad_frac:
            accept, reason = False, "too much motion"
        elif prior is not None and abs(meas - prior) > cfg.max_jump_bpm:
            # A big jump is accepted only if the same new rate shows up in several
            # consecutive windows; isolated glitches are ignored.
            if self._jump_cand is not None and abs(meas - self._jump_cand) <= cfg.jump_tol_bpm:
                self._jump_n += 1
            else:
                self._jump_cand, self._jump_n = meas, 1
            if self._jump_n >= cfg.jump_confirm_windows:
                self.x = None                     # re-lock on the new rate
                self._jump_cand, self._jump_n = None, 0
            else:
                accept, reason = False, "unconfirmed jump"
        else:
            self._jump_cand, self._jump_n = None, 0

        if accept:
            R = (cfg.meas_sigma_bpm / max(conf, 0.15)) ** 2
            if self.x is None:
                self.x, self.P = meas, R
            else:
                self.P += cfg.kalman_q * max(t - self.t_last, 0.0)
                K = self.P / (self.P + R)
                self.x += K * (meas - self.x)
                self.P *= (1.0 - K)
            self.t_accept = t
            status = "tracking"
        else:
            status = "holding" if self.x is not None else "no_signal"
        self.t_last = t
        return dict(bpm=self.x, raw_bpm=meas, confidence=conf, status=status, reason=reason)

    def reject(self, t: float, reason: str) -> dict:
        """Window unusable (no face, no regions...). Hold or drop."""
        if self.x is not None and t - self.t_accept > self.cfg.hold_s:
            self.x = None
        status = "holding" if self.x is not None else "no_signal"
        return dict(bpm=self.x, raw_bpm=None, confidence=0.0, status=status, reason=reason)

    @staticmethod
    def _raw_argmax(freqs, psd) -> float:
        return refine_peak(freqs, psd, int(np.argmax(psd))) * 60.0
