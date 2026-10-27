"""Turns buffered per-region RGB traces into a BPM estimate.

Steps: window -> mask bad frames -> resample to uniform time grid -> POS per region ->
band-pass -> per-region SNR -> quality-weighted fusion -> spectrum -> BPM tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .bpm import BPMTracker
from .config import Config
from .dsp import bandpass, peak_snr_db, pos, spectrum


@dataclass
class Estimate:
    t: float = 0.0
    status: str = "warming_up"      # warming_up | tracking | holding | no_signal | low_fps
    reason: str = ""
    bpm: Optional[float] = None     # smoothed output
    raw_bpm: Optional[float] = None  # this window's own peak
    confidence: float = 0.0
    snr_db: float = float("nan")
    motion_frac: float = 0.0
    progress: float = 0.0           # warm-up progress 0..1
    n_regions: int = 0
    region_snr: List[float] = field(default_factory=list)
    waveform: Optional[np.ndarray] = None
    fs: float = 0.0


class Estimator:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.tracker = BPMTracker(self.cfg)

    def reset(self):
        self.tracker.reset()

    def _finish(self, base: Estimate, d: dict) -> Estimate:
        base.bpm, base.raw_bpm = d["bpm"], d["raw_bpm"]
        base.confidence, base.status, base.reason = d["confidence"], d["status"], d["reason"]
        return base

    def estimate(self, t, rgb, bad, now: Optional[float] = None) -> Estimate:
        """t [N] seconds, rgb [N, R, 3] (NaN where unusable), bad [N] bool (motion-gated)."""
        cfg = self.cfg
        t = np.asarray(t, float)
        if len(t) == 0:
            return Estimate()
        now = float(t[-1] if now is None else now)
        sel = (t >= now - cfg.window_s) & (t <= now)
        t, rgb, bad = t[sel], np.asarray(rgb)[sel], np.asarray(bad)[sel]
        est = Estimate(t=now)
        if len(t) < 10:
            return est
        keep = np.concatenate([[True], np.diff(t) > 0])           # strictly increasing time
        t, rgb, bad = t[keep], rgb[keep], bad[keep]
        span = t[-1] - t[0]
        est.progress = float(min(1.0, span / cfg.min_window_s))
        if span < cfg.min_window_s:
            return est
        fps = (len(t) - 1) / span
        if fps < cfg.min_fps:
            est.status, est.reason = "low_fps", f"{fps:.1f} fps"
            return est

        has_face = np.isfinite(rgb).any(axis=(1, 2))
        bad_total = bad | ~has_face
        est.motion_frac = float(bad_total.mean())
        if (~bad_total).sum() < 0.3 * len(t):
            reason = "no face" if has_face.mean() < 0.5 else "too much motion"
            return self._finish(est, self.tracker.reject(now, reason))

        fs = float(min(cfg.target_fs, fps))
        tu = np.arange(t[0], t[-1], 1.0 / fs)
        sigs, snrs = [], []
        for r in range(rgb.shape[1]):
            ok = np.isfinite(rgb[:, r, :]).all(axis=1) & ~bad
            if ok.sum() < 20 or ok.mean() < 0.5:
                continue
            x = np.stack([np.interp(tu, t[ok], rgb[ok, r, c]) for c in range(3)], axis=1)
            s = pos(x, fs)
            s = None if s is None else bandpass(s, fs, cfg.band_hz)
            if s is None or s.std() < 1e-9:
                continue
            s = s / s.std()
            f, p = spectrum(s, fs, cfg.band_hz)
            snrs.append(peak_snr_db(f, p, cfg.region_half_bw_hz)[1])
            sigs.append(s)
        est.n_regions = len(sigs)
        est.region_snr = [float(v) for v in snrs]
        est.fs = fs
        if not sigs:
            return self._finish(est, self.tracker.reject(now, "no usable regions"))

        sigs, snrs = np.array(sigs), np.array(snrs)
        if cfg.use_quality_weights:
            w = np.where(snrs >= cfg.min_region_snr_db, np.clip(10 ** (snrs / 10.0), 0, 20), 0.0)
            if w.sum() <= 0:
                w = np.ones(len(sigs))
        else:
            w = np.ones(len(sigs))
        fused = (w[:, None] * sigs).sum(axis=0) / w.sum()
        est.waveform = fused

        f, p = spectrum(fused, fs, cfg.band_hz)
        _, est.snr_db = peak_snr_db(f, p, cfg.region_half_bw_hz)
        return self._finish(est, self.tracker.update(now, f, p, est.snr_db, est.motion_frac))
