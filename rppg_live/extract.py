"""Frame -> per-region RGB samples. Handles tracking, smoothing, motion gating."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .face import Detector, FaceTracker, LandmarkSmoother, head_metrics
from .skin import N_REGIONS, SkinSegmenter


@dataclass
class FrameInfo:
    t: float
    detected: bool = False
    bridged: bool = False
    bad: bool = True
    speed: float = 0.0
    yaw: float = 0.0
    landmarks: Optional[np.ndarray] = None
    polygons: List[np.ndarray] = field(default_factory=list)
    overlay: Optional[np.ndarray] = None
    crop_origin: Tuple[int, int] = (0, 0)
    region_ok: np.ndarray = field(default_factory=lambda: np.zeros(N_REGIONS, bool))


class Extractor:
    def __init__(self, cfg: Optional[Config] = None, detector: Optional[Detector] = None,
                 keep_all: bool = False):
        self.cfg = cfg or Config()
        self.detector = detector if detector is not None else FaceTracker()
        self.keep_all = keep_all
        self.smoother = LandmarkSmoother(max_gap=self.cfg.max_gap_frames)
        self.skin = SkinSegmenter(self.cfg)
        self._t, self._rgb, self._bad = deque(), deque(), deque()
        self.reset_motion()

    def reset_motion(self):
        self._prev_anchor = None
        self._prev_t = None
        self._speed = 0.0

    def reset(self):
        self.smoother.reset()
        self.skin.reset()
        self._t.clear(); self._rgb.clear(); self._bad.clear()
        self.reset_motion()

    # ------------------------------------------------------------------
    def _push(self, t, rgb, bad):
        self._t.append(t); self._rgb.append(rgb); self._bad.append(bad)
        if not self.keep_all:
            while self._t and t - self._t[0] > self.cfg.max_buffer_s:
                self._t.popleft(); self._rgb.popleft(); self._bad.popleft()

    def arrays(self):
        if not self._t:
            return np.zeros(0), np.zeros((0, N_REGIONS, 3)), np.zeros(0, bool)
        return np.array(self._t), np.stack(self._rgb), np.array(self._bad, bool)

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, t: float) -> FrameInfo:
        cfg = self.cfg
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw = self.detector.detect(rgb, int(round(t * 1000)))
        lm, bridged = self.smoother.update(raw)
        info = FrameInfo(t=t, detected=raw is not None, bridged=bridged)

        if lm is None:
            self.reset_motion()
            self._push(t, np.full((N_REGIONS, 3), np.nan), True)
            return info

        width, yaw, anchor = head_metrics(lm)
        speed = 0.0
        if self._prev_anchor is not None and t > self._prev_t:
            inst = np.linalg.norm(anchor - self._prev_anchor) / width / (t - self._prev_t)
            self._speed = 0.5 * self._speed + 0.5 * inst
            speed = self._speed
        self._prev_anchor, self._prev_t = anchor, t

        res = self.skin.analyze(rgb, lm)
        bad = bridged or speed > cfg.max_speed or abs(yaw) > cfg.max_yaw
        self._push(t, res.means, bool(bad))

        info.bad, info.speed, info.yaw = bool(bad), speed, yaw
        info.landmarks, info.polygons = lm, res.polygons
        info.overlay, info.crop_origin = res.overlay, res.crop_origin
        info.region_ok = np.isfinite(res.means).all(axis=1)
        return info
