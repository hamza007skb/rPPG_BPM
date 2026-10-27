"""RPPGEngine: feed frames in, get BPM estimates out."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import Config
from .estimator import Estimate, Estimator
from .extract import Extractor, FrameInfo
from .face import Detector


class RPPGEngine:
    def __init__(self, cfg: Optional[Config] = None, detector: Optional[Detector] = None):
        self.cfg = cfg or Config()
        self.extractor = Extractor(self.cfg, detector)
        self.estimator = Estimator(self.cfg)
        self.latest = Estimate()
        self.history: List[Estimate] = []
        self._last_update = -np.inf

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameInfo:
        """Process one BGR frame captured at time t (seconds). Updates `latest` once per second."""
        info = self.extractor.process(frame_bgr, t)
        if t - self._last_update >= self.cfg.update_every_s:
            ts, rgb, bad = self.extractor.arrays()
            self.latest = self.estimator.estimate(ts, rgb, bad, now=t)
            self.history.append(self.latest)
            self._last_update = t
        return info

    def reset(self):
        self.extractor.reset()
        self.estimator.reset()
        self.latest = Estimate()
        self.history.clear()
        self._last_update = -np.inf
