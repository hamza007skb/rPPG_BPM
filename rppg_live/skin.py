"""ROI definition and adaptive skin segmentation.

For every frame:
  1. Build polygons for 5 skin regions from MediaPipe landmarks.
  2. Cut out eyes, eyebrows and lips (dilated) so they never enter a region.
  3. Fit a Gaussian to the (Cr, Cb) chroma and the luma of the *current face*
     (robustly, with temporal smoothing) and keep only pixels that match it.
     This adapts to skin tone and lighting without fixed colour thresholds.
  4. Drop specular highlights, shadows/hair (luma outliers) and clipped pixels.
  5. Average RGB over what is left, per region.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config

# ---- MediaPipe Face Mesh landmark groups ---------------------------------
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
             378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37,
        39, 40, 185]
EXCLUDE_GROUPS = [RIGHT_EYE, LEFT_EYE, RIGHT_BROW, LEFT_BROW, LIPS]

# "r"/"l" = the subject's right/left (the subject's right appears on the image left).
REGIONS = {
    "forehead_r": [21, 54, 103, 67, 109, 10, 151, 108, 69, 104, 68, 71],
    "forehead_l": [251, 284, 332, 297, 338, 10, 151, 337, 299, 333, 298, 301],
    "cheek_r": [116, 117, 118, 119, 100, 36, 205, 187, 147, 123, 50, 101],
    "cheek_l": [345, 346, 347, 348, 329, 266, 425, 411, 376, 352, 280, 330],
    "nose": [6, 197, 195, 5, 4, 1, 98, 327, 129, 358],
}
REGION_NAMES: List[str] = list(REGIONS)
N_REGIONS = len(REGIONS)
CORE_REGIONS = ("forehead_r", "forehead_l", "cheek_r", "cheek_l")  # used to fit the skin model


@dataclass
class SkinResult:
    means: np.ndarray                      # [N_REGIONS, 3] mean RGB, NaN if unusable
    counts: np.ndarray                     # [N_REGIONS] skin pixels used
    polygons: List[np.ndarray]             # region hulls, full-frame pixel coords
    overlay: Optional[np.ndarray]          # bool mask (crop coords) of pixels used
    crop_origin: Tuple[int, int]           # (x0, y0) of the crop in the frame


def _hull(lm: np.ndarray, idx: List[int]) -> np.ndarray:
    return cv2.convexHull(lm[idx].astype(np.float32)).reshape(-1, 2)


def _mahalanobis2(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(cov)
    d = x - mu
    return np.einsum("...i,ij,...j->...", d, inv, d)


class SkinSegmenter:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.reset()

    def reset(self):
        self._mu: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._luma: Optional[Tuple[float, float]] = None  # (median, sigma)

    # ------------------------------------------------------------------
    def _update_model(self, cc: np.ndarray, y: np.ndarray, core: np.ndarray) -> bool:
        """Robustly fit chroma Gaussian + luma stats on the core region."""
        seed = cc[core]
        if seed.shape[0] < 80:
            return self._mu is not None   # reuse the previous model if we have one
        floor = np.eye(2) * 6.0           # variance floor: ~2.5 chroma levels
        mu, cov = seed.mean(0), np.cov(seed.T) + floor
        for _ in range(2):                # trim outliers (lips, hair, glasses) and refit
            keep = _mahalanobis2(seed, mu, cov) < 6.0
            if keep.sum() < 50:
                break
            mu, cov = seed[keep].mean(0), np.cov(seed[keep].T) + floor
        ys = y[core]
        med = float(np.median(ys))
        sig = max(1.4826 * float(np.median(np.abs(ys - med))), 4.0)
        a = self.cfg.skin_ema
        if self._mu is None:
            self._mu, self._cov, self._luma = mu, cov, (med, sig)
        else:
            self._mu = (1 - a) * self._mu + a * mu
            self._cov = (1 - a) * self._cov + a * cov
            self._luma = ((1 - a) * self._luma[0] + a * med, (1 - a) * self._luma[1] + a * sig)
        return True

    # ------------------------------------------------------------------
    def analyze(self, frame_rgb: np.ndarray, lm: np.ndarray) -> SkinResult:
        cfg = self.cfg
        H, W = frame_rgb.shape[:2]
        nan = SkinResult(np.full((N_REGIONS, 3), np.nan), np.zeros(N_REGIONS, int), [], None, (0, 0))

        oval = lm[FACE_OVAL]
        x0, y0 = np.floor(oval.min(0)).astype(int)
        x1, y1 = np.ceil(oval.max(0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return nan
        crop = np.ascontiguousarray(frame_rgb[y0:y1, x0:x1])
        h, w = crop.shape[:2]
        off = np.array([x0, y0], dtype=np.float32)
        hulls = {n: _hull(lm, idx) for n, idx in REGIONS.items()}
        polygons = [hulls[n].astype(np.int32) for n in REGION_NAMES]

        region_masks = {}
        for n, hp in hulls.items():
            m = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(m, np.rint(hp - off).astype(np.int32), 1)
            region_masks[n] = m

        if not cfg.use_skin_mask:         # baseline: raw polygons only
            return self._reduce(crop, region_masks, None, polygons, (x0, y0), None, cfg)

        # face minus eyes / brows / lips
        face = np.zeros((h, w), np.uint8)
        cv2.fillPoly(face, [np.rint(oval - off).astype(np.int32)], 1)
        excl = np.zeros((h, w), np.uint8)
        for grp in EXCLUDE_GROUPS:
            cv2.fillConvexPoly(excl, np.rint(_hull(lm, grp) - off).astype(np.int32), 1)
        face_w = float(oval[:, 0].max() - oval[:, 0].min())
        k = max(3, int(round(0.03 * face_w)) | 1)
        excl = cv2.dilate(excl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        allowed = (face > 0) & (excl == 0)

        ycc = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)
        y = ycc[..., 0].astype(np.float32)
        cc = ycc[..., 1:].astype(np.float32)          # (Cr, Cb)

        core = np.zeros((h, w), bool)
        for n in CORE_REGIONS:
            core |= region_masks[n] > 0
        core &= allowed
        if not self._update_model(cc, y, core):
            return nan

        d2 = _mahalanobis2(cc, self._mu, self._cov)
        med, sig = self._luma
        skin = (
            allowed
            & (d2 < cfg.skin_dist ** 2)
            & (y > med - 3.5 * sig) & (y < med + 3.0 * sig)   # shadows / specular highlights
            & (crop.max(axis=2) < 250) & (crop.min(axis=2) > 5)  # clipped pixels
        )
        used = np.zeros((h, w), bool)
        for m in region_masks.values():
            used |= m > 0
        return self._reduce(crop, region_masks, skin, polygons, (x0, y0), used & skin, cfg)

    # ------------------------------------------------------------------
    @staticmethod
    def _reduce(crop, region_masks, skin, polygons, origin, overlay, cfg) -> SkinResult:
        means = np.full((N_REGIONS, 3), np.nan)
        counts = np.zeros(N_REGIONS, int)
        for r, n in enumerate(REGION_NAMES):
            m = region_masks[n] > 0
            area = int(m.sum())
            if skin is not None:
                m = m & skin
            cnt = int(m.sum())
            counts[r] = cnt
            if skin is not None and (cnt < cfg.min_region_pixels or cnt < cfg.min_region_frac * area):
                continue
            if cnt == 0:
                continue
            means[r] = cv2.mean(crop, mask=m.astype(np.uint8))[:3]
        if overlay is None and skin is None:
            overlay = np.zeros(crop.shape[:2], bool)
            for m in region_masks.values():
                overlay |= m > 0
        return SkinResult(means, counts, polygons, overlay, origin)
