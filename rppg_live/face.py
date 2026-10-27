"""Face landmark detection (MediaPipe FaceLandmarker) and landmark smoothing."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np

log = logging.getLogger("rppg_live")

N_LANDMARKS = 468
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def ensure_model(path=None) -> Path:
    """Return the model path, downloading it on first use."""
    path = Path(path) if path else default_model_path()
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading face landmarker model to %s", path)
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    except Exception as exc:  # pragma: no cover - network dependent
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the face model ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\nand save it as:\n  {path}"
        ) from exc
    return path


class Detector(Protocol):
    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        """Return landmarks [468, 2] in pixel coordinates, or None."""


class FaceTracker:
    """MediaPipe FaceLandmarker in VIDEO mode (tracks across frames, 1 face)."""

    def __init__(self, model_path=None, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._lm = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        ts = max(int(timestamp_ms), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        h, w = frame_rgb.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(frame_rgb))
        res = self._lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None
        pts = res.face_landmarks[0][:N_LANDMARKS]
        return np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float64)

    def close(self):
        self._lm.close()


class LandmarkSmoother:
    """EMA smoothing, reset on large jumps, and short gap bridging."""

    def __init__(self, alpha: float = 0.5, max_gap: int = 8, jump_frac: float = 0.2):
        self.alpha, self.max_gap, self.jump_frac = alpha, max_gap, jump_frac
        self.reset()

    def reset(self):
        self._prev: Optional[np.ndarray] = None
        self._gap = 0

    def update(self, lm: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], bool]:
        """Return (landmarks or None, bridged). bridged=True means a stale estimate."""
        if lm is not None:
            if self._prev is None:
                self._prev = lm.copy()
            else:
                width = max(np.ptp(lm[:, 0]), 1.0)
                jump = np.linalg.norm(lm - self._prev, axis=1).mean() / width
                self._prev = lm.copy() if jump > self.jump_frac else (
                    self.alpha * lm + (1 - self.alpha) * self._prev)
            self._gap = 0
            return self._prev.copy(), False
        self._gap += 1
        if self._prev is not None and self._gap <= self.max_gap:
            return self._prev.copy(), True
        self._prev = None
        return None, False


def head_metrics(lm: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """(face_width_px, yaw_proxy, anchor_centroid). Yaw 0 = frontal."""
    left, right, nose = lm[234], lm[454], lm[1]
    width = max(abs(right[0] - left[0]), 1.0)
    yaw = ((nose[0] - left[0]) - (right[0] - nose[0])) / width
    anchor = lm[[1, 10, 152, 234, 454]].mean(axis=0)
    return float(width), float(yaw), anchor
