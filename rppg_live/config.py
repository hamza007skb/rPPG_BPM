"""All tunable parameters in one place."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # ---- signal window -------------------------------------------------
    band_hz: tuple = (0.7, 3.0)      # 42-180 BPM
    window_s: float = 12.0           # analysis window length
    min_window_s: float = 8.0        # warm-up before the first estimate
    update_every_s: float = 1.0      # estimate rate
    target_fs: float = 30.0          # uniform resampling rate (capped by real fps)
    min_fps: float = 10.0            # below this we refuse to estimate
    max_buffer_s: float = 40.0       # live buffer length

    # ---- skin segmentation --------------------------------------------
    use_skin_mask: bool = True       # False = plain landmark polygons (baseline)
    skin_dist: float = 2.5           # Mahalanobis radius in (Cr, Cb) space
    skin_ema: float = 0.15           # temporal smoothing of the skin colour model
    min_region_pixels: int = 100     # a region needs this many skin pixels
    min_region_frac: float = 0.25    # ...and this fraction of its polygon area

    # ---- motion gating -------------------------------------------------
    max_speed: float = 0.35          # face-widths / second
    max_yaw: float = 0.30            # |yaw| proxy, 0 = frontal
    max_gap_frames: int = 8          # bridge this many missed detections
    max_bad_frac: float = 0.40       # window rejected if more frames are bad

    # ---- region fusion -------------------------------------------------
    use_quality_weights: bool = True  # False = equal weights (baseline)
    min_region_snr_db: float = -3.0   # regions below this get weight 0
    region_half_bw_hz: float = 0.15   # half bandwidth for SNR computation

    # ---- BPM tracking --------------------------------------------------
    use_tracker: bool = True          # False = plain per-window argmax (baseline)
    accept_snr_db: float = 0.0        # fused SNR needed to accept a measurement
    hold_s: float = 8.0               # keep showing the last BPM this long
    track_sigma_bpm: float = 6.0      # width of the prior around the last BPM
    track_growth_bpm_per_s: float = 1.0
    max_jump_bpm: float = 12.0        # larger jumps must be confirmed...
    jump_confirm_windows: int = 3     # ...by this many consecutive windows
    jump_tol_bpm: float = 8.0         # that agree within this tolerance
    kalman_q: float = 2.0             # process noise, BPM^2 per second
    meas_sigma_bpm: float = 3.0       # measurement noise at confidence 1
