"""Real-time heart-rate monitor: face tracking + skin segmentation + tracked BPM.

    python app.py                       # default webcam
    python app.py --camera 1 --mirror
    python app.py --video clip.mp4      # play a file (timestamps from the file)
    python app.py --video clip.mp4 --no-display --csv out.csv   # headless

Keys:  q / Esc quit   s toggle skin overlay   r reset   space pause (video files)
"""
from __future__ import annotations

import argparse
import csv
import time
from typing import Optional

import cv2
import numpy as np

from rppg_live import Config, RPPGEngine
from rppg_live.estimator import Estimate
from rppg_live.extract import FrameInfo
from rppg_live.face import FaceTracker

STATUS_COLOR = {  # BGR
    "tracking": (80, 220, 80), "holding": (0, 200, 255), "no_signal": (60, 60, 255),
    "warming_up": (200, 200, 200), "low_fps": (60, 60, 255),
}


def draw_overlay(frame, info: FrameInfo, show_skin: bool):
    if info.landmarks is None:
        return
    if show_skin and info.overlay is not None:
        x0, y0 = info.crop_origin
        h, w = info.overlay.shape
        roi = frame[y0:y0 + h, x0:x0 + w]
        tint = roi.copy()
        tint[info.overlay] = (0, 255, 0)
        cv2.addWeighted(tint, 0.35, roi, 0.65, 0, roi)
    for poly, ok in zip(info.polygons, info.region_ok):
        cv2.polylines(frame, [poly], True, (255, 255, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
    if info.bad:
        cv2.putText(frame, "MOTION", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_panel(frame, est: Estimate, fps: float):
    h, w = frame.shape[:2]
    col = STATUS_COLOR.get(est.status, (255, 255, 255))
    panel = frame[h - 110:h, :].copy()
    cv2.rectangle(frame, (0, h - 110), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.25, frame[h - 110:h, :], 0.75, 0, frame[h - 110:h, :])

    if est.bpm is not None:
        cv2.putText(frame, f"{est.bpm:5.1f}", (12, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 3, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (170, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    elif est.status == "warming_up":
        cv2.putText(frame, f"warming up {est.progress * 100:3.0f}%", (12, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "--", (12, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 3, cv2.LINE_AA)

    status = est.status + (f" ({est.reason})" if est.reason else "")
    cv2.putText(frame, status, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"SNR {est.snr_db:4.1f} dB  conf {est.confidence:.2f}  regions {est.n_regions}  {fps:.0f} fps",
                (12, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # pulse waveform
    if est.waveform is not None and len(est.waveform) > 10:
        n = int(min(len(est.waveform), 8 * est.fs))
        y = est.waveform[-n:]
        y = (y - y.mean()) / (np.abs(y - y.mean()).max() + 1e-9)
        xs = np.linspace(260, w - 12, n)
        ys = (h - 55) - y * 38
        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, col, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--mirror", action="store_true", help="flip horizontally (selfie view)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--no-skin", action="store_true", help="disable skin segmentation (baseline)")
    ap.add_argument("--csv", type=str, default=None, help="write per-second estimates to CSV")
    ap.add_argument("--model", type=str, default=None, help="path to face_landmarker.task")
    args = ap.parse_args()

    cfg = Config(use_skin_mask=not args.no_skin)
    tracker = FaceTracker(args.model)
    engine = RPPGEngine(cfg, detector=tracker)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera / video.")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    show_skin, paused = True, False
    t0, n, fps_ui, last_tick = time.monotonic(), 0, 0.0, time.monotonic()
    frame_idx = 0
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                t = frame_idx / file_fps if args.video else time.monotonic() - t0
                frame_idx += 1
                info = engine.process(frame, t)
                n += 1
                now = time.monotonic()
                if now - last_tick >= 1.0:
                    fps_ui, n, last_tick = n / (now - last_tick), 0, now
                if not args.no_display:
                    draw_overlay(frame, info, show_skin)
                    draw_panel(frame, engine.latest, fps_ui)
                elif frame_idx % 150 == 0:
                    e = engine.latest
                    print(f"t={t:6.1f}s  bpm={e.bpm if e.bpm is None else round(e.bpm, 1)}  {e.status}  snr={e.snr_db:.1f}")
            if not args.no_display:
                cv2.imshow("rppg_live", frame)
                key = cv2.waitKey(1 if not (paused and args.video) else 30) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    show_skin = not show_skin
                elif key == ord("r"):
                    engine.reset()
                elif key == ord(" ") and args.video:
                    paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        if args.csv:
            with open(args.csv, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t", "bpm", "raw_bpm", "snr_db", "confidence", "motion_frac", "status", "reason"])
                for e in engine.history:
                    w.writerow([f"{e.t:.2f}", e.bpm, e.raw_bpm, f"{e.snr_db:.2f}", f"{e.confidence:.3f}",
                                f"{e.motion_frac:.2f}", e.status, e.reason])
            print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
