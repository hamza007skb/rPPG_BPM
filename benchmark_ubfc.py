"""Sliding-window benchmark on UBFC-rPPG (DATASET_2 layout: <subject>/vid.avi + ground_truth.txt).

    python benchmark_ubfc.py --data-dir data/ubfc
    python benchmark_ubfc.py --data-dir data/ubfc --ablate       # baseline -> +skin -> +weights -> full
    python benchmark_ubfc.py --data-dir data/ubfc --max-videos 5

How it is evaluated
  * Video is processed exactly like the live app (same extractor / estimator, video-time clock).
  * Every `update_every_s` seconds after warm-up the estimate is compared with the mean
    oximeter heart rate over the same trailing window.
  * Reported: MAE, RMSE, Pearson r, % within 5 BPM, and coverage (fraction of windows where
    the method produced a number; the tracker can abstain, the baseline never does).
  * Dataset 1 (gtdump.xmp ground truth) is not supported by this script.
Cite: Bobbia et al., "Unsupervised skin tissue segmentation for remote photoplethysmography",
Pattern Recognition Letters, 2017.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from rppg_live import Config
from rppg_live.estimator import Estimator
from rppg_live.extract import Extractor
from rppg_live.face import FaceTracker

VARIANTS = {   # name: (use_skin_mask, use_quality_weights, use_tracker)
    "baseline": (False, False, False),
    "+skin": (True, False, False),
    "+skin+weights": (True, True, False),
    "full": (True, True, True),
}


def parse_ground_truth(path: Path):
    """UBFC DATASET_2 ground_truth.txt: line 1 = PPG, line 2 = heart rate, line 3 = time (s)."""
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError(f"{path}: expected 3 lines (PPG, HR, time), got {len(lines)}")
    hr = np.array(lines[1].replace(",", " ").split(), dtype=float)
    ts = np.array(lines[2].replace(",", " ").split(), dtype=float)
    n = min(len(hr), len(ts))
    return ts[:n], hr[:n]


def _ffmpeg_frames(video: Path):
    """Fallback reader: decode through the ffmpeg CLI (no transcoding, the file is untouched).
    Yields (frame_bgr, fps)."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError("OpenCV can't open the video and ffmpeg/ffprobe are not on PATH (brew install ffmpeg)")
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate", "-of", "json", str(video)]))["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    num, den = info["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy(), fps
    finally:
        p.stdout.close(); p.kill(); p.wait()


def _frames(video: Path):
    """Yield (frame_bgr, fps). OpenCV first, ffmpeg pipe if OpenCV can't open/decode the file."""
    cap = cv2.VideoCapture(str(video))
    got = 0
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            got += 1
            yield frame, fps
    cap.release()
    if got == 0:
        print(f"[{video.parent.name}] OpenCV read 0 frames, falling back to ffmpeg", flush=True)
        yield from _ffmpeg_frames(video)


def extract_video(video: Path, cfg: Config, cache: Path | None):
    key = f"{video.parent.name}_{'skin' if cfg.use_skin_mask else 'noskin'}.npz"
    if cache is not None and (cache / key).exists():
        d = np.load(cache / key)
        if len(d["t"]):
            return d["t"], d["rgb"], d["bad"]
    tracker = FaceTracker()
    ext = Extractor(cfg, tracker, keep_all=True)
    for i, (frame, fps) in enumerate(_frames(video)):
        ext.process(frame, i / fps)
    tracker.close()
    t, rgb, bad = ext.arrays()
    if cache is not None and len(t):
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache / key, t=t, rgb=rgb, bad=bad)
    return t, rgb, bad


def replay(t, rgb, bad, cfg: Config, gt_t, gt_hr):
    est = Estimator(cfg)
    rows = []
    now = cfg.min_window_s
    while now <= t[-1]:
        e = est.estimate(t, rgb, bad, now=now)
        m = (gt_t >= now - cfg.window_s) & (gt_t <= now)
        if m.any() and e.status != "warming_up":
            rows.append((now, float(gt_hr[m].mean()), e.bpm, e.snr_db, e.status))
        now += cfg.update_every_s
    return rows


def metrics(pairs):
    if not pairs:
        return dict(n=0, mae=np.nan, rmse=np.nan, r=np.nan, within5=np.nan)
    g, p = np.array(pairs).T
    err = p - g
    return dict(n=len(g), mae=np.abs(err).mean(), rmse=np.sqrt((err ** 2).mean()),
                r=np.corrcoef(g, p)[0, 1] if len(g) > 2 and g.std() > 0 and p.std() > 0 else np.nan,
                within5=(np.abs(err) <= 5).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="data/cache", help="ROI trace cache (delete after changing skin/ROI code)")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--variants", default=None, help="comma list from: " + ", ".join(VARIANTS))
    ap.add_argument("--max-videos", type=int, default=None)
    args = ap.parse_args()

    names = (args.variants.split(",") if args.variants else (list(VARIANTS) if args.ablate else ["full"]))
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant {n!r}")

    videos = sorted(Path(args.data_dir).rglob("vid.avi"))
    videos = [v for v in videos if (v.parent / "ground_truth.txt").exists()]
    if not videos:
        raise SystemExit(f"No <subject>/vid.avi + ground_truth.txt found under {args.data_dir}.\n"
                         "Download UBFC-rPPG DATASET_2 and point --data-dir at it.")
    if args.max_videos:
        videos = videos[: args.max_videos]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else None

    per_variant = {n: [] for n in names}      # (gt, est) pairs for windows with output
    windows = {n: 0 for n in names}           # all windows (for coverage)
    per_subject = {n: {} for n in names}
    rows_csv = []

    for v in videos:
        subj = v.parent.name
        gt_t, gt_hr = parse_ground_truth(v.parent / "ground_truth.txt")
        traces = {}
        for n in names:
            use_skin = VARIANTS[n][0]
            if use_skin not in traces:
                print(f"[{subj}] extracting (skin={use_skin}) ...", flush=True)
                traces[use_skin] = extract_video(v, Config(use_skin_mask=use_skin), cache)
            t, rgb, bad = traces[use_skin]
            if len(t) == 0:
                print(f"[{subj}] no frames read, skipping")
                continue
            skin, weights, trk = VARIANTS[n]
            cfg = Config(use_skin_mask=skin, use_quality_weights=weights, use_tracker=trk)
            rows = replay(t, rgb, bad, cfg, gt_t, gt_hr)
            pairs = [(g, b) for (_, g, b, _, _) in rows if b is not None]
            windows[n] += len(rows)
            per_variant[n] += pairs
            per_subject[n][subj] = metrics(pairs)["mae"]
            rows_csv += [(n, subj, f"{now:.1f}", g, b, snr, st) for (now, g, b, snr, st) in rows]
            print(f"[{subj}] {n:14s} MAE={per_subject[n][subj]:.2f}  ({len(pairs)}/{len(rows)} windows)")

    with open(out / "ubfc_windows.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "subject", "t", "gt_bpm", "est_bpm", "snr_db", "status"])
        w.writerows(rows_csv)

    print("\n=== Summary (pooled over all windows) ===")
    print(f"{'variant':14s} {'MAE':>6s} {'RMSE':>6s} {'r':>6s} {'<=5BPM':>7s} {'coverage':>9s}")
    for n in names:
        m = metrics(per_variant[n])
        cov = m["n"] / windows[n] if windows[n] else np.nan
        print(f"{n:14s} {m['mae']:6.2f} {m['rmse']:6.2f} {m['r']:6.2f} {m['within5']:7.1%} {cov:9.1%}")
    print(f"\nvideos: {len(videos)}   per-window results: {out / 'ubfc_windows.csv'}")


if __name__ == "__main__":
    main()