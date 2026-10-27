# rppg_live: real-time face ROI tracking, skin segmentation and heart-rate (BPM)

A from-scratch pipeline for camera-based pulse estimation.

```
camera frame
  -> MediaPipe face landmarks (smoothed, short gaps bridged)
  -> 5 ROIs (2x forehead, 2x cheek, nose) minus eyes / brows / lips
  -> adaptive skin segmentation (per-face chroma + luma model; drops hair, shadows, specular, clipped pixels)
  -> mean RGB per region  ---- motion gate (head speed, yaw) marks bad frames
  -> [every 1 s] last 12 s -> uniform resample -> POS per region -> band-pass 0.7-3 Hz
  -> per-region SNR -> quality-weighted fusion -> spectrum
  -> BPM tracker (prior near last BPM, SNR/motion gate, jump confirmation, Kalman smoothing)
  -> BPM + confidence + status (tracking / holding / no_signal)
```

## Run

```bash
pip install -r requirements.txt
python app.py                      # webcam; q quits, s toggles skin overlay, r resets
```
The face model (`models/face_landmarker.task`, ~4 MB) downloads on first run; if the download is
blocked, fetch it manually from the URL printed in the error.
First reading appears after ~8 s (warm-up). Sit still, face the light, avoid backlighting.

## Benchmark on UBFC-rPPG

Download DATASET_2 (https://sites.google.com/view/ybenezeth/ubfcrppg) into `data/ubfc/`
(`subjectN/vid.avi` + `ground_truth.txt`), then:

```bash
python benchmark_ubfc.py --data-dir data/ubfc --ablate
```
`--ablate` reports baseline -> +skin -> +weights -> full so you can see what each part contributes.
Output: MAE / RMSE / Pearson r / % within 5 BPM / coverage, plus `results/ubfc_windows.csv`.
Delete `data/cache` after changing any ROI or skin code (traces are cached per video).
Dataset 1 (different ground-truth format) is not supported by this script.

## What has and has not been verified

Verified (tests + manual): signal chain recovers 54-150 BPM from synthetic traces; irregular
(~14 fps) timestamps are handled; motion bursts are held, not reported; the tracker rejects isolated
glitches and follows sustained changes; ROIs and skin mask land on skin on a real face photo; the app
and benchmark run end-to-end on a synthetic video.

**Not verified: accuracy on real data.** Defaults such as `accept_snr_db`, `skin_dist`,
`max_speed` and the Kalman settings were set from simulations. Tune them on UBFC with `--ablate`
before claiming any numbers. Expect weak spots with heavy beards, glasses, hair over the forehead,
very low light, and strong backlighting.

## Files

| File | Role |
|---|---|
| `rppg_live/face.py` | MediaPipe tracker, landmark smoothing, head pose proxy |
| `rppg_live/skin.py` | ROI polygons + adaptive skin segmentation |
| `rppg_live/extract.py` | frame -> per-region RGB samples, motion gating |
| `rppg_live/dsp.py` | POS, band-pass, spectrum, SNR, peak refinement |
| `rppg_live/estimator.py` | window -> resample -> POS -> weighted fusion -> BPM |
| `rppg_live/bpm.py` | BPM tracker (peak selection, gating, Kalman) |
| `rppg_live/config.py` | every tunable parameter |
| `app.py` / `benchmark_ubfc.py` | live app / dataset evaluation |
