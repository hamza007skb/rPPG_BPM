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

**Real-data result (single video).** `python benchmark_ubfc.py --data-dir data/ubfc --ablate` on UBFC-rPPG
DATASET_2 `subject4` (46 s, 38 sliding windows, 12 s window, 1 s step, oximeter HR as ground truth):

| Variant | MAE (BPM) | RMSE (BPM) | Pearson r | within 5 BPM | Coverage |
|---|---|---|---|---|---|
| baseline | 7.27 | 10.02 | -0.24 | 50.0% | 100.0% |
| +skin | 7.24 | 9.99 | -0.24 | 50.0% | 100.0% |
| +skin+weights | 7.22 | 9.95 | -0.24 | 50.0% | 100.0% |
| full (+tracker) | 2.03 | 2.88 | 0.48 | 86.8% | 100.0% |

Report:
- The BPM tracker accounts for essentially all of the gain: MAE drops from 7.22 to 2.03 BPM and
  windows within 5 BPM rise from 50% to 86.8%. The negative r of the first three variants suggests the
  raw spectral peak often lands on the wrong component (harmonic or motion) and the tracker's prior,
  gating and smoothing correct this.
- Skin segmentation and quality weighting changed MAE by only 0.05 BPM in total, which is within noise
  on this video. No benefit from them is demonstrated here.
- Coverage is 100% for every variant, so the tracker never abstained and MAE is not hiding dropped windows.
- r = 0.48 for `full` is low next to its MAE. Heart rate varies little within one UBFC video, so r is
  unreliable here; MAE and within-5 are the more meaningful numbers.

**Not verified: generalisation.** These numbers come from one subject. Consecutive windows overlap
heavily, so the 38 windows are not independent samples and the result is closer to n = 1 than n = 38.
Do not read it as a benchmark of the method, and do not conclude from it that skin segmentation or
weighting are useless; they may matter under motion, hair, glasses or uneven lighting that this video
does not stress. Defaults such as `accept_snr_db`, `skin_dist`, `max_speed` and the Kalman settings were
set from simulations and have not been tuned on UBFC. Expect weak spots with heavy beards, glasses, hair
over the forehead, very low light, and strong backlighting.

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