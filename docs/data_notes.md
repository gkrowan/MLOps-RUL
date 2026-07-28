# FEMTO data notes

Decisions and quirks discovered while verifying the raw dataset (`scripts/verify_data.py`).
Keeping this current as we go so the Week 4 README write-up is assembly, not archaeology.

## Layout

- `data/Training_set/` — 6 bearings, full run-to-failure. This is the labeled training data.
- `data/Test_set/` — 11 bearings, **truncated** partway through their run. This is what a
  submission/model would see at inference time.
- `data/Validation_Set/` — the same 11 bearings as `Test_set`, but the **full** run-to-failure.
  This is the released ground truth (PHM12 challenge answer key) — `Test_set` is confirmed to be
  a truncated prefix of it (verified: identical accelerometer readings up to the truncation
  point, for all 11 bearings). **Do not train on this** — it's for computing true RUL /
  scoring, not a third data split to fit on.
- All three splits cover the same 3 operating conditions (1/2/3), encoded in the bearing
  directory name `Bearing{condition}_{unit}`.
- Every bearing's `acc_*.csv` / `temp_*.csv` sequence is complete (no missing file-index gaps),
  confirmed across the whole dataset.

## RUL ground truth for Test_set bearings

For a `Test_set` bearing, true remaining life = (file count in matching `Validation_Set`
bearing − file count in `Test_set` bearing) × 10 seconds per file. E.g. `Bearing1_3` has 1802
files in `Test_set` vs 2375 in `Validation_Set` → 573 files / 5730s of RUL was held out.

## File formats

- `acc_*.csv`: one row per accelerometer sample, 2560 rows (0.1s @ 25.6kHz), 6 columns
  — `hour, minute, second, microsecond, horiz_accel_g, vert_accel_g`. A new file is written
  roughly every 10s of run time.
- `temp_*.csv`: one row per temperature sample, nominally 600 rows (60s @ 10Hz), 5 columns
  — `hour, minute, second, hundredth_second, temperature_c`.

## Known quirks (handled in `femto_rul.ingestion.raw_loader`, not bugs to fix)

1. **Delimiter is inconsistent across the dataset.** Most files are comma-delimited, but some
   bearings ship `acc_*.csv` and/or `temp_*.csv` as semicolon-delimited instead — e.g.
   `Validation_Set/Bearing1_4`'s acc files are all semicolons, while `Test_set/Bearing1_4`
   (same bearing, truncated copy) is comma-delimited. It's consistent *within* one file and one
   (bearing, file-type) pair — we never observed a mid-run switch — but never assume comma
   globally. The loader sniffs the delimiter per file.
2. **Temp files are sometimes shorter than 600 rows**, always at a run boundary (last file of
   the run almost always, occasionally the first). This is the sensor's partial final/initial
   buffer flush, not corruption — `Test_set` and `Validation_Set` agree on the same short files
   for shared bearings. Don't assert exactly 600 rows; the loader just reads whatever's there.
3. **5 bearings have zero temperature data**: `Bearing2_2`, `Bearing3_2` (Training),
   `Bearing1_3`, `Bearing2_3`, `Bearing2_6` (Validation/Test). This is a real absence in the
   raw release, not a naming mismatch — the loader returns an empty (but correctly-shaped)
   DataFrame for these.

## RUL labeling scheme (decided)

Raw time-to-failure in seconds: `RUL(file_index) = (total_snapshots_in_full_run − file_index) × 10`.
Chosen over a 0–1 life-percentage scheme because it matches the units the scoring metrics
(RMSE + PHM12 asymmetric scoring) actually operate on. Percentage-of-life labels would need a
separate estimate of a test bearing's total lifespan to convert back to seconds at inference
time — which is exactly the unknown the challenge is asking us to predict — so they don't map
cleanly onto the eval metric. Implemented in `femto_rul.labeling.rul`:
- `label_full_run_bearing(bearing_dir)` — for `Training_set` bearings and `Validation_Set`
  bearings considered on their own (both are full runs, so total length is just the bearing's
  own file count).
- `label_truncated_bearing(test_bearing_dir, validation_bearing_dir)` — for `Test_set` bearings,
  using the matching `Validation_Set` bearing for the true total run length.

Not yet implemented: a piecewise/capped RUL (flat ceiling during the healthy-life region, where
the vibration signature carries no degradation signal to learn from). Worth revisiting if
baseline RMSE is dominated by early-life predictions.

## Feature extraction schema (decided)

One row per `(split, bearing, file_index)` snapshot. Implemented in `femto_rul.pipeline`
(`build_bearing_dataset` / `build_full_dataset`). Columns:

- Metadata: `split`, `condition`, `bearing`, `elapsed_time_seconds`, `file_index`
- Per channel (`_horiz` / `_vert` suffix) — time domain (`femto_rul.features.time_domain`):
  `rms`, `kurtosis` (excess/Fisher, Gaussian ≈ 0), `skewness`, `crest_factor`
- Per channel — frequency domain (`femto_rul.features.frequency_domain`): `fft_band_0`..`fft_band_7`,
  8 equal-width bins from 0Hz to the 12.8kHz Nyquist frequency. Chose generic equal-width bins
  over bearing-fault-frequency-targeted bins (BPFO/BPFI/BSF/FTF) since the latter needs the
  PRONOSTIA bearing geometry spec verified first — worth revisiting as a stretch goal.
- Label: `rul_seconds`

This is the schema to hand off to Person B's orchestrator — 30 columns total (5 metadata + 24
feature + 1 label). Run `python scripts/extract_features.py` to regenerate
`data/processed/features.parquet` (gitignored, ~20min single-threaded over the full dataset).

## Open decisions (not yet made — flag here once decided)

- [ ] Bearing-level train/test split strategy within `Training_set` for local validation
      (separate from the official `Test_set`/`Validation_Set` held-out bearings)
- [ ] Whether to add a piecewise/capped RUL label for the healthy-life region
- [ ] Whether to pursue bearing-fault-frequency-targeted FFT bins (needs geometry spec lookup)
