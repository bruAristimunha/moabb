# UNITS_FIXES_HIGH — high-amplitude / mixed-unit suspects (corpus amplitude audit)

Audit context: cached (4–38 Hz bandpassed) EEG amplitudes, `amp_p50` in µV;
healthy corpus median ≈ 4 µV, acceptable band ≈ 1–50 µV. Fix pattern follows the
established exemplars (yang2025, zhang2017, tavakolan2017, wrcc2023_mi_a,
hygrip2020, aguilera_rodriguez2025). All fixes are code-only edits, nothing
committed.

## Summary table

| Dataset | Cached amp_p50 (µV) | Root cause | Fix | Expected after fix | Confidence |
|---|---|---|---|---|---|
| Peterson2022 | 13,529,297 | EDF physical-dimension fields blank → MNE reads µV numbers as V (×1e6 high) | `moabb/datasets/peterson2022.py:221-229` — scale EEG picks ×1e-6 after BIDS load | ~3–14 µV | certain (source EDF header + MNE load verified) |
| Perdikis2018 | 4,648,617 | CNBI GDF2: phys-dim code 0 (unrecognised) → MNE unit scale 1; float samples calibrated dig ±3.4e38 → phys ±262144 (µV numbers) read as V | `moabb/datasets/perdikis2018.py:364-371` — ×1e-6 after channel cleanup | ~4.6 µV | certain (GDF header + MNE load verified: 4.58e6 → 4.58 µV) |
| Sun2026 | 4,283,351 | BrainVision IEEE_FLOAT_32 written with µV-scale numbers as if volts (channels.tsv units "V"; creators fed µV into a volts-expecting pybv pipeline) | `moabb/datasets/sun2026.py:274-283` — ×1e-6 on all non-stim channels after `read_raw_bids` | ~4–10 µV | certain (source .vhdr/.eeg bytes + MNE load verified: 9.86e6 → 9.9 µV on sub-01 ses-01 run-01) |
| SpinalStim2025 | 3,238,196 | CNBI GDF2: phys-dim code 0, identity calibration (dig = phys = ±3.4e38) → stored µV floats read as V | `moabb/datasets/spinalstim2025.py:309-316` — ×1e-6 after reorder (EEG + sens EOG) | ~3–6 µV | certain (GDF header + data section + MNE load verified: 6.27e6 → 6.27 µV on Subject_501 r001) |
| Patel2025 | 4,551 | .mat trial values are nV-scale (µV×1000, e.g. undivided gain in the processed redistribution); loader applied the µV assumption ×1e-6 → ×1e3 high | `moabb/datasets/patel2025.py:323-330` — factor 1e-6 → 1e-9 | ~4.6 µV | likely (exact ×1000 amplitude arithmetic; .mat not probed) |
| Rozado2015 | 0.146 (ratio 38) | XDF EEG stream carries µV (standard LSL BioSemi output), loader assumed raw 24-bit counts (LSB 31.25 nV) → ×32 low (1e-6/31.25e-9 = 32; observed deficit ~27×) | `moabb/datasets/rozado2015.py:277-284` — 31.25e-9 → 1e-6 | ~4.7 µV | likely (exact ×32 arithmetic; all subjects low-side so spread=38 is inter-subject variability, not mixed units) |
| Vasilyev2021 | 1,247 (ratio 16,015) | NOT a clean unit factor — see diagnosis below | none applied | n/a | needs-source-probe |
| AlexMI | 0.10 (ratio 247) | NOT a loader/source units bug — source fifs are healthy; anomaly is cache-side | none applied | n/a | certain loader is clean; cache anomaly needs-cache-probe |

## Files edited

- `moabb/datasets/peterson2022.py` (UNITS FIX at line 221; scale at line 229)
- `moabb/datasets/perdikis2018.py` (line 364; scale at 371)
- `moabb/datasets/sun2026.py` (line 274; scale at 283)
- `moabb/datasets/spinalstim2025.py` (line 309; scale at 316)
- `moabb/datasets/patel2025.py` (line 323; factor change at 330)
- `moabb/datasets/rozado2015.py` (line 277; factor change at 284)

Not edited: `moabb/datasets/vasilyev2021.py`, `moabb/datasets/alex_mi.py` (see below).

## Per-dataset detail

### Peterson2022 — CERTAIN, fixed
- Loader: `BaseBIDSDataset` → `mne_bids.read_raw_bids` → `read_raw_edf`; no scaling in loader.
- Evidence: EDF header of `ds003810 sub-02_task-MIvsRest_run-1_eeg.edf` has an
  EMPTY 8-char physical-dimension field for all 16 signals (phys ranges like
  12482..15501 = µV numbers with DC offset). MNE maps empty units to scale 1
  (`_orig_units` = 'n/a') → values stay numerically µV but are labelled volts.
  Full-file MNE check: 4–38 Hz median |x| = 3.26e6 µV; ×1e-6 → 3.26 µV.
- Fix: in `_get_single_subject_data`, per run, `raw._data[eeg_picks] *= 1e-6`
  before annotations→stim conversion.

### Perdikis2018 — CERTAIN, fixed
- Loader: `mne.io.read_raw_gdf`, no scaling.
- Evidence: real GDF captured from `MA25VE.tar.gz`
  (`MA25VE/MA25VE_20160811/MA25VE.20160811.144550.offline.mi.mi_bhbfrst.gdf`):
  GDF 2.00, 17 ch (`eeg:1..16`, `trigger:1`), ALL phys-dim codes = 0. MNE maps
  code 0 → scale 1 (only 4275=µV→1e-6, 4274=mV→1e-3 are converted). The float32
  samples are calibrated by MNE from digital ±3.4028e38 to physical ±262144 —
  i.e. numerically µV (g.USBamp ±262 mV input range in µV). MNE load of the
  padded capture: 4.5795e6 µV bandpassed (audit says 4.65e6 ✓); ×1e-6 → 4.58 µV.
- Fix: `_read_calibration_run`, after dropping non-EEG channels: `raw._data[:] *= 1e-6`.

### Sun2026 — CERTAIN, fixed
- Loader: `read_raw_bids` on BrainVision BIDS; no scaling.
- Evidence: `.vhdr` (pybv 0.7.6) declares `IEEE_FLOAT_32`, channels `0.1,µV`;
  `channels.tsv` declares units `V`. Stored float32 values are enormous
  (median raw value ~6e10, demeaned ~6.2e8): the creators passed
  microvolt-scale arrays into a volts-expecting writer (pybv scales V→stored
  by ×1e7 for the 0.1 µV resolution), so the file decodes — per its own
  header — to values exactly 1e6 above volts. Verified end-to-end with
  `read_raw_brainvision` on sub-01 ses-01 run-01: 4–38 Hz median 9.86e6 µV
  (audit dataset-wide 4.28e6 ✓ same order); ×1e-6 → 9.9 µV.
- Fix: `_get_single_subject_data`, after aux retype: `raw._data[non_stim_picks] *= 1e-6`
  (HEO/VEO/EKG/EMG share the amplifier scale; Trigger→stim excluded).

### SpinalStim2025 — CERTAIN, fixed
- Loader: `mne.io.read_raw_gdf`, no scaling.
- Evidence: real GDF header from the Zenodo d3 archive
  (`Subject_501_SinglePulse_Offline__s001_r001_...gdf`): GDF 2.00, 68 ch, ALL
  phys-dim codes = 0, digital = physical = ±3.4028e38 (identity calibration →
  stored floats pass through). Data section decodes to demeaned per-channel
  medians ~10–35 (numerically µV; DC offsets up to ~1.7e5 µV = 170 mV, normal
  for unfiltered amplifier µV). MNE load of the padded capture: 6.268e6 µV
  bandpassed; ×1e-6 → 6.27 µV (audit p50 3.24e6 → ~3.2 µV ✓).
- Fix: `_read_run`, after `reorder_channels`: `raw._data[:] *= 1e-6` (all
  remaining channels are EEG + sens7-9 EOG; Status/trigger dropped earlier).

### Patel2025 — LIKELY, fixed
- Loader: `sio.loadmat` trials → `* 1e-6` (µV assumption) → `RawArray`.
- Evidence (amplitude arithmetic, code-only): cached 4551 µV is ×1e3 above the
  ~4.5 µV physiological target, i.e. the .mat numbers are nanovolt-scale
  (µV×1000 — consistent with an undivided gain-1000 amplifier stage in the
  processed Figshare redistribution of the Geronimo 2016 g.USBamp/BCI2000
  recordings). Alternative explanations (V-scale or raw counts) are excluded
  because they would not land within the 1–50 µV band under any power-of-10.
- Fix: factor 1e-6 → 1e-9. Expected cached amplitude after fix ≈ 4.55 µV.
- Residual risk: none identified for the factor; the nV interpretation label
  ("undivided gain" vs deliberate nV export) is unverified — does not affect
  the factor.

### Rozado2015 — LIKELY, fixed
- Loader: pyxdf → `31.25e-9 * eeg_data` ("BioSemi 24-bit counts, LSB 31.25 nV").
- Evidence (amplitude arithmetic, code-only): cached 0.146 µV is ~27× LOW; the
  ratio between the µV interpretation (1e-6) and the count interpretation
  (31.25e-9) is exactly 32. LSL/XDF BioSemi acquisition streams standardly emit
  µV, not raw counts. After ×32, cached median ≈ 4.7 µV. The within-dataset
  max/min ratio 38 has every file on the LOW side (max ≈ 0.8 µV cached), which
  fits one uniform wrong factor + inter-subject variability and does NOT fit a
  counts/µV mixture (a genuine counts file would have sat at ~4 µV cached,
  giving a far larger ratio).
- Fix: 31.25e-9 → 1e-6.
- Ideal follow-up (not required for the factor): read the per-stream `unit`
  field from the XDF stream info and dispatch on it; the archives (RAR on
  Harvard Dataverse) were not probed.

### Vasilyev2021 — NEEDS-SOURCE-PROBE, not edited
- Loader: custom BCI2000 `.dat` parser; calibrates `(raw − offset) × gain × 1e-6`
  with header `SourceChGain`/`SourceChOffset` (µV assumption); silent fallback
  to gain=1/offset=0 if the header list is malformed (`_param_float_list`).
- Evidence from 26 sampled run headers+data (all 7 subjects, sessions 001/006,
  probed before the code-only directive): every sampled file is uniform —
  `BCI2000V=1.1, HeaderLen=768, SourceCh=30, DataFormat=int32,
  SourceChGain= 30 0.02 ×29 + 0.003 (last ch), SourceChOffset= −30`. The
  loader's parsing handles these headers CORRECTLY (no fallback triggered), yet
  the calibrated 4–38 Hz amplitude is 500–2060 µV per file (~×150–450 high,
  matching the audit p50 1247 µV). Dividing by ~256 lands every sampled file at
  2.5–7.0 µV, but a 24-bit-in-int32 bit-shift is EXCLUDED (low byte zero in
  only 0.4% of samples = chance). So the header's nominal 0.02 µV/count does
  not describe the true scale, and the needed correction is NOT a clean power
  of 10 — it looks like ~1/256 (amplifier-specific, possibly an NVX gain
  convention), but that cannot be asserted from code alone.
- The audit's max/min ratio 16,015 is NOT reproduced in the 26-file sample
  (spread only ~4×), so other files (e.g. the ~700 unsampled runs, feedback
  sessions) must carry a different scale — per-file dimension-aware handling
  is required, keyed on the actual per-file calibrated amplitude or on header
  fields not yet inventoried across the full archive.
- Proposed fix (pending a full-archive probe): after calibration, compute
  per-file `med = median(|signal − mean|)`; if `med > 1e-4` V (100 µV — no
  clean bandpassed EEG median sits there), apply the amplifier-specific
  correction (÷256 if confirmed by a raw-count → µV calibration table for the
  NVX/actiCHamp hardware; otherwise the nearest power-of-2/10 that brings
  `med` into 1–50 µV, logged per file). Do not ship a blanket factor.

### AlexMI — LOADER CLEAN (certain), audit anomaly is cache-side, not edited
- Loader (`moabb/datasets/alex_mi.py`, local copy of the classic MOABB
  dataset): `mne.io.Raw(fif)` — scaling comes from the fif calibration only.
- Evidence: all 8 Zenodo source fifs downloaded and measured (before the
  code-only directive): 4–38 Hz medians 2.70–4.13 µV, cal/range = 1.0
  everywhere, max/min ratio 1.53. The audit's 0.10 µV / ratio 247 (subjects at
  0.02 and 5.0 µV) is NOT producible by this loader from these files.
- Conclusion: no units fix belongs in the loader. The cached artifact was
  built from something else or corrupted in caching — the pattern (some
  subjects ~250× low) resembles the known channel-union zero-padding cache
  bug (cf. Dan2023/Schwarz2020), or a stale cache from an older loader.
  Action item: regenerate/probe the AlexMI cache, not the loader.

## Verification notes
- All six edited files pass `python -m py_compile`.
- The four "certain" fixes were validated by loading real source bytes through
  the same MNE readers the loaders use and reproducing the audit's cached
  amplitudes before the fix and 1–50 µV amplitudes after it.
- No other datasets were touched.
