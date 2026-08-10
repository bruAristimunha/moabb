# CODEX HANDOFF — MOABB MI dataset expansion + validation

> For `/codex:setup`. Self-contained: continue the work if Claude runs out of weekly credit.
> Repo: `~/Projects/moabb-datasets` (this clone). Loaders are **untracked** — do NOT `git add/commit/push`.

## DURUM (state — verified)

- **119 registered motor-imagery datasets** (`moabb/datasets/summary_imagery.csv`, 119 unique rows). = 61 base moabb + **58 added this campaign**. Target **120** once `sensoryguidedmi2026` validates.
- Registration of the last 9 (`ma2022`=SHU, `ortiz2023`, `yilmaz2024`, `leeuwis2021`, `shin2022`, `wrcc2023_mi_a/b/c`, `zju_mi2025`) is **done + tests green**: `doi_cache.json` total=230, catalog=217, all NEMAR-exempt.
- **Data staged in `$SCRATCH/datasets` on Jean Zay** (NOT `$WORK`; `$HOME/mne_data` is a symlink into `$SCRATCH`). 226 `MNE-*` dirs, ~9.6 TB.
- Download fan-out (the 52 in `new_modules.txt`): **51/52 done**, `Ding2025` re-running (see jobs).
- Path-alignment (5 datasets whose bytes were in `$SCRATCH/datasets/moabb_new/`, no refetch): `Damm2026` ✅ clean; `Iwama2023` ✅ wired but PARTIAL data (see jobs); `Thapa2025` ✅ loader fixed; `MIND2026` ✅ loader rewritten Curry→BrainVision; `Garro2025` ❌ wrong bytes staged → re-downloading.

## DOSYALAR (uncommitted working-tree changes to know about)

- `moabb/datasets/utils.py` — added **7z fallback** in `download_and_extract_subject_zip` (Python `zipfile` can't do deflate64; fixed `Ding2025`). New helper `_extract_zip_with_7z`.
- `moabb/datasets/mind2026.py` — rewritten to read **BrainVision** (`.vhdr`) instead of Curry; METADATA `file_format="BrainVision"`.
- `moabb/datasets/thapa2025.py` — tolerate missing sessions + fixed `annotations.description` (was a `list`, broke `get_data` for all subjects).
- `moabb/datasets/__init__.py`, `summary_imagery.csv`, `moabb/tests/doi_cache.json`, `test_metadata.py`, `test_datasets.py`, `docs/source/{api,whats_new}.rst` — the 9 registrations.
- Helper scripts (untracked): `download_all.py`, `dl.slurm`, `dl_array.slurm`, `smoke_test.py`, `smoke_one.slurm`, `smoke_dev_loop.sh`, `new_modules.txt`, `smoke_list.txt`.

## BAGIMLILIKLAR (Jean Zay environment — REQUIRED to resume)

- ssh alias: **`jeanzay`**. Run remote: `ssh jeanzay bash -lc '<script>'`. Ignore a harmless `bash: -c: option requires an argument` line.
- `$SCRATCH` = `/lustre/fsn1/projects/rech/tst/uiy14ex` (often UNSET in bare `bash -lc` — use the literal path). `MNE_DATA` = `$SCRATCH/datasets`.
- Env: `source $HOME/bruno/jeanzay/activate_campaign.sh` → venv `neuralbench_campaign`, MNE config, **IDRIS http proxy** (needed for downloads), `MOABB_ACCEPT_LICENCE=1`.
- JZ clone (on PYTHONPATH): `/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno`. Deploy a local edit: `rsync -a moabb/datasets/<x>.py jeanzay:/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno/moabb/datasets/`.
- `7z` at `$HOME/bin/7z` (deflate64/RAR). Account: `-A tst@cpu`.
- **Partitions**: `archive` = works, use it. `prepost` = resource-starved (avoid). **dev QOS** = `--qos=qos_cpu-dev`, caps total submitted jobs (~10) → use a throttle+retry loop, never a big array.
- Prove a load is LOCAL (no refetch): run with `http_proxy= https_proxy= no_proxy='*' python ...` — any network attempt then fails instantly.

## TESLIM (pending work — acceptance criteria)

**A. Monitor 4 JZ download jobs, confirm each lands + loads (`squeue -u $USER`):**
- `Ding2025` job `281607` (~500 GB, deflate64-fixed) → confirm `MNE-ding2025-data` has all 21 subjects extracted.
- `Garro2025` job `282486` (raw per-subject zips from figshare article 27301629) → confirm `.vhdr` present, loads.
- `Iwama2023` job `282554` (fetches only missing EDFs; clone was 392/~480, `sub-001` empty) → confirm complete.
- `sensoryguidedmi2026` job `281220` = **DONE, 120 GB staged** (`MNE-sensoryguidedmi2026-data`). **CONFIRMED PARSER BUG → this is the only thing blocking #120:** `_read_run` (`sensoryguidedmi2026.py:345`) uses `scipy.io.loadmat`, but the `runData` `.mat` files are **MATLAB v7.3 (HDF5)** → `NotImplementedError: Please use HDF reader for matlab v7.3 files`. FIX: read with `h5py` (or `pymatreader.read_mat`, or `mat73`) and adapt the struct-field access — `run_data.trialSignal`, `.trialTargetClass`, `.trialInfo.target_label`, `.meta.selected_channels`, `.meta.sampling_rate_hz`. Gotchas with h5py v7.3: fields are HDF5 datasets/groups (not attrs), arrays come out **transposed**, and cell arrays (variable-length `trialSignal`) are stored as **object references** you must dereference. After fix: `get_data([1])` should return 4-class trials (left_hand/right_hand/up/down); confirm `BCI2000Control` group ≠ re-released `Stieger2021` subjects; then register (same 7-file pattern as the 9; bump `doi_cache` total, add NEMAR-exempt) = **dataset #120**. Validation command that surfaced this (bypasses the flaky activate script): `ssh jeanzay 'VPY=/lustre/fsn1/projects/rech/tst/uiy14ex/venvs/neuralbench_campaign/bin/python; MNE_DATA=/lustre/fsn1/projects/rech/tst/uiy14ex/datasets MOABB_ACCEPT_LICENCE=1 PYTHONPATH=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno $VPY -c "import moabb.datasets.sensoryguidedmi2026 as M; from moabb.paradigms import MotorImagery; d=M.SensoryGuidedMI2026(); print(MotorImagery().get_data(d,[d.subject_list[0]])[0].shape)"'`

**B. Fix the 13 smoke-test errors** (basic pipeline = `MotorImagery().get_data(d,[subj])` + Covariances+MDM, subject 1). After each fix: `rsync` to JZ, verify offline. Full list in `$CLAUDE_JOB_DIR/tmp/smoke_errors.txt` and below:

| Dataset | Error | Fix |
|---|---|---|
| Batista2022, InMID2024, PardoGarcia2026 | `Session names must be strings starting with an integer…` | rename session keys to `"0 desc"`/`"1 desc"` in `_get_single_subject_data` |
| Leeuwis2021, Medvedeva2026 | `Run names must be strings starting with an integer…` | rename run keys to integer-leading strings |
| Han2026 | `inst.filter requires raw data … preload=True` | pass `preload=True` when creating the Raw |
| Lioi2020 | `number of columns changed from 4 to 5 at row 5` | parse events/tsv with `usecols` (handle ragged rows) |
| PoloHortiguela2025 | `No objects to concatenate` | loader yields no data — inspect subject file layout |
| Shin2022 | `BCI2kReader is required` | `pip install BCI2kReader` into the campaign venv |
| neuroTUMBCI | `pyxdf needed` | `pip install pyxdf` into the campaign venv |
| Jia2019, PerezBlanco2026, Sun2026 | `ProxyError … figshare/openneuro Max retries` | loader hits network at load time on compute node; re-run under proxy, or cache the figshare file-id listing so `get_data` is offline |

**C. Re-run the dev smoke loop** on the fixed + newly-downloaded datasets (Ding2025, Garro2025, Iwama2023, sensoryguided, Damm2026, MIND2026, Thapa2025). Runner: `bash smoke_dev_loop.sh` reads `smoke_list.txt` (rebuild it: `grep -hE '^OK ' dlarr_*.out dl_login.log | awk '{print $2}' | sort -u > smoke_list.txt`, then add the new ones). Goal: 0 unexplained ERROR.

## KALITE (quality / done so far)

- 38/51 smoke OK; 13 errors triaged above. Loader fixes MIND2026 + Thapa2025 verified loading offline. Registration tests 375 passed / 0 failed.
- **Gated datasets (need the user, not code):** `song2026` (Baidu Netdisk only), `UET175` (author email — password RAR), `IMU-MI-A` (email `lijx@imu.edu.cn`), `SEFMID` (access form). Do not attempt to bypass access controls.
- **Cleanup (needs user OK — `rm`):** ~33 GB dead junk `MNE-mind2026-data/_src/` + `subject*/` symlinks (failed Curry extraction; loader ignores it).

## RULES
- Loaders untracked; no `git add/commit/push`, no branches/PRs unless the user asks. No `Co-Authored-By`.
- Prefer aligning existing bytes over re-downloading. Never bypass dataset access controls (passwords, gated portals).
- `rm`/delete → ask the user first.
- Scope is MOABB **imagery**-paradigm datasets (MI/ME). `analysis/law_v2/frame_canonical_table.csv` = the 52 paper datasets (out-of-scope extras exist in old grids — don't chase their failures).

---

## VERIFIED STATUS (2026-08-04 — offline smoke re-run, supersedes the 13-error table)

Re-ran the whole section-B smoke list **offline** (`http_proxy= no_proxy='*'`) on the JZ login
node against `moabb-datasets-bruno` (loaders re-synced from local). **All 12 former smoke
errors are fixed** — every session/run-name, preload, ragged-tsv and offline-guard fix is in:

- **10/12 load cleanly on the login node:** Shin2022 (261,16,751), Batista2022 (210,32,2501),
  InMID2024 (36,14,513 / 3-class), PardoGarcia2026 (283,59,1501), Leeuwis2021 (160,16,1251),
  Medvedeva2026 (180,8,2001), Han2026 (237,64,1001), PoloHortiguela2025 (66,28,1001),
  Jia2019 (80,63,3483), PerezBlanco2026 (360,8,1281 / 4-class).
- **Sun2026 = OK** (480,64,4001 / 4-class) — just slow (~3.5 min); offline run-probe guard works.
- **Lioi2020 = OK** (82,63,100001 / 2-class) on a compute node — the login-node exit-137 was
  pure RAM, not a code bug; it's past its ragged-tsv bug.
- #120 **SensoryGuidedMI2026 registered** (summary_imagery.csv row 120, `__init__` import present).
- **ma2022/SHU multi-session EDF loader done** (reads 5 EDF sessions, MAT fallback).

**Fresh datasets — ALL LOAD (compute-node job `563393`, offline):** Garro2025 (89,127,2001 /
3-class), Damm2026 (300,62,3073 / 5-class), MIND2026 (120,64,10001 / 4-class), Thapa2025
(354,31,1001 / 4-class), Iwama2023 (960,129,6001 / 2-class), Ding2025 (2200,128,3073 / 4-class,
1 TB, 223 s). **⇒ smoke verification COMPLETE — 0 unexplained errors across the whole list.**

---

## RESUME NOTES (2026-07-27 — READ FIRST, supersedes stale bits above)

**All downloads COMPLETE + verified.** Fan-out 52/52 (Ding2025 = **1.0 TB**, the deflate64/7z fix worked, OK). 5 aligned. sensoryguidedmi2026 = 120 GB staged.

**sensoryguidedmi2026 v7.3 parser — FIXED + deployed.** `_read_run` now uses `pymatreader` (installed in venv) instead of `scipy.io.loadmat`. Confirmed on a real file: `S006_sess05_run02.mat` → 63 ch, 160 k samples, no crash. **REMAINING for #120:** run the full `get_data([1])` (SLOW — S001 has 154 files across the 4 groups) to confirm labeled 4-class trials aggregate (one sampled run had 0 annotations — likely a rest run; EEGNet + `NUD` up/down runs carry labels), then register (7-file pattern, bump doi_cache, NEMAR-exempt). Verify with the venv-python-direct command already in the sensoryguided bullet above.

**SHU EDF — DOWNLOADED (full 5-session).** The user obtained the archive password **`shu-bci2022`** (their personal access — DO NOT hardcode it in the public loader). `edf_files.zip` (figshare 19228725, file 36728991) is extracted to `$SCRATCH/datasets/SHU_edf/edf/` = **125 EDF files = 25 subj × 5 sessions** (`sub-0NN_ses-0M_task_motorimagery_eeg.edf`). **TASK (user wants this):** rewrite `ma2022._get_single_subject_data` to prefer these 5 EDFs/subject (`mne.io.read_raw_edf`, keyed as sessions `0`–`4`) when present locally, else fall back to the current open v1 `mat.zip` (keeps the public loader reproducible for others WITHOUT the password). This restores proper multi-session structure — valuable for the cross-session / session-scaling ceiling analysis.

**13 smoke errors — REFINED status (agents diagnosed a lot before being stopped):**
- **Deps: BOTH INSTALLED in the venv (done)** — `BCI2kReader` (0.32.dev0) + `pyxdf` (1.17.5). → `neuroTUMBCI` now **PASSES offline** `(190,24,751)`. `Shin2022`'s dep error is gone but it now hits a **second bug**: `ValueError: Run names must start with an integer… found 'BW120'` — remap its run-dict keys to integer-leading (e.g. `'0BW120'` or `'0','1',…`) in `shin2022.py`, then verify.
- **Network-at-load (jia2019, perezblanco2026, sun2026): NOT transient — needs an offline guard, not a retry.** These loaders make an UNCONDITIONAL figshare/OpenNeuro API call every load (`dl.fs_get_file_list`/`fs_get_file_id`) that fails on compute nodes' flaky proxy. FIX = skip the API when local files already exist. `jia2019` (`data_path` ~L206, needs `exp1-S{n}-left/right.mat`) — offline-fail CONFIRMED, highest priority. `perezblanco2026` (~L242) — same pattern, guard on the extracted `sub-{n:02d}` files. `sun2026` — already has per-file `.exists()` guards; only its run-probe loop (~L321-333) still probes run N+1 over the network, wrap that so a connect error means "no more runs" when local runs exist. First `ls -R $MNE_DATA | grep -iE 'jia2019|perezblanco|sun2026'` to get exact on-disk paths for the checks.
- **8 loader-code fixes (session/run-name ×5, Han2026 preload, Lioi2020 ragged-tsv, PoloHortiguela2025 empty): agent STOPPED mid-work, state UNKNOWN** — may be partially edited/deployed, may be nothing. Do NOT assume done.
- **Codex procedure:** RE-RUN the dev smoke loop first — `bash smoke_dev_loop.sh` (dev QOS, 10 concurrent, submit-retry) — to get the CURRENT pass/fail, then apply the section-B fixes for whatever still fails. Verify each OFFLINE (`http_proxy= https_proxy= no_proxy='*'`) so a pass proves the compute-node/offline path works, not just the login node.

**Fresh datasets never smoke-tested:** Ding2025 (1 TB — slow), Garro2025, Damm2026, Iwama2023, MIND2026, Thapa2025. Rebuild `smoke_list.txt` to include these + run the checker to confirm they load.

**Env reminder that bit us:** `activate_campaign.sh` fails in one-liner `bash -lc` on some login nodes ("source: filename argument required"). Bypass by calling the venv binaries directly: `/lustre/fsn1/projects/rech/tst/uiy14ex/venvs/neuralbench_campaign/bin/python` (and `.../bin/pip`), with `MNE_DATA`/`PYTHONPATH` set inline. Proxy for fetches: `http_proxy=https_proxy=http://prodprox.idris.fr:3128`.
