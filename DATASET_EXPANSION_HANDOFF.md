# Dataset-Expansion Handoff — deep-ceiling MI benchmark

**Purpose:** resume the MOABB MI/ME dataset-expansion campaign from a cold start (new session / after credit runs out). Written 2026-07-25. Repo: this clone `~/Projects/moabb-datasets` (MOABB develop). Paper repo: `~/Projects/papers/deep-ceiling`.

**One-line status:** growing the benchmark past 100 MI/ME datasets by writing new MOABB `BaseDataset` loaders. Loaders are UNTRACKED / uncommitted (no PR, no push).

**STATE (2026-07-25): ~99 clean imagery datasets; catalog 197.** Batch-2 (22) + Batch-3 (8) registered green, THEN a `/code-review max` over all 40 active loaders (43 findings, 6 adversarially-confirmed) + a fix pass: **9 fixed, 2 UNREGISTERED**. Doi_cache total 203, test_metadata catalog 197.
- Batch-3 registered (7): han2026/Han2026[ds007327, MO-vs-MI -> STILL flagged CUT], batista2022, inear_mi2026[26subj 7ch ear-EEG; FIXED v5-fallback], pan2023[Harvard dvn/251now, != Pan2025/gh74zg], pardogarcia2026[FIXED: EMG-confirmed REAL motor not SSVEP; sessions vary], kodera2023, sun2026.
- Code-review FIXES applied (loader files only): hygrip2020 (data is MILLIVOLTS not uV -> factor 1e-3), medvedeva2026 (EMG-confirmed motor, kwarg forwarded), inear_mi2026 (MAT v5/v7.3 branch), neurotumbci (sessions_per_subject 5->3=min), sitstand2026 (uV->V + ch-name validation), moving2024 (trigger-match guard), rehab2025openbci (verified 1000Hz IS correct - Cyton WiFi), mind2026 (dropped unused return_all_modalities), pardogarcia2026 (variable sessions).
- **UNREGISTERED (confirmed unusable, .py left on disk like reyesjimenez2026): li2021** (Zenodo 4699203 has NO per-trial labels - MARKER all-zero, no annotations), **forenzo2025** (continuous 2D-cursor/robot-arm control, not discrete-trial MI).
- **DEFERRED to download-test rigor pass** (~15 label-inference mediums needing real-data confirmation): song2026 (digit-guess label map), peterson2022 (repurposed GDF codes), daly2020 (alpha-based labels), cai2026 (token validation), damm2026 (onset*sfreq^2), dfki2023 (substring marker), farabbi2020 (positional ch rename), pan2023 (label truncation), etc.
- **MARGIN available to exceed 100 comfortably:** ~15-25 genuinely-new OPEN datasets still unimplemented in `openalex_new.json` (only 8 of ~25-35 usable done); + IEEE 5 (pending AWS_SECRET); + Han2026 keep decision. Path to 105-112 is clear via another implement wave (systematic engine + Zenodo token both working).

**Systematic discovery engine (works, reusable):** OpenAlex `type:dataset` + `title_and_abstract.search` over the corpus-derived vocabulary (EEG/BCI/motor imagery/sensorimotor rhythm/ERD-ERS/attempted movement/reach-grasp/kinesthetic/grip-force/gait) -> dedup vs the accession index -> Zenodo API enrichment (needs ZENODO_TOKEN in .dataset_keys.env; public API 403s). Saved candidate list: `/Users/bruaristimunha/.claude/jobs/c6fa8046/tmp/openalex_new.json` (82 canonical, ~25-35 genuinely usable after collapsing NEMAR mirrors + figshare .vN + reprocessed). API keys in `.dataset_keys.env` (gitignored): ZENODO_TOKEN + ELSEVIER_API_KEY filled; AWS_ACCESS_KEY_ID filled, AWS_SECRET_ACCESS_KEY empty (BLOCKER for IEEE).

---

## ROUND 4 (2026-07-26): ~99 -> ~110 registering + IEEE downloading
- Track-B workflow wt15092zg: 16 candidates -> **12 implemented, 3 already_in_moabb, 1 excluded (no-data)**. Registering **11** via agent a9e4ac6b (round 4): martinezpeon2024/MartinezPeon2024, russo2024, wang2026, inmid2024[cross-cites Wirawan2024 zs25xxjkm9 in docstring - NOT a dup, downloads rjx76wd5v6], patel2025, vagaja2023, ding2025[21s/128ch/4-finger], jia2019, vasilyev2021, li2026[IMU-MI_A 5-subj SAMPLE, laterality unverified], perdikis2018[CNBI Cybathlon, 2 subj]. **EXCLUDED openclose_hands_c4** (n_subjects=1, n_channels=1 - can't support multi-ch law features; .py left unregistered). => ~110 imagery datasets.
- **IEEE download SOLVED**: creds stored (gitignored .dataset_keys.env + JZ ~/.aws/credentials); IEEE S3 needs auth (no anon, GetObject-only, ListBucket denied). S3 paths from each dataset's "ACCESS ON AWS" modal (JS-extract `s3://ieee-dataport/data/<data-id>/...`): Chu&Bi `data/1159659/Setdata_IEEE.zip` (DONE 1.6GB), Emotiv `data/109082/101215/EDF.rar`, EMGEEG `data/1164744/BMIS_EEG_DATA.zip`, Chu2024 `data/1235195/EEG.zip`, ULMEE `data/1402708/92674/ULMEE.zip`. Download via **archive SLURM job (account tst@cpu)**, NOT the login node (flaky S3) and NOT $SCRATCH/bin in the script ($SCRATCH unset in batch -> use absolute /lustre/fsn1/projects/rech/tst/uiy14ex/bin/s5cmd). Latest job **277794**. Verify per-file rc in dl_ieee_<job>.out; then implement 5 IEEE loaders vs the real zips.
- NEXT: registration round-4 green -> code-review + download-test rigor pass on rounds 3-4 loaders (same as rounds 1-2). Reuse implement script: wf scriptPath implement-new-mi-datasets-wf_71c1f2e2-f2f.js.

## 0. HOW TO RESUME (do this first)

```bash
cd ~/Projects/moabb-datasets
git status --porcelain | grep '^??' | grep datasets/    # untracked new loaders
git diff --stat -- moabb/datasets/__init__.py moabb/tests/doi_cache.json \
  moabb/datasets/summary_imagery.csv moabb/tests/test_metadata.py \
  moabb/tests/test_datasets.py docs/source/api.rst docs/source/whats_new.rst
python -c "import moabb.datasets"                        # must import clean
```
Then read this file top-to-bottom and pick up at "PENDING WORK".

**Registration pattern (7 files per dataset)** — learn it from the uncommitted batch-1 diff, then replicate:
1. `moabb/datasets/<module>.py` (loader — usually already written)
2. `moabb/datasets/__init__.py` — `from .<module> import <Class>` (alphabetical)
3. `moabb/datasets/summary_imagery.csv` — one row, columns per header, values from `<Class>.METADATA`
4. `moabb/tests/doi_cache.json` — one entry + bump `_metadata.total`
5. `moabb/tests/test_metadata.py` — bump the single hardcoded catalog-count assertion
6. `moabb/tests/test_datasets.py` — add to `NEMAR_ID_EXEMPT` ONLY if not an OpenNeuro dataset
7. `docs/source/api.rst` + `docs/source/whats_new.rst`

Gate: `python -m pytest moabb/tests/test_metadata.py moabb/tests/test_datasets.py -x -q`

---

## 1. RUNNING JOBS (check on resume; may have finished)

| What | ID | Check |
|---|---|---|
| Registration agent (22 loaders) | agent `a9e4ac6b2db97d889` | `git diff --stat`; corrected to DROP upperlimbrehab2025 |
| To-100 screening workflow (17 cands) | workflow `wompkw5e3` | `/workflows`; returns IMPLEMENT/ON_REQUEST/EXCLUDE/ALREADY_IN_MOABB |
| MIND2026 single-stream download | SLURM `245759` on JZ | see §5 |

`ssh jeanzay 'bash -lc "squeue -u \$USER"'` — NOTE the `bash -lc` wrapper (see §4 gotcha).

---

## 2. REGISTER SET — 22 loaders (this batch)

**15 deep-screen** (all smoke-tested vs real data): cai2026/Cai2026(ds006840), moving2024/MOVING2024, peterson2022/Peterson2022(ds003810), dfki2023/DFKI2023, lee2022/Lee2022(ds004022), lu2026/Lu2026, lomele2026/Lomele2026, openvibe/OpenViBE, hygrip2020/HYGRIP2020, forenzo2025/Forenzo2025, pan2025/Pan2025, lioixp1/LioiXP1(ds002336), rehab2025openbci/Rehab2025OpenBCI, polohortiguela2025/PoloHortiguela2025, neurotumbci/neuroTUMBCI.
**6 reserve-clean → 5** after dedup: farabbi2020/Farabbi2020, lioi2020/Lioi2020(ds002338), medvedeva2026/Medvedeva2026, sitstand2026/SitStand2026, spinalstim2025/SpinalStim2025.
**+ wirawan2024/Wirawan2024** (honest 3-class folder {left_hand,right_hand,trunk}), **+ li2021/Li2021** (PROVISIONAL — validate in download-test or prune).

OpenNeuro nemar_ids (do NOT NEMAR-exempt): cai2026=ds006840, peterson2022=ds003810, lee2022=ds004022, lioixp1=ds002336, lioi2020=ds002338. All others EXEMPT.

## 2b. EXCLUDED (leave .py unregistered; do not re-add)
- **upperlimbrehab2025** = DUPLICATE of registered `chang2025` (figshare 28831730.v2, same file IDs).
- **quirogaforero2025** = 5-ch .npy disproven as 5-channel (single-channel trials, unreconstructable electrodes) → can't support multi-ch law features.
- **mousecursor2025** = marker columns empty for every subject (labels not data-borne).
- **luciw2014** (WAY-EEG-GAL) = weight/friction object conditions, not movement classes.
- **forenzo2024** = in MOABB `_REMOVED_DATASETS`.
- **SignEEG2024** = signature biometric, not MI.

## 2c. ALREADY IN CLONE (prior wave — never re-implement; DOI-dedup catches these)
Yi2025(s41597-025-05286-0), Yang2025(s41597-025-04826-y, WBCIC-SHU), HefmiIch2025(s41597-025-06100-7), Chang2025(figshare 28831730). Plus the whole existing MOABB catalog (~120 subclasses, 276 accessions indexed).

---

## 3. PENDING WORK (ordered)

1. **Finish registration** of the 22 → gate green (agent a9e4ac6b).
2. **Rigor pass** on the 22 (same as batch-1): `/code-review max`-style review → apply fixes → REAL download-test via `Dataset().get_data([subject])`. **Li2021 first** (validate or prune). Prune any that fail like ReyesJimenez2026 was.
3. **To-100 push:** take workflow `wompkw5e3`'s IMPLEMENT list (~15 open candidates: Ahn2013, Spuler2018, Lee2016[check vs Lee2019!], DFKI2024/zenodo.8345429, Xu2021, Hooks2023, Chu&Bi 6-class/ieee 8qw6-f578, Sun2024, Saichoo2026, Li2025, Chu2024/ieee f1m3-fh49, Mensah/ds006126, Toni2026, ULMEE2025/ieee efjn-d211, Emotiv-Insight-Thwe2026/ieee c8f1-eg58, EMG-EEG-Lee2023/ieee 5ztn-4k41) → implement + register + rigor pass. Lands ~100–108.
4. **Systematic keyword sweep (user's methodology ask):** harvested "beyond-MI" vocabulary from our own catalog — kinesthetic, ERD/ERS, sensorimotor rhythm, mu rhythm, attempted movement, reach-to-grasp, grasp-and-lift, finger tapping, dorsiflexion/plantarflexion, gait/walking imagery, neurofeedback, self-paced, exoskeleton/orthosis, same-limb, elbow/wrist/shoulder/ankle, imagined speech(exclude), grip force. Run each term × repository (OpenNeuro, Zenodo, figshare, Mendeley, ScienceDB, IEEE-DataPort, OpenReview, Dryad) → dedup vs the 276-accession index → screen → new candidates. This is the exhaustive PRISMA-style sweep beyond plain "motor imagery".
5. **On-request pool** (needs the USER to request access; do NOT auto-download or submit personal-data forms): SEFMID (ScienceDB sciencedb.30795, kaiwu@scut.edu.cn), IMU-MI-A (emailed), UET175, Ma2022/SHU, Bodda2022, REH-MI/Altaheri, LopezLarraz2015/2025.

---

## 4. GOTCHAS

- **`$SCRATCH` does NOT expand over non-interactive `ssh jeanzay '...'`** → wrap in `bash -lc "..."`. Real path `/lustre/fsn1/projects/rech/tst/uiy14ex`.
- **Always DOI/accession-dedup new loaders vs the clone's full catalog before registering** (prior waves exist; 2 dups caught this session). Index = all DOIs + `ds######` + `zenodo:` + `figshare:` + `ieee:` + class names.
- **Labels must be data-borne** (events/markers/folders/label array/stim channel), never inferred from acquisition order (Song2026/MouseCursor2025/Wirawan2024 traps).
- Nature `www.nature.com` auth-walls scrapers (303→idp) → use Europe PMC / PMC / the repository page.
- **NEVER** submit ScienceDB/restricted access personal-data forms or create accounts on the user's behalf — flag for the user.
- IEEE-DataPort datasets: often need a free login → treat as ON_REQUEST unless a direct open URL exists.
- Loaders stay UNTRACKED — no git add/commit/push/PR unless the user explicitly asks.

---

## 5. JEAN ZAY DOWNLOAD STATE

Dir: `$SCRATCH/datasets/moabb_new/` (~111 GB). Done: MILimbEEG.zip, KMIHandGrip2025.zip, PerezBlanco2026, Damm2026 (BIDS sub-01..18), OpenNeuro/Mendeley batch (job 245269 COMPLETED).
In flight: **MIND2026_V3.zip** (single-stream `aria2c -x1 -s1`, ~74 GB target; SLURM `245759`). MIND corrupts under parallel ranges (ScienceDB on-the-fly zip) — MUST stay single-stream. Verify with python `zipfile`, NOT `unzip` (Info-ZIP mis-flags 74 GB ZIP64).
Queues with internet: prepost, archive (login node frontal/jean-zay1 also allowed). Compute nodes have NO internet. Tools: aria2 module, s5cmd at `$SCRATCH/bin`, aws-cli/rclone modules. OpenNeuro = `s5cmd --no-sign-request cp 's3://openneuro.org/<ds>/*' .`.

---

## 6. KEY FILES / POOLS

- This handoff + `MEMORY.md` → memory `moabb-dataset-expansion-run.md` (fuller narrative), `dataset-paradigm-audit.md`, `only-52-datasets-scope.md`.
- Candidate pools: `/tmp/mi_final.json` (83), `/tmp/mi_tier3.json` (33), `/tmp/mi_verified2.json` (5 on-request). Many already implemented — dedup first.
- Plan: `docs/superpowers/plans/2026-07-25-moabb-dataset-expansion.md`.
- Papers: `~/Projects/moabb-datasets/dataset_papers/` (7, all OA).
- Downloaded pages: `~/Projects/papers/deep-ceiling/Dataset Downloads _ OpenViBE.html`.

---

## 7. SCOPE NOTE

The paper's *original* canonical law used 52 consolidated datasets (`~/Projects/papers/deep-ceiling/analysis/law_v2/frame_canonical_table.csv`). The current push is a deliberate BENCHMARK expansion (user wants ≥100) — more datasets raise the law's statistical power (n was the limiter). Tail datasets skew small-n / niche / gated: keep for breadth, but flag each so the user can keep-or-cut before the law re-fit.
