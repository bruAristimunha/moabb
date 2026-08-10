"""Continuous reaching-and-grasping EEG-BCI dataset with a robotic arm.

Forenzo, Zhang, Wittenberg and He (2025), medRxiv.
Article DOI: 10.1101/2025.04.16.25325551
Data DOI: 10.1184/R1/28452131 (KiltHub / CMU figshare)
"""

import logging
from pathlib import Path

import mne
import numpy as np
from pymatreader import read_mat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import download_and_extract_subject_zip, set_neuroscan_montage


log = logging.getLogger(__name__)

# KiltHub (CMU figshare) per-subject ZIP file IDs (article 28452131, v1).
# Verified against https://api.figshare.com/v2/articles/28452131 (2026-07).
_FIGSHARE_BASE = "https://ndownloader.figshare.com/files/"
_FILE_IDS = {
    1: 52503659,
    2: 52503671,
    3: 52503665,
    4: 52503668,
    5: 52503653,
    6: 52503650,
    7: 52503647,
    8: 52503674,
    9: 52503662,
    10: 52503656,
}

# Subjects S01-S03 are stroke survivors, S04-S10 are healthy.
_STROKE_SUBJECTS = frozenset({1, 2, 3})

# The released data has no discrete class labels: the cursor moves continuously
# and the target drifts, so the only per-run markers are TrialStart / TrialEnd.
# We therefore expose a single event marking the onset of each 60 s trial.
_EVENTS = {"trial": 1}

_SFREQ = 1000.0
_N_EEG = 62


class Forenzo2025(BaseDataset):
    """Continuous reaching-and-grasping MI-BCI dataset from Forenzo et al. 2025.

    Dataset from the study *Continuous Reaching and Grasping with a BCI
    Controlled Robotic Arm in Healthy and Stroke-Affected Individuals* [1]_.

    Subjects performed motor imagery (MI) to continuously control a virtual
    cursor and/or a robotic arm in 2-D. The underlying MI control scheme is a
    5-state paradigm (left-hand MI -> left, right-hand MI -> right, both-hands
    MI -> up, resting -> down, foot MI -> click), but the *released* data does
    not contain discrete per-class trial labels: the cursor and target move
    continuously and the only markers per run are ``TrialStart`` / ``TrialEnd``.

    Ten subjects took part: **S01-S03 are stroke survivors** and **S04-S10 are
    healthy**. EEG was recorded with a 64-electrode Neuroscan Quik-Cap using
    the BCI2000 platform; the two mastoid electrodes (M1/M2) were not
    collected, so 62 channels are provided, sampled at 1 kHz (band-pass
    0.1-200 Hz, 60 Hz notch).

    Each session comprises groups of five back-to-back 60 s trials (a *run*)
    plus a 13th "chance level" run (screen off, no MI, random cursor motion).
    Five sessions per subject:

    - **Sessions 1-4 (Virtual Click Task)**: guide a cursor to a circular
      target and click (foot MI); Se01-Se03 cursor only, Se04 adds robotic-arm
      feedback.
    - **Session 5 (Cup Task)**: control the robotic arm to pick up, move and
      place three cups on shelves.

    Three online decoders were used across runs: ``AR`` (BCI2000
    autoregressive power spectrum), ``DL`` (EEGNet, standard training) and
    ``CL`` (EEGNet with mid-session recalibration); the decoder is encoded in
    each ``.mat`` filename.

    Because the target moves continuously, this is a continuous
    (regression-style) decoding dataset carrying **no discrete class labels**.
    This loader annotates each 60 s trial onset with a single ``"trial"``
    event; it does not map onto a standard N-class MI classification task
    without deriving labels from the target/cursor time series
    (``targetpos`` / ``cursorpos``), which is not attempted here.

    Notes
    -----
    The chance-level runs (no MI performed) are included as-is; take care when
    using them in analyses. Sessions/runs are keyed off the ``session`` / ``run``
    struct fields in each ``.mat`` file, robust to filename variation.

    References
    ----------
    .. [1] Forenzo, D., Zhang, Y., Wittenberg, G. F., & He, B. (2025).
           Continuous Reaching and Grasping with a BCI Controlled Robotic Arm
           in Healthy and Stroke-Affected Individuals. medRxiv.
           https://doi.org/10.1101/2025.04.16.25325551
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="standard_1005",
            hardware="Neuroscan Quik-Cap 64-ch (M1/M2 not recorded), BCI2000",
            sensor_type="Ag/AgCl",
            filters={"bandpass": [0.1, 200], "notch_hz": 60},
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="mixed (S01-S03 stroke survivors, S04-S10 healthy)",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=1,
            class_labels=["trial"],
            trial_duration=60.0,
            study_design=(
                "Continuous reaching/grasping MI-BCI controlling a virtual "
                "cursor and a robotic arm. 5-state MI control scheme "
                "(left/right/both-hands/rest/foot-click), but the release "
                "carries only continuous cursor/target time series and "
                "TrialStart/TrialEnd markers (no discrete class labels). "
                "Sessions 1-4: Virtual Click Task (Se04 adds arm feedback); "
                "Session 5: Cup Task. Online decoders AR / DL (EEGNet) / CL "
                "(EEGNet recalibrated). 12 task runs + 1 chance run per session, "
                "5 x 60 s trials per run."
            ),
            feedback_type="cursor",
            stimulus_type="continuous cursor / robotic arm",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1101/2025.04.16.25325551",
            investigators=[
                "Dylan Forenzo",
                "Yang Zhang",
                "George F. Wittenberg",
                "Bin He",
            ],
            institution_department="Department of Biomedical Engineering",
            institution="Carnegie Mellon University",
            country="US",
            data_url="https://kilthub.cmu.edu/articles/dataset/28452131",
            publication_year=2025,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=5,
        runs_per_session=12,
        tags=Tags(
            pathology=["Healthy", "Stroke"],
            modality=["Motor"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["trial"],
            imagery_duration_s=60.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=3900,
            trials_context=(
                "10 subjects x 5 sessions x ~12 runs x 5 trials = ~3000 task "
                "trials, plus chance-level runs; counts vary by subject/session."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["AR_linear_decoder", "EEGNet"],
            feature_extraction=["AR_spectral_estimation", "deep_learning"],
            frequency_bands={"alpha_mu": [8.0, 13.0]},
            spatial_filters=["Laplacian", "CAR"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject", "cross_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["cursor_control", "robotic_arm"],
            environment="laboratory",
            online_feedback=True,
        ),
        data_processed=False,
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=5,
            events=dict(_EVENTS),
            code="Forenzo2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1101/2025.04.16.25325551",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject grouped by session and run."""
        base = Path(self.data_path(subject)[0])
        mat_files = sorted(base.rglob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(
                f"No .mat files found for subject {subject} in {base}"
            )

        # Group runs by session, keyed on the eeg.session / eeg.run struct
        # fields (robust to whatever filenames the ZIP uses). Multiple .mat
        # files can share the same (session, run) numbers -- e.g. the three
        # online decoders AR / DL / CL, or a chance-level run reusing a run id.
        # Key the inner dict by (run_num, filename) so distinct files are never
        # silently overwritten.
        session_runs = {}
        for mf in mat_files:
            try:
                raw, sess_num, run_num = self._load_run(mf)
            except Exception as e:  # noqa: BLE001
                log.warning("Skipping %s: %s", mf.name, e)
                continue
            session_runs.setdefault(sess_num, {})[(run_num, mf.stem)] = raw

        if not session_runs:
            raise FileNotFoundError(f"No loadable runs for subject {subject}")

        sessions = {}
        for sess_num, runs in sorted(session_runs.items()):
            sessions[str(sess_num - 1)] = {
                str(run_idx): runs[key]
                for run_idx, key in enumerate(sorted(runs))
            }
        return sessions

    @staticmethod
    def _parse_int(value, default):
        """Extract the trailing integer from an id like 'Se04' / 'R02'."""
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else default

    def _load_run(self, mat_path):
        """Load one run .mat into an MNE Raw; return (raw, session, run)."""
        mat = read_mat(str(mat_path))
        eeg = mat.get("eeg", mat)

        data = np.asarray(eeg["data"], dtype=float)  # (62 x n_samples)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Ensure (channels x samples); the file stores channels along rows.
        if data.shape[0] != _N_EEG and data.shape[1] == _N_EEG:
            data = data.T

        # Channel names from the eeg.channellabels cell vector.
        labels = eeg.get("channellabels", None)
        if labels is not None:
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            ch_names = [str(c).strip() for c in labels]
        else:
            ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]
        # Keep only labelled EEG rows if extra rows are present.
        if data.shape[0] > len(ch_names):
            data = data[: len(ch_names), :]
        elif data.shape[0] < len(ch_names):
            ch_names = ch_names[: data.shape[0]]

        fs = float(eeg.get("fs", _SFREQ))
        times = np.asarray(eeg.get("times", []), dtype=float).ravel()

        # Scale microvolts -> volts.
        if np.abs(data).max() > 1e-3:
            data = data * 1e-6

        # Build a stim channel: one marker per TrialStart onset.
        stim = np.zeros((1, data.shape[1]))
        event = eeg.get("event", None)
        for latency in self._trial_start_latencies(event):
            if times.size:
                idx = int(np.searchsorted(times, latency))
            else:
                idx = int(round(float(latency) * fs))
            if 0 <= idx < data.shape[1]:
                stim[0, idx] = _EVENTS["trial"]

        info = mne.create_info(
            ch_names=ch_names + ["STI"],
            ch_types=["eeg"] * len(ch_names) + ["stim"],
            sfreq=fs,
        )
        raw = mne.io.RawArray(
            data=np.concatenate([data, stim], axis=0), info=info, verbose=False
        )
        # Neuroscan ALL_CAPS labels -> standard_1005 (stim channel untouched).
        set_neuroscan_montage(raw)

        sess_num = self._parse_int(eeg.get("session", ""), 1)
        run_num = self._parse_int(eeg.get("run", ""), 1)
        return raw, sess_num, run_num

    @staticmethod
    def _trial_start_latencies(event):
        """Yield the latency of each TrialStart event from the eeg.event struct."""
        if event is None:
            return
        # pymatreader returns a MATLAB struct array as a dict of parallel lists.
        types = event.get("type", None) if isinstance(event, dict) else None
        lats = event.get("latency", None) if isinstance(event, dict) else None
        if types is None or lats is None:
            return
        if not isinstance(types, (list, np.ndarray)):
            types, lats = [types], [lats]
        for t, lat in zip(types, lats):
            if str(t).strip().lower() == "trialstart":
                yield float(lat)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"S{subject:02d}"

        if subj_dir.exists() and list(subj_dir.rglob("*.mat")):
            return [str(subj_dir)]

        file_id = _FILE_IDS.get(subject)
        if file_id is None:
            raise ValueError(f"No download URL for subject {subject}")

        url = f"{_FIGSHARE_BASE}{file_id}"
        download_and_extract_subject_zip(url, sign, data_dir, path, force_update, verbose)
        return [str(subj_dir)] if subj_dir.exists() else [str(data_dir)]
