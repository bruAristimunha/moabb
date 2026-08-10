"""EEG Motor Intention Dataset for Rehabilitation-Oriented BCIs (Quiroga Forero et al., 2025)."""

import logging
from pathlib import Path

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# Zenodo record 17980608. The curated low-density subset used in the
# accompanying article: 24 subjects, five motor-related electrodes, binary
# rest-vs-move, distributed as one ``.npy`` per subject (~15 MB total). This
# avoids the ~3.2 GB Full_Dataset.rar, whose _raw.fif files carry no event
# markers and are therefore not epochable.
QUIROGAFORERO2025_URL = (
    "https://zenodo.org/api/records/17980608/files/5_Channels_Dataset.rar/content"
)

# Real subject identifiers of the 24 ``.npy`` files inside the archive. The
# curated subset dropped 6 of the 30 recorded volunteers (3, 12, 13, 20, 22,
# 25) for signal quality, so the file numbering is not contiguous. MOABB
# subject i (1..24) maps to QUIROGAFORERO2025_SUBJECT_IDS[i - 1].
QUIROGAFORERO2025_SUBJECT_IDS = [
    1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15,
    16, 17, 18, 19, 21, 23, 24, 26, 27, 28, 29, 30,
]

# The five motor-related electrodes documented for the curated subset. The
# distributed .npy stores one channel's time series per row, and the per-row
# electrode identity is not preserved (see the class docstring), so the loader
# exposes a single representative motor channel; Cz is the vertex electrode for
# the lower-limb (ankle dorsiflexion) task the accompanying article analyses.
QUIROGAFORERO2025_ELECTRODES = ["Cz", "C3", "C4", "Fz", "Pz"]
QUIROGAFORERO2025_CHANNEL = "Cz"

# Native acquisition rate (Hz). Each curated trial is 1024 samples long
# (2.048 s at 500 Hz), stored as a row of the subject .npy.
QUIROGAFORERO2025_SFREQ = 500.0
QUIROGAFORERO2025_TRIAL_SAMPLES = 1024


class QuirogaForero2025(BaseDataset):
    """Motor-intention (rehabilitation) EEG dataset from Quiroga Forero et al., 2025.

    .. admonition:: Dataset summary

        =============== ======= ======= ========== ================= ============ ============= ===========
        Name              #Subj   #Chan   #Classes   #Trials / class   Trials len   Sampling rate   #Sessions
        =============== ======= ======= ========== ================= ============ ============= ===========
        QuirogaForero2025    24       1          2               40          2.048 s        500 Hz           1
        =============== ======= ======= ========== ================= ============ ============= ===========

    **Dataset description**

    EEG recordings acquired at CIRINS (Facultad de Ingenieria, Universidad Nacional
    de Entre Rios, Argentina) from 30 healthy volunteers (16 F / 14 M, mean age
    28 +/- 6.6 years) performing cued *motor-intention / motor-execution* tasks of
    the dominant side (lower-limb ankle dorsiflexion or upper-limb hand
    open/close). Signals were recorded with an ANT Neuro ``eego`` amplifier and a
    32-channel cap (international 10-20 system) at 500 Hz, average-referenced.

    This loader targets the **curated low-density subset** distributed as
    ``5_Channels_Dataset.rar``, which is the data used in the accompanying
    article. It restricts the recording to five motor-related electrodes
    (Cz, C3, C4, Fz and Pz) and keeps only 24 of the 30 volunteers, selected on
    signal quality. Each subject is provided as a single ``<id>.npy`` file of
    shape ``(80, 1025)``: 80 single-trial rows (the first 40 are movement, the
    last 40 are rest), each row holding 1024 signal samples followed by a binary
    label in the last column. The loader reads that label column directly to
    build a 2-class **rest vs move** task (``rest`` -> 1, ``move`` -> 2).

    .. note::

        The full 32-channel ``Full_Dataset.rar`` was *not* used: as inspected
        for subject S01 its ``_raw.fif`` files carry no cue/event markers and
        are therefore not epochable. The curated ``.npy`` subset is the only
        distributed source with recoverable per-trial labels.

    .. warning::

        The distributed ``.npy`` stores one channel's time series per row
        (1024 samples, not divisible by 5), and the rows are **not** a
        time-aligned five-channel recording: rows show no cross-channel
        correlation within a trial and no recoverable channel grouping, so the
        original five-electrode spatial arrangement cannot be reconstructed
        from the array. The loader consequently exposes the signal as a single
        motor channel labelled ``Cz`` (the vertex electrode for the analysed
        lower-limb task). The binary rest/move labels are exact; the channel
        identity is a documented best-effort inference. The label-to-class
        assignment (last column: 1 -> ``move``, 0 -> ``rest``) is confirmed by
        the movement rows carrying lower band power than the rest rows in 23 of
        24 subjects (event-related desynchronisation).

    Parameters
    ----------
    subjects : list of int | None
        Restrict to a subset of the 24 subjects (1..24). ``None`` loads all.

    References
    ----------

    .. [1] Quiroga Forero, A., Acevedo, R. C., Rufiner, H. L. (2025).
       EEG Motor Intention Dataset for Rehabilitation-Oriented BCIs. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.17980608

    Notes
    -----
    .. versionadded:: 1.1.1
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=1,
            channel_types={"eeg": 1},
            montage="10-20",
            hardware="ANT Neuro eego amplifier, 32-channel EEG cap",
            reference="average",
            ground=None,
            impedance_threshold_kohm=5,
            sensors=[QUIROGAFORERO2025_CHANNEL],
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=24,
            health_status="healthy",
            gender={"female": 16, "male": 14},
            age_mean=28.0,
            age_std=6.6,
            age_min=18.0,
            age_max=60.0,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["rest", "move"],
            trial_duration=2.048,
            study_design=(
                "Cued motor-intention/execution of the dominant side. Lower-limb: "
                "ankle dorsiflexion; upper-limb: hand open/close. Visually and "
                "auditory cued with a random 5-6 s rest and a movement window. The "
                "curated subset collapses left/right movement into a single move "
                "class and provides a balanced binary rest-vs-move task (40 move "
                "and 40 rest single-trial epochs per subject)."
            ),
            stimulus_type="visual and auditory",
            stimulus_modalities=["visual", "audio"],
            synchronicity="cue-based",
            mode="offline",
            events={"rest": 1, "move": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.17980608",
            description=(
                "EEG motor-intention dataset for rehabilitation-oriented BCIs: "
                "curated low-density subset of 24 volunteers, five motor-related "
                "electrodes reduced to a single motor channel, binary rest-vs-move "
                "task provided as per-subject .npy trial arrays."
            ),
            investigators=[
                "Alejandro Quiroga Forero",
                "Ruben Carlos Acevedo",
                "Hugo Leonardo Rufiner",
            ],
            institution=(
                "Centro en Ingenieria en Rehabilitacion e Investigaciones "
                "Neuromusculares y Sensoriales (CIRINS), Facultad de Ingenieria, "
                "Universidad Nacional de Entre Rios"
            ),
            country="AR",
            data_url="https://doi.org/10.5281/zenodo.17980608",
            publication_year=2025,
            ethics_approval=[
                "Central Bioethics Committee of the Province of Entre Rios, Argentina "
                "(IS004678); ClinicalTrials.gov NCT06861517"
            ],
            keywords=[
                "motor intention",
                "motor execution",
                "rehabilitation",
                "BCI",
                "EEG",
                "ankle dorsiflexion",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(modality=["Motor"], type=["Motor Execution"]),
        file_format="NPY",
    )

    def __init__(self, subjects=None, *, return_all_modalities=False):
        self.events = {"rest": 1, "move": 2}
        super().__init__(
            subjects=list(range(1, 24 + 1)),
            sessions_per_subject=1,
            events=self.events,
            code="QuirogaForero2025",
            # tmax is one sample short of the full 1024-sample trial so the
            # inclusive-tmax epoch spans exactly one trial without bleeding into
            # the next concatenated trial (and keeps the final epoch in range).
            interval=(0, (QUIROGAFORERO2025_TRIAL_SAMPLES - 1) / QUIROGAFORERO2025_SFREQ),
            paradigm="imagery",
            doi="10.5281/zenodo.17980608",
            selected_subjects=subjects,
            return_all_modalities=return_all_modalities,
        )

    def _extract_archive(self, path_rar, path_folder):
        """Extract the RAR archive into ``path_folder`` if not already done."""
        if (path_folder / "5_Channels_Dataset").is_dir():
            return
        try:
            import rarfile

            with rarfile.RarFile(str(path_rar)) as rf:
                rf.extractall(str(path_folder))
            return
        except Exception as exc:  # pragma: no cover - env dependent
            log.warning("rarfile extraction failed (%s); trying patoolib.", exc)
        try:
            import patoolib

            patoolib.extract_archive(
                str(path_rar), outdir=str(path_folder), verbosity=-1
            )
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "Could not extract QuirogaForero2025 RAR archive. Install a RAR "
                "backend (the `unar`/`unrar` binary plus the `rarfile` or "
                "`patool` Python package) and retry."
            ) from exc

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the ``.npy`` file path for a single subject.

        Parameters
        ----------
        subject : int
            The MOABB subject number (1-24) to fetch data for.
        path : None | str
            Location where the data is stored / will be downloaded.
        force_update : bool
            Force re-download even if a local copy exists.
        update_path : bool | None
            Update the MNE config path.
        verbose : bool, str, int, or None
            Override the default verbose level.

        Returns
        -------
        list of str
            A one-element list with the path to the subject's ``.npy`` file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path_rar = Path(
            dl.data_dl(
                QUIROGAFORERO2025_URL,
                self.code,
                path=path,
                force_update=force_update,
                verbose=verbose,
            )
        )
        path_folder = path_rar.parent
        self._extract_archive(path_rar, path_folder)

        real_id = QUIROGAFORERO2025_SUBJECT_IDS[subject - 1]
        npy_path = path_folder / "5_Channels_Dataset" / f"{real_id}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(
                f"Missing curated file for subject {subject} "
                f"(expected {npy_path})."
            )
        return [str(npy_path)]

    def _get_single_subject_data(self, subject):
        """Return ``{"0": {"0": mne.io.RawArray}}`` for one subject.

        The subject ``.npy`` holds 80 single-trial rows of 1024 samples plus a
        binary label column. Rows are concatenated into one continuous
        single-channel recording, and an event is placed at each trial onset:
        ``move`` (label 1 -> code 2) or ``rest`` (label 0 -> code 1).
        """
        file_path = self.data_path(subject)[0]

        arr = np.load(file_path)
        signal = np.asarray(arr[:, :-1], dtype=np.float64)  # (n_trials, 1024), volts
        labels = np.asarray(arr[:, -1], dtype=int)  # 1 -> move, 0 -> rest
        n_trials, n_samples = signal.shape

        # Concatenate the trials into one continuous single-channel signal.
        data = signal.reshape(1, n_trials * n_samples)

        info = mne.create_info(
            ch_names=[QUIROGAFORERO2025_CHANNEL],
            sfreq=QUIROGAFORERO2025_SFREQ,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(data=data, info=info, verbose=False)
        try:
            raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
        except Exception:  # pragma: no cover - montage best-effort
            pass

        onset_samples = np.arange(n_trials, dtype=int) * n_samples
        onset_codes = np.where(labels == 1, self.events["move"], self.events["rest"])
        events = np.column_stack(
            (
                onset_samples,
                np.zeros(n_trials, dtype=int),
                onset_codes.astype(int),
            )
        )
        event_desc = {code: name for name, code in self.events.items()}
        annotations = mne.annotations_from_events(
            events, sfreq=raw.info["sfreq"], event_desc=event_desc, verbose=False
        )
        raw.set_annotations(annotations)

        return {"0": {"0": raw}}
