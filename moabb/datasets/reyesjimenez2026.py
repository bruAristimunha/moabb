"""ReyesJimenez2026 movement-related cortical potential (MRCP) dataset."""

import warnings
import zipfile as z
from pathlib import Path
from zipfile import BadZipFile

import mne
import numpy as np
import pandas as pd

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)


# Mendeley Data archive (whole-dataset zip cache). The per-file API for this
# record currently returns an empty listing, so we target the standard Mendeley
# S3 zip-cache URL pattern <id>-<version>.zip. See the needs-data note below.
REYESJIMENEZ2026_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/"
    "y23s2xg6x4-1.zip"
)

# 32 EEG channels, EMOTIV Flex 2 Saline, international 10-10 system.
# The article lists 31 names explicitly plus CPz among the nine
# motor electrodes of interest; CPz is inserted in the central-parietal row to
# reach the documented 32-channel montage.
REYESJIMENEZ2026_CHANNELS = [
    "AF3",
    "AF4",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "PO3",
    "POz",
    "PO4",
    "O1",
    "O2",
]

# Trigger integer codes embedded in the EEG/EMG CSV "Trigger" column. Per the
# data descriptor these mark the phases of each movement trial (preparation,
# movement onset, completion). 771 is treated as the movement-trial anchor.
REYESJIMENEZ2026_MOVE_TRIGGER = 771


class ReyesJimenez2026(BaseDataset):
    """Motor-execution MRCP dataset [1]_ (right-hand fist closure).

    **Dataset description**

    This dataset contains simultaneous electroencephalography (EEG) and
    electromyography (EMG) recordings acquired while healthy participants
    executed a voluntary right-hand fist-closure movement, designed to elicit
    movement-related cortical potentials (MRCP). Forty healthy right-handed
    participants (20 female, 20 male, aged 18-30) each completed five sessions.
    Within each session subjects performed ten right-hand fist-closure trials
    and ten resting periods, guided by a custom Python visual interface that
    displayed a gradually filling green arrow to cue the execution phase, with a
    one-second pause between repetitions.

    EEG was recorded with a 32-channel EMOTIV FLEX 2 Saline wireless system
    (34 electrodes: 32 recording + CMS/DRL reference at TP9/TP10) following the
    international 10-10 system, at a 128 Hz sampling rate and 16-bit resolution
    (1 LSB = 0.51 uV). EMG was recorded from the flexor carpi radialis and
    palmaris longus of the right forearm in a bipolar configuration at 440 Hz.
    All recordings are stored as raw, unfiltered CSV files. Only the EEG stream
    is loaded here; the 440 Hz EMG stream is not merged because of its different
    sampling rate.

    This loader reframes the single-gesture protocol as a two-class
    **movement-vs-rest** problem for the imagery paradigm: a "movement" epoch is
    anchored at each movement-phase trigger (code 771) and a "rest" epoch is
    taken from the pre-movement baseline (``rest_offset`` seconds before the
    movement anchor). The default interval ``(0, 2)`` covers the MRCP window,
    which the descriptor's validation figure spans from MRCP onset (0 s) to the
    expected moment of movement execution (2 s).

    References
    ----------

    .. [1] Reyes-Jimenez, F., Rosas-Agraz, F., Macias-Naranjo, E.,
       Alvarado-Rodriguez, F. J., Velez-Perez, H., Romo-Vazquez, R., &
       Guzman-Quezada, E. (2026). EEG and EMG dataset for analyzing
       movement-related cortical potentials in hand gesture tasks. Data in
       Brief, 65, 112596. DOI: https://doi.org/10.1016/j.dib.2026.112596
       Dataset: Mendeley Data, DOI: https://doi.org/10.17632/y23s2xg6x4.1

    Notes
    -----
    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="10-10",
            hardware="EMOTIV FLEX 2 Saline 32-channel wireless EEG system",
            cap_manufacturer="EMOTIV",
            cap_model="FLEX 2 Saline",
            sensor_type="saline",
            electrode_type="saline",
            reference="CMS/DRL at TP9/TP10",
            ground="DRL (Drive Right Leg) at TP10",
            software="EMOTIV Launcher / EMOTIV PRO",
            sensors=REYESJIMENEZ2026_CHANNELS,
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_emg=True,
                emg_channels=1,
                other_physiological=[
                    "bipolar forearm EMG (flexor carpi radialis / palmaris "
                    "longus) at 440 Hz, stored in separate CSV files"
                ],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=40,
            health_status="healthy",
            gender={"male": 20, "female": 20},
            age_min=18.0,
            age_max=30.0,
            handedness="right",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="motor execution",
            n_classes=2,
            class_labels=["movement", "rest"],
            trials_per_class={"movement": 50, "rest": 50},
            study_design=(
                "Right-hand fist-closure execution guided by a Python visual "
                "interface. Five sessions per subject, each with ten movement "
                "trials and ten resting periods, reframed as move-vs-rest."
            ),
            feedback_type="none",
            stimulus_type="visual cue (filling green arrow)",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            has_training_test_split=False,
            events={"movement": 1, "rest": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.dib.2026.112596",
            description=(
                "EEG (32ch, 128 Hz) and EMG (440 Hz) recordings from 40 healthy "
                "right-handed adults executing right-hand fist closures to elicit "
                "movement-related cortical potentials (MRCP)."
            ),
            investigators=[
                "Fernanda Reyes-Jimenez",
                "Fernanda Rosas-Agraz",
                "Eduardo Macias-Naranjo",
                "Francisco J. Alvarado-Rodriguez",
                "Hugo Velez-Perez",
                "Rebeca Romo-Vazquez",
                "Erick Guzman-Quezada",
            ],
            senior_author="Erick Guzman-Quezada",
            contact_info=["erick.guzman@edu.uag.mx"],
            institution="Universidad Autonoma de Guadalajara",
            institution_department="Departamento de Electromecanica",
            institution_address="Zapopan, Jalisco, Mexico",
            country="MX",
            data_url="https://data.mendeley.com/datasets/y23s2xg6x4/1",
            associated_paper_doi="10.17632/y23s2xg6x4.1",
            publication_year=2026,
            ethics_approval=[
                "Ethics Committee of the Universidad Autonoma de Ciudad Juarez, "
                "protocol CEI-2025-1-77"
            ],
            keywords=[
                "BCI",
                "EEG-based prosthetics",
                "hand gesture analysis",
                "MRCP",
                "neural signal processing",
                "machine learning for neurorehabilitation",
            ],
            license="CC-BY-NC-4.0",
            repository="Mendeley Data",
        ),
        sessions_per_subject=5,
        runs_per_session=1,
        tags=Tags(modality=["Motor"], type=["Motor Execution"]),
        file_format="CSV",
        data_processed=False,
        contributing_labs=["Universidad Autonoma de Guadalajara"],
        n_contributing_labs=1,
        abstract=(
            "This dataset contains electroencephalography (EEG) and "
            "electromyography (EMG) recordings acquired during the execution of "
            "specific motor tasks aimed at eliciting movement-related cortical "
            "potentials (MRCP). Data were collected from 40 healthy participants "
            "aged 18-30 across five sessions, each comprising ten right-hand "
            "fist-closure movements guided by a custom Python visual interface. "
            "EEG was recorded with a 32-channel EMOTIV Flex 2 wireless system at "
            "128 Hz; raw EEG, raw EMG and event triggers were stored in CSV."
        ),
    )

    def __init__(
        self,
        subjects=None,
        sessions=None,
        rest_offset=4.0,
        *,
        return_all_modalities=False,
    ):
        self.rest_offset = rest_offset
        super().__init__(
            subjects=list(range(1, 40 + 1)),
            sessions_per_subject=5,
            events={"movement": 1, "rest": 2},
            code="ReyesJimenez2026",
            interval=(0, 2),
            paradigm="imagery",
            doi="10.1016/j.dib.2026.112596",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local EEG CSV paths for a single subject (one per session).

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data. If None, the MNE default is
            used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused; kept for signature compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list of str
            Paths to the five per-session EEG CSV files for this subject.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path_zip = Path(
            dl.data_dl(REYESJIMENEZ2026_URL, self.code, path, force_update, verbose)
        )
        path_folder = path_zip.parent

        # Extract once.
        marker = path_folder / "SUBJECT01"
        if not marker.is_dir():
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)
            except BadZipFile:
                warnings.warn(
                    "Corrupted zip file detected, re-downloading...", stacklevel=2
                )
                path_zip.unlink(missing_ok=True)
                path_zip = Path(
                    dl.data_dl(REYESJIMENEZ2026_URL, self.code, path, True, verbose)
                )
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)

        sub = f"SUBJECT{subject:02d}"
        subject_paths = []
        for session in range(1, self.n_sessions + 1):
            fname = f"{sub}_Session_{session:02d}_EEG.csv"
            # Files live under SUBJECTXX/SUBJECTXX_Session_0X/ per the descriptor;
            # fall back to a flat SUBJECTXX/ layout if the session subfolder is
            # absent.
            candidate = path_folder / sub / f"{sub}_Session_{session:02d}" / fname
            if not candidate.exists():
                candidate = path_folder / sub / fname
            subject_paths.append(str(candidate))

        return subject_paths

    def _read_eeg_csv(self, file_path):
        """Read one EEG CSV into (data[V], trigger[int])."""
        df = pd.read_csv(file_path)
        cols = {c.strip().lower(): c for c in df.columns}

        # Locate the 32 EEG channel columns by 10-10 name (case-insensitive),
        # tolerating an "EEG." / "EEG_" prefix from the EMOTIV export.
        data = np.zeros((len(REYESJIMENEZ2026_CHANNELS), len(df)), dtype=float)
        found = 0
        missing = []
        for i, ch in enumerate(REYESJIMENEZ2026_CHANNELS):
            key = ch.lower()
            src = (
                cols.get(key)
                or cols.get(f"eeg.{key}")
                or cols.get(f"eeg_{key}")
                or cols.get(f"eeg {key}")
            )
            if src is not None:
                data[i] = pd.to_numeric(df[src], errors="coerce").to_numpy()
                found += 1
            else:
                missing.append(ch)

        # Fall back to positional selection of the first 32 numeric columns
        # if channel names could not be matched (unknown exact CSV header).
        if found < len(REYESJIMENEZ2026_CHANNELS):
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] >= len(REYESJIMENEZ2026_CHANNELS):
                data = (
                    numeric.iloc[:, : len(REYESJIMENEZ2026_CHANNELS)]
                    .to_numpy()
                    .T.astype(float)
                )
            else:
                warnings.warn(
                    f"Only matched {found}/{len(REYESJIMENEZ2026_CHANNELS)} EEG "
                    f"channels in {Path(file_path).name}; missing {missing}.",
                    stacklevel=2,
                )

        # CSV values are microvolts -> convert to volts for MNE.
        data = np.nan_to_num(data) * 1e-6

        # Trigger column.
        trig_col = None
        for k, v in cols.items():
            if "trigger" in k or k == "stim":
                trig_col = v
                break
        if trig_col is not None:
            trigger = (
                pd.to_numeric(df[trig_col], errors="coerce")
                .fillna(0)
                .to_numpy()
                .astype(int)
            )
        else:
            warnings.warn(
                f"No trigger column found in {Path(file_path).name}; "
                "no events will be created.",
                stacklevel=2,
            )
            trigger = np.zeros(len(df), dtype=int)

        return data, trigger

    def _build_stim(self, trigger):
        """Map raw phase triggers to a movement/rest stim channel.

        Movement events (code 1) are placed at the leading edge of each
        movement-phase trigger (771). Rest events (code 2) are synthesised from
        the pre-movement baseline, ``rest_offset`` seconds earlier.
        """
        sfreq = 128.0
        stim = np.zeros_like(trigger)

        move_mask = trigger == REYESJIMENEZ2026_MOVE_TRIGGER
        onsets = np.where(move_mask & ~np.r_[False, move_mask[:-1]])[0]

        offset = int(round(self.rest_offset * sfreq))
        for onset in onsets:
            stim[onset] = 1  # movement
            rest_idx = onset - offset
            if rest_idx >= 0 and stim[rest_idx] == 0:
                stim[rest_idx] = 2  # rest baseline

        return stim.astype(float)

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}} for a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            Nested session/run dictionary of :class:`mne.io.RawArray`.
        """
        file_paths = self.data_path(subject)

        montage = mne.channels.make_standard_montage("standard_1005")
        sessions = {}
        for session_idx, file_path in enumerate(file_paths):
            data, trigger = self._read_eeg_csv(file_path)
            stim = self._build_stim(trigger)

            ch_names = list(REYESJIMENEZ2026_CHANNELS) + ["STI"]
            ch_types = ["eeg"] * len(REYESJIMENEZ2026_CHANNELS) + ["stim"]
            info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types=ch_types)

            raw_data = np.vstack([data, stim[np.newaxis, :]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.RawArray(raw_data, info, verbose=False)
                raw.set_montage(montage, on_missing="ignore", verbose=False)

            sessions[str(session_idx)] = {"0": raw}

        return sessions
