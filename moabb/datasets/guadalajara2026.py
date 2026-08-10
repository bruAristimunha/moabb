"""Guadalajara2026 hand-gesture MRCP (motor execution) dataset."""

import warnings
import zipfile as z
from pathlib import Path

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


# Canonical whole-dataset archive for the Mendeley record (DOI 10.17632/y23s2xg6x4.1).
# NOTE (needs-data): as of writing, the Mendeley public API reports this record with
# zero published files (empty file list, size 0) and the S3 zip cache returns 403, so
# the archive below could not be downloaded or its internal layout verified. The URL
# follows Mendeley's standard cache pattern and is the best-known download entry point.
GUADALAJARA2026_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/"
    "y23s2xg6x4-1.zip"
)

# Sampling rate is NOT documented in the Mendeley record. This is an UNVERIFIED
# placeholder used only to build the RawArray; replace once the record is inspected.
SFREQ = 512.0

# The nine fronto-central EEG electrodes explicitly named in the record; the record
# states 32 EEG electrodes on the 10-10 system but does not list the remaining 23.
CONFIRMED_EEG_CHANNELS = [
    "FC3",
    "FC1",
    "FCz",
    "C3",
    "C1",
    "Cz",
    "CP3",
    "CP1",
    "CPz",
]


class Guadalajara2026(BaseDataset):
    """Hand-gesture MRCP (motor execution) dataset [1]_.

    .. admonition:: Dataset summary

        =============== ======= ======= ================ =============== =============== ===========
        Name            #Subj   #Chan   #Classes         #Trials / class Trials len      Sampling rate
        =============== ======= ======= ================ =============== =============== ===========
        Guadalajara2026 40      32      2                unknown         unknown         unknown
        =============== ======= ======= ================ =============== =============== ===========

    **Dataset description**

    EEG and EMG recordings acquired during the detection of movement-related cortical
    potentials (MRCPs) in healthy subjects. The experimental protocol consisted of a
    self-paced voluntary right-hand movement (fist clenching, a single gesture). EEG
    was acquired with 32 electrodes placed according to the international 10-10 system,
    with particular focus on the fronto-central regions (FC3, FC1, FCz, C3, C1, Cz,
    CP3, CP1, CPz). Complementary surface EMG was recorded from the right forearm
    muscles to validate the presence and characteristics of the motor activity.

    In the MOABB framing this single-gesture protocol is treated as a two-class
    ``move`` vs ``rest`` problem, decodable from the pre-movement MRCP: epochs time-locked
    to (or preceding) movement onset are labelled ``move`` and baseline/inter-trial
    windows are labelled ``rest``.

    .. warning::

        **needs-data.** The underlying files could not be resolved. The Mendeley
        record (DOI 10.17632/y23s2xg6x4.1) exposes no downloadable files through its
        public API (empty file list, reported size 0) and the standard archive/zip
        endpoints are inaccessible. The following are therefore **unverified** and
        assumed from the record's prose, to be corrected once the files are available:

        * the number of subjects (40 assumed) and per-subject file naming,
        * the file format (assumed one CSV per subject with a header row of channel
          names plus a marker/trigger column),
        * the sampling rate (placeholder ``512`` Hz),
        * the full 32-channel montage (only the 9 fronto-central channels are named),
        * the marker codes distinguishing movement onset from rest.

    Parameters
    ----------
    subjects : list of int | None
        Subset of subjects to load. ``None`` loads all subjects.
    sessions : list of str | None
        Subset of sessions to load. ``None`` loads all sessions.

    References
    ----------
    .. [1] Reyes-Jiménez, F., Rosas-Agraz, F., Macias-Naranjo, E., Romo-Vázquez, R.,
       Vélez-Pérez, H., Alvarado-Rodríguez, F. J., & Guzman, E. E. (2025). EEG and EMG
       Dataset for Analyzing Movement-Related Cortical Potentials in Hand Gesture
       Tasks. Mendeley Data, V1. DOI: https://doi.org/10.17632/y23s2xg6x4.1

    Notes
    -----
    .. versionadded:: 1.1.1
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=None,  # not documented in the Mendeley record
            n_channels=32,
            channel_types={"eeg": 32, "emg": 1},
            montage="10-10",
            reference=None,
            ground=None,
            sensor_type="EEG",
            sensors=CONFIRMED_EEG_CHANNELS,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_emg=True,
                emg_channels=1,
                other_physiological=["right forearm surface EMG"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=40,
            health_status="healthy",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="motor execution",
            n_classes=2,
            class_labels=["move", "rest"],
            trial_duration=None,
            study_design="Self-paced voluntary right-hand movement (fist clenching), "
            "a single gesture, recorded with EEG + right-forearm EMG to characterise "
            "movement-related cortical potentials (MRCPs). Framed as move vs rest.",
            stimulus_type="self-paced",
            synchronicity="self-paced",
            mode="offline",
            events={"move": 1, "rest": 2},
            instructions="Perform a voluntary right-hand fist clench at a self-selected pace.",
        ),
        documentation=DocumentationMetadata(
            doi="10.17632/y23s2xg6x4.1",
            description="EEG and EMG recordings during self-paced right-hand fist "
            "clenching for the study of movement-related cortical potentials (MRCPs) "
            "in healthy subjects; 32-channel 10-10 EEG plus right-forearm EMG.",
            investigators=[
                "Fernanda Reyes-Jiménez",
                "Fernanda Rosas-Agraz",
                "Eduardo Macias-Naranjo",
                "Rebeca Romo-Vázquez",
                "Hugo Vélez-Pérez",
                "Francisco J. Alvarado-Rodríguez",
                "Erick Eduardo Guzman",
            ],
            institution="Universidad de Guadalajara; Universidad Autónoma de Guadalajara",
            country="MX",
            repository="Mendeley Data",
            data_url="https://data.mendeley.com/datasets/y23s2xg6x4/1",
            license="CC-BY-NC-SA-4.0",
            publication_year=2025,
            keywords=[
                "MRCP",
                "movement-related cortical potentials",
                "motor execution",
                "hand gesture",
                "EEG",
                "EMG",
                "brain-computer interface",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(modality=["Motor"], type=["Motor Execution"]),
    )

    def __init__(self, subjects=None, sessions=None, **kwargs):
        self.events = {"move": 1, "rest": 2}
        super().__init__(
            subjects=list(range(1, 40 + 1)),
            sessions_per_subject=1,
            events=self.events,
            code="Guadalajara2026",
            interval=[-2.0, 0.0],  # MRCP window preceding movement onset
            paradigm="imagery",
            doi="10.17632/y23s2xg6x4.1",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path to a single subject's data file.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location where the data is stored. If None, the MNE default is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Whether to update the MNE config path.
        verbose : bool, str, int, or None
            Override the default verbosity.

        Returns
        -------
        list
            A single-element list with the path to the subject's data file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path_zip = Path(
            dl.data_dl(GUADALAJARA2026_URL, self.code, path, force_update, verbose)
        )
        path_folder = path_zip.parent / "MendeleyDatasets"

        if not path_folder.is_dir():
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(path_folder)

        # Assumed per-subject CSV naming; corrected once the record layout is known.
        candidates = sorted(path_folder.rglob(f"*{subject:02d}*.csv")) + sorted(
            path_folder.rglob(f"*{subject}*.csv")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No CSV file found for subject {subject} under {path_folder}. "
                "The Guadalajara2026 file layout is unresolved (needs-data); update "
                "data_path once the Mendeley record exposes its files."
            )
        return [str(candidates[0])]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{session: {run: mne.io.Raw}}`` for the requested subject.
        """
        file_path = self.data_path(subject)[0]

        df = pd.read_csv(file_path)
        columns = list(df.columns)

        # Identify a marker/trigger column (assumed name) and EMG column(s).
        marker_col = next(
            (c for c in columns if c.lower() in ("marker", "trigger", "stim", "label")),
            None,
        )
        emg_cols = [c for c in columns if "emg" in c.lower()]
        data_cols = [c for c in columns if c not in ([marker_col] if marker_col else [])]
        eeg_cols = [c for c in data_cols if c not in emg_cols]

        ch_names = eeg_cols + emg_cols
        ch_types = ["eeg"] * len(eeg_cols) + ["emg"] * len(emg_cols)
        info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)

        data = df[ch_names].to_numpy().T.astype(float)
        raw = mne.io.RawArray(data, info, verbose=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
            except Exception:
                pass

        # Build move/rest annotations from the marker column when available.
        if marker_col is not None:
            markers = df[marker_col].to_numpy()
            onsets = np.flatnonzero(np.asarray(markers) != 0)
            if onsets.size:
                mapping = {1: "move", 2: "rest"}
                descs = [mapping.get(int(markers[i]), str(int(markers[i]))) for i in onsets]
                annotations = mne.Annotations(
                    onset=onsets / SFREQ,
                    duration=np.zeros(onsets.size),
                    description=descs,
                )
                raw.set_annotations(annotations)

        return {"0": {"0": raw}}
