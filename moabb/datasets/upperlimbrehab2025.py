"""UpperLimbRehab2025 multi-paradigm upper-limb rehabilitation dataset."""

import warnings
import zipfile as z
from pathlib import Path
from zipfile import BadZipFile

import mne

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


# Figshare article 28831730 (v2). Raw EEG data is split across six zip files,
# each holding five subjects (S201-S230, i.e. "S{200 + subject}").
UPPERLIMBREHAB2025_FILES = {
    0: "https://ndownloader.figshare.com/files/57432889",  # subjects 1-5
    1: "https://ndownloader.figshare.com/files/57433693",  # subjects 6-10
    2: "https://ndownloader.figshare.com/files/57435691",  # subjects 11-15
    3: "https://ndownloader.figshare.com/files/57435694",  # subjects 16-20
    4: "https://ndownloader.figshare.com/files/57560707",  # subjects 21-25
    5: "https://ndownloader.figshare.com/files/57560710",  # subjects 26-30
}

# Motor-execution runs: four continuous recordings per subject, each cueing the
# three rehabilitation gestures 10 times (marker codes 1, 2, 3).
UPPERLIMBREHAB2025_RUNS = ["ME1", "ME2", "ME3", "ME4"]

# 59 scalp EEG channels (10-10 layout) as stored in the EEGLAB chanlocs.
UPPERLIMBREHAB2025_CHANNELS = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2",
    "F3", "F4", "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3",
    "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4",
    "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    "TP7", "TP8", "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz",
    "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2",
]


class UpperLimbRehab2025(BaseDataset):
    """Motor-execution subset of a multi-paradigm upper-limb rehabilitation EEG dataset [1]_.

    **Dataset description**

    A multi-paradigm EEG dataset recorded from 28 healthy participants performing
    upper-limb rehabilitation exercises under six feedback conditions (motor
    execution ``ME``, motor imagery ``MI``, mental imagery ``Image``, mirror
    therapy ``Mirror``, auxiliary/assisted ``Aux`` and virtual reality ``VR``).
    Each subject folder ``S{200 + subject}`` (e.g. ``S206`` for subject 6) stores
    the recordings as EEGLAB ``.set``/``.fdt`` pairs.

    This loader exposes the **motor-execution (``ME``) paradigm**, which contains
    the three rehabilitation gestures as three balanced classes. There are four
    ``ME`` runs per subject (``ME1``-``ME4``); each run is a continuous recording
    in which the three gesture cues (marker codes 1, 2 and 3) are each presented
    ten times, giving 30 cues per run. EEG was recorded from 59 scalp electrodes
    (10-10 layout) at 1000 Hz.

    The exact human-readable mapping of gesture cue codes 1/2/3 to specific
    grasp/release exercises is not documented in the repository README; classes
    are therefore exposed as ``gesture_1``/``gesture_2``/``gesture_3`` following
    the marker codes. Confirm the semantic labels with the authors before
    publishing class-name interpretations.

    References
    ----------

    .. [1] Chang, W., Yan, G., Lv, R., Du, K., & Kong, W. (2025).
        A Multi-Paradigm EEG Dataset for Studying Upper Limb Rehabilitation
        Exercises (Raw EEG Dataset). figshare. Dataset.
        DOI: https://doi.org/10.6084/m9.figshare.28831730.v2

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    nemar_id = "EXEMPT"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=59,
            channel_types={"eeg": 59},
            montage="10-10",
            reference=None,
            ground=None,
            sensors=UPPERLIMBREHAB2025_CHANNELS,
        ),
        participants=ParticipantMetadata(
            n_subjects=28,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["gesture_1", "gesture_2", "gesture_3"],
            events={"gesture_1": 1, "gesture_2": 2, "gesture_3": 3},
        ),
        documentation=DocumentationMetadata(
            doi="10.6084/m9.figshare.28831730.v2",
            description=(
                "Multi-paradigm EEG dataset of 28 participants performing upper-limb "
                "rehabilitation exercises under six feedback conditions (motor "
                "execution, motor imagery, mental imagery, mirror therapy, assisted "
                "movement and virtual reality). This loader exposes the motor-execution "
                "three-gesture paradigm (59 EEG channels, 1000 Hz)."
            ),
            investigators=[
                "Wenwen Chang",
                "Guanghui Yan",
                "Renjie Lv",
                "Kaiyue Du",
                "Weixuan Kong",
            ],
            country="CN",
            data_url="https://doi.org/10.6084/m9.figshare.28831730.v2",
            publication_year=2025,
            license="CC-BY-4.0",
            repository="Figshare",
            keywords=[
                "EEG",
                "upper limb rehabilitation",
                "motor execution",
                "motor imagery",
                "BCI",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=4,
        tags=Tags(modality=["Motor"], type=["Motor Execution"]),
    )

    def __init__(self):
        self.events = {"gesture_1": 1, "gesture_2": 2, "gesture_3": 3}
        super().__init__(
            subjects=list(range(1, 28 + 1)),
            sessions_per_subject=1,
            events=self.events,
            code="UpperLimbRehab2025",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.6084/m9.figshare.28831730.v2",
        )

    def _zip_url(self, subject):
        """Return the figshare download URL of the zip holding ``subject``."""
        return UPPERLIMBREHAB2025_FILES[(subject - 1) // 5]

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the four motor-execution ``.set`` paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
            If it doesn't exist, the "~/mne_data" directory is used. If the
            dataset is not found under the given path, the data will be
            automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python config
            to the given path.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list
            A list containing the paths to the subject's four motor-execution
            ``.set`` files.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path_zip = Path(
            dl.data_dl(self._zip_url(subject), self.code, force_update=force_update)
        )
        path_folder = path_zip.parent
        sub = f"S{200 + subject:03d}"

        # Extract the zip once; subjects share a zip, so guard on the subject dir.
        if not (path_folder / sub).is_dir():
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)
            except BadZipFile:
                warnings.warn(
                    "Corrupted zip file detected, re-downloading...", stacklevel=2
                )
                path_zip.unlink(missing_ok=True)
                path_zip = Path(
                    dl.data_dl(self._zip_url(subject), self.code, force_update=True)
                )
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)

        subject_paths = [
            str(path_folder / sub / f"{sub}_{run}.set")
            for run in UPPERLIMBREHAB2025_RUNS
        ]
        return subject_paths

    def _get_single_subject_data(self, subject):
        """Return the motor-execution data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{"0": {"0": raw, "1": raw, "2": raw, "3": raw}}`` where each raw is
            a continuous recording of one motor-execution run.
        """
        file_path_list = self.data_path(subject)
        # Marker codes stored in the EEGLAB events -> MOABB class labels.
        rename = {"1": "gesture_1", "2": "gesture_2", "3": "gesture_3"}

        runs = {}
        for run_idx, set_path in enumerate(file_path_list):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

            # EEGLAB stores gesture cues as numeric annotation descriptions; map
            # them to the class labels expected by the paradigm.
            raw.annotations.rename(
                {k: v for k, v in rename.items() if k in set(raw.annotations.description)}
            )
            raw = raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = raw.set_montage("standard_1005", on_missing="ignore", verbose=False)

            runs[str(run_idx)] = raw

        return {"0": runs}
