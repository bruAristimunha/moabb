"""Li2021 same-joint (shoulder) motor imagery dataset."""

import logging
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


log = logging.getLogger(__name__)

# Zenodo record 10.5281/zenodo.4699203 (single archive "dataset.zip").
LI2021_URL = "https://zenodo.org/records/4699203/files/dataset.zip"

# The seven subjects are stored as folders A-G inside the archive.
LI2021_SUBJECT_FOLDERS = {i + 1: letter for i, letter in enumerate("ABCDEFG")}

# The 14 EEG channels of the Emotiv EPOC+ (order given in the CSV header).
LI2021_EEG_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]


class Li2021(BaseDataset):
    """Motor imagery of three movements of the same (shoulder) joint [1]_ [2]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Li2021           7       14           3                 20  5 s           128 Hz                     1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    Seven right-handed subjects (five male, two female, aged 23-28) imagined three
    movements of the *same* shoulder joint: abduction, extension and flexion. A trial
    started with a fixation cross and beep at t=1 s; at t=3 s an arrow pointed left,
    right or top (at random) to cue abduction, extension or flexion respectively, and
    disappeared after 1.25 s; motor imagery stopped at t=7 s, followed by a 2 s
    inter-trial interval. Sixty trials per subject were collected (20 per class).

    EEG was recorded with an Emotiv EPOC+ (14 channels: AF3, F7, F3, FC5, T7, P7, O1,
    O2, P8, T8, FC6, F4, F8, AF4; international 10-20 layout) at 128 Hz. In the Zenodo
    archive each subject is a folder (A-G) holding one ~5 s recording (both ``.csv``
    and ``.edf``) per trial.

    .. warning::

        **Per-trial class labels are not distributed with the Zenodo release.** The
        cue was randomized per trial, the ``.edf``/``.csv`` files contain a MARKER
        channel that is uniformly zero, and no separate label/key file is included in
        the archive. The three-class (abduction/extension/flexion) mapping of the 60
        recordings per subject therefore cannot be reconstructed from the published
        data alone. This loader reads and returns the raw 14-channel EEG correctly
        (one run per trial) but cannot attach reliable class annotations. The
        contributing authors' trial-order key is required to make this a usable
        supervised motor-imagery dataset.

    References
    ----------

    .. [1] Li, J., Guan, S., Wang, F., Yuan, Z., Kang, X., & Lu, B. (2021).
       Motor imagery EEG data of the same joint [Data set]. Zenodo.
       https://doi.org/10.5281/zenodo.4699203

    .. [2] Wang, F., Guan, S., Li, J., ... & Lu, B. (2021). Discriminating three
       motor imagery states of the same joint for brain-computer interface. PeerJ,
       9, e12027. https://doi.org/10.7717/peerj.12027

    Notes
    -----

    .. versionadded:: 1.2.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            n_channels=14,
            channel_types={"eeg": 14},
            montage="10-20",
            hardware="Emotiv EPOC+",
            cap_manufacturer="Emotiv",
            cap_model="EPOC+",
            reference=None,
            ground=None,
            sensors=list(LI2021_EEG_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=7,
            health_status="healthy",
            gender={"male": 5, "female": 2},
            age_min=23.0,
            age_max=28.0,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["abduction", "extension", "flexion"],
            trials_per_class={"abduction": 20, "extension": 20, "flexion": 20},
            trial_duration=5.0,
            study_design="Motor imagery of three movements of the same shoulder joint "
            "(abduction, extension, flexion). Left/right/top arrow cue presented at "
            "random per trial; 60 trials per subject (20 per class).",
            feedback_type="none",
            stimulus_type="visual arrow cue",
            stimulus_modalities=["visual", "audio"],
            synchronicity="cue-based",
            mode="offline",
            events={"abduction": 1, "extension": 2, "flexion": 3},
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.4699203",
            related_paper_dois=["10.7717/peerj.12027"],
            description="Three-class motor imagery of the same shoulder joint "
            "(abduction, extension, flexion) from seven subjects, recorded with a "
            "14-channel Emotiv EPOC+ at 128 Hz.",
            investigators=[
                "Jixian Li",
                "Shan Guan",
                "Fuwang Wang",
                "Zhen Yuan",
                "Xiaogang Kang",
                "Bin Lu",
            ],
            institution="Northeast Electric Power University",
            institution_address="Jilin City, China",
            country="CN",
            data_url="https://doi.org/10.5281/zenodo.4699203",
            publication_year=2021,
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "same joint",
                "shoulder",
                "Emotiv EPOC+",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="CSV and EDF",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 7 + 1)),
            sessions_per_subject=1,
            events={"abduction": 1, "extension": 2, "flexion": 3},
            code="Li2021",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.5281/zenodo.4699203",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the list of trial data files for a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for (1-7).
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Deprecated, unused.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list
            Sorted list of the subject's ``.edf`` trial-file paths.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download the single archive shared by all subjects.
        path_zip = Path(dl.data_dl(LI2021_URL, self.code, force_update=force_update))
        path_folder = path_zip.parent
        extract_root = path_folder / "dataset"

        # Extract once; the archive holds one top-level "dataset/" directory.
        if not extract_root.is_dir():
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)
            except BadZipFile:
                warnings.warn(
                    "Corrupted zip file detected, re-downloading...", stacklevel=2
                )
                path_zip.unlink(missing_ok=True)
                path_zip = Path(dl.data_dl(LI2021_URL, self.code, force_update=True))
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)

        subject_folder = extract_root / LI2021_SUBJECT_FOLDERS[subject]
        subject_paths = sorted(str(p) for p in subject_folder.glob("*.edf"))
        return subject_paths

    def _get_single_subject_data(self, subject):
        """Return the raw data of a single subject.

        Each ``.edf`` trial is returned as a separate run under session ``"0"``.
        Only the 14 EEG channels are kept and the standard 10-20 montage is set.

        Notes
        -----
        Per-trial class labels are not part of the Zenodo release (see the class
        docstring warning), so no class annotations are attached here.
        """
        file_paths = self.data_path(subject)

        log.warning(
            "Li2021: per-trial class labels (abduction/extension/flexion) are not "
            "distributed with the Zenodo record; returning unlabelled raw EEG only."
        )

        runs = {}
        for run_idx, file_path in enumerate(file_paths):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            # Keep only the 14 EEG channels present in the Emotiv export.
            picks = [ch for ch in LI2021_EEG_CHANNELS if ch in raw.ch_names]
            raw = raw.pick(picks)
            raw.set_channel_types(dict.fromkeys(raw.ch_names, "eeg"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = raw.set_montage("standard_1020", on_missing="ignore")
            runs[str(run_idx)] = raw

        return {"0": runs}
