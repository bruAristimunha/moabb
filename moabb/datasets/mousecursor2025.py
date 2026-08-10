"""MouseCursor2025 multidirectional motor-imagery mouse-cursor dataset."""

import mne
import pandas as pd

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


# Emotiv EPOC+ 14 scalp-EEG channel names (standard 10-20 labels), in the
# fixed column order used by the EmotivPro CSV export ("EEG.<name>" columns).
EMOTIV_EEG_CHANNELS = [
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

SFREQ = 128.0

# Per-subject Mendeley public-file download URLs (record c92c2n5t34, version 2).
# Only the 18 EmotivPro (14-channel, 128 Hz) subjects are exposed here; subjects
# 19-21 in the record are a different acquisition system (OpenBCI Cyton, 8
# channels, 250 Hz) and are intentionally excluded (see class docstring).
MOUSECURSOR2025_URLS = {
    1: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/1ac3b956-02b0-4351-ab88-b5bd0d150951/file_downloaded",
    2: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/5e18a7df-ba34-41af-b974-6871ecb3b009/file_downloaded",
    3: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/233f0b31-e909-4ffb-9495-e7f9d0d231e5/file_downloaded",
    4: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/bc08ec29-93ac-4390-8f1d-d9f2718f7332/file_downloaded",
    5: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/b3902e54-1239-4cac-9fa9-780a666be6f0/file_downloaded",
    6: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/0d87c649-9c12-4de4-8f42-7118edbad5cf/file_downloaded",
    7: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/8328d5a1-ab32-47a4-9ed0-f1c3a3ce5c99/file_downloaded",
    8: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/96c4cb27-9c3a-4f6f-8b4e-771748a31e2a/file_downloaded",
    9: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/0fd4210b-22a3-44ff-a543-103fc430b1e3/file_downloaded",
    10: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/47ad1e88-bc63-4677-b797-33c0688116c7/file_downloaded",
    11: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/b77e8ac5-748d-4e91-8e29-36deaa6d139a/file_downloaded",
    12: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/ccd7a123-a517-48fe-a6f4-9dd3a2464b1d/file_downloaded",
    13: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/b906d7c8-1bca-4442-80f7-1fcf848d4c60/file_downloaded",
    14: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/99fbdad9-7b98-42d7-b0d4-3e057476bb30/file_downloaded",
    15: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/4ef01aef-7ebb-49c8-97ff-8e900d8ee795/file_downloaded",
    16: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/bc4c8cf6-4320-41f3-af4a-ee5d5c49755a/file_downloaded",
    17: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/4f2c5999-1130-436d-a045-093ac97ffc58/file_downloaded",
    18: "https://data.mendeley.com/public-files/datasets/c92c2n5t34/files/6e79372f-322f-43bf-af6e-38a80500d3c1/file_downloaded",
}


class MouseCursor2025(BaseDataset):
    """Multidirectional MI-BCI mouse-cursor dataset [1]_.

    .. admonition:: Dataset summary (status: needs-data)

        This loader is **provisional** because the shared CSV files do not
        contain trial labels. See the "Unresolved" note below before use.

    **Dataset description**

    EEG recordings for a multidirectional motor-imagery brain-computer
    interface (MI-BCI) designed for mouse-cursor control, collected at the
    Biomedical Instrumentation and Signal Processing Laboratory (BISPL),
    Independent University, Bangladesh. Volunteers performed six task classes:
    right-hand movement, right-leg movement, left-hand movement, left-leg
    movement, eye blinks, and resting state.

    Data were acquired with an EMOTIV EPOC+ 14-channel wireless headset and
    exported from EmotivPro as a wide CSV (one file per subject, 167 columns:
    14 scalp-EEG channels plus motion, contact/signal-quality, performance-
    metric, and band-power derived columns). Only the 14 EEG columns
    (``EEG.AF3 ... EEG.AF4``, standard 10-20 positions) are loaded here, at the
    EEG sampling rate of 128 Hz. Each recording is a single continuous ~20-min
    session.

    **Subjects exposed by this loader**

    The Mendeley record lists 21 files, but they are heterogeneous. Subjects
    1-18 are the EmotivPro 14-channel / 128 Hz recordings described above and
    are the only ones exposed (``subject_list = 1..18``). Subjects 19-21 (the
    ``*.csv.csv`` files) are a **different acquisition system** -- OpenBCI Cyton,
    8 EXG channels at 250 Hz -- with no shared montage or sampling rate, and are
    excluded so the dataset presents a single consistent channel set.

    **Unresolved (why status = needs-data)**

    The EmotivPro CSV files contain marker columns (``MarkerIndex``,
    ``MarkerType``, ``MarkerValueInt``, ``EEG.MarkerHardware``) but they are
    **empty across the entire recording** for every subject inspected (1, 2, 10
    verified sample-by-sample). No separate events/labels file is distributed
    and the record description gives no trial timing or class ordering. As a
    result, the mapping from time to the six task classes cannot be recovered
    from the public data, so this loader returns the continuous raw EEG **with
    no class annotations**. The six-class ``events`` mapping declared below is
    the intended protocol, not something present in the files. Labeling requires
    a trial-timing/marker file from the authors.

    References
    ----------

    .. [1] Rafique, S., Saif, Z., Roja, S. T., & Islam, K. (2025).
       EEG MI-BCI Multidirectional Mouse Cursor Dataset. Mendeley Data, V2.
       DOI: https://doi.org/10.17632/c92c2n5t34.2

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            n_channels=14,
            channel_types={"eeg": 14},
            montage="standard_1020",
            hardware="EMOTIV EPOC+ 14-channel wireless EEG headset",
            software="EmotivPro",
            cap_manufacturer="Emotiv",
            cap_model="EPOC+",
            sensor_type="saline felt",
            electrode_material="Ag/AgCl",
            reference="CMS/DRL (P3/P4)",
            line_freq=50.0,
            sensors=EMOTIV_EEG_CHANNELS,
        ),
        participants=ParticipantMetadata(
            n_subjects=18,
            health_status="healthy",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=6,
            class_labels=[
                "right_hand",
                "right_leg",
                "left_hand",
                "left_leg",
                "eye_blink",
                "rest",
            ],
            events={
                "right_hand": 1,
                "right_leg": 2,
                "left_hand": 3,
                "left_leg": 4,
                "eye_blink": 5,
                "rest": 6,
            },
            study_design="Multidirectional motor imagery for mouse-cursor control.",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.17632/c92c2n5t34.2",
            description=(
                "EEG recordings for a multidirectional motor-imagery BCI for "
                "mouse-cursor control (EMOTIV EPOC+, 14 channels, 128 Hz), six "
                "task classes."
            ),
            investigators=[
                "Sayem Rafique",
                "Zawwad Saif",
                "Saima Tasfia Roja",
                "Kafiul Islam",
            ],
            institution="Independent University, Bangladesh",
            institution_department=(
                "Biomedical Instrumentation and Signal Processing Laboratory (BISPL)"
            ),
            country="BD",
            data_url="https://doi.org/10.17632/c92c2n5t34.2",
            publication_year=2025,
            license="CC-BY-4.0",
            repository="Mendeley Data",
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "mouse cursor",
                "EMOTIV EPOC+",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(
            pathology=["healthy"], modality=["motor"], type=["imagery"]
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=[
                "right_hand",
                "right_leg",
                "left_hand",
                "left_leg",
                "eye_blink",
                "rest",
            ],
        ),
        file_format="CSV",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None):
        self.events = {
            "right_hand": 1,
            "right_leg": 2,
            "left_hand": 3,
            "left_leg": 4,
            "eye_blink": 5,
            "rest": 6,
        }
        super().__init__(
            subjects=list(range(1, 18 + 1)),
            sessions_per_subject=1,
            events=self.events,
            code="MouseCursor2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.17632/c92c2n5t34.2",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path to a single subject's CSV file.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused; kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list
            A one-element list with the path to the subject's CSV file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        url = MOUSECURSOR2025_URLS[subject]
        local_path = dl.data_dl(
            url, self.code, path=path, force_update=force_update, verbose=verbose
        )
        return [local_path]

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{session: {run: mne.io.Raw}}`` with a single session/run holding
            the continuous 14-channel EEG. No class annotations are attached
            because the shared files contain no trial labels (see class
            docstring, "Unresolved").
        """
        file_path = self.data_path(subject)[0]

        # EmotivPro export: first line is a metadata header, second line is the
        # column header row. Load only the 14 scalp-EEG columns.
        eeg_columns = ["EEG." + ch for ch in EMOTIV_EEG_CHANNELS]
        df = pd.read_csv(
            file_path, skiprows=1, usecols=eeg_columns, low_memory=False
        )

        # EmotivPro exports EEG amplitudes in microvolts; MNE expects volts.
        data = df[eeg_columns].to_numpy().T.astype(float) * 1e-6

        info = mne.create_info(
            ch_names=list(EMOTIV_EEG_CHANNELS),
            sfreq=SFREQ,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage("standard_1020", verbose=False)

        return {"0": {"0": raw}}
