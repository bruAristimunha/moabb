"""Single-channel (C4) motor-imagery open/close hands dataset (Mendeley).

Carretero Perez, A. (2024). "EEG Motor imagery open/close hands with C4
electrode dataset." Mendeley Data, V1.
Data DOI: 10.17632/bfn2pksz45.1
"""

import json

import mne
import numpy as np
from mne.channels import make_standard_montage
from scipy.io import loadmat

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


# Mendeley Data public-api record. The returned JSON carries a long-lived
# signed ``download_url`` for every file in its ``files`` array.
OPENCLOSE_C4_RECORD = "https://data.mendeley.com/public-api/datasets/bfn2pksz45"

# One EEG electrode at C4, referenced behind the ear.
OPENCLOSE_C4_CHANNELS = ["C4"]

# Each record is 2 s long and stored as 20006 samples; the README states the
# last 6 samples "can be erased", giving 20000 usable samples -> 10000 Hz.
OPENCLOSE_C4_RAW_LEN = 20006
OPENCLOSE_C4_N_SAMPLES = 20000
OPENCLOSE_C4_TRIAL_SEC = 2.0
OPENCLOSE_C4_SFREQ = OPENCLOSE_C4_N_SAMPLES / OPENCLOSE_C4_TRIAL_SEC  # 10000.0 Hz

# Signals are stored as raw ~12-bit ADC counts (uint16, centred near 1975).
# The amplifier gain / ADC reference is not documented, so the absolute scale
# is unknown: the per-record DC offset is removed and the residual counts are
# placed on a nominal volt scale (values land in a plausible +/- 200 uV range).
OPENCLOSE_C4_ADC_TO_VOLT = 1e-6

# Class definition. Each Mendeley .mat file is a MATLAB workspace dump that, in
# this v1 record, happens to contain every array; to stay faithful to the
# documented file -> class mapping we read each class from its own file and its
# self-describing variable name (README codebook, "_anacp" = the recordist's
# initials). Movement setups hold 4 trials each; the rest setup holds 10.
#   (class_label, event_id, mat_filename, mat_variable)
OPENCLOSE_C4_CLASSES = [
    ("close_right_hand", 1, "setup1.mat", "setup1_anacp"),
    ("open_right_hand", 2, "setup2.mat", "setup2_anacp"),
    ("close_left_hand", 3, "setup3.mat", "setup3_anacp"),
    ("open_left_hand", 4, "setup4.mat", "setup4_anacp"),
    ("rest", 5, "setup1_rest.mat", "setup1_anacp_rest"),
]

OPENCLOSE_C4_EVENTS = {label: code for (label, code, _f, _v) in OPENCLOSE_C4_CLASSES}


class OpenCloseHandsC4(BaseDataset):
    """Single-channel (C4) motor imagery of opening/closing the hands [1]_.

    .. admonition:: Dataset summary

        ================  =======  =======  ==========  =================  ============  ===============  ===========
        Name                #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        ================  =======  =======  ==========  =================  ============  ===============  ===========
        OpenCloseHandsC4        1        1           5            4 (rest 10)          2s           10000 Hz            1
        ================  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    A compact single-subject, single-electrode motor-imagery recording shared on
    Mendeley Data. The participant imagined opening and closing the right and
    left hands (the fists), plus a no-movement (rest) baseline. Signals were
    acquired from one electrode placed at C4, with the reference electrode placed
    behind the ear.

    Data are organised into five conditions, each provided as a named array in a
    MATLAB file (the class of every trial is therefore data-borne: it is the
    array it belongs to, not its acquisition order):

    * ``setup1`` -- close right hand (4 trials)
    * ``setup2`` -- open right hand (4 trials)
    * ``setup3`` -- close left hand (4 trials)
    * ``setup4`` -- open left hand (4 trials)
    * ``setup1_rest`` -- rest / no movement (10 trials)

    This yields 16 movement trials and 10 rest trials (26 total). Each record
    lasts 2 seconds and is stored as 20006 samples; following the README, the
    trailing 6 samples are dropped, leaving 20000 samples, i.e. a 10000 Hz
    sampling rate. Every condition is exposed as one run of a single session,
    with one 2 s annotation per trial carrying the condition label.

    The signals are stored as raw ~12-bit ADC counts. Because the amplifier gain
    and ADC reference voltage are not documented, the absolute amplitude is
    uncalibrated: this loader removes the per-record DC offset and rescales the
    residual counts onto a nominal volt scale (they fall in a plausible EEG
    micro-volt range), which is adequate for band-power / spectral analysis but
    should not be read as an absolute voltage.

    References
    ----------

    .. [1] Carretero Perez, A. (2024). EEG Motor imagery open/close hands with
       C4 electrode dataset. Mendeley Data, V1.
       DOI: https://doi.org/10.17632/bfn2pksz45.1

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=OPENCLOSE_C4_SFREQ,
            n_channels=1,
            channel_types={"eeg": 1},
            sensors=list(OPENCLOSE_C4_CHANNELS),
            montage="standard_1020",
            reference="behind-the-ear (mastoid)",
            electrode_type=None,
            hardware=None,
        ),
        participants=ParticipantMetadata(
            n_subjects=1,
            health_status="healthy",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=5,
            class_labels=[
                "close_right_hand",
                "open_right_hand",
                "close_left_hand",
                "open_left_hand",
                "rest",
            ],
            trials_per_class={
                "close_right_hand": 4,
                "open_right_hand": 4,
                "close_left_hand": 4,
                "open_left_hand": 4,
                "rest": 10,
            },
            trial_duration=OPENCLOSE_C4_TRIAL_SEC,
            study_design=(
                "Single subject, single C4 electrode (reference behind the ear). "
                "Imagined open/close of the right and left hands plus a rest "
                "baseline, each condition stored as a named MATLAB array of 2 s "
                "records (4 movement records per condition, 10 rest records)."
            ),
            feedback_type="none",
            synchronicity="cue-based",
            mode="offline",
            events=dict(OPENCLOSE_C4_EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.17632/bfn2pksz45.1",
            description=(
                "Single-subject, single-channel (C4) EEG motor imagery of "
                "opening/closing the right and left hands plus rest; raw 12-bit "
                "ADC records of 2 s at 10000 Hz, stored in MATLAB files."
            ),
            investigators=["Ana Carretero Perez"],
            institution="Universidad Politecnica de Madrid",
            country="ES",
            repository="Mendeley Data",
            data_url="https://data.mendeley.com/datasets/bfn2pksz45/1",
            license="CC-BY-4.0",
            publication_year=2024,
            keywords=[
                "motor imagery",
                "EEG",
                "hand",
                "open close",
                "C4",
                "single channel",
                "BCI",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=len(OPENCLOSE_C4_CLASSES),
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="MATLAB",
    )

    def __init__(self):
        super().__init__(
            subjects=[1],
            sessions_per_subject=1,
            events=dict(OPENCLOSE_C4_EVENTS),
            code="OpenCloseHandsC4",
            interval=(0, OPENCLOSE_C4_TRIAL_SEC),
            paradigm="imagery",
            doi="10.17632/bfn2pksz45.1",
        )

    def _file_map(self, path=None, force_update=False, verbose=None):
        """Return a mapping ``{filename: download_url}`` from the record JSON."""
        record_path = dl.data_dl(
            OPENCLOSE_C4_RECORD,
            self.code,
            path=path,
            force_update=force_update,
            verbose=verbose,
        )
        if isinstance(record_path, (list, tuple)):
            record_path = record_path[0]
        with open(record_path, encoding="utf-8") as fid:
            record = json.load(fid)
        return {
            f["filename"]: f["content_details"]["download_url"]
            for f in record["files"]
        }

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local paths of the five MATLAB files (one per condition).

        Parameters
        ----------
        subject : int
            Subject number (only ``1`` is available).
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list of str
            The five local MATLAB paths, ordered as ``OPENCLOSE_C4_CLASSES``.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        file_map = self._file_map(path=path, force_update=force_update, verbose=verbose)
        paths = []
        for _label, _code, filename, _var in OPENCLOSE_C4_CLASSES:
            url = file_map[filename]
            local = dl.data_dl(
                url, self.code, path=path, force_update=force_update, verbose=verbose
            )
            if isinstance(local, (list, tuple)):
                local = local[0]
            paths.append(str(local))
        return paths

    def _read_condition(self, mat_path, var_name, label):
        """Build one ``Raw`` for a condition by concatenating its 2 s trials."""
        mat = loadmat(mat_path)
        arr = np.asarray(mat[var_name], dtype=np.float64)  # (n_trials, 20006)
        arr = arr[:, :OPENCLOSE_C4_N_SAMPLES]  # drop the trailing 6 samples
        # Remove the per-record DC offset (uncalibrated counts) and rescale onto
        # a nominal volt scale; absolute amplitude is not physically meaningful.
        arr = (arr - arr.mean(axis=1, keepdims=True)) * OPENCLOSE_C4_ADC_TO_VOLT
        n_trials = arr.shape[0]

        # Concatenate the trials along time: (1, n_trials * n_samples).
        data = arr.reshape(1, -1)
        info = mne.create_info(
            ch_names=list(OPENCLOSE_C4_CHANNELS),
            sfreq=OPENCLOSE_C4_SFREQ,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage(
            make_standard_montage("standard_1020"), on_missing="ignore", verbose=False
        )

        onsets = [i * OPENCLOSE_C4_TRIAL_SEC for i in range(n_trials)]
        annotations = mne.Annotations(
            onset=onsets,
            duration=[OPENCLOSE_C4_TRIAL_SEC] * n_trials,
            description=[label] * n_trials,
        )
        raw.set_annotations(annotations, verbose=False)
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of the single subject as ``{session: {run: Raw}}``."""
        paths = self.data_path(subject)
        runs = {}
        for run_idx, ((label, _code, _f, var), mat_path) in enumerate(
            zip(OPENCLOSE_C4_CLASSES, paths)
        ):
            runs[str(run_idx)] = self._read_condition(mat_path, var, label)
        return {"0": runs}
