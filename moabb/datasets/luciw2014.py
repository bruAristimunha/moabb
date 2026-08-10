"""WAY-EEG-GAL grasp-and-lift dataset (Luciw et al., 2014)."""

import zipfile as z
from pathlib import Path

import mne
import numpy as np
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
from moabb.datasets.utils import stim_channels_with_selected_ids


# One Figshare article (single PX.zip file) per participant, collection 988376.
LUCIW2014_FILE_IDS = {
    1: 3229301,
    2: 3229304,
    3: 3229307,
    4: 3229310,
    5: 3229313,
    6: 3209486,
    7: 3209501,
    8: 3209504,
    9: 3209495,
    10: 3209492,
    11: 3209498,
    12: 3209489,
}

LUCIW2014_BASE_URL = "https://ndownloader.figshare.com/files/"

# 32 EEG channels, order as in the dataset (chanlabels_32channel.xyz).
LUCIW2014_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
]

# Weight condition coding in P.AllLifts CurW column (1=165 g, 2=330 g, 3=660 g).
LUCIW2014_WEIGHT_EVENTS = {"weight_165g": 1, "weight_330g": 2, "weight_660g": 3}

N_SERIES = 9


class Luciw2014(BaseDataset):
    """WAY-EEG-GAL grasp-and-lift dataset [1]_ [2]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Luciw2014       12       32           3               ~110            3s           500Hz              1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    WAY-EEG-GAL (Wearable interfaces for hAnd function recoverY - EEG - Grasp
    And Lift) is a dataset designed to allow tests of techniques to decode
    sensation, intention, and action from scalp EEG recorded while participants
    perform a cued reach, grasp-and-lift task. Twelve right-handed participants
    each performed 328 grasp-and-lift trials (3,936 in total) across 9 series
    (runs). During each trial the participant reached for a small object,
    grasped it with the thumb and index finger, lifted it a few centimeters,
    held it briefly, replaced it, released it, and returned the hand to a rest
    position, cued by an LED.

    The object's physical properties were varied unpredictably between trials:
    the weight took three levels (165, 330 and 660 g) and the contact-surface
    friction took three levels (sandpaper, suede and silk). Series were of three
    kinds: weight series (only weight varies), friction series (only surface
    varies) and mixed series (both vary).

    Signals recorded: 32-channel scalp EEG (10-20 montage, 500 Hz), 5-channel
    EMG of arm and hand muscles, 3D kinematics of the hand and object, and
    force/torque at the two contact plates. This loader exposes only the EEG.

    .. note::

        This is a motor **execution** grasp-and-lift task, not motor imagery,
        and the dataset has no single native discrete class. This loader frames
        the inherent (recorded, not invented) object-**weight** condition as the
        3-class decoding target, with one event placed at the object lift-off of
        each trial. The friction condition and the movement-phase event times
        (LED on/off, hand start, first digit touch, load-phase onset, lift-off,
        replace, release) are stored in the source ``P.AllLifts`` matrix and
        could support alternative framings.

    References
    ----------

    .. [1] Luciw, M., Jarocka, E., Edin, B. (2014). WAY-EEG-GAL: Multi-channel
       EEG recordings during 3,936 grasp and lift trials with varying weight and
       friction. figshare. Collection.
       DOI: https://doi.org/10.6084/m9.figshare.988376

    .. [2] Luciw, M. D., Jarocka, E., & Edin, B. B. (2014). Multi-channel EEG
       recordings during 3,936 grasp and lift trials with varying weight and
       friction. Scientific Data, 1, 140047.
       DOI: https://doi.org/10.1038/sdata.2014.47

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="10-20",
            reference=None,
            ground=None,
            sensors=LUCIW2014_CHANNELS,
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(n_subjects=12, species="homo sapiens"),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["weight_165g", "weight_330g", "weight_660g"],
            trial_duration=3.0,
            study_design=(
                "Cued reach, grasp-and-lift task (motor execution). Object weight "
                "(165/330/660 g) and grasp-surface friction (sandpaper/suede/silk) "
                "varied unpredictably across 328 trials per participant (9 series)."
            ),
            stimulus_type="LED cue",
            mode="offline",
            events=LUCIW2014_WEIGHT_EVENTS,
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/sdata.2014.47",
            description=(
                "WAY-EEG-GAL: 32-channel EEG (plus EMG, kinematics and kinetics) "
                "recorded from 12 participants during 3,936 grasp-and-lift trials "
                "with varying object weight and grasp-surface friction."
            ),
            investigators=["Matthew D. Luciw", "Ewa Jarocka", "Benoni B. Edin"],
            country="SE",
            license="CC0-1.0",
            repository="Figshare",
            data_url="https://doi.org/10.6084/m9.figshare.988376",
            publication_year=2014,
        ),
        sessions_per_subject=1,
        runs_per_session=N_SERIES,
        tags=Tags(modality=["Motor"], type=["Motor Execution"]),
        file_format="MAT",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 12 + 1)),
            sessions_per_subject=1,
            events=dict(LUCIW2014_WEIGHT_EVENTS),
            code="Luciw2014",
            interval=(-1.0, 2.0),
            paradigm="imagery",
            doi="10.1038/sdata.2014.47",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download (if needed) and return the HS ``.mat`` paths of one subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list
            Paths to the 9 ``HS_P{subject}_S{series}.mat`` files, in series order.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        url = LUCIW2014_BASE_URL + str(LUCIW2014_FILE_IDS[subject])
        path_zip = Path(dl.data_dl(url, self.code, path=path, force_update=force_update))
        extract_dir = path_zip.parent / f"P{subject}"

        if force_update or not extract_dir.is_dir():
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(path_zip.parent)

        subject_paths = []
        for series in range(1, N_SERIES + 1):
            matches = sorted(path_zip.parent.rglob(f"HS_P{subject}_S{series}.mat"))
            if not matches:
                raise FileNotFoundError(
                    f"HS_P{subject}_S{series}.mat not found after extraction"
                )
            subject_paths.append(str(matches[0]))
        return subject_paths

    def _all_lifts_path(self, subject):
        """Return the path to the ``P{subject}_AllLifts.mat`` trial-info file."""
        any_hs = Path(self.data_path(subject)[0])
        matches = sorted(any_hs.parent.rglob(f"P{subject}_AllLifts.mat"))
        if not matches:
            raise FileNotFoundError(f"P{subject}_AllLifts.mat not found")
        return str(matches[0])

    def _get_single_subject_data(self, subject):
        """Return the EEG data of a single subject.

        Returns
        -------
        dict
            ``{"0": {"<series>": mne.io.Raw}}`` with one run per series.
        """
        hs_paths = self.data_path(subject)

        # Trial info: one matrix P.AllLifts with named columns (P.ColNames).
        lifts_mat = loadmat(
            self._all_lifts_path(subject), struct_as_record=False, squeeze_me=True
        )
        p_struct = lifts_mat["P"]
        col_names = [str(c).strip() for c in np.atleast_1d(p_struct.ColNames)]
        all_lifts = np.atleast_2d(np.asarray(p_struct.AllLifts, dtype=float))
        col = {name: i for i, name in enumerate(col_names)}
        c_run = col["Run"]
        c_start = col["StartTime"]
        c_liftoff = col["tLiftOff"]
        c_weight = col["CurW"]

        runs = {}
        for series_idx, hs_path in enumerate(hs_paths, start=1):
            hs_mat = loadmat(hs_path, struct_as_record=False, squeeze_me=True)
            hs = hs_mat["hs"]
            sfreq = float(hs.eeg.samplingrate)

            # hs.eeg.sig is (n_samples, n_channels); 0.1 * sig gives microvolts.
            sig = np.asarray(hs.eeg.sig, dtype=float).T * 0.1e-6  # -> volts
            info = mne.create_info(
                ch_names=list(LUCIW2014_CHANNELS), sfreq=sfreq, ch_types="eeg"
            )
            raw = mne.io.RawArray(sig, info, verbose=False)
            raw.set_montage("standard_1020", on_missing="warn", verbose=False)

            # Events for this series: one marker per lift, at object lift-off.
            mask = all_lifts[:, c_run] == series_idx
            onsets = all_lifts[mask, c_start] + all_lifts[mask, c_liftoff]
            weights = all_lifts[mask, c_weight].astype(int)
            samples = np.round(onsets * sfreq).astype(int)

            valid = (samples >= 0) & (samples < raw.n_times)
            samples, weights = samples[valid], weights[valid]

            events = np.column_stack([samples, np.zeros_like(samples), weights])
            event_desc = {v: k for k, v in self.event_id.items()}
            annotations = mne.annotations_from_events(
                events, sfreq=sfreq, event_desc=event_desc
            )
            raw.set_annotations(annotations)

            runs[str(series_idx)] = stim_channels_with_selected_ids(raw, self.event_id)

        return {"0": runs}
