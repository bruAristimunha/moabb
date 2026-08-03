import mne
import numpy as np
import pytest
from scipy.io import savemat

from moabb.datasets.bids_interface import BIDSInterfaceRawEDF, StepType
from moabb.datasets.preprocessing import FixedPipeline, SetRawAnnotations
from moabb.datasets.wrcc2023_mi_a import (
    WRCC2023_MI_A,
    WRCC2023_MI_A_CACHE_VERSION,
    WRCC2023_MI_A_CHANNELS,
)
from moabb.datasets.wrcc2023_mi_b import WRCC2023_MI_B
from moabb.datasets.wrcc2023_mi_c import WRCC2023_MI_C


@pytest.mark.parametrize(
    ("dataset_class", "include_sfreq"),
    [(WRCC2023_MI_A, True), (WRCC2023_MI_B, False), (WRCC2023_MI_C, True)],
)
def test_all_wrcc_trials_survive_epoch_boundaries(tmp_path, dataset_class, include_sfreq):
    """Leading and trailing pads retain every stored trial."""
    n_trials, n_channels, n_samples = 3, len(WRCC2023_MI_A_CHANNELS), 4
    contents = {
        "data": np.ones((n_trials, n_channels, n_samples)),
        "label": np.array([1, 2, 3]),
    }
    if include_sfreq:
        contents["fs"] = np.array([[1000.0]])
    path = tmp_path / "subject.mat"
    savemat(path, contents)

    raw = dataset_class._mat_to_raw(path)
    events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"left_hand": 1, "right_hand": 2, "feet": 3},
        tmin=0,
        tmax=n_samples / 1000.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    assert len(events) == n_trials
    assert len(epochs) == n_trials
    assert epochs.drop_log == ((), (), ())


def test_wrcc_companion_loaders_preserve_the_same_volt_scale(tmp_path):
    """MI-A/B/C payloads share a volts-at-source amplitude contract."""
    n_trials, n_channels, n_samples = 3, len(WRCC2023_MI_A_CHANNELS), 4
    source_volts = np.linspace(
        -25e-6, 25e-6, num=n_trials * n_channels * n_samples
    ).reshape(n_trials, n_channels, n_samples)
    path = tmp_path / "subject.mat"
    savemat(
        path,
        {"data": source_volts, "label": np.array([1, 2, 3]), "fs": np.array([[1000.0]])},
    )
    expected = np.transpose(source_volts, (1, 0, 2)).reshape(
        n_channels, n_trials * n_samples
    )

    loaded = [
        dataset_class._mat_to_raw(path).get_data(picks="eeg")[:, 1:-1]
        for dataset_class in (WRCC2023_MI_A, WRCC2023_MI_B, WRCC2023_MI_C)
    ]

    for actual in loaded:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_wrcc_mi_a_unit_fix_changes_raw_cache_fingerprint(tmp_path):
    """Corrected MI-A data must not reuse legacy rescaled raw caches."""
    dataset = WRCC2023_MI_A()
    corrected_pipeline = dataset._create_process_pipeline()
    legacy_pipeline = FixedPipeline(
        [(StepType.RAW, SetRawAnnotations(dataset.event_id, interval=dataset.interval))]
    )
    corrected = BIDSInterfaceRawEDF(
        dataset, 1, path=tmp_path, process_pipeline=corrected_pipeline
    )
    legacy = BIDSInterfaceRawEDF(
        dataset, 1, path=tmp_path, process_pipeline=legacy_pipeline
    )

    assert (
        corrected_pipeline.get_params()["Raw__cache_version"]
        == WRCC2023_MI_A_CACHE_VERSION
    )
    assert corrected.root == legacy.root
    assert corrected.desc != legacy.desc
    assert corrected._lock_file("0") != legacy._lock_file("0")


def test_wrcc_mi_b_uses_companion_neuracle_montage(tmp_path):
    """MI-B shares the documented 59-channel WRCC acquisition order."""
    path = tmp_path / "subject.mat"
    savemat(
        path,
        {"data": np.ones((2, len(WRCC2023_MI_A_CHANNELS), 4)), "label": np.array([1, 2])},
    )

    raw = WRCC2023_MI_B._mat_to_raw(path)
    eeg = raw.copy().pick("eeg")
    positioned = [
        channel
        for channel in eeg.ch_names
        if np.any(eeg.info["chs"][eeg.ch_names.index(channel)]["loc"][:3])
    ]

    assert eeg.ch_names == WRCC2023_MI_A_CHANNELS
    assert {"C3", "Cz", "C4"}.issubset(eeg.ch_names)
    assert len(positioned) == len(WRCC2023_MI_A_CHANNELS)
