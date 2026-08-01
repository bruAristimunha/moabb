"""Regression tests for loaders whose published payload is in microvolts."""

from types import SimpleNamespace

import mne
import numpy as np
import pytest

from moabb.datasets.bids_interface import BIDSInterfaceRawEDF, StepType
from moabb.datasets.peterson2022 import PETERSON2022_CACHE_VERSION, Peterson2022
from moabb.datasets.preprocessing import FixedPipeline, SetRawAnnotations
from moabb.datasets.spinalstim2025 import (
    _AUX_CHANNELS,
    _EEG_CHANNELS,
    SPINALSTIM2025_CACHE_VERSION,
    SPINALSTIM2025_EEG_SCALE_TO_VOLTS,
    SpinalStim2025,
)
from moabb.datasets.sun2026 import (
    _AUX_TYPES,
    _EEG_CH_NAMES,
    SUN2026_CACHE_VERSION,
    SUN2026_EEG_SCALE_TO_VOLTS,
    Sun2026,
)


def test_peterson_blank_edf_physical_dimension_is_declared_as_microvolts():
    dataset = Peterson2022()

    assert dataset._get_read_extra_params(2) == {"units": "uV"}


def test_spinalstim_gdf_microvolts_are_converted_to_volts(tmp_path, monkeypatch):
    channel_names = [*_EEG_CHANNELS, *_AUX_CHANNELS]
    source_uv = np.vstack(
        [np.full((len(_EEG_CHANNELS), 32), 20.0), np.full((len(_AUX_CHANNELS), 32), 5.0)]
    )
    raw = mne.io.RawArray(
        source_uv.copy(),
        mne.create_info(channel_names, sfreq=512.0, ch_types="eeg"),
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([0.0], [0.0], ["769"]))
    monkeypatch.setattr(
        "moabb.datasets.spinalstim2025.mne.io.read_raw_gdf", lambda *args, **kwargs: raw
    )

    corrected = SpinalStim2025._read_run(tmp_path / "offline.gdf")

    np.testing.assert_allclose(
        corrected.get_data(picks=_EEG_CHANNELS),
        source_uv[: len(_EEG_CHANNELS)] * SPINALSTIM2025_EEG_SCALE_TO_VOLTS,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        corrected.get_data(picks=_AUX_CHANNELS),
        source_uv[len(_EEG_CHANNELS) :],
        rtol=0,
        atol=0,
    )


def test_spinalstim_subject_roots_prevent_cross_subject_duplicates(tmp_path, monkeypatch):
    archive_root = tmp_path / "d3_SinglePulse_n5"
    correct_503 = (
        archive_root
        / "Offline_Recordings/Subject_503_SinglePulse_Offline"
        / "Subject_503_Session_001_SinglePulse_Offline_Visual/correct_503.gdf"
    )
    mislabeled_under_504 = (
        archive_root
        / "Offline_Recordings/Subject_504_SinglePulse_Offline"
        / "Subject_503_Session_001_SinglePulse_Offline_Visual/belongs_to_504.gdf"
    )
    for path in (correct_503, mislabeled_under_504):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        "moabb.datasets.spinalstim2025.dl.data_dl",
        lambda *args, **kwargs: str(tmp_path / "d3_SinglePulse_n5.zip"),
    )
    dataset = SpinalStim2025()

    assert dataset.data_path(23) == [str(correct_503)]
    assert dataset.data_path(24) == [str(mislabeled_under_504)]


def test_spinalstim_preserves_both_d4_offline_roots(tmp_path, monkeypatch):
    archive_root = tmp_path / "d4_SCI_patients"
    rest = archive_root / "Patient_1/Subject_0001_REST_Offline/rest.gdf"
    tess = archive_root / "Patient_1/Subject_0001_TESS_Offline/tess.gdf"
    for path in (rest, tess):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        "moabb.datasets.spinalstim2025.dl.data_dl",
        lambda *args, **kwargs: str(tmp_path / "d4_SCI_patients.zip"),
    )

    assert SpinalStim2025().data_path(26) == [str(rest), str(tess)]


def test_sun_brainvision_microvolts_are_converted_without_scaling_trigger(monkeypatch):
    channel_names = [*_EEG_CH_NAMES, *_AUX_TYPES]
    source_uv = np.vstack(
        [
            np.full((len(_EEG_CH_NAMES), 64), 25.0),
            np.full((len(_AUX_TYPES) - 1, 64), 7.0),
            np.full((1, 64), 123.0),
        ]
    )
    raw = mne.io.RawArray(
        source_uv.copy(),
        mne.create_info(channel_names, sfreq=1000.0, ch_types="eeg"),
        verbose=False,
    )
    bids_path = SimpleNamespace(session="01", run="01")
    monkeypatch.setattr(Sun2026, "data_path", lambda _self, _subject: [bids_path])
    monkeypatch.setattr(
        "moabb.datasets.sun2026.read_raw_bids", lambda *args, **kwargs: raw
    )

    corrected = Sun2026()._get_single_subject_data(1)["01"]["01"]

    np.testing.assert_allclose(
        corrected.get_data(picks=_EEG_CH_NAMES),
        source_uv[: len(_EEG_CH_NAMES)] * SUN2026_EEG_SCALE_TO_VOLTS,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        corrected.get_data(picks=["HEO", "VEO", "EKG", "EMG"]),
        source_uv[len(_EEG_CH_NAMES) : -1],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(corrected.get_data(picks=["Trigger"]), 123.0)
    assert corrected.get_channel_types()[-5:] == ["eog", "eog", "ecg", "emg", "stim"]


@pytest.mark.parametrize(
    ("dataset", "subject", "cache_version"),
    [
        (Peterson2022(), 2, PETERSON2022_CACHE_VERSION),
        (SpinalStim2025(), 1, SPINALSTIM2025_CACHE_VERSION),
        (Sun2026(), 1, SUN2026_CACHE_VERSION),
    ],
)
def test_unit_repairs_change_raw_cache_fingerprint(
    tmp_path, dataset, subject, cache_version
):
    corrected_pipeline = dataset._create_process_pipeline()
    legacy_pipeline = FixedPipeline(
        [(StepType.RAW, SetRawAnnotations(dataset.event_id, interval=dataset.interval))]
    )
    corrected = BIDSInterfaceRawEDF(
        dataset, subject, path=tmp_path, process_pipeline=corrected_pipeline
    )
    legacy = BIDSInterfaceRawEDF(
        dataset, subject, path=tmp_path, process_pipeline=legacy_pipeline
    )

    assert corrected_pipeline.get_params()["Raw__cache_version"] == cache_version
    assert corrected.desc != legacy.desc
    assert corrected._lock_file("0") != legacy._lock_file("0")
