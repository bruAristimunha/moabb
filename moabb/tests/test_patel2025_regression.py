"""Download-free regressions for the Patel2025 trial-validity repair."""

from collections import Counter
from types import SimpleNamespace

import mne
import numpy as np

from moabb.datasets.bids_interface import BIDSInterfaceRawEDF, StepType
from moabb.datasets.patel2025 import (
    EEG_CHANNELS,
    EOG_CHANNELS,
    PATEL2025_CACHE_VERSION,
    Patel2025,
)
from moabb.datasets.preprocessing import FixedPipeline, SetRawAnnotations


def _trial(n_samples, value):
    return np.full((n_samples, len(EEG_CHANNELS) + len(EOG_CHANNELS)), value)


def test_patel_skips_only_field_shorter_than_analysis_interval(monkeypatch, caplog):
    elements = np.array(
        [
            SimpleNamespace(L=_trial(256, 1), R=_trial(300, 2), Re=_trial(256, 3)),
            SimpleNamespace(L=_trial(119, 9), R=_trial(256, 4), Re=_trial(512, 5)),
        ],
        dtype=object,
    )
    monkeypatch.setattr(
        "moabb.datasets.patel2025.sio.loadmat",
        lambda *args, **kwargs: {"Subject2": elements},
    )

    with caplog.at_level("WARNING", logger="moabb.datasets.patel2025"):
        raw = Patel2025._mat_to_raw("Subject2.mat")

    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose=False)
    expected_codes = [1, 2, 3, 2, 3]
    assert events[:, 0].tolist() == [1, 513, 1025, 1537, 2049]
    assert events[:, 2].tolist() == expected_codes
    assert Counter(events[:, 2]) == Counter({1: 1, 2: 2, 3: 2})
    assert raw.n_times == 1 + len(expected_codes) * 512

    data = raw.get_data(picks=EEG_CHANNELS + EOG_CHANNELS)
    block_starts = data[:, events[:, 0]]
    expected_values = np.array([1, 2, 3, 4, 5], dtype=float) * 1e-6
    expected_data = np.tile(expected_values, (data.shape[0], 1))
    np.testing.assert_allclose(block_starts, expected_data)
    assert "skipped 1 trial(s)" in caplog.text
    assert "shorter than the 256 samples" in caplog.text


def test_patel_trial_repair_changes_raw_cache_fingerprint(tmp_path):
    dataset = Patel2025()
    corrected_pipeline = dataset._create_process_pipeline()
    legacy_pipeline = FixedPipeline(
        [(StepType.RAW, SetRawAnnotations(dataset.event_id, interval=dataset.interval))]
    )
    corrected = BIDSInterfaceRawEDF(
        dataset, 2, path=tmp_path, process_pipeline=corrected_pipeline
    )
    legacy = BIDSInterfaceRawEDF(
        dataset, 2, path=tmp_path, process_pipeline=legacy_pipeline
    )

    assert (
        corrected_pipeline.get_params()["Raw__cache_version"] == PATEL2025_CACHE_VERSION
    )
    assert corrected.root == legacy.root
    assert corrected.desc != legacy.desc
    assert corrected._lock_file("0") != legacy._lock_file("0")
