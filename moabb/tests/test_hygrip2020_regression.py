"""Regression tests for HYGRIP's mislabeled EEG unit metadata."""

import h5py
import numpy as np

from moabb.datasets.hygrip2020 import EEG_CH_NAMES, HYGRIP2020


def test_hygrip_numeric_payload_is_already_volts(tmp_path, monkeypatch):
    sfreq = 1_000.0
    times = np.arange(2_000) / sfreq
    # A physiological 8-uV, 10-Hz signal makes a 1e-3 unit error obvious.
    signal = 8e-6 * np.sin(2 * np.pi * 10 * times)
    eeg = np.tile(signal, (len(EEG_CH_NAMES), 1))
    path = tmp_path / "hygrip.h5"
    with h5py.File(path, "w") as hdf:
        hdf.attrs["eeg_sfreq"] = sfreq
        hdf.attrs["eeg_units"] = b"milivolts"  # codespell:ignore
        dataset = hdf.create_dataset("A/eeg", data=eeg)
        dataset.attrs["events"] = np.array([[0.25, 0], [1.25, 1]])

    hygrip = HYGRIP2020()
    monkeypatch.setattr(hygrip, "data_path", lambda subject: [str(path)])

    raw = hygrip._get_single_subject_data(1)["0"]["0"]
    observed = raw.get_data(picks="eeg")

    np.testing.assert_allclose(observed, eeg, rtol=0, atol=0)
    rms_uv = np.sqrt(np.mean(observed**2)) * 1e6
    assert 5.0 < rms_uv < 6.5
