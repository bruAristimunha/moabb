#!/usr/bin/env python
"""Run a basic MI pipeline on ONE subject of a single dataset; report OK/ERROR.

Usage: smoke_test.py <DatasetClassName>

Loads via the moabb MotorImagery paradigm, then fits Covariances+MDM
(fallback CSP+LDA) with up-to-3-fold CV on subject 1. Prints a one-line
``SMOKE {json}`` summary and writes ``smoke_<name>.json`` (with traceback).
Assumes data is already present locally (does not intend to download).
"""
import json
import sys
import time
import traceback

name = sys.argv[1]
res = {"dataset": name}
t0 = time.time()
try:
    import mne

    mne.set_log_level("ERROR")
    import numpy as np
    import moabb
    import moabb.datasets as D
    from moabb.paradigms import MotorImagery
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder

    moabb.set_log_level("error")

    cls = getattr(D, name)
    d = cls()
    subj = d.subject_list[0]
    X, y, meta = MotorImagery().get_data(d, [subj])
    ye = LabelEncoder().fit_transform(y)
    counts = np.bincount(ye)
    res.update(
        n_trials=int(len(y)),
        n_classes=int(len(counts)),
        n_chan=int(X.shape[1]),
        n_times=int(X.shape[2]),
    )
    if len(counts) < 2 or counts.min() < 2:
        res.update(status="LOAD_ONLY", note="too few trials/classes to classify")
    else:
        try:
            from pyriemann.classification import MDM
            from pyriemann.estimation import Covariances

            clf = make_pipeline(Covariances("oas"), MDM())
        except Exception:
            from mne.decoding import CSP
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            clf = make_pipeline(
                CSP(n_components=min(4, X.shape[1])), LinearDiscriminantAnalysis()
            )
        cv = int(min(3, counts.min()))
        scores = cross_val_score(clf, X, ye, cv=cv)
        res.update(status="OK", cv=cv, acc=round(float(np.mean(scores)), 3))
except Exception as e:  # noqa: BLE001
    res.update(
        status="ERROR",
        etype=type(e).__name__,
        error=str(e)[:400],
        trace=traceback.format_exc()[-1500:],
    )
res["t"] = round(time.time() - t0, 1)
print("SMOKE " + json.dumps({k: v for k, v in res.items() if k != "trace"}), flush=True)
with open(f"smoke_{name}.json", "w") as f:
    json.dump(res, f, indent=2)
