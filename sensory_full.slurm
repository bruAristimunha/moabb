#!/bin/bash
#SBATCH --job-name=sgfull
#SBATCH --partition=archive
#SBATCH -A tst@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=sensory_full_%j.out

B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
VPY=/lustre/fsn1/projects/rech/tst/uiy14ex/venvs/neuralbench_campaign/bin/python
export MNE_DATA=/lustre/fsn1/projects/rech/tst/uiy14ex/datasets
export MOABB_ACCEPT_LICENCE=1
export PYTHONPATH=$B
export MPLCONFIGDIR=/tmp
export OPENBLAS_NUM_THREADS=10
export OMP_NUM_THREADS=10
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy='*'
export NO_PROXY='*'

cd "$B"
"$VPY" - <<'PY'
from collections import Counter

from moabb.datasets.sensoryguidedmi2026 import (
    SUBJECT_MAP,
    SensoryGuidedMI2026,
)
from moabb.paradigms import MotorImagery


dataset = SensoryGuidedMI2026()
X, y, metadata = MotorImagery().get_data(dataset, [1])
counts = Counter(map(str, y))
expected = {"left_hand", "right_hand", "up", "down"}
assert set(counts) == expected, (counts, expected)
assert len(y) == len(metadata) == X.shape[0]
print(
    "SENSORY_FULL",
    {
        "subject": 1,
        "source": SUBJECT_MAP[1],
        "shape": X.shape,
        "labels": dict(sorted(counts.items())),
        "sessions": sorted(map(str, metadata.session.unique())),
        "runs": metadata.run.nunique(),
    },
    flush=True,
)
PY
