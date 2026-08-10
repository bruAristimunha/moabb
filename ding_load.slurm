#!/bin/bash
#SBATCH --job-name=dingload
#SBATCH --partition=archive
#SBATCH -A tst@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=ding_load_%j.out

B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
VPY=/lustre/fsn1/projects/rech/tst/uiy14ex/venvs/neuralbench_campaign/bin/python
export MNE_DATA=/lustre/fsn1/projects/rech/tst/uiy14ex/datasets
export MOABB_ACCEPT_LICENCE=1
export PYTHONPATH="$B"
export MPLCONFIGDIR=/tmp
export OPENBLAS_NUM_THREADS=10
export OMP_NUM_THREADS=10
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy='*'
export NO_PROXY='*'

cd "$B"
"$VPY" - <<'PY'
from collections import Counter

from moabb.datasets import Ding2025
from moabb.paradigms import MotorImagery


dataset = Ding2025()
X, y, metadata = MotorImagery().get_data(dataset, [dataset.subject_list[0]])
print(
    "DING_LOAD",
    {
        "shape": X.shape,
        "labels": dict(sorted(Counter(y).items())),
        "sessions": sorted(metadata["session"].unique()),
        "runs": metadata["run"].nunique(),
    },
    flush=True,
)
PY
