#!/bin/bash
#SBATCH --job-name=unitcheck
#SBATCH -A tst@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=unit_check_%j.out

B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
VPY=/lustre/fsn1/projects/rech/tst/uiy14ex/venvs/neuralbench_campaign/bin/python
export PYTHONPATH=$B
export MPLCONFIGDIR=/tmp
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy='*'
export NO_PROXY='*'

cd "$B"
"$VPY" - <<'PY'
import inspect
import tempfile
from pathlib import Path

from moabb.tests import test_dataset_fixes


tests = [
    (name, function)
    for name, function in inspect.getmembers(test_dataset_fixes, inspect.isfunction)
    if name.startswith("test_")
]
for name, function in tests:
    parameters = inspect.signature(function).parameters
    if "tmp_path" in parameters:
        with tempfile.TemporaryDirectory() as directory:
            function(tmp_path=Path(directory))
    else:
        function()
    print("PASS", name, flush=True)
print("UNIT_CHECK", len(tests), "passed", flush=True)
PY
