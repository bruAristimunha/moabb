#!/bin/bash
#SBATCH --job-name=boardverify
#SBATCH --partition=archive
#SBATCH -A tst@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=verify_board_%j.out

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
for dataset in \
    Sun2026 \
    Ding2025 \
    Garro2025 \
    Damm2026 \
    Iwama2023 \
    MIND2026 \
    Thapa2025 \
    SensoryGuidedMI2026
do
    echo "VERIFY_START $dataset"
    "$VPY" smoke_test.py "$dataset"
done

bash unit_check.slurm
echo "BOARD_VERIFY_DONE"
