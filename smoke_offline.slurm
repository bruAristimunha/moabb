#!/bin/bash
#SBATCH --job-name=smokeoff
#SBATCH -A tst@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=smoke_offline_%x_%j.out

B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
VPY=/lustre/fsn1/projects/rech/tst/uiy14ex/venvs/neuralbench_campaign/bin/python
export MNE_DATA=/lustre/fsn1/projects/rech/tst/uiy14ex/datasets
export MOABB_ACCEPT_LICENCE=1
export PYTHONPATH=$B
export MPLCONFIGDIR=/tmp
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy='*'
export NO_PROXY='*'

cd "$B"
"$VPY" smoke_test.py "$1"
