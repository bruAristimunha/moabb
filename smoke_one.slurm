#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH -A tst@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=smoke_j_%j.out
source $HOME/bruno/jeanzay/activate_campaign.sh
B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
export PYTHONPATH=$B:$PYTHONPATH
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd $B
python smoke_test.py "$1"
