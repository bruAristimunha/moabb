#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH -A tst@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=smoke_%A_%a.out
# One dataset per array task. No internet needed (data already local).
source $HOME/bruno/jeanzay/activate_campaign.sh
B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
export PYTHONPATH=$B:$PYTHONPATH
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd $B
NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" smoke_list.txt)
echo "[task ${SLURM_ARRAY_TASK_ID}] smoke $NAME"
python smoke_test.py "$NAME"
