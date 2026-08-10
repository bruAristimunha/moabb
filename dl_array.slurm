#!/bin/bash
#SBATCH --job-name=dlarr
#SBATCH -A tst@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=dlarr_%A_%a.out
# One dataset per array task. Partition/QOS set at submit time.
source $HOME/bruno/jeanzay/activate_campaign.sh
B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
export PYTHONPATH=$B:$PYTHONPATH
MOD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $B/new_modules.txt)
echo "[task ${SLURM_ARRAY_TASK_ID}] proxy=$https_proxy module=$MOD"
python $B/download_all.py "$MOD"
