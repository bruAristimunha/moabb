#!/bin/bash
#SBATCH --job-name=dl_part
#SBATCH -A tst@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
# Partition is set at submit time (--partition=prepost|archive).
# activate_campaign.sh exports the IDRIS http proxy (SLURM jobs need it), the
# neuralbench venv, MNE_DATA=$SCRATCH/datasets, MOABB_ACCEPT_LICENCE=1.
source $HOME/bruno/jeanzay/activate_campaign.sh
export PYTHONPATH=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno:$PYTHONPATH
python /lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno/download_all.py "$1"
