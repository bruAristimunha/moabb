#!/bin/bash
#SBATCH --job-name=dl_all
#SBATCH --partition=prepost
#SBATCH -A tst@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
# Campaign env sets the IDRIS http proxy (needed for downloads in SLURM jobs),
# the neuralbench venv, MNE_DATA=$SCRATCH/datasets, and MOABB_ACCEPT_LICENCE=1.
source $HOME/bruno/jeanzay/activate_campaign.sh
export PYTHONPATH=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno:$PYTHONPATH
echo "proxy=$https_proxy  MNE_DATA=$(python -c 'import mne;print(mne.get_config(\"MNE_DATA\"))')"
python /lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno/download_all.py
