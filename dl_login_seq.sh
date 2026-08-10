#!/bin/bash
# Gentle SEQUENTIAL downloader for the login node (run inside tmux so it
# survives ssh logout). Usage: dl_login_seq.sh START END
source $HOME/bruno/jeanzay/activate_campaign.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
export PYTHONPATH=$B:$PYTHONPATH
cd $B
for i in $(seq "$1" "$2"); do
  MOD=$(sed -n "${i}p" new_modules.txt)
  [ -z "$MOD" ] && continue
  echo "=== [$(date +%H:%M:%S)] ($i) $MOD ==="
  python download_all.py "$MOD"
done
echo "=== LOGIN SEQ DONE $(date +%H:%M:%S) ==="
