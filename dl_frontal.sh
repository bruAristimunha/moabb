#!/bin/bash
# Launch parallel downloads on the frontal login node. Usage: dl_frontal.sh START END
source $HOME/bruno/jeanzay/activate_campaign.sh
# Login node caps per-user threads; downloads need no BLAS threads. Cap to 1 each.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
export PYTHONPATH=$B:$PYTHONPATH
cd $B
for i in $(seq "$1" "$2"); do
  MOD=$(sed -n "${i}p" new_modules.txt)
  [ -z "$MOD" ] && continue
  nohup python download_all.py "$MOD" > "dl_frontal_${MOD}.log" 2>&1 &
  echo "frontal launched: $MOD (pid $!)"
done
echo "frontal procs running: $(jobs -rp | wc -l)"
