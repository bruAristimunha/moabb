#!/bin/bash
# Run all datasets in smoke_list.txt through smoke_test.py on the dev QOS,
# keeping ~10 jobs in flight (dev caps total submitted jobs, so we throttle +
# retry). Run inside tmux on a login node so it survives logout.
source $HOME/bruno/jeanzay/activate_campaign.sh
B=/lustre/fsn1/projects/rech/tst/uiy14ex/moabb-datasets-bruno
cd "$B"
rm -f smoke_*.json smoke_j_*.out
N=$(grep -c . smoke_list.txt)
echo "=== dev loop: $N datasets, target 10 concurrent, $(date +%H:%M:%S) ==="
while read -r NAME; do
  [ -z "$NAME" ] && continue
  # hold at <=10 of my smoke jobs queued/running
  while [ "$(squeue -u "$USER" -n smoke -h 2>/dev/null | wc -l)" -ge 10 ]; do sleep 10; done
  # submit; retry until dev accepts it (rides out QOSMaxSubmitJobPerUserLimit)
  until sbatch --qos=qos_cpu-dev smoke_one.slurm "$NAME" >/dev/null 2>&1; do sleep 10; done
  echo "submitted $NAME"
  sleep 1
done < smoke_list.txt
echo "=== all submitted; draining $(date +%H:%M:%S) ==="
while [ "$(squeue -u "$USER" -n smoke -h 2>/dev/null | wc -l)" -gt 0 ]; do sleep 20; done
echo "=== SMOKE_LOOP_DONE $(date +%H:%M:%S) ==="
python - <<'PY'
import glob, json
ok = err = load = 0
errors = []
for f in glob.glob("smoke_*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    s = d.get("status")
    if s == "OK":
        ok += 1
    elif s == "LOAD_ONLY":
        load += 1
    else:
        err += 1
        errors.append((d.get("dataset"), d.get("etype"), d.get("error")))
print(f"SMOKE SUMMARY: OK={ok} LOAD_ONLY={load} ERROR={err}")
for name, et, e in sorted(errors):
    print(f"  ERROR {name}: {et}: {e}")
PY
