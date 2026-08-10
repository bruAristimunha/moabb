#!/usr/bin/env python
"""Pre-stage every new-loader dataset's data into MNE_DATA on Jean Zay.

Imports each new loader module directly (so unregistered ones work too),
instantiates its BaseDataset subclass, and calls download() for all subjects.
MOABB skips anything already cached. Per-dataset try/except so one failure
never stops the rest; prints OK/FAIL/SKIP with elapsed time and any error.
"""
import importlib
import sys
import time
import os

from moabb.datasets.base import BaseDataset

HERE = os.path.dirname(os.path.abspath(__file__))
_args = [a for a in sys.argv[1:] if a.strip()]
if len(_args) == 1 and os.path.isfile(_args[0]):
    mods = [l.strip() for l in open(_args[0]) if l.strip()]
elif _args:
    mods = _args  # explicit module names (one-per-array-task fan-out)
else:
    mods = [l.strip() for l in open(os.path.join(HERE, "new_modules.txt")) if l.strip()]
print(f"=== download {len(mods)} module(s): {mods} ===", flush=True)

# datasets that cannot be fetched non-interactively -> skip with a note
SKIP = {"song2026": "Baidu Netdisk only", "mind2026": "already staged (74GB)"}

ok = fail = skip = 0
for mod in mods:
    if mod in SKIP:
        print(f"SKIP {mod}: {SKIP[mod]}", flush=True)
        skip += 1
        continue
    try:
        m = importlib.import_module(f"moabb.datasets.{mod}")
    except Exception as e:  # noqa: BLE001
        print(f"IMPORT_FAIL {mod}: {str(e)[:160]}", flush=True)
        fail += 1
        continue
    classes = [
        getattr(m, n)
        for n in dir(m)
        if isinstance(getattr(m, n), type)
        and issubclass(getattr(m, n), BaseDataset)
        and getattr(m, n) is not BaseDataset
        and getattr(getattr(m, n), "__module__", "") == m.__name__
    ]
    for cls in classes:
        name = cls.__name__
        t0 = time.time()
        try:
            d = cls()
            # download() fetches all subjects; accept license non-interactively
            try:
                d.download(accept=True)
            except TypeError:
                d.download()
            print(f"OK {name} ({mod}) {time.time()-t0:.0f}s", flush=True)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name} ({mod}) {time.time()-t0:.0f}s: {str(e)[:180]}", flush=True)
            fail += 1

print(f"ALL_DONE ok={ok} fail={fail} skip={skip}", flush=True)
