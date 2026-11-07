"""All J/psi candidates of the per-candidate tables (jpsi_cands) of a ttJ output directory in one parquet file per channel,
with the data set and year: for the unbinned fits and for looking at the e + jets events. MC (a data set name without
"Run<year>"): the year of TTJ_MC_YEAR (default 2024); the table has the MC truth column (truth) and the event weight, and
<dest dir>/sumw.json the sum of the generator weights per data set (for the cross-section normalisation).
usage: python dump_cands.py <out dir> <dest dir>"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from coffea.util import load

OUT, DEST = sys.argv[1], sys.argv[2]


def one(name):
    p = f"{OUT}/{name}/{name}.coffea"
    if not os.path.exists(p):
        return {}
    try:
        o = load(p)
    except Exception as err:  # a file still being written by a running job: skipped, and named
        print(f"skipped {p}: {type(err).__name__}", flush=True)
        return {}
    r = {"_sumw": {}}
    for ds in o:
        sw = [x["sumw"] for x in o[ds].values() if isinstance(x, dict) and "sumw" in x]
        r["_sumw"][ds] = float(sw[0]) if sw else 0.0
        for ch, x in o[ds].items():
            if not isinstance(x, dict) or "jpsi_cands" not in x:
                continue
            t = {k: np.asarray(v.value) for k, v in x["jpsi_cands"].items()}
            if not len(t.get("mass", [])):
                continue
            df = pd.DataFrame(t)
            df["dataset"] = ds
            df["year"] = int(ds.split("Run")[1][:4]) if "Run20" in ds else int(os.environ.get("TTJ_MC_YEAR", "2024"))
            df["job"] = name
            r.setdefault(ch, []).append(df)
    return r


with Pool(10) as pool:
    parts = pool.map(one, sorted(os.listdir(OUT)), chunksize=8)
os.makedirs(DEST, exist_ok=True)
SW = {}
for r in parts:
    for ds, v in r.pop("_sumw", {}).items():
        SW[ds] = SW.get(ds, 0.0) + v
json.dump(SW, open(os.path.join(DEST, "sumw.json"), "w"), indent=1)
for ch in sorted({c for r in parts for c in r}):
    df = pd.concat([d for r in parts for d in r.get(ch, [])], ignore_index=True)
    df.to_parquet(os.path.join(DEST, f"cands_{ch}.parquet"), index=False)
    print(ch, len(df), "candidates,", df["job"].nunique(), "jobs")
