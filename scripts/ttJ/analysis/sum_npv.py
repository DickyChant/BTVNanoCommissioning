"""Sum of the npv histograms (good primary vertices of all selected ttbar events) of a ttJ output directory per channel and
year -> json {channel: {year: {"n": events, "mean_npv": ...}}}.   usage: python sum_npv.py <out dir> <out.json>"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from coffea.util import load

OUT = sys.argv[1]


def one(name):
    p = f"{OUT}/{name}/{name}.coffea"
    r = {}
    if not os.path.exists(p):
        return r
    o = load(p)
    for ds in o:
        y = ds.split("Run")[1][:4] if "Run20" in ds else os.environ.get("TTJ_MC_YEAR", "2024")
        for ch, x in o[ds].items():
            if isinstance(x, dict) and "npv" in x:
                h = x["npv"]
                v = h.values()
                c = h.axes[0].centers
                a = r.setdefault((ch, y), np.zeros(len(v)))
                a += v
                r[(ch, y), "c"] = c
    return r


with Pool(10) as pool:
    parts = pool.map(one, sorted(os.listdir(OUT)), chunksize=8)
tot, cen = {}, None
for r in parts:
    for k, v in r.items():
        if isinstance(k[1], str) and k[1] == "c":
            cen = v
            continue
        tot[k] = tot.get(k, 0) + v
res = {}
for (ch, y), v in sorted(tot.items()):
    res.setdefault(ch, {})[y] = {"n": float(v.sum()), "mean_npv": float((v * cen).sum() / v.sum()), "hist": v.tolist()}
    print(ch, y, f"{v.sum():.0f} events, <npv> {(v * cen).sum() / v.sum():.2f}")
json.dump(res, open(sys.argv[2], "w"))
