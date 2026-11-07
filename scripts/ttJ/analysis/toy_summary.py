"""Summary of the toy closure study (toy_closure.py -> toys.jsonl): per scenario, injected yield and fit, the truth, the mean
fitted yield, the bias (fit - truth) with its error, the spread, and the pull (fit - truth) / (the fit's error).
The truth of each fit is its own convention: full mass range for the 2D and unbinned fits, window - sidebands / 2 for the
l_xy fit.
-> toy_closure.json   usage: python3 toy_summary.py <toys.jsonl> <out.json>"""
import json
import sys

import numpy as np

IN, OUT = sys.argv[1:3]
T = [json.loads(l) for l in open(IN) if l.strip()]
# a toy rerun after a failure (toy_closure.py --retry-failed) appears twice: keep its successful line
best = {}
for t in T:
    k = (t["scenario"], t["inject"], t["toy"])
    if k not in best or "error" in best[k]:
        best[k] = t
T = list(best.values())
ok = [t for t in T if "error" not in t]
FITS = {"lxy": ("l_xy fit", "truth_ws"), "fit2d": ("binned 2D fit", "truth_full"), "unbinned": ("unbinned fit", "truth_full"),
        "unbinned_global": ("unbinned fit, one resolution", "truth_full")}
PE = np.linspace(-5, 5, 21)  # pull histograms (under- and overflow in the end bins)
R = {"pull_edges": PE.tolist(), "n_toys": len(T), "n_failed": len(T) - len(ok), "fits": {k: v[0] for k, v in FITS.items()}, "points": []}
for sc in sorted({t["scenario"] for t in ok}):
    for inj in sorted({t["inject"] for t in ok if t["scenario"] == sc}):
        g = [t for t in ok if t["scenario"] == sc and t["inject"] == inj]
        row = {"scenario": sc, "inject": inj, "n": len(g)}
        for k, (_, tk) in FITS.items():
            f = np.array([t[k] for t in g])
            e = np.array([t[f"{k}_err"] for t in g])
            tr = np.array([t[tk] for t in g], float)
            dlt = f - tr
            pl = dlt / np.where(e > 0, e, np.nan)
            row[k] = {"truth": float(tr.mean()), "fit": float(f.mean()), "bias": float(dlt.mean()), "bias_err": float(dlt.std(ddof=1) / np.sqrt(len(g))),
                      "rms": float(dlt.std(ddof=1)), "pull_mean": float(np.nanmean(pl)), "pull_width": float(np.nanstd(pl, ddof=1)),
                      "mean_err": float(e.mean()),
                      "pull_hist": np.histogram(np.clip(pl[np.isfinite(pl)], -4.99, 4.99), PE)[0].tolist()}
        R["points"].append(row)
json.dump(R, open(OUT, "w"), indent=1)
print(f"{len(T)} toys, {R['n_failed']} failed")
for row in R["points"]:
    print(f"{row['scenario']:8s} inject {row['inject']:3d} ({row['n']} toys): " + " | ".join(
        f"{k} truth {row[k]['truth']:.0f} bias {row[k]['bias']:+.1f}+-{row[k]['bias_err']:.1f} rms {row[k]['rms']:.0f} pull {row[k]['pull_mean']:+.2f}/{row[k]['pull_width']:.2f}" for k in FITS))
