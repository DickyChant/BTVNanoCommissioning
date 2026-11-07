"""Which prompt-J/psi fit closes on the truth? Toy experiments from the CMS ttbar simulation run through the full chain of the
data (life_from_cands.py -> fit_lxy.py, fit_jpsi.py -> splot_prompt.py), with a known number of prompt J/psi.
Each toy:
  - draws, without replacement, the number of candidates of the data in each stratum (channel x category x sign; category:
    b jet / other ttbar jet / outside the ttbar jets) from the simulation's candidate tables (MC truth kept): the toys have the
    data's size and the simulation's composition. No stratum in a variable of the fits: the data's counts per l_xy class give
    the simulated b decays the data's l_xy spectrum (about 50 extra short b decays per toy outside the ttbar jets, a 4%
    prompt-like share in the b jets), and per mass region (window, sidebands) a continuum under the peak that the sidebands do
    not describe (the data has more continuum per J/psi: about 100 candidates over-subtracted per toy);
  - injects n_inj truth-prompt J/psi of the simulation (all categories, primary vertex; with replacement) into the signal
    category (J/psi outside the ttbar jets);
  - scenario "datalike" also injects prompt J/psi into the b jets, prompt / b decay = 0.05 as in the data b jets (the
    prompt-like share the fits give to every b-decay template); scenario "mc": the b jets as simulated (no prompt-like part);
  - runs the fits and records the prompt yield of each against the truth (injected + the simulation's own truth-prompt
    J/psi of the drawn signal candidates; full mass range for the 2D and unbinned fits, window - sidebands / 2 for the l_xy
    fit, which counts J/psi in the window).
-> <out dir>/toys.jsonl (one line per toy)
usage: python3 toy_closure.py <MC cands_assoc> <data cands_assoc> <MC results_asdata.json> <out dir> [--ntoys 20] [--inject 0,50,100,150]
       [--scenarios mc,datalike] [--workers 10] [--seed 1]"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CH = ("mujets", "ejets", "emu")
WIN, SBS = (2.95, 3.25), ((2.60, 2.90), (3.30, 3.60))
ap = argparse.ArgumentParser()
ap.add_argument("mc")
ap.add_argument("data")
ap.add_argument("summary")
ap.add_argument("out")
ap.add_argument("--ntoys", type=int, default=20)
ap.add_argument("--inject", default="0,50,100,150")
ap.add_argument("--scenarios", default="mc,datalike")
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--retry-failed", action="store_true", help="rerun only the toys of <out dir>/toys.jsonl that failed")
ap.add_argument("--npv", default=os.path.join(HERE, "mc", "npv.json"), help="pileup vertices of the simulation (splot_prompt.py reads it next to its output)")
A = ap.parse_args()


def grp(d):
    return np.where(d.cat == 0, 0, np.where(d.jcat == 1, 1, 2))


def load(dd):
    out = []
    for ch in CH:
        d = pd.read_parquet(os.path.join(dd, f"cands_{ch}.parquet"))
        d["ch"] = ch
        d["grp"] = grp(d)
        out.append(d)
    return pd.concat(out, ignore_index=True)


MC, DA = load(A.mc), load(A.data)
KEY = ["ch", "grp", "os"]  # not l_xy or mass: the fitted variables keep the shapes of the simulation
NDATA = DA.groupby(KEY).size()
STRATA = {k: g.index.to_numpy() for k, g in MC.groupby(KEY)}
FIT = (MC.os == 1) & (MC.nb >= 1) & (MC.life <= 2) & (MC.lxy > -0.5) & (MC.lxy < 1.5) & (MC.mass > 2.6) & (MC.mass < 3.6)
POOL = MC.index[FIT & (MC.truth == 0)].to_numpy()  # the truth-prompt J/psi of the simulation, all categories
print(f"{len(MC)} simulated candidates in {len(STRATA)} strata; {len(DA)} data candidates; prompt pool {len(POOL)}", flush=True)


def sbw(m):
    return np.where((m > WIN[0]) & (m < WIN[1]), 1.0, 0.0) - 0.5 * np.where(((m > SBS[0][0]) & (m < SBS[0][1])) | ((m > SBS[1][0]) & (m < SBS[1][1])), 1.0, 0.0)


def truth(t):
    """truth-prompt J/psi of the signal selection of the fits: full mass range, and window - sidebands / 2"""
    m = (t.os == 1) & (t.nb >= 1) & (t.life <= 2) & (t.lxy > -0.5) & (t.lxy < 1.5) & (t.mass > 2.6) & (t.mass < 3.6) & (t.jcat >= 2) & (t.truth == 0)
    return int(m.sum()), float(sbw(t.mass[m].to_numpy()).sum())


def toy(job):
    scen, ninj, it = job
    rng = np.random.default_rng(A.seed * 100003 + ninj * 101 + it * 7 + (0 if scen == "mc" else 50021))
    parts = []
    for k, nd in NDATA.items():
        idx = STRATA.get(k)
        if idx is None or not len(idx):
            continue
        n = min(rng.poisson(nd), len(idx))
        parts.append(MC.loc[rng.choice(idx, n, replace=False)])
    t = pd.concat(parts, ignore_index=True)
    inj = []
    if ninj:  # prompt J/psi outside the ttbar jets: in no selected jet
        p = MC.loc[rng.choice(POOL, rng.poisson(ninj), replace=True)].copy()
        p["cat"], p["jcat"], p["tt_sel"], p["tt_chi2"], p["ttjet"] = 2, 3, False, False, False
        inj.append(p)
    if scen == "datalike":  # prompt J/psi in the b jets: prompt / b decay = 0.05, as in the data
        nb_ = int(((t.grp == 0) & (t.truth == 1) & (t.life <= 2) & (t.os == 1)).sum())
        p = MC.loc[rng.choice(POOL, rng.poisson(0.05 * nb_), replace=True)].copy()
        p["cat"], p["jcat"], p["grp"] = 0, 0, 0
        inj.append(p)
    t = pd.concat([t] + inj, ignore_index=True)
    tr_full, tr_ws = truth(t)
    with tempfile.TemporaryDirectory(dir=A.out) as w:
        cd = os.path.join(w, "cands")
        os.makedirs(cd)
        shutil.copy(A.npv, os.path.join(w, "npv.json"))
        for ch in CH:
            t[t.ch == ch].drop(columns=["ch", "grp"]).to_parquet(os.path.join(cd, f"cands_{ch}.parquet"))
        run = lambda *a, env=None: subprocess.run([sys.executable, *a], cwd=HERE, capture_output=True, text=True,  # noqa: E731
                                                   env={**os.environ, **(env or {})})
        r = run("life_from_cands.py", A.summary, cd, f"{w}/results.json", "--signal", "assoc")
        r = r.returncode == 0 and run("fit_lxy.py", f"{w}/results.json", f"{w}/fit_lxy.json")
        r = r and r.returncode == 0 and run("fit_jpsi.py", f"{w}/results.json", f"{w}/fit_jpsi.json")
        r = r and r.returncode == 0 and run("splot_prompt.py", cd, f"{w}/fit_jpsi.json", f"{w}/results.json", f"{w}/splot.json", f"{w}/sw.parquet", env={"SIGNAL": "assoc"})
        if not r or r.returncode != 0:
            return {"scenario": scen, "inject": ninj, "toy": it, "error": (r.stderr[-400:] if r else "life/fit failed")}
        FL, FJ, SP = (json.load(open(f"{w}/{n}.json")) for n in ("fit_lxy", "fit_jpsi", "splot"))
    L2 = FJ["lt2d"]["samples"]["notb"]
    res = {"scenario": scen, "inject": ninj, "toy": it, "truth_full": tr_full, "truth_ws": tr_ws,
           "lxy": FL["params"]["fp_notb"] * FL["params"]["N_notb"], "lxy_err": FL["errors"]["fp_notb"] * FL["params"]["N_notb"],
           "fit2d": L2["N_P"], "fit2d_err": L2["fp_err"] * (L2["N_P"] + L2["N_NP"]),
           "unbinned": SP["fits"]["notb"]["N"][0], "unbinned_err": SP["fits"]["notb"]["N_err"][0],
           "unbinned_global": SP["fits"]["notb_globalres"]["N"][0], "unbinned_global_err": SP["fits"]["notb_globalres"]["N_err"][0],
           "fp_inb_lxy": FL["params"]["fp_inb"]}
    return res


if __name__ == "__main__":
    os.makedirs(A.out, exist_ok=True)
    jobs = [(s, int(n), i) for s in A.scenarios.split(",") for n in A.inject.split(",") for i in range(A.ntoys)]
    if A.retry_failed:  # the same seeds: the toys that failed, unless a later line has them
        done = {}
        for l in open(os.path.join(A.out, "toys.jsonl")):
            t = json.loads(l)
            k = (t["scenario"], t["inject"], t["toy"])
            done[k] = done.get(k, False) or "error" not in t
        jobs = [k for k, ok_ in done.items() if not ok_]
        print(len(jobs), "failed toys to rerun", flush=True)
    with Pool(A.workers) as pool, open(os.path.join(A.out, "toys.jsonl"), "a") as fo:
        for res in pool.imap_unordered(toy, jobs):
            fo.write(json.dumps(res) + "\n")
            fo.flush()
            if "error" in res:
                print("toy failed", res["scenario"], res["inject"], res["toy"], res["error"][-200:], flush=True)
            else:
                print(f"{res['scenario']:8s} inject {res['inject']:3d} toy {res['toy']:2d}: truth {res['truth_full']} ({res['truth_ws']:.0f}) | l_xy {res['lxy']:.0f}, 2D {res['fit2d']:.0f},"
                      f" unbinned {res['unbinned']:.0f}, global {res['unbinned_global']:.0f}", flush=True)
