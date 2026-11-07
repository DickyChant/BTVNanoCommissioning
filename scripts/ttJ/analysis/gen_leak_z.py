"""The leak of top-decay J/psi through the veto, with the Pythia8 + LETO J/psi reweighted to the momentum share z = pT(J/psi) /
pT(jet) measured in data. LETO puts the J/psi at a much larger z inside its jet than the data show for the vetoed prompt J/psi in
the ttbar jets, and the J/psi that pass the veto are mostly those at large z (the jet fails the ID through the J/psi muons, or
the J/psi carries most of a soft jet). Weights w(z) = data / Pythia of the J/psi in the other ttbar jets (jcat 1, l + jets; data:
prompt sWeights), applied to every Pythia J/psi inside a jet (a selected jet, or the nearest gen jet above 15 GeV, |eta| < 2.5,
within 0.4); the J/psi in no jet keep weight 1 (variation: 0.5 and 1.5).
-> gen_leak_z.json   usage: python3 gen_leak_z.py <gen table parquet (gen_assoc.py, GEN_TABLE)> <sweights.parquet> <out.json>"""
import json
import sys

import numpy as np
import pandas as pd

GT, SWQ, OUTJ = sys.argv[1:4]
G = pd.read_parquet(GT)
G = G[G.prompt].reset_index(drop=True) if "prompt" in G else G  # the table of gen_assoc.py also holds the b-decay J/psi
S = pd.read_parquet(SWQ)
# z >= 0.6 in one bin: the jet ID (muon energy fraction < 0.8) leaves almost no selected jet with z > 0.8, in data and in
# Pythia alike, so a weight of its own there would be 0 / 0
ZE = np.array([0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01])


def zsel(q):
    """z with respect to the selected jet within 0.4 of the J/psi (-1 if none)"""
    pt = np.stack([q[f"jet{k}_pt"].to_numpy(float) for k in range(8)], 1)
    eta = np.stack([q[f"jet{k}_eta"].to_numpy(float) for k in range(8)], 1)
    phi = np.stack([q[f"jet{k}_phi"].to_numpy(float) for k in range(8)], 1)
    dph = (phi - q.phi.to_numpy(float)[:, None] + np.pi) % (2 * np.pi) - np.pi
    dr = np.where(pt > 0, np.hypot(eta - q.eta.to_numpy(float)[:, None], dph), 99)
    j = dr.argmin(1)
    r = np.arange(len(q))
    return np.where(dr[r, j] < 0.4, q.pt.to_numpy(float) / pt[r, j], -1.0)


def zany(q):
    """z with respect to the selected jet, else the nearest jet above 15 GeV within 0.4; -1 in no jet"""
    z = zsel(q)
    sj = (q.sj_dr.to_numpy(float) >= 0) & (q.sj_dr.to_numpy(float) < 0.4)
    return np.where(z > 0, z, np.where(sj, q.pt.to_numpy(float) / np.maximum(q.sj_pt.to_numpy(float), 1e-6), -1.0))


def hist(z, w):
    return np.histogram(np.clip(z, 0, 1.0), ZE, weights=w)[0]


d = S[(S["sample"] == "ttj") & S.ch.isin(["mujets", "ejets"])]
zd, wd = zsel(d), d.w_prompt.to_numpy(float)
hd = hist(zd[zd > 0], wd[zd > 0])
hd2 = hist(zd[zd > 0], wd[zd > 0] ** 2)
g = G[G.jcat == 1]
zg = zsel(g)
hg = hist(zg[zg > 0], np.ones((zg > 0).sum()))
W = np.maximum(hd / hd.sum(), 0) / np.maximum(hg / hg.sum(), 1e-9)
OUT = {"z_edges": ZE.tolist(), "data_ttj": (hd / hd.sum()).tolist(), "data_ttj_err": (np.sqrt(hd2) / hd.sum()).tolist(), "pythia_ttj": (hg / hg.sum()).tolist(),
       "weights": W.tolist(), "mean_z_data": float((zd[zd > 0] * wd[zd > 0]).sum() / wd[zd > 0].sum()), "mean_z_pythia": float(zg[zg > 0].mean()), "variants": {}}
z = zany(G)
ib = np.clip(np.digitize(np.clip(z, 0, 1.0), ZE) - 1, 0, len(W) - 1)
for tag, wno in (("nominal", 1.0), ("nojet_x0.5", 0.5), ("nojet_x1.5", 1.5), ("raw", None)):
    w = np.ones(len(G)) if wno is None else np.where(z > 0, W[ib], wno)
    top, isr, sig = (G.origin == 1).to_numpy(), (G.origin == 3).to_numpy(), (G.jcat >= 2).to_numpy()
    eff = w[top & sig].sum() / w[top].sum()
    share = w[top & sig].sum() / w[sig].sum()
    per_ttj = w[top & sig].sum() / w[(G.jcat == 1).to_numpy()].sum()  # leaked top-decay J/psi per vetoed J/psi in the other ttbar jets
    OUT["variants"][tag] = {"top_decay_pass": float(eff), "isr_pass": float(w[isr & sig].sum() / w[isr].sum()), "signal_top_decay_share": float(share),
                            "leak_per_vetoed_ttjet": float(per_ttj), "ratio_ttjet_to_signal": float(w[(G.jcat == 1).to_numpy()].sum() / w[sig].sum())}
    print(f"{tag:11s}: top-decay J/psi passing {eff:.3f}, ISR passing {OUT['variants'][tag]['isr_pass']:.3f}; top-decay share of the signal {share:.3f};"
          f" leaked per vetoed other-ttbar-jet J/psi {per_ttj:.3f}; vetoed / signal {OUT['variants'][tag]['ratio_ttjet_to_signal']:.2f}")
print("z data", np.round(OUT["data_ttj"], 2), "mean", round(OUT["mean_z_data"], 2), "| Pythia", np.round(OUT["pythia_ttj"], 2), "mean", round(OUT["mean_z_pythia"], 2))
json.dump(OUT, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
