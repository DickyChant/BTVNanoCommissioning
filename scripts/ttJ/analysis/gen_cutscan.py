"""Tighter signal selections against the top-decay J/psi that pass the veto, in Pythia8 + LETO (gen table of gen_assoc.py), and
their cost on the data: per selection, the top-decay share of the signal and the efficiency for ISR J/psi (Pythia), the prompt
sWeights kept in data and for the prompt pileup J/psi (random direction with respect to the ttbar).
-> gen_cutscan.json   usage: python3 gen_cutscan.py <gen table parquet> <sweights.parquet> <out.json>"""
import json
import sys

import numpy as np
import pandas as pd

GT, SWQ, OUTJ = sys.argv[1:4]
G = pd.read_parquet(GT)
G = G[G.prompt].reset_index(drop=True) if "prompt" in G else G  # the table of gen_assoc.py also holds the b-decay J/psi
S = pd.read_parquet(SWQ)


def insj(q):
    return ((q.sj_dr >= 0) & (q.sj_dr < 0.4)).to_numpy()


def mjj_win(q):
    return ((q.tt_mjj > 65) & (q.tt_mjj < 95)).to_numpy() | (q.ch == "emu").to_numpy()


CUTS = {
    "veto (nominal)": lambda q: np.ones(len(q), bool),
    "ΔR(J/ψ, t̅t jets) > 0.8": lambda q: (q.tt_drmin > 0.8).to_numpy(),
    "ΔR(J/ψ, t̅t jets) > 1.2": lambda q: (q.tt_drmin > 1.2).to_numpy(),
    "in no jet above 15 GeV": lambda q: ~insj(q) & (q.jcat == 3).to_numpy(),
    "in no jet above 15 GeV, ΔR > 0.8": lambda q: ~insj(q) & (q.jcat == 3).to_numpy() & (q.tt_drmin > 0.8).to_numpy(),
    "W pair 65–95 GeV": mjj_win,
    "W pair 65–95 GeV, ΔR > 0.8": lambda q: mjj_win(q) & (q.tt_drmin > 0.8).to_numpy(),
}
OUT = {"cuts": {}}
g = G[G.jcat >= 2]
top, isr = (g.origin == 1).to_numpy(), (g.origin == 3).to_numpy()
ntop0, nisr0 = int(((G.origin == 1) & (G.jcat >= 2)).sum()), int(((G.origin == 3) & (G.jcat >= 2)).sum())
sig = S[S["sample"] == "notb"]
pu = S[S["sample"] == "pileup"]
for nm, f in CUTS.items():
    m = f(g)
    a, b = int((m & top).sum()), int((m & isr).sum())
    ws, wp = sig.w_prompt.to_numpy(float), pu.w_prompt.to_numpy(float)
    ms, mp = f(sig), f(pu)
    OUT["cuts"][nm] = {"top": a, "isr": b, "top_share": a / max(a + b, 1), "isr_eff": b / max(nisr0, 1), "top_eff": a / max(ntop0, 1),
                       "data_prompt": float(ws[ms].sum()), "data_prompt_frac": float(ws[ms].sum() / ws.sum()), "pileup_frac": float(wp[mp].sum() / wp.sum())}
    o = OUT["cuts"][nm]
    print(f"{nm:36s} Pythia: top decay {a:4d} ISR {b:4d}, top-decay share {o['top_share']:.2f}, ISR kept {o['isr_eff']:.2f}, top decay kept {o['top_eff']:.2f}"
          f" | data: prompt {o['data_prompt']:5.1f} ({o['data_prompt_frac']:.2f}), pileup J/psi kept {o['pileup_frac']:.2f}")
json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False)
print("wrote", OUTJ)
