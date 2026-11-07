"""The b-decay l_xy shape of each category from the ttbar simulation (MC truth b-hadron decays): the b tag makes the b-jet
decays longer than the b decays outside the ttbar jets or in untagged jets, so one b-jet template does not fit every category.
Per category (b jet: cat 0; other ttbar jet: jcat 1; outside the ttbar jets: jcat >= 2), the selection of the fits (opposite
sign, N_b >= 1, |z0| < 0.1 cm, J/psi window, -0.5 < l_xy < 1.5 mm), an unbinned fit of the model of the unbinned m x l_xy fit:
    f G(0, S sigma_i) + (1 - f) exp(lambda) x G(0, S sigma_i),     sigma_i the l_xy error of the candidate
gives the short-decay share f and the decay length lambda. The fits use them (TTJ_BTEMPLATE=<this json>) with lambda scaled by
lambda(b jets, the dataset) / lambda(b jets, the simulation), the same fit on both: lam_ref_2d (fit_jpsi.py) and lam_ref_lxy
(fit_lxy.py) of the simulation are stored here. For the figures, per category the l_xy histogram of the truth b decays with the
projection of its fitted model and of the b-jet model (both with the category's own per-candidate errors).
-> btemplate.json   usage: python3 btemplate_mc.py <MC cands_assoc> <MC fit_jpsi.json> <MC fit_lxy.json> <out.json>"""
import json
import os
import sys

import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy.special import log_ndtr, ndtr

MCD, FJM, FLM, OUTJ = sys.argv[1:5]
L_LO, L_HI = -0.5, 1.5
WIN = (2.95, 3.25)
EDGES = np.array([-0.5, -0.2, -0.1, -0.05, -0.025, 0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5])


def gaus_l(x, s):
    return np.exp(-0.5 * (x / s) ** 2) / (np.sqrt(2 * np.pi) * s) / (ndtr(L_HI / s) - ndtr(L_LO / s))


def cdf_decay(x, lam, s):
    return ndtr(x / s) - np.exp(s * s / (2 * lam * lam) - x / lam + log_ndtr(x / s - s / lam))


def decay_l(x, lam, s):
    return np.exp(s * s / (2 * lam * lam) - x / lam + log_ndtr(x / s - s / lam)) / lam / (cdf_decay(L_HI, lam, s) - cdf_decay(L_LO, lam, s))


d = pd.concat([pd.read_parquet(os.path.join(MCD, f"cands_{c}.parquet"), columns=["os", "nb", "life", "lxy", "lxyerr", "mass", "cat", "jcat", "truth"])
               for c in ("mujets", "ejets", "emu")], ignore_index=True)
d = d[(d.os == 1) & (d.nb >= 1) & (d.life <= 2) & (d.lxy > L_LO) & (d.lxy < L_HI) & (d.mass > WIN[0]) & (d.mass < WIN[1]) & (d.truth == 1)]
CATS = {"inb": d.cat == 0, "ttj": d.jcat == 1, "signal": d.jcat >= 2}
OUT = {"model": "f G(0, S err) + (1 - f) exp(lambda) x G(0, S err), MC truth b decays in the J/psi window", "cats": {},
       "lam_ref_2d": json.load(open(FJM))["lt2d"]["params"]["lam_inb"], "lam_ref_lxy": json.load(open(FLM))["params"]["lam_inb"]}


def proj(er, f, lam, S):
    """expected count per bin of EDGES: the sum over the candidates of their bin probabilities"""
    s = (S * er)[:, None]
    g = (ndtr(EDGES / s) - ndtr(L_LO / s)) / (ndtr(L_HI / s) - ndtr(L_LO / s))
    c = (cdf_decay(EDGES, lam, s) - cdf_decay(L_LO, lam, s)) / (cdf_decay(L_HI, lam, s) - cdf_decay(L_LO, lam, s))
    return np.diff(f * g + (1 - f) * c, axis=1).sum(axis=0)


OUT["edges"] = EDGES.tolist()
for cat, m in CATS.items():
    q = d[m]
    x, er = q.lxy.to_numpy(float), np.maximum(q.lxyerr.to_numpy(float), 0.003)

    def nll(f, lam, S):
        s = S * er
        return -np.log(np.maximum(f * gaus_l(x, s) + (1 - f) * decay_l(x, lam, s), 1e-300)).sum()

    mi = Minuit(nll, f=0.05, lam=0.4, S=1.3)
    mi.errordef = Minuit.LIKELIHOOD
    mi.limits["f"], mi.limits["lam"], mi.limits["S"] = (0, 1), (0.02, 3), (0.5, 4)
    mi.migrad()
    mi.hesse()
    OUT["cats"][cat] = {"n": int(len(q)), "f": float(mi.values["f"]), "f_err": float(mi.errors["f"]), "lam": float(mi.values["lam"]),
                        "lam_err": float(mi.errors["lam"]), "S": float(mi.values["S"]), "valid": bool(mi.valid),
                        "hist": np.histogram(x, EDGES)[0].tolist(), "model": proj(er, mi.values["f"], mi.values["lam"], mi.values["S"]).tolist()}
    print(f"{cat:6s}: {len(q):6d} b decays: short share f = {mi.values['f']:.3f} +- {mi.errors['f']:.3f}, lambda = {mi.values['lam']:.3f} +- {mi.errors['lam']:.3f} mm,"
          f" S = {mi.values['S']:.2f}; valid {mi.valid}")
B = OUT["cats"]["inb"]
for cat, m in CATS.items():
    OUT["cats"][cat]["model_bjet"] = proj(np.maximum(d[m].lxyerr.to_numpy(float), 0.003), B["f"], B["lam"], B["S"]).tolist()
print(f"reference lambda of the simulation's b jets: 2D fit {OUT['lam_ref_2d']:.3f}, l_xy fit {OUT['lam_ref_lxy']:.3f} mm")
json.dump(OUT, open(OUTJ + ".tmp", "w"), indent=1)
os.replace(OUTJ + ".tmp", OUTJ)  # the fits read it (TTJ_BTEMPLATE) while toys run
print("wrote", OUTJ)
