"""Prompt J/psi with the b-decay l_xy shape of the simulation of the same category: binned fit of the l_xy of the J/psi
(window - sidebands / 2, the selection of the unbinned fit) to N_p x prompt + N_b x b decay, the templates from the CMS
ttbar simulation (MC truth: b decay of the category; prompt of all categories, few per category), chi2 with the data
and template errors. Why: in the other ttbar jets (vetoed) the J/psi of b decays sit in jets that failed the b tag, which
happens when the b hadron flew little, so their l_xy is short and the b-jet template of the unbinned fit calls most of them
prompt (closure on the simulation). Run on the data and, as a closure, on the simulation itself (unweighted, its truth
known); variations: the prompt template 10% wider / narrower (data resolution vs simulation); and the b-decay yield
constrained (Gaussian, 20%) to the simulation's, scaled by data / simulation of the b-decay J/psi fitted the same way in
the b jets and outside the ttbar jets.
-> mc_template.json   usage: python3 mc_template_fit.py <data cands_assoc dir> <MC cands_assoc dir> <MC sumw.json>
                             <xsec_map.json> <MC summary json> <data results json> <out.json>"""
import json
import os
import sys

import numpy as np
import pandas as pd
from iminuit import Minuit

DD, MCD, SWJ, XSJ, MCR, DRJ, OUTJ = sys.argv[1:8]
CH = ("mujets", "ejets", "emu")
M_LO, M_HI, L_LO, L_HI = 2.6, 3.6, -0.5, 1.5
WIN, SBS = (2.95, 3.25), ((2.60, 2.90), (3.30, 3.60))
E = np.array([-0.5, -0.2, -0.1, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.5])
GRP = {"all": CH, "ljets": ("mujets", "ejets")}
SW, XS = json.load(open(SWJ)), json.load(open(XSJ))
MR, DR = json.load(open(MCR)), json.load(open(DRJ))


def ntt(R, ch, key):
    return float(R["channels"][ch]["yields"][key]["n_tt"])


def load(d_, mc):
    out = []
    for ch in CH:
        d = pd.read_parquet(os.path.join(d_, f"cands_{ch}.parquet"))
        d = d[(d.os == 1) & (d.nb >= 1) & (d.life <= 2) & (d.lxy > L_LO) & (d.lxy < L_HI) & (d.mass > M_LO) & (d.mass < M_HI)].copy()
        d["ch"] = ch
        if mc:  # normalised to the luminosity of the data, 310 fb^-1
            d["wd"] = d.weight.to_numpy(float) * np.array([XS[ds] / SW[ds] for ds in d.dataset]) * 310.0e3
        out.append(d)
    return pd.concat(out, ignore_index=True)


def sbw(m):
    m = np.asarray(m, float)
    return np.where((m > WIN[0]) & (m < WIN[1]), 1.0, 0.0) - 0.5 * np.where(((m > SBS[0][0]) & (m < SBS[0][1])) | ((m > SBS[1][0]) & (m < SBS[1][1])), 1.0, 0.0)


def catmask(d):
    return {"inb": (d.cat == 0).to_numpy(), "ttj": (d.jcat == 1).to_numpy(), "signal": (d.jcat >= 2).to_numpy()}


def hist(x, w):
    return np.histogram(x, E, weights=w)[0], np.sqrt(np.histogram(x, E, weights=w ** 2)[0])


def template(x, w):
    h, e = hist(x, w)
    s = h.sum()
    return h / s, e / s


def fit(h, eh, B, eB, P, eP, nb_con=None):
    def chi2(Np, Nb):
        v = eh ** 2 + (Nb * eB) ** 2 + (Np * eP) ** 2
        c = float((((h - Np * P - Nb * B) ** 2) / np.maximum(v, 1e-12)).sum())
        return c + (((Nb - nb_con[0]) / nb_con[1]) ** 2 if nb_con else 0.0)
    n0 = max(h.sum(), 1.0)
    best = None
    for f0 in (0.05, 0.2, 0.5):  # starts: the prompt share; keep the lowest valid minimum
        mi = Minuit(chi2, Np=f0 * n0, Nb=(1 - f0) * n0)
        mi.errordef = Minuit.LEAST_SQUARES
        mi.simplex()
        mi.migrad()
        mi.hesse()
        if mi.valid and (best is None or mi.fval < best.fval):
            best = mi
    mi = best or mi
    return {"N_p": mi.values["Np"], "N_p_err": mi.errors["Np"], "N_b": mi.values["Nb"], "N_b_err": mi.errors["Nb"], "rho": float(mi.covariance.correlation()["Np", "Nb"]),
            "chi2": mi.fval, "ndf": len(h) - 2, "valid": bool(mi.valid)}


DA, MC = load(DD, False), load(MCD, True)
cmD, cmM = catmask(DA), catmask(MC)
wM, wMs = MC.wd.to_numpy(float), MC.wd.to_numpy(float) * sbw(MC.mass)
tru = MC.truth.to_numpy()
win = ((MC.mass > WIN[0]) & (MC.mass < WIN[1])).to_numpy()
OUT = {"edges": E.tolist(), "fits": {}, "closure": {}, "hists": {}, "templates": {}}
for k_ in (1.0, 0.9, 1.1):  # prompt template: all categories, l_xy scaled by k_
    pm = win & (tru == 0)
    OUT["templates"][f"prompt_{k_:g}"] = [a.tolist() for a in template(MC.lxy[pm] * k_, wM[pm])]
for cat in ("ttj", "signal", "inb"):
    bm = cmM[cat] & win & (tru == 1)
    B, eB = template(MC.lxy[bm], wM[bm])
    OUT["templates"][f"bdecay_{cat}"] = [B.tolist(), eB.tolist()]
    for gn, chs in GRP.items():
        gD, gM = DA.ch.isin(chs).to_numpy(), MC.ch.isin(chs).to_numpy()
        d = DA[cmD[cat] & gD]
        h, eh = hist(d.lxy, sbw(d.mass))
        key = f"{cat}_{gn}"
        OUT["hists"][key] = {"data": [h.tolist(), eh.tolist()]}
        for k_ in (1.0, 0.9, 1.1):
            P, eP = (np.asarray(a) for a in OUT["templates"][f"prompt_{k_:g}"])
            OUT["fits"][f"{key}_w{k_:g}"] = fit(h, eh, B, eB, P, eP)
        # simulation scaled to the data ttbar events: the prediction, by truth
        for t, tn in ((1, "bdecay"), (0, "prompt")):
            m = cmM[cat] & gM & (tru == t)
            OUT["hists"][key][f"mc_{tn}"] = [a.tolist() for a in hist(MC.lxy[m], wMs[m])]
        OUT["hists"][key]["mc_bdecay_n"] = float(wMs[cmM[cat] & gM & (tru == 1)].sum())  # window - sidebands, as the data
        # closure: the simulation as pseudo-data (unweighted, all truths, window - sidebands) against its truth
        q = MC[cmM[cat] & gM]
        hq, ehq = hist(q.lxy, sbw(q.mass))
        P, eP = (np.asarray(a) for a in OUT["templates"]["prompt_1"])
        F = fit(hq, ehq, B, eB, P, eP)
        qw = (q.mass > WIN[0]) & (q.mass < WIN[1])
        F["truth_prompt"] = float((sbw(q.mass)[(q.truth == 0).to_numpy()]).sum())
        F["truth_bdecay"] = float((sbw(q.mass)[(q.truth == 1).to_numpy()]).sum())
        F["n_window"] = int(qw.sum())
        OUT["closure"][key] = F

# the b-decay yield of the vetoed category from the simulation, scaled by data / simulation in the other two categories
for gn in GRP:
    sc = {c: OUT["fits"][f"{c}_{gn}_w1"]["N_b"] / OUT["hists"][f"{c}_{gn}"]["mc_bdecay_n"] for c in ("inb", "signal")}
    s_ = float(np.mean(list(sc.values())))
    pred = s_ * OUT["hists"][f"ttj_{gn}"]["mc_bdecay_n"]
    d = OUT["hists"][f"ttj_{gn}"]["data"]
    P, eP = (np.asarray(a) for a in OUT["templates"]["prompt_1"])
    B, eB = (np.asarray(a) for a in OUT["templates"]["bdecay_ttj"])
    F = fit(np.asarray(d[0]), np.asarray(d[1]), B, eB, P, eP, nb_con=(pred, 0.2 * pred))
    F.update({"scale": sc, "scale_mean": s_, "nb_pred": pred})
    OUT["fits"][f"ttj_{gn}_nbcon"] = F
for k, F in OUT["fits"].items():
    print(f"data {k:16s}: N_p {F['N_p']:7.1f} +- {F['N_p_err']:5.1f}, N_b {F['N_b']:7.1f} +- {F['N_b_err']:5.1f}, rho {F['rho']:+.2f}, chi2 {F['chi2']:.1f} / {F['ndf']}")
for k, F in OUT["closure"].items():
    print(f"closure {k:12s}: N_p {F['N_p']:7.1f} +- {F['N_p_err']:5.1f} (truth {F['truth_prompt']:.1f}), N_b {F['N_b']:7.1f} (truth {F['truth_bdecay']:.1f}), chi2 {F['chi2']:.1f} / {F['ndf']}")
for k, H in OUT["hists"].items():
    print(f"simulation {k:12s}: b-decay J/psi per data ttbar sample {H['mc_bdecay_n']:.1f}")
json.dump(OUT, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
