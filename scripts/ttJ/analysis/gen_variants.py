"""Prompt J/psi of the ttbar simulation per generator / tune against the data: is the simulation short of prompt J/psi
everywhere? None of the samples has onia in the parton shower; their prompt J/psi come from hadronisation (a low-mass
c c-bar system collapsing into a J/psi: Lund strings in Pythia, clusters in Sherpa AHADIC), so the colour reconnection
and the hadronisation model can change them. Samples (Summer24, NanoAODv15, the code of the data pass; l + jets, each
normalised to the sigma(ttbar) x BR of the POWHEG sample x 310 fb^-1, one-charge samples x 1/2):
    POWHEG + Pythia 8 CP5 (nominal), CP5 CR1 (QCD-inspired), CP5 CR2 (gluon move), CP5 with early resonance decays (ERD on),
    Sherpa (default hadronisation), Sherpa AHADIC (cluster), MadGraph5 MLM (up to 3 jets) and aMC@NLO FxFx (up to 2 jets),
    both with Pythia 8 CP5.
Per category (b jet: cat 0; other ttbar jet: jcat 1; outside the ttbar jets: jcat >= 2), the selection of the unbinned fit
(opposite sign, N_b >= 1, |z0| < 0.1 cm, 2.6 < m < 3.6 GeV, -0.5 < l_xy < 1.5 mm): MC truth prompt and b-decay J/psi
against the data sWeights (l + jets). The data prompt yield in the other ttbar jets is biased by short b decays (closure).
The l + jets channels with the l + jets samples; "full" scales each to the whole ttbar prediction of the l + jets channels with
the nominal ratio (l + jets + dilepton samples) / (l + jets sample). The e mu channel separately ("emu" block, ratio keys
"<gen>|<cat>|emu") with the dilepton samples of the same generators (cluster 897660) against the e mu sWeights.
-> gen_variants.json   usage: python3 gen_variants.py <nominal cands_assoc> <nominal sumw.json> <variants cands_assoc>
                              <variants sumw.json> <xsec_map.json> <data sweights.parquet> <out.json>"""
import json
import os
import sys

import numpy as np
import pandas as pd

NOMD, NOMSW, ALTD, ALTSW, XSJ, SWQ, OUTJ = sys.argv[1:8]
LJ = ("mujets", "ejets")
M_LO, M_HI, L_LO, L_HI = 2.6, 3.6, -0.5, 1.5
LUMI_PB = 310.0e3
XS = json.load(open(XSJ))
GEN = {  # name: data-set name prefixes (l + jets)
    "Pythia 8 CP5": ("TTtoLNu2Q_TuneCP5_13p6TeV",),
    "CP5 CR1": ("TTtoLNu2Q_TuneCP5CR1_",),
    "CP5 CR2": ("TTtoLNu2Q_TuneCP5CR2_",),
    "CP5 ERD on": ("TTtoLNu2Q_Par-ERD-On_",),
    "Sherpa": ("TTtoLplusNuQ-4Jets-1NLO3LO_TuneSherpaDef",),
    "Sherpa AHADIC": ("TTtoLminusNu2Q-4Jets-1NLO3LO_TuneAHADIC",),
    "MG5 MLM": ("TTtoLplusNu2Q-3Jets_TuneCP5", "TTtoLminusNu2Q-3Jets_TuneCP5"),
    "aMC@NLO FxFx": ("TTtoLplusNu2Q-2Jets_TuneCP5", "TTtoLminusNu2Q-2Jets_TuneCP5"),
}
GEN_DL = {  # the dilepton samples of the same generators (e mu)
    "Pythia 8 CP5": ("TTto2L2Nu_TuneCP5_13p6TeV",),
    "CP5 CR1": ("TTto2L2Nu_TuneCP5CR1_",),
    "CP5 CR2": ("TTto2L2Nu_TuneCP5CR2_",),
    "CP5 ERD on": ("TTto2L2Nu_Par-ERD-On_",),
    "Sherpa": ("TTto2L2Nu-4Jets-1NLO3LO_TuneSherpaDef",),
    "Sherpa AHADIC": ("TTto2L2Nu-4Jets-1NLO3LO_TuneAHADIC",),
    "MG5 MLM": ("TTto2L2Nu-3Jets_TuneCP5",),
    "aMC@NLO FxFx": ("TTto2L2Nu-2Jets_TuneCP5",),
}
CATS = ("inb", "ttj", "signal")


def load(d_, swj, chans=LJ):
    if not os.path.exists(swj):
        return pd.DataFrame()
    SW = json.load(open(swj))
    out = []
    for ch in chans:
        p = os.path.join(d_, f"cands_{ch}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d = d[(d.os == 1) & (d.nb >= 1) & (d.life <= 2) & (d.lxy > L_LO) & (d.lxy < L_HI) & (d.mass > M_LO) & (d.mass < M_HI)].copy()
        d = d[d.dataset.isin([k for k in SW if SW[k] > 0 and k in XS])]
        d["w"] = d.weight.to_numpy(float) * np.array([XS[ds] / SW[ds] for ds in d.dataset]) * LUMI_PB
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


NOM_ALL = load(NOMD, NOMSW)
A = pd.concat([NOM_ALL, load(ALTD, ALTSW)], ignore_index=True)
S = pd.read_parquet(SWQ)
S = S[S.ch.isin(LJ)]
SMP = {"inb": "inb", "ttj": "ttj", "signal": "notb"}
OUT = {"lumi_pb": LUMI_PB, "data": {}, "gen": {}, "datasets": {}, "full_over_lj": {}}


def data_yields(S_):
    """prompt and b-decay sWeights per category"""
    out = {}
    for cat in CATS:
        ds = S_[S_["sample"] == SMP[cat]]
        out[cat] = {sp: [float(ds[f"w_{sp}"].sum()), float(np.sqrt((ds[f"w_{sp}"] ** 2).sum()))] for sp in ("prompt", "nonprompt")}
    return out


def by_truth(A_, gens, out):
    """weighted MC truth prompt and b-decay J/psi per category of each generator: [sum, error, raw count]"""
    for g, pre in gens.items():
        q = A_[A_.dataset.str.startswith(pre)] if len(A_) else A_
        if not len(q):
            continue
        out["datasets"][g] = sorted(q.dataset.unique().tolist())
        cm = {"inb": (q.cat == 0).to_numpy(), "ttj": (q.jcat == 1).to_numpy(), "signal": (q.jcat >= 2).to_numpy()}
        r = {}
        for cat in CATS:
            for t, tn in ((0, "prompt"), (1, "bdecay")):
                k = cm[cat] & (q.truth == t).to_numpy()
                w = q.w.to_numpy(float)[k]
                r[f"{cat}_{tn}"] = [float(w.sum()), float(np.sqrt((w ** 2).sum())), int(k.sum())]
        out["gen"][g] = r


def add_ratios(blk, sfx=""):
    """prompt J/psi per b-decay J/psi within each sample: independent of the normalisation (cross section, sum of weights, event
    weights); no prompt J/psi in the sample: 90% CL upper limit 2.3 / (raw b-decay count) x (weighted / raw b decays)"""
    for g, r in blk["gen"].items():
        for cat in CATS:
            p_, b_ = r[f"{cat}_prompt"], r[f"{cat}_bdecay"]
            if b_[0] <= 0 or b_[2] == 0:
                continue
            if p_[2] > 0 and p_[0] > 0:
                v = p_[0] / b_[0]
                OUT["ratio"][f"{g}|{cat}{sfx}"] = {"r": v, "err": v * ((p_[1] / p_[0]) ** 2 + (b_[1] / b_[0]) ** 2) ** 0.5, "n_p": p_[2], "n_b": b_[2], "upper": None}
            else:
                OUT["ratio"][f"{g}|{cat}{sfx}"] = {"r": 0.0, "err": 0.0, "n_p": p_[2], "n_b": b_[2], "upper": 2.3 / b_[2]}
    for cat in CATS:
        dp, db = blk["data"][cat]["prompt"], blk["data"][cat]["nonprompt"]
        if db[0] > 0 and dp[0] > 0:
            v = dp[0] / db[0]
            OUT["ratio"][f"data|{cat}{sfx}"] = {"r": v, "err": v * ((dp[1] / dp[0]) ** 2 + (db[1] / db[0]) ** 2) ** 0.5}


OUT["data"] = data_yields(S)
by_truth(A, GEN, OUT)
OUT["ratio"] = {}
add_ratios(OUT)
# the e mu channel: the dilepton samples of the same generators against the e mu sWeights
SE = pd.read_parquet(SWQ)
OUT["emu"] = {"data": data_yields(SE[SE.ch == "emu"]), "gen": {}, "datasets": {}}
by_truth(pd.concat([load(NOMD, NOMSW, ("emu",)), load(ALTD, ALTSW, ("emu",))], ignore_index=True), GEN_DL, OUT["emu"])
add_ratios(OUT["emu"], "|emu")
print("prompt / b decay:", {k: (round(v["r"], 4), round(v["err"], 4), v.get("upper") and round(v["upper"], 4)) for k, v in OUT["ratio"].items() if not k.endswith("|ttj")})
nom = OUT["gen"].get("Pythia 8 CP5")
if nom:  # the dilepton events in the l + jets channels: nominal (all samples) / (l + jets sample), per category and species
    q = NOM_ALL
    cm = {"inb": (q.cat == 0).to_numpy(), "ttj": (q.jcat == 1).to_numpy(), "signal": (q.jcat >= 2).to_numpy()}
    for cat in CATS:
        for t, tn in ((0, "prompt"), (1, "bdecay")):
            full = float(q.w.to_numpy(float)[cm[cat] & (q.truth == t).to_numpy()].sum())
            OUT["full_over_lj"][f"{cat}_{tn}"] = full / nom[f"{cat}_{tn}"][0] if nom[f"{cat}_{tn}"][0] else None
    for g, r in OUT["gen"].items():
        for k, v in list(r.items()):
            fz = OUT["full_over_lj"].get(k)
            if fz:
                r[f"{k}_full"] = [v[0] * fz, v[1] * fz]
    print("full / l + jets sample:", {k: round(v, 3) for k, v in OUT["full_over_lj"].items() if v})
print(f"{'':16s}" + "".join(f"{c + ' prompt':>22s}{c + ' b decay':>22s}" for c in CATS))
print(f"{'data (sPlot)':16s}" + "".join(f"{OUT['data'][c]['prompt'][0]:12.1f} ± {OUT['data'][c]['prompt'][1]:6.1f}  {OUT['data'][c]['nonprompt'][0]:12.1f} ± {OUT['data'][c]['nonprompt'][1]:6.1f}  " for c in CATS))
for g, r in OUT["gen"].items():
    line = f"{g:16s}"
    for c in CATS:
        for tn in ("prompt", "bdecay"):
            v = r[f"{c}_{tn}"]
            line += f"{v[0]:12.1f} ± {v[1]:6.1f} ({v[2]:4d})"
    print(line)
    if nom and g != "Pythia 8 CP5":
        print(f"{'  / nominal':16s}" + "".join(f"{r[f'{c}_{tn}'][0] / nom[f'{c}_{tn}'][0]:20.2f}  " if nom[f"{c}_{tn}"][0] else f"{'-':>22s}" for c in CATS for tn in ("prompt", "bdecay")))
print("e mu, dilepton samples (prompt / b decay outside the ttbar jets; in b jets):")
for g in ["data"] + list(OUT["emu"]["gen"]):
    a, b = OUT["ratio"].get(f"{g}|signal|emu"), OUT["ratio"].get(f"{g}|inb|emu")
    fm = lambda x: "-" if not x else (f"< {x['upper']:.3f}" if x.get("upper") else f"{x['r']:.4f} +- {x['err']:.4f}")  # noqa: E731
    print(f"  {g:16s} {fm(a):>20s} {fm(b):>20s}")
json.dump(OUT, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
