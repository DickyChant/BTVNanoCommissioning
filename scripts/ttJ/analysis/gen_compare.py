"""Data against Pythia8 ttbar with J/psi from the parton shower (LETO; genstudy/gen_ttjpsi.py, gen table of gen_assoc.py) for
the prompt J/psi of l + jets: outside the ttbar jets (the signal, jcat >= 2) and in the other ttbar jets (vetoed, jcat 1).
Normalisation without a detector simulation: the J/psi from b decays (all categories) in bins of pT(J/psi),
    k(pT) = N_data(b decay, sWeights) / N_Pythia(b decay),
absorbs the luminosity, the ttbar cross section and selection and the J/psi -> mu mu efficiency; each prompt Pythia J/psi gets
k(pT) / ldmeFac (the onia-shower rate is linear in the LDMEs: OniaShower:ldmeFac = 20 in the sample). The second parton
scattering (DPS) is not in the Pythia sample (its MPI showers give no J/psi in the acceptance): it is the prompt pileup J/psi
of the data (another collision, the J/psi taken out of the MET), at the rate of splot_prompt.py for sigma_eff = 15 mb.
Pythia variants: nominal, and "z" with each J/psi inside a jet reweighted to the momentum share z of the data (gen_leak_z.py).
Gen-level ttbar observables from the same reconstruction as the data (ttreco.py, the J/psi jet left out), gen jets, MET =
the neutrino sum smeared by 20 GeV per component.
-> gen_compare.json   usage: python3 gen_compare.py <gen table> <sweights.parquet> <splot.json> <gen_leak_z.json> <out.json>"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttreco import observables, rapidity, reconstruct  # noqa: E402

GT, SWQ, SPJ, LZJ, OUTJ = sys.argv[1:6]
LJ = ("mujets", "ejets")
G = pd.read_parquet(GT)
S = pd.read_parquet(SWQ)
S = S[S.ch.isin(LJ)].reset_index(drop=True)
SP, LZ = json.load(open(SPJ)), json.load(open(LZJ))
LDME = 20.0
SIG_EFF = "15"
ORIG = {1: "top decay", 3: "ISR"}  # top radiation (2 J/psi of 3464) goes with ISR: radiated in the ttbar production
PTN = np.array([5, 8, 10, 13, 17, 22, 30, 45, 70, 400.])  # normalisation bins (b-decay J/psi start at 5 GeV)


def zany(q):
    """z with respect to the selected jet within 0.4, else the nearest jet above 15 GeV within 0.4; -1 in no jet"""
    z = q.z.to_numpy(float)
    sj = (q.sj_dr.to_numpy(float) >= 0) & (q.sj_dr.to_numpy(float) < 0.4)
    return np.where(z > 0, z, np.where(sj, q.pt.to_numpy(float) / np.maximum(q.sj_pt.to_numpy(float), 1e-6), -1.0))


# ---------------- normalisation
nd = np.histogram(S.pt, PTN, weights=S.w_nonprompt)[0]
nd2 = np.histogram(S.pt, PTN, weights=S.w_nonprompt ** 2)[0]
gnp = G[~G.prompt]
ng = np.histogram(gnp.pt, PTN)[0].astype(float)
K = nd / np.maximum(ng, 1)
K_err = K * np.sqrt(nd2 / np.maximum(nd, 1e-9) ** 2 + 1 / np.maximum(ng, 1))
K_all = nd.sum() / ng.sum()
print(f"b-decay J/psi: data {nd.sum():.0f}, Pythia {ng.sum():.0f}; k = {K_all:.3f}; per pT bin {np.round(K, 2).tolist()}")

P = G[G.prompt].reset_index(drop=True)
P["origin2"] = np.where(P.origin == 1, 1, 3)
ib = np.clip(np.digitize(P.pt, PTN) - 1, 0, len(K) - 1)
w_nom = K[ib] / LDME
ZE, ZW = np.asarray(LZ["z_edges"]), np.asarray(LZ["weights"])
zp = zany(P)
w_z = w_nom * np.where(zp > 0, ZW[np.clip(np.digitize(np.clip(zp, 0, 1.0), ZE) - 1, 0, len(ZW) - 1)], 1.0)
WV = {"nominal": w_nom, "z": w_z}

# ---------------- ttbar observables (the J/psi jet left out), data and Pythia
def ttobs(d, ch, nojpsi=False):
    R = reconstruct(d, ch, True, nojpsi=nojpsi)
    ob = observables(R, d.pt.to_numpy(float), d.phi.to_numpy(float), rapidity(R["jp"]))
    return {**ob, "ok": R["ok"]}


def add_obs(d):
    out = []
    for ch in LJ:
        q = d[d.ch == ch].copy()
        for nj, sfx in ((False, ""), (True, "_noj")):
            for k, a in ttobs(q, ch, nj).items():
                q[f"{k}{sfx}"] = a
        out.append(q)
    return pd.concat(out).sort_index()


S = add_obs(S)
P = add_obs(P)
for v in WV:
    WV[v] = np.asarray(WV[v])
P["zany"], S["zany"] = zany(P), zany(S)

# ---------------- samples: (frame, weight); data sWeights, Pythia by origin and variant, DPS = prompt pileup J/psi (J/psi
# out of the MET for the ttbar observables) scaled to the predicted DPS yield of l + jets
DATA = {"signal": S[S["sample"] == "notb"], "ttj": S[S["sample"] == "ttj"]}
PU = S[S["sample"] == "pileup"]
dps_all = SP["dps"]["pred_prompt_notb_by_sigma_eff_mb"]
pu_all = SP["dps"]["pileup_prompt_notb"]
f_lj = PU.w_prompt.sum() / pu_all  # the l + jets share of the prompt pileup J/psi (the DPS prediction is for all channels)
N_DPS = {k: float(v) * f_lj for k, v in dps_all.items()}
w_dps = PU.w_prompt.to_numpy(float) / PU.w_prompt.sum() * N_DPS[SIG_EFF]
CAT = {"signal": (P.jcat >= 2).to_numpy(), "ttj": (P.jcat == 1).to_numpy()}

VARS = {
    "pt": (np.array([3, 6, 8, 10, 13, 17, 22, 30, 45, 100.]), "pt"),
    "zany": (np.array([-0.2, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]), "zany"),
    "drtt": (np.array([0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]), "tt_drmin"),
    "dphi": (np.linspace(0, np.pi, 7), "dphi"),
    "ppar": (np.array([-150, -70, -30, -10, 10, 30, 70, 150.]), "ppar"),
    "dy": (np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]), "dy"),
}
TTOBS = ("dphi", "ppar", "dy")


def col(q, v, noj=False):
    c = VARS[v][1]
    return q[f"{c}_noj" if (noj and v in TTOBS) else c].to_numpy(float)


def okm(q, v, noj=False):
    return q[f"ok{'_noj' if noj else ''}"].to_numpy(bool) if v in TTOBS else np.ones(len(q), bool)


def hist(x, w, e):
    x = np.clip(x, e[0] + 1e-9, e[-1] - 1e-9)
    return np.histogram(x, e, weights=w)[0], np.sqrt(np.histogram(x, e, weights=w ** 2)[0])


def wmean(x, w):
    m = (w * x).sum() / w.sum()
    return float(m), float(np.sqrt((w ** 2 * (x - m) ** 2).sum()) / abs(w.sum()))


OUT = {"ldme_fac": LDME, "norm": {"pt_edges": PTN.tolist(), "k": K.tolist(), "k_err": K_err.tolist(), "k_all": float(K_all), "data_bdecay": nd.tolist(), "pythia_bdecay": ng.tolist(),
                "mean_pt_data": float((S.w_nonprompt * S.pt).sum() / S.w_nonprompt.sum()), "mean_pt_pythia": float(gnp.pt.mean())},
       "vars": {v: e.tolist() for v, (e, _) in VARS.items()}, "dps": {"sigma_eff_mb": SIG_EFF, "n_lj": N_DPS, "f_lj": float(f_lj)}, "hists": {}, "yields": {}, "means": {}}
for cat, d in DATA.items():
    wd = d.w_prompt.to_numpy(float)
    OUT["yields"][f"data_{cat}"] = [float(wd.sum()), float(np.sqrt((wd ** 2).sum()))]
    if cat == "signal":
        sp = SP["fits"]["notb"]
        OUT["yields"]["data_signal_fit_all_channels"] = [sp["N"][0], sp["N_err"][0]]
    m = CAT[cat]
    for vn, w in WV.items():
        for o, on in ORIG.items():
            s_ = m & (P.origin2 == o).to_numpy()
            OUT["yields"][f"pythia_{vn}_{cat}_{on}"] = [float(w[s_].sum()), float(np.sqrt((w[s_] ** 2).sum()))]
    for v, (e, _) in VARS.items():
        ok = okm(d, v)
        h, he = hist(col(d, v)[ok], wd[ok], e)
        OUT["hists"][f"data_{cat}_{v}"] = [h.tolist(), he.tolist()]
        OUT["means"][f"data_{cat}_{v}"] = wmean(col(d, v)[ok], wd[ok])
        for vn, w in WV.items():
            for o, on in ORIG.items():
                s_ = m & (P.origin2 == o).to_numpy() & okm(P, v)
                h, he = hist(col(P, v)[s_], w[s_], e)
                OUT["hists"][f"pythia_{vn}_{cat}_{on}_{v}"] = [h.tolist(), he.tolist()]
            s_ = m & okm(P, v)
            OUT["means"][f"pythia_{vn}_{cat}_{v}"] = wmean(col(P, v)[s_], w[s_])
            for o, on in ORIG.items():
                s_ = m & (P.origin2 == o).to_numpy() & okm(P, v)
                OUT["means"][f"pythia_{vn}_{cat}_{on}_{v}"] = wmean(col(P, v)[s_], w[s_])
        if cat == "signal":
            ok = okm(PU, v, True)
            h, he = hist(col(PU, v, True)[ok], w_dps[ok], e)
            OUT["hists"][f"dps_{v}"] = [h.tolist(), he.tolist()]
            OUT["means"][f"dps_{v}"] = wmean(col(PU, v, True)[ok], w_dps[ok])


# ---------------- the share of DPS from the shapes: data = N [f x DPS + (1 - f) x Pythia (ttbar collision, both origins)],
# each template at unit area; chi2 with the data sWeight errors and the template statistics (as ttbar_balance.frac_fit)
def frac_fit(v, vn):
    hd, ed = (np.asarray(a, float) for a in OUT["hists"][f"data_signal_{v}"])
    hp = sum(np.asarray(OUT["hists"][f"pythia_{vn}_signal_{on}_{v}"][0], float) for on in ORIG.values())
    ep = np.sqrt(sum(np.asarray(OUT["hists"][f"pythia_{vn}_signal_{on}_{v}"][1], float) ** 2 for on in ORIG.values()))
    hq, eq = (np.asarray(a, float) for a in OUT["hists"][f"dps_{v}"])
    N, tp, tq, etp, etq = hd.sum(), hp / hp.sum(), hq / hq.sum(), ep / hp.sum(), eq / hq.sum()
    fs = np.linspace(-1.0, 2.0, 3001)
    pred = N * (fs[:, None] * tq + (1 - fs[:, None]) * tp)
    var = np.maximum(ed ** 2 + N ** 2 * (fs[:, None] ** 2 * etq ** 2 + (1 - fs[:, None]) ** 2 * etp ** 2), 1e-9)
    c2 = ((hd - pred) ** 2 / var).sum(1)
    k = c2.argmin()
    inside = fs[c2 <= c2[k] + 1]
    at = lambda f: float(np.interp(f, fs, c2))  # noqa: E731
    return {"f_dps": float(fs[k]), "err": [float(inside.min() - fs[k]), float(inside.max() - fs[k])], "chi2_min": float(c2[k]), "ndf": int(len(hd) - 1),
            "chi2_pythia_only": at(0.0), "chi2_dps_only": at(1.0), "z_vs_pythia_only": float(np.sqrt(max(at(0.0) - c2[k], 0))),
            "z_vs_dps_only": float(np.sqrt(max(at(1.0) - c2[k], 0)))}


# (not z: the muons of a pileup J/psi are taken out of the jets by PUPPI, so the pileup J/psi are never inside a jet)
OUT["fit"] = {f"{v}_{vn}": frac_fit(v, vn) for v in ("pt", "drtt", "dphi", "ppar", "dy") for vn in WV}

# ---------------- Pythia scaled to the vetoed J/psi of the data (the overall LDME scale), and what that leaves for DPS
# "_bcorr": the vetoed prompt J/psi refitted with the b-decay l_xy shape of the simulation in that category (mc_template_fit.py,
# b-decay yield constrained to the simulation): the J/psi of b decays in untagged b jets are short and the b-jet template
# of the unbinned fit counts most of them as prompt (closure on the CMS ttbar simulation)
Y = OUT["yields"]
OUT["scaled"] = {}
MT = os.path.join(os.path.dirname(os.path.abspath(OUTJ)), "mc_template.json")
TTJ_REFIT = json.load(open(MT))["fits"]["ttj_ljets_nbcon"] if os.path.exists(MT) else None


def scaled(vn, dttj):
    pt_ = {cat: sum(Y[f"pythia_{vn}_{cat}_{on}"][0] for on in ORIG.values()) for cat in DATA}
    dat = {"ttj": dttj, "signal": Y["data_signal"]}
    sc = dttj[0] / pt_["ttj"]
    sc_e = sc * dttj[1] / dttj[0]
    sps = sc * pt_["signal"]
    dps = Y["data_signal"][0] - sps
    dps_e = np.hypot(Y["data_signal"][1], sc_e * pt_["signal"])
    return {"data_ttj": list(dttj), "scale": sc, "scale_err": sc_e, "sps_signal": sps, "sps_signal_err": sc_e * pt_["signal"], "top_decay_signal": sc * Y[f"pythia_{vn}_signal_top decay"][0],
            "top_decay_signal_err": sc_e * Y[f"pythia_{vn}_signal_top decay"][0], "dps_left": float(dps), "dps_left_err": float(dps_e),
            "sigma_eff_mb": float(N_DPS[SIG_EFF] * float(SIG_EFF) / dps) if dps > 0 else None,
            "sigma_eff_mb_95_lower": float(N_DPS[SIG_EFF] * float(SIG_EFF) / (dps + 1.645 * dps_e)),
            "ratio_data_to_pythia": {cat: [dat[cat][0] / pt_[cat], dat[cat][1] / pt_[cat]] for cat in DATA}}


for vn in WV:
    OUT["scaled"][vn] = scaled(vn, Y["data_ttj"])
    if TTJ_REFIT:
        OUT["scaled"][f"{vn}_bcorr"] = scaled(vn, (TTJ_REFIT["N_p"], TTJ_REFIT["N_p_err"]))

# ---------------- printout
for cat in DATA:
    d_ = Y[f"data_{cat}"]
    line = f"{cat:7s} data {d_[0]:6.1f} +- {d_[1]:4.1f}"
    for vn in WV:
        ps = [Y[f"pythia_{vn}_{cat}_{on}"][0] for on in ORIG.values()]
        line += f" | Pythia {vn}: " + " + ".join(f"{x:.1f} ({on})" for x, on in zip(ps, ORIG.values())) + f" = {sum(ps):.1f}"
    if cat == "signal":
        line += f" | DPS ({SIG_EFF} mb) {N_DPS[SIG_EFF]:.1f}"
    print(line)
for v in VARS:
    M_ = OUT["means"]
    print(f"<{v}>: data {M_[f'data_signal_{v}'][0]:.2f} +- {M_[f'data_signal_{v}'][1]:.2f}; Pythia nominal {M_[f'pythia_nominal_signal_{v}'][0]:.2f},"
          f" z {M_[f'pythia_z_signal_{v}'][0]:.2f} (ISR {M_[f'pythia_z_signal_ISR_{v}'][0]:.2f}, top decay {M_[f'pythia_z_signal_top decay_{v}'][0]:.2f}); DPS {M_[f'dps_{v}'][0]:.2f}"
          f" | vetoed: data {M_[f'data_ttj_{v}'][0]:.2f}, Pythia {M_[f'pythia_nominal_ttj_{v}'][0]:.2f} / z {M_[f'pythia_z_ttj_{v}'][0]:.2f}")
for k, F in OUT["fit"].items():
    print(f"f(DPS) from {k:14s} {F['f_dps']:+.2f} {F['err'][0]:+.2f} {F['err'][1]:+.2f}; chi2 {F['chi2_min']:.1f} / {F['ndf']}; Pythia only {F['z_vs_pythia_only']:.1f} sigma,"
          f" DPS only {F['z_vs_dps_only']:.1f} sigma")
for vn, C in OUT["scaled"].items():
    print(f"Pythia {vn} x {C['scale']:.2f} +- {C['scale_err']:.2f} (to the vetoed J/psi): outside the ttbar jets {C['sps_signal']:.1f} +- {C['sps_signal_err']:.1f}"
          f" (top decay {C['top_decay_signal']:.1f}); left for DPS {C['dps_left']:.1f} +- {C['dps_left_err']:.1f} -> sigma_eff {C['sigma_eff_mb'] or float('inf'):.0f} mb"
          f" (> {C['sigma_eff_mb_95_lower']:.1f} mb at 95% CL, one-sided)")
json.dump(OUT, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
