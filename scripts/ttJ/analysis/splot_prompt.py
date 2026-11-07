"""sPlot of the prompt J/psi in ttbar events. An unbinned extended fit of m(mu mu) x l_xy to the J/psi candidates from the
primary vertex (opposite sign, N_b >= 1, |z0| < 0.1 cm, 2.6 < m < 3.6 GeV, -0.5 < l_xy < 1.5 mm; mu + jets, e + jets and
e mu), per category (not b-matched: the signal; b-matched: the J/psi of the ttbar b jets), with three species:
    prompt J/psi      S(m) R(l)
    non-prompt J/psi  S(m) D(l)     not b-matched: D = the b-matched J/psi l_xy shape (both b quarks are from the ttbar
                                    decays, as in fit_jpsi.py); b-matched: D = E_lambda * R
    continuum         B(m) C(l)     C: narrow part with the resolution R, wide Gaussian, decay part (fit_jpsi.py)
S, R, D and the start values from the binned mass x l_xy fit (fit_jpsi.json); the continuum shape (mass slope and C) is
refitted unbinned, then fixed, and the three yields refitted for the sWeights (Pivk and Le Diberder, NIM A 555 (2005) 356):
    w_n(e) = sum_j V_nj f_j(e) / sum_k N_k f_k(e),   V^-1_ij = sum_e f_i(e) f_j(e) / (sum_k N_k f_k(e))^2
The sWeights give the distribution of a variable that does not depend on m and l_xy within each species: the J/psi pT and
rapidity, and the ttbar observables of the event (m(jj), m(jjb), m(l, b), MET, ...). R is a Gaussian of width S times the
l_xy error of the candidate (S fitted), since the resolution is worse at low pT; one Gaussian for all candidates is the
cross-check ("globalres"), and so is the sideband-subtracted prompt-region count (|l_xy| < 0.06 mm) per pT bin.
The pileup J/psi (|z0| >= 0.1 cm, window - sidebands) are the reference for a J/psi not correlated with the ttbar.
-> splot.json: fit results, sWeighted histograms per species and channel, the cross-check, the all-ttbar reference shapes.
With a fifth argument, the candidates of the fits with their sWeights go to that parquet file (for ttbar_balance.py).
SIGNAL=assoc (on the tables of assoc_veto.py): the signal is the J/psi outside every ttbar decay jet (jcat >= 2: not in a
b jet, not in the four selected jets of l + jets, not in a jet of the chi2 ttbar fit), the search for associated
production; the J/psi in the other ttbar jets (jcat 1, mostly the W pair) get their own fit ("ttj"). Default: not b-matched.
usage: python3 splot_prompt.py <cands dir> <fit_jpsi.json> <results.json> <out.json> [sweights.parquet]"""
import json
import os
import sys

import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy.special import log_ndtr, ndtr

CD, FJ, RJ, OUTJ = sys.argv[1:5]
SWQ = sys.argv[5] if len(sys.argv) > 5 else None
FJ, RJ = json.load(open(FJ)), json.load(open(RJ))
CH = ("mujets", "ejets", "emu")
M_LO, M_HI, L_LO, L_HI = 2.6, 3.6, -0.5, 1.5
WIN, SBS = (2.95, 3.25), ((2.60, 2.90), (3.30, 3.60))
SH = FJ["shape"]
SIGNAL = os.environ.get("SIGNAL", "notb")


def signal(d):
    return (d.cat != 0) if SIGNAL == "notb" else (d.jcat >= 2)

P2 = FJ["lt2d"]["params"]
S1 = P2["s1"]

# ---------------- candidates
cols = {}
for ch in CH:
    d = pd.read_parquet(os.path.join(CD, f"cands_{ch}.parquet"))
    d = d[(d.os == 1) & (d.nb >= 1) & (((d.life <= 2) & (d.lxy > L_LO) & (d.lxy < L_HI)) | (d.life == 3)) & (d.mass > M_LO) & (d.mass < M_HI)].copy()
    d["ch"] = ch
    # the ttbar lepton of the channel (e mu: the muon, which fired the trigger)
    lep = "e" if ch == "ejets" else "mu"
    for k in ("mlj", "tt_mlb", "rside"):
        d[k] = d[f"{k}_{lep}"]
    d["lep_pt"], d["lep_mT"] = d[f"{lep}_pt"], d[f"{lep}_mT"]
    cols[ch] = d
D = pd.concat(cols.values(), ignore_index=True)
# pileup J/psi (|z0| >= 0.1 cm): inclusive J/psi from another collision, the reference for a J/psi not correlated with the
# ttbar (as from a second parton scattering); window - sidebands
PU = D[D.life == 3].copy()
D = D[D.life <= 2].copy()
D["notb"] = signal(D)
print({ch: int(len(d)) for ch, d in cols.items()}, "candidates;", int(D.notb.sum()), "not b-matched,", len(PU), "pileup")


# ---------------- pdfs, normalised on the fit ranges
def cb_pdf(m, mu, s, a, n):
    t = (m - mu) / s
    A, B = (n / a) ** n * np.exp(-0.5 * a * a), n / a - a
    tot = A * (B + a) ** (1 - n) / (n - 1) + np.sqrt(2 * np.pi) * (1 - ndtr(-a))
    core = np.exp(-0.5 * t * t)
    tail = A * np.power(np.maximum(B - t, 1e-12), -n)
    return np.where(t > -a, core, tail) / (tot * s)


def cb_cdf(m, mu, s, a, n):
    t = (np.asarray(m, float) - mu) / s
    A, B = (n / a) ** n * np.exp(-0.5 * a * a), n / a - a
    tail = lambda u: A * (B - u) ** (1 - n) / (n - 1)  # noqa: E731
    core = lambda u: np.sqrt(2 * np.pi) * (ndtr(u) - ndtr(-a))  # noqa: E731
    tot = tail(-a) + np.sqrt(2 * np.pi) * (1 - ndtr(-a))
    return np.where(t <= -a, tail(np.minimum(t, -a)), tail(-a) + core(np.maximum(t, -a))) / tot


def sig_m(m):
    p = SH
    f = p["f1"] * cb_pdf(m, p["mu"], p["s1"], p["a"], p["n"]) + (1 - p["f1"]) * np.exp(-0.5 * ((m - p["mu"]) / p["s2"]) ** 2) / (np.sqrt(2 * np.pi) * p["s2"])
    cdf = lambda x: p["f1"] * cb_cdf(x, p["mu"], p["s1"], p["a"], p["n"]) + (1 - p["f1"]) * ndtr((x - p["mu"]) / p["s2"])  # noqa: E731
    return f / (cdf(M_HI) - cdf(M_LO))


def bkg_m(m, k):
    if abs(k) < 1e-6:
        return np.full_like(m, 1.0 / (M_HI - M_LO))
    return k * np.exp(-k * (m - M_LO)) / (1 - np.exp(-k * (M_HI - M_LO)))


def gaus_l(x, s):
    return np.exp(-0.5 * (x / s) ** 2) / (np.sqrt(2 * np.pi) * s) / (ndtr(L_HI / s) - ndtr(L_LO / s))


def gaus_c(e, s):
    """CDF of gaus_l at e (e: edges, broadcast against s)"""
    return (ndtr(e / s) - ndtr(L_LO / s)) / (ndtr(L_HI / s) - ndtr(L_LO / s))


def cdf_decay(x, lam, s):
    return ndtr(x / s) - np.exp(s * s / (2 * lam * lam) - x / lam + log_ndtr(x / s - s / lam))


def decay_l(x, lam, s):
    """exp(-l / lam) / lam (l > 0) convolved with G(0, s), normalised on [L_LO, L_HI]"""
    return np.exp(s * s / (2 * lam * lam) - x / lam + log_ndtr(x / s - s / lam)) / lam / (cdf_decay(L_HI, lam, s) - cdf_decay(L_LO, lam, s))


def decay_c(e, lam, s):
    return (cdf_decay(e, lam, s) - cdf_decay(L_LO, lam, s)) / (cdf_decay(L_HI, lam, s) - cdf_decay(L_LO, lam, s))


NPI, NNPI = P2["NP_inb"], P2["NNP_inb"]
FPI = NPI / (NPI + NNPI)


# l_xy pdfs given the resolution s of each candidate (an array): prompt, non-prompt (cat), continuum; and their CDFs
def lpdfs(x, s, cat, fn, fw, sw, lb):
    npr = decay_l(x, P2["lam_inb"], s) if cat in ("inb", "pileup") else FPI * gaus_l(x, s) + (1 - FPI) * decay_l(x, P2["lam_inb"], s)
    return gaus_l(x, s), npr, fn * gaus_l(x, s) + (1 - fn) * (fw * gaus_l(x, sw) + (1 - fw) * decay_l(x, lb, s))


def lcdfs(e, s, cat, fn, fw, sw, lb):
    npr = decay_c(e, P2["lam_inb"], s) if cat in ("inb", "pileup") else FPI * gaus_c(e, s) + (1 - FPI) * decay_c(e, P2["lam_inb"], s)
    return gaus_c(e, s), npr, fn * gaus_c(e, s) + (1 - fn) * (fw * gaus_c(e, sw) + (1 - fw) * decay_c(e, lb, s))


# ---------------- unbinned fit and sWeights per category
# resolution: "percand" (nominal), S times the l_xy error of the candidate, S fitted: the resolution follows the J/psi pT
# (the error is 32 um at 5-8 GeV, 13 um above 50 GeV), so the prompt sWeights do not depend on pT through it;
# "global": one Gaussian of the binned fit (sigma = s1 of fit_jpsi.json) for all candidates, as a cross-check
SP = ("prompt", "nonprompt", "cont")
OUT = {"selection": "opposite sign, N_b >= 1, |z0| < 0.1 cm, 2.6 < m < 3.6 GeV, -0.5 < l_xy < 1.5 mm", "signal": SIGNAL, "fits": {}, "hists": {}, "check": {},
       "resolution": "nominal: S x the l_xy error per candidate (S fitted); variation 'globalres': one Gaussian of the binned fit"}
SHAPE_PARS = ("k", "fn", "fw", "sw", "lb", "S")
LIMS = dict(NP=(0, None), NNP=(0, None), NB=(0, None), k=(-10, 10), fn=(0, 1), fw=(0, 1), sw=(0.03, 1.0), lb=(0.01, 5), S=(0.5, 3.0))


def fit(cat, mode="percand", d=None, key=None):
    if d is None:
        d = D[D.notb] if cat == "notb" else D[D.cat == 0]
    m, x, er = d.mass.to_numpy(float), d.lxy.to_numpy(float), np.maximum(d.lxyerr.to_numpy(float), 0.003)
    sm = sig_m(m)
    res = (lambda S: S * er) if mode == "percand" else (lambda S: np.full(len(x), S1))  # noqa: E731

    def comps(k, fn, fw, sw, lb, S):
        a, b, c = lpdfs(x, res(S), cat, fn, fw, sw, lb)
        return sm * a, sm * b, bkg_m(m, k) * c

    def nll(NP, NNP, NB, k, fn, fw, sw, lb, S):
        a, b, c = comps(k, fn, fw, sw, lb, S)
        tot = NP * a + NNP * b + NB * c
        return (NP + NNP + NB) - np.log(np.maximum(tot, 1e-300)).sum()

    pc = "notb" if cat in ("pileup", "ttj") else cat  # start values
    sc = 0.5 if cat == "pileup" else (len(d) / max(int((D.cat != 0).sum()), 1) if pc == "notb" else 1.0)  # the share of the binned-fit sample
    st = dict(NP=sc * P2[f"NP_{pc}"], NNP=sc * P2[f"NNP_{pc}"],
              NB=max(len(d) - sc * (P2[f"NP_{pc}"] + P2[f"NNP_{pc}"]), 10.0),
              k=FJ["mass"][f"{pc}_pv"]["k"], fn=P2[f"fn_{pc}"], fw=P2[f"fw_{pc}"], sw=P2[f"sw_{pc}"], lb=P2[f"lb_{pc}"],
              S=S1 / np.median(er) if mode == "percand" else 1.0)

    def minuit(start, fix=()):
        mi = Minuit(nll, **start)
        mi.errordef = Minuit.LIKELIHOOD
        for k_, v in LIMS.items():
            mi.limits[k_] = v
        if mode != "percand":
            mi.fixed["S"] = True
        for k_ in fix:
            mi.fixed[k_] = True
        mi.migrad(); mi.migrad()
        return mi

    mf = minuit(st)
    full = {k_: float(mf.values[k_]) for k_ in mf.parameters}
    # yields only, the shapes fixed: the sWeights
    mi = minuit(full, fix=SHAPE_PARS)
    mi.hesse(); mi.minos("NP")
    F = np.vstack(comps(*[full[k_] for k_ in SHAPE_PARS]))  # (3, n)
    N = np.array([mi.values[k_] for k_ in ("NP", "NNP", "NB")])
    tot = N @ F
    V = np.linalg.inv((F[:, None, :] * F[None, :, :] / tot ** 2).sum(axis=2))
    W = (V @ F) / tot  # (3, n)
    # likelihood ratio against N_P = 0, the continuum shape (and S) free in both fits
    m0 = minuit({**full, "NP": 0.0}, fix=("NP",))
    z = float(np.sqrt(max(2 * (m0.fval - mf.fval), 0)))
    key = key or (cat if mode == "percand" else f"{cat}_globalres")
    OUT["fits"][key] = {"n": int(len(d)), "params": full, "N": N.tolist(), "N_err": [float(mi.errors[k_]) for k_ in ("NP", "NNP", "NB")],
                        "NP_minos": [float(mi.merrors["NP"].lower), float(mi.merrors["NP"].upper)], "V": V.tolist(),
                        "sum_w": W.sum(axis=1).tolist(), "signif_prompt": z, "valid": bool(mf.valid),
                        "binned_2d": {"NP": P2[f"NP_{pc}"], "NNP": P2[f"NNP_{pc}"]} if cat != "pileup" else {}}
    print(f"{key}: {len(d)} candidates; N_P {N[0]:.1f} {mi.merrors['NP'].lower:+.1f} {mi.merrors['NP'].upper:+.1f}, N_NP {N[1]:.1f} +- {mi.errors['NNP']:.1f},"
          f" N_B {N[2]:.1f}; S {full['S']:.2f}; prompt {z:.1f} sigma (binned 2D: N_P {P2[f'NP_{pc}']:.1f}, N_NP {P2[f'NNP_{pc}']:.1f});"
          f" sum of sWeights {W.sum(axis=1).round(1)}; valid {mf.valid}")
    return d.assign(w_prompt=W[0], w_nonprompt=W[1], w_cont=W[2]), (m, x, res(full["S"]), full, N)


dn, PN = fit("notb")
di, PI = fit("inb")
dng, _ = fit("notb", "global")
if SIGNAL != "notb":
    # the J/psi in the other ttbar jets (vetoed): mostly the W pair of l + jets
    dt, PT = fit("ttj", d=D[D.jcat == 1])

# fit projections (mass and l_xy) of the unbinned fits, for the backup: N_k times the pdf integrated over each bin,
# the l_xy pdf averaged over the resolutions of the candidates
ME, LEP = np.linspace(M_LO, M_HI, 41), np.linspace(L_LO, L_HI, 101)
OUT["proj"] = {"mass_edges": ME.tolist(), "lxy_edges": LEP.tolist()}
mm = np.linspace(M_LO, M_HI, 4001)
for cat, (m, x, sres, full, N) in (("notb", PN), ("inb", PI)) + ((("ttj", PT),) if SIGNAL != "notb" else ()):
    pr = {"data_m": np.histogram(m, ME)[0].tolist(), "data_l": np.histogram(x, LEP)[0].tolist()}
    pm = {"prompt": sig_m(mm), "nonprompt": sig_m(mm), "cont": bkg_m(mm, full["k"])}
    cl = lcdfs(LEP[None, :], sres[:, None], cat, full["fn"], full["fw"], full["sw"], full["lb"])
    for j, s_ in enumerate(SP):
        cm = np.concatenate([[0], np.cumsum(0.5 * (pm[s_][1:] + pm[s_][:-1]) * np.diff(mm))])
        pr[f"{s_}_m"] = (N[j] * np.diff(np.interp(ME, mm, cm))).tolist()
        pr[f"{s_}_l"] = (N[j] * np.diff(cl[j], axis=1).mean(axis=0)).tolist()
    OUT["proj"][cat] = pr

# ---------------- sWeighted histograms
VARS = {
    "pt": ([3, 5, 8, 12, 16, 20, 25, 30, 40, 60, 100], "p<sub>T</sub>(J/ψ) [GeV]"),
    "absy": ([0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4], "|y(J/ψ)|"),
    "tt_mjj": (None, "m(jj) [GeV]"),
    "tt_mjjb": (None, "m(jjb) [GeV]"),
    "tt_mlb": (None, "m(ℓ, b) [GeV]"),
    "mlj": ([0, 25, 50, 75, 100, 125, 150, 200, 250, 350], "m(ℓ, J/ψ) [GeV]"),
    "met": ([0, 25, 50, 75, 100, 150, 200, 300], "p<sub>T</sub><sup>miss</sup> [GeV]"),
    "lep_pt": ([30, 40, 50, 60, 80, 100, 150, 250], "p<sub>T</sub>(ℓ) [GeV]"),
    "lep_mT": ([0, 20, 40, 60, 80, 100, 120, 160, 250], "m<sub>T</sub>(ℓ, p<sub>T</sub><sup>miss</sup>) [GeV]"),
    "nb": ([0.5, 1.5, 2.5, 3.5, 4.5], "N<sub>b</sub>"),
    "ntrack": ([0, 20, 30, 40, 50, 60, 80, 120], "n<sub>track</sub>"),
    "sj_dr": ([0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 2.0, 4.0], "ΔR(J/ψ, nearest jet)"),
    # in a selected jet (Delta R < 0.4): the J/psi share of the jet pT (fragmentation of a gluon or charm jet: z well below 1;
    # a jet made of the two muons alone: z near 1); -1 when in no selected jet
    "z": ([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2], "z = p<sub>T</sub>(J/ψ) / p<sub>T</sub>(jet)"),
    "cat": ([0.5, 1.5, 2.5], "in a selected jet (1) or in none (2)"),
    "sj_muef": ([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001], "muon energy fraction of the nearest jet"),
}
TK = RJ["channels"]["mujets"]["ttkin"]
# the binning of the all-ttbar summaries, merged by REB (20 GeV for m(jj) and m(jjb), 30 GeV for m(l, b)): ~170 weighted J/psi
REB = {"tt_mjj": 5, "tt_mjjb": 4, "tt_mlb": 3}
for v, key in (("tt_mjj", "mjj"), ("tt_mjjb", "mjjb"), ("tt_mlb", "mlb")):
    VARS[v] = (TK[f"{key}_edges"][::REB[v]], VARS[v][1])
OUT["vars"] = {v: {"edges": [float(e) for e in E], "label": lab} for v, (E, lab) in VARS.items()}


def hist(d, v, wcol):
    E = np.asarray(VARS[v][0], float)
    x, w = d[v].to_numpy(float), d[wcol].to_numpy(float)
    ok = x > E[0] - 1e-9 if v not in ("tt_mjj", "tt_mjjb", "tt_mlb", "mlj", "z", "sj_muef") else x > 0  # -1: not defined (e mu: no hadronic W)
    h = np.histogram(x[ok], E, weights=w[ok])[0]
    h2 = np.histogram(x[ok], E, weights=w[ok] ** 2)[0]
    return h.tolist(), np.sqrt(h2).tolist()


for cat, d in (("notb", dn), ("inb", di)) + ((("ttj", dt),) if SIGNAL != "notb" else ()):
    for s in SP:
        for v in VARS:
            for ch in ("all",) + CH:
                dd = d if ch == "all" else d[d.ch == ch]
                OUT["hists"][f"{cat}_{s}_{v}_{ch}"] = hist(dd, v, f"w_{s}")
# the resolution cross-check: the prompt sWeights with one Gaussian for all candidates
for v in VARS:
    OUT["hists"][f"notb_prompt_globalres_{v}_all"] = hist(dng, v, "w_prompt")

PU["w_pileup"] = np.where((PU.mass > WIN[0]) & (PU.mass < WIN[1]), 1.0, 0.0) - 0.5 * np.where(
    ((PU.mass > SBS[0][0]) & (PU.mass < SBS[0][1])) | ((PU.mass > SBS[1][0]) & (PU.mass < SBS[1][1])), 1.0, 0.0)
OUT["fits"]["pileup"] = {"n": int(len(PU)), "sub": float(PU.w_pileup.sum())}
for v in VARS:
    for ch in ("all",) + CH:
        dd = PU if ch == "all" else PU[PU.ch == ch]
        OUT["hists"][f"pileup_{v}_{ch}"] = hist(dd, v, "w_pileup")

# all ttbar events (N_b >= 1) for the ttbar observables, from the binned summaries
OUT["ttbar"] = {}
for v, key in (("tt_mjj", "mjj"), ("tt_mjjb", "mjjb"), ("tt_mlb", "mlb")):
    for ch in ("all",) + CH:
        chs = CH if ch == "all" else (ch,)
        tot = 0
        for c in chs:
            T = RJ["channels"][c]["ttkin"]
            k = f"{key}_ttbar" if f"{key}_ttbar" in T else ("mlb_mu_ttbar" if key == "mlb" else None)
            if k:
                tot = tot + np.asarray(T[k], float)
        tot = np.asarray(tot, float)
        OUT["ttbar"][f"{v}_{ch}"] = tot.reshape(-1, REB[v]).sum(axis=1).tolist() if tot.ndim else []

# cross-check of the pT shape: window - sidebands in the prompt region (|l_xy| < 0.06 mm) and the non-prompt region (> 0.1 mm)
E = np.asarray(VARS["pt"][0], float)
for reg, sel in (("prompt_region", dn.lxy.abs() < 0.06), ("nonprompt_region", dn.lxy > 0.1)):
    d = dn[sel]
    win = (d.mass > WIN[0]) & (d.mass < WIN[1])
    sb = np.zeros(len(d), bool)
    for lo, hi in SBS:
        sb |= (d.mass > lo) & (d.mass < hi)
    hw, hs = np.histogram(d.pt[win], E)[0], np.histogram(d.pt[sb], E)[0]
    OUT["check"][reg] = {"sub": (hw - 0.5 * hs).tolist(), "err": np.sqrt(hw + 0.25 * hs).tolist()}

# the prompt-J/psi candidates with the largest sWeights would be the events to look at; keep run / lumi / event of the top 30
top = dn.sort_values("w_prompt", ascending=False).head(30)
OUT["top_prompt"] = top[["ch", "year", "run", "lumi", "event", "mass", "lxy", "pt", "w_prompt"]].to_dict(orient="records")
# ---------------- the J/psi rate of a second parton scattering, from the pileup J/psi
# P(J/psi | ttbar collision) = sigma(J/psi) / sigma_eff = r x sigma_vis / sigma_eff, with r the J/psi per pileup collision
# that has a good vertex (npvsGood - 1 per event) and sigma_vis = eps x sigma_inel the cross section of such collisions.
# The pileup J/psi take the selection of the signal: not b-matched (cat != 0), prompt from their own m x l_xy fit (their
# l_xy is measured from the primary vertex, which is within ~10 um of theirs in x-y). Pileup collisions within 1 mm of the
# primary vertex (|z0| < 0.1 cm) are counted as the primary vertex: about 1.5% of them for a 3.7 cm luminous region.
PUF = PU[(PU.lxy > L_LO) & (PU.lxy < L_HI)]
pu_all, _ = fit("pileup", d=PUF, key="pileup_all")
pu_notb, _ = fit("pileup", d=PUF[signal(PUF)], key="pileup_notb")
# pileup J/psi under the primary vertex: the prompt pileup J/psi in |z0| >= 0.1 cm follow the distance between two collisions,
# a Gaussian in z0 of width sqrt(2) sigma_z; fitted to their sWeighted |z0| distribution, it gives how many fall in |z0| < 0.1 cm
ZE = np.array([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0])
az = pu_notb.z0.abs().to_numpy(float)
hz = np.histogram(az, ZE, weights=pu_notb.w_prompt)[0]
hz2 = np.histogram(az, ZE, weights=pu_notb.w_prompt ** 2)[0]


def zchi2(A, sz):
    pred = A * (ndtr(ZE[1:] / sz) - ndtr(ZE[:-1] / sz)) * 2
    return (((hz - pred) ** 2) / np.maximum(hz2, 1.0)).sum()


mz = Minuit(zchi2, A=hz.sum(), sz=5.0)
mz.limits["sz"] = (0.5, 20)
mz.migrad(); mz.hesse()
A, sz = mz.values["A"], mz.values["sz"]
leak = A * (2 * ndtr(0.1 / sz) - 1)
leak_err = leak * np.hypot(mz.errors["A"] / A, mz.errors["sz"] / sz)
# pileup J/psi closer than 0.3 cm: the first two bins against the Gaussian; if the missing ones moved into |z0| < 0.1 cm
# (a merged vertex), they add to the leak: the upper bound
exp13 = A * 2 * (ndtr(0.3 / sz) - ndtr(0.1 / sz))
obs13 = float(hz[:2].sum())
OUT["pu_leak_bound"] = {"expected_01_03": exp13, "observed_01_03": obs13, "observed_err": float(np.sqrt(hz2[:2].sum()))}
OUT["pu_leak"] = {"z0_edges": ZE.tolist(), "prompt_pileup_z0": hz.tolist(), "prompt_pileup_z0_err": np.sqrt(hz2).tolist(), "sigma_dz_cm": sz,
                  "sigma_dz_err": float(mz.errors["sz"]), "n_in_pv": leak, "n_in_pv_err": leak_err, "chi2": float(mz.fval), "nbins": len(hz)}
print(f"pileup under the primary vertex: prompt pileup J/psi |z0| fit sigma(dz) {sz:.2f} +- {mz.errors['sz']:.2f} cm, chi2 {mz.fval:.1f} / {len(hz) - 2};"
      f" in |z0| < 0.1 cm: {leak:.1f} +- {leak_err:.1f} prompt not-b-matched J/psi (of {OUT['fits']['notb']['N'][0]:.0f}); 0.1-0.3 cm: {obs13:.1f} +- {np.sqrt(hz2[:2].sum()):.1f} for {exp13:.1f} expected")

# the prompt pileup J/psi (sWeights), the template of a J/psi not correlated with the ttbar
for v in VARS:
    for ch in ("all",) + CH:
        dd = pu_notb if ch == "all" else pu_notb[pu_notb.ch == ch]
        OUT["hists"][f"pileupfit_prompt_{v}_{ch}"] = hist(dd, v, "w_prompt")
NPV = json.load(open(os.path.join(os.path.dirname(os.path.abspath(OUTJ)), "npv.json")))
nev = sum(v["n"] for ch in NPV.values() for v in ch.values())
npu = sum(v["n"] * (v["mean_npv"] - 1) for ch in NPV.values() for v in ch.values()) / nev
ntt = sum(C["yields"]["data_os_all_ge1b"]["n_tt"] for C in RJ["channels"].values())
SIG_INEL, EPS, SIG_EFF = 80.0, 0.7, (5.0, 15.0, 25.0)  # mb: inelastic pp at 13.6 TeV; good-vertex efficiency; sigma_eff range
fpu = OUT["fits"]["pileup_notb"]
r = fpu["N"][0] / (ntt * npu)
r_err = 0.5 * (fpu["NP_minos"][1] - fpu["NP_minos"][0]) / (ntt * npu)
pred = {f"{se:g}": r * EPS * SIG_INEL / se * ntt for se in SIG_EFF}
OUT["dps"] = {"n_tt": ntt, "mean_npu": npu, "pileup_prompt_notb": fpu["N"][0], "pileup_prompt_notb_err": 0.5 * (fpu["NP_minos"][1] - fpu["NP_minos"][0]),
              "r_per_pu_vertex": r, "r_err": r_err, "sigma_inel_mb": SIG_INEL, "eps_vtx": EPS, "pred_prompt_notb_by_sigma_eff_mb": pred,
              "measured_prompt_notb_splot": OUT["fits"]["notb"]["N"][0], "measured_prompt_notb_binned_lxy": None}
print(f"DPS: {fpu['N'][0]:.0f} prompt not-b-matched pileup J/psi in {ntt / 1e6:.2f}M ttbar events x {npu:.1f} pileup vertices: r = {r:.2e} per vertex;"
      f" predicted prompt not-b-matched J/psi from DPS {', '.join(f'{v:.0f} (sigma_eff {k} mb)' for k, v in pred.items())}; measured {OUT['fits']['notb']['N'][0]:.0f}")
json.dump(OUT, open(OUTJ, "w"), indent=1, default=float)
if SWQ:
    # sample: notb / inb (signal-region fits), pileup (the prompt-pileup fit, not b-matched)
    parts = [dn.assign(sample="notb"), di.assign(sample="inb"), pu_notb.assign(sample="pileup")] + ([dt.assign(sample="ttj")] if SIGNAL != "notb" else [])
    pd.concat(parts, ignore_index=True).to_parquet(SWQ, index=False)
    print("wrote", SWQ)
print("wrote", OUTJ)
