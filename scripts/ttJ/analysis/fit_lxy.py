"""Prompt fraction of the ttbar J/psi from a fit of the pseudo-proper decay length l_xy (opposite sign, N_b >= 1,
|z0| < 0.1 cm: from the primary vertex), mu + jets, e + jets and e mu summed (build_ttJ.py results, life section).
  b-matched:      N [f R(l) + (1 - f) (E_lambda * R)(l)]          (b decays of the ttbar b quarks)
  not b-matched:  N [f_p R(l) + (1 - f_p) B(l)],  B = the fitted b-matched shape (its prompt-like part f included):
                  in ttbar both b quarks come from the top decays, so the not-b-matched J/psi from b decays (b jets out
                  of the selection) share it, and f_p is the prompt excess over the b-matched J/psi
    R         prompt resolution, G(0, sigma), shared by all samples
    E_lambda  exp(-l / lambda) / lambda for l > 0, convolved with R
  per l_xy bin: window counts n ~ Poisson(s + b), sideband counts m ~ Poisson(2 b) (the two sidebands are twice as wide
  as the window), the background b profiled bin by bin. The b-matched (mostly b decays) and the not-b-matched J/psi are
  fitted together. Then, per channel, N and f_p of the not-b-matched J/psi with the b-matched shape fixed, in 9 coarse
  l_xy bins (a few tens of J/psi per channel: too few for a background term in each of the 100 bins).
Variations: "exp", the not-b-matched b decays as their own exponential (free lambda) instead of the b-matched shape;
"tail", R with a fixed 20% second Gaussian of 3 sigma; "coarse", the combined fit in the coarse bins; "mu_emu", mu + jets
and e mu only (e + jets has more not-b-matched J/psi from the primary vertex per ttbar event, 1.5 times after the jet-cleaning fix),
(On these data a free second Gaussian goes to zero fraction.)
-> <out.json>: parameters (MINOS errors on f_p), the variations, sideband-subtracted data and the fitted components per bin.
usage: python3 fit_lxy.py <results.json> <out.json>"""
import json
import sys

import numpy as np
from iminuit import Minuit
from scipy.special import log_ndtr, ndtr, xlogy

R = json.load(open(sys.argv[1]))
CH = ("mujets", "ejets", "emu")
E = np.asarray(R["channels"]["mujets"]["life"]["lxy_edges"], float)
TAU = 2.0
CATS = ("inb", "notb")


def counts(cat, chans=CH):
    n = sum(np.asarray(R["channels"][c]["life"][f"lxy_{cat}_pv_win"]) for c in chans)
    m = sum(np.asarray(R["channels"][c]["life"][f"lxy_{cat}_pv_sb"]) for c in chans)
    return n, m


def cdf_decay(x, lam, s):
    """CDF of exp(-l / lam) / lam (l > 0) convolved with G(0, s)"""
    return ndtr(x / s) - np.exp(s * s / (2 * lam * lam) - x / lam + log_ndtr(x / s - s / lam))


def shapes(s1, lam, tail=(0.0, 3.0), e=E):
    """per-bin probabilities of the prompt and the b-decay components; tail = (fraction, width ratio) of a second Gaussian"""
    ft, r = tail
    p = np.diff((1 - ft) * ndtr(e / s1) + ft * ndtr(e / (r * s1)))
    d = np.diff((1 - ft) * cdf_decay(e, lam, s1) + ft * cdf_decay(e, lam, r * s1))
    return p, d


# coarse bins (mm) for the low-count fits: the per-channel ones and a cross-check of the combined fit
COARSE = [-0.5, -0.1, -0.04, 0.0, 0.04, 0.1, 0.2, 0.4, 0.8, 1.5]
IC = [int(np.argmin(np.abs(E - x))) for x in COARSE]
EC = E[IC]


def coarse(a):
    return np.add.reduceat(np.asarray(a, float), IC[:-1])


def nll_bins(s, n, m):
    """-ln L of the window (s + b) and the sidebands (TAU b), b profiled per bin"""
    s = np.maximum(s, 1e-12)
    a = (1 + TAU) * s - n - m
    b = (-a + np.sqrt(a * a + 4 * (1 + TAU) * m * s)) / (2 * (1 + TAU))
    mu = s + b
    return -(xlogy(n, mu) - mu + xlogy(m, TAU * b) - TAU * b).sum()


DATA = {c: counts(c) for c in CATS}
DATA_C = {c: tuple(coarse(x) for x in DATA[c]) for c in CATS}


def fit(tail=(0.0, 3.0), same_lam=False, coarse_bins=False, inb_template=True, chans=CH, no_prompt=False):
    """inb_template: the b-decay shape of the not-b-matched J/psi is the whole b-matched shape (its prompt-like part
    included), so fp_notb is the prompt excess over the b-matched J/psi; chans: the channels summed"""
    data = DATA if chans == CH else {c: counts(c, chans) for c in CATS}
    e = E
    if coarse_bins:
        data, e = {c: tuple(coarse(x) for x in data[c]) for c in CATS}, EC

    def nll(s1, N_inb, fp_inb, lam_inb, N_notb, fp_notb, lam_notb):
        p, d = shapes(s1, lam_inb, tail, e)
        t = nll_bins(N_inb * (fp_inb * p + (1 - fp_inb) * d), *data["inb"])
        if inb_template:
            d = fp_inb * p + (1 - fp_inb) * d
        elif not same_lam:
            p, d = shapes(s1, lam_notb, tail, e)
        return t + nll_bins(N_notb * (fp_notb * p + (1 - fp_notb) * d), *data["notb"])

    st = {c: max(float(DATA[c][0].sum() - DATA[c][1].sum() / TAU), 10.0) for c in CATS}
    mi = Minuit(nll, s1=0.02, N_inb=st["inb"], fp_inb=0.1, lam_inb=0.35, N_notb=st["notb"], fp_notb=0.3, lam_notb=0.35)
    mi.errordef = Minuit.LIKELIHOOD
    LIM = {"s1": (0.003, 0.1), "fp_inb": (0, 1), "fp_notb": (0, 1), "lam_inb": (0.02, 3.0), "lam_notb": (0.02, 3.0), "N_inb": (0, np.inf),
           "N_notb": (0, np.inf)}
    for k, lim in LIM.items():  # one by one: assigning a dict to Minuit.limits hangs in iminuit 2.32
        mi.limits[k] = lim
    if same_lam or inb_template:
        mi.fixed["lam_notb"] = True
    if no_prompt:  # the null hypothesis of the significance: no prompt not-b-matched J/psi
        mi.values["fp_notb"] = 0.0
        mi.fixed["fp_notb"] = True
    mi.migrad()
    mi.hesse()
    safe_minos(mi, "fp_inb", *(() if no_prompt else ("fp_notb",)))
    return mi


def safe_minos(mi, *pars):
    """MINOS where the minimum is valid; otherwise (e.g. a fraction at its limit of 0 in a toy) the HESSE errors stand in"""
    try:
        mi.minos(*pars)
    except RuntimeError:
        pass


def merr(mi, k):
    """[lower, upper] error: MINOS if it ran, else -/+ the HESSE error"""
    try:
        return [float(mi.merrors[k].lower), float(mi.merrors[k].upper)]
    except (KeyError, RuntimeError):
        return [-float(mi.errors[k]), float(mi.errors[k])]


def signif(**kw):
    """likelihood-ratio significance of the prompt not-b-matched component"""
    return float(np.sqrt(max(2 * (fit(no_prompt=True, **kw).fval - fit(**kw).fval), 0)))


mi = fit()
print(mi.fmin)
P = {k: float(mi.values[k]) for k in mi.parameters}
ERR = {k: float(mi.errors[k]) for k in mi.parameters}
MINOS = {k: merr(mi, k) for k in ("fp_inb", "fp_notb")}
out = {"edges": E.tolist(), "params": P, "errors": ERR, "minos": MINOS, "valid": bool(mi.valid and mi.fmin.has_accurate_covar),
       "minos_valid": {k: bool(k in mi.merrors and mi.merrors[k].is_valid) for k in MINOS}, "samples": {}, "channels": {}, "variations": {}}
for c in CATS:
    n, m = DATA[c]
    p, d = shapes(P["s1"], P["lam_inb"])
    if c == "notb":  # the b-matched shape
        d = P["fp_inb"] * p + (1 - P["fp_inb"]) * d
    N, fp = P[f"N_{c}"], P[f"fp_{c}"]
    sub, err = n - m / TAU, np.sqrt(n + m / TAU**2)
    mod = N * (fp * p + (1 - fp) * d)
    k = err > 0
    out["samples"][c] = {"data": sub.tolist(), "data_err": err.tolist(), "prompt": (N * fp * p).tolist(), "nonprompt": (N * (1 - fp) * d).tolist(),
                         "chi2": float((((sub - mod) / np.where(k, err, 1)) ** 2)[k].sum()), "nbins": int(k.sum())}
    print(f"{c}: N {N:.1f} +- {ERR[f'N_{c}']:.1f}, f_prompt {fp:.3f} {MINOS[f'fp_{c}'][0]:+.3f} {MINOS[f'fp_{c}'][1]:+.3f}, lambda {P['lam_inb']:.3f} +- {ERR['lam_inb']:.3f} mm,"
          f" chi2 (sub.) {out['samples'][c]['chi2']:.1f} / {out['samples'][c]['nbins']} bins")
print(f"resolution sigma {1e3 * P['s1']:.1f} +- {1e3 * ERR['s1']:.1f} um; fit valid {out['valid']}, MINOS valid {out['minos_valid']}")
for name, kw in (("exp", {"inb_template": False}), ("tail", {"tail": (0.2, 3.0)}), ("coarse", {"coarse_bins": True}),
                 ("mu_emu", {"chans": ("mujets", "emu")})):
    v = fit(**kw)
    out["variations"][name] = {k: float(v.values[k]) for k in v.parameters} | {"valid": bool(v.valid)}
    out["variations"][name]["fp_notb_minos"] = merr(v, "fp_notb")
    out["variations"][name]["N_notb_err"] = float(v.errors["N_notb"])
    out["variations"][name]["errors"] = {k: float(v.errors[k]) for k in v.parameters}
    print(f"variation {name}: f_prompt inb {v.values['fp_inb']:.3f}, not b-matched {v.values['fp_notb']:.3f} (nominal {P['fp_notb']:.3f}); valid {v.valid}")

# per channel, not b-matched, in the coarse bins: N and f_p free, the b-matched shape fixed from the combined fit
p, d = shapes(P["s1"], P["lam_inb"], e=EC)
d = P["fp_inb"] * p + (1 - P["fp_inb"]) * d


def make_nll(n, m):
    def nll_ch(N, fp):
        return nll_bins(N * (fp * p + (1 - fp) * d), n, m)
    return nll_ch


for ch in CH:
    n, m = (coarse(x) for x in counts("notb", (ch,)))
    mc = Minuit(make_nll(n, m), N=max(float(n.sum() - m.sum() / TAU), 5.0), fp=0.3)
    mc.errordef = Minuit.LIKELIHOOD
    mc.limits["N"], mc.limits["fp"] = (0, np.inf), (0, 1)
    mc.migrad(); mc.hesse(); safe_minos(mc, "fp")
    out["channels"][ch] = {"N": float(mc.values["N"]), "N_err": float(mc.errors["N"]), "fp": float(mc.values["fp"]),
                           "fp_minos": merr(mc, "fp"), "sub": float(n.sum() - m.sum() / TAU),
                           # the canvas of the fit: window - sidebands / TAU per coarse bin, and the fitted components
                           "edges": list(map(float, EC)), "data": (n - m / TAU).tolist(), "data_err": np.sqrt(n + m / TAU ** 2).tolist(),
                           "prompt": (mc.values["N"] * mc.values["fp"] * p).tolist(), "nonprompt": (mc.values["N"] * (1 - mc.values["fp"]) * d).tolist()}
    print(f"  {ch} not b-matched: N {mc.values['N']:.1f} +- {mc.errors['N']:.1f} (window - sidebands {n.sum() - m.sum() / TAU:.1f}),"
          f" f_prompt {mc.values['fp']:.3f} {merr(mc, 'fp')[0]:+.3f} {merr(mc, 'fp')[1]:+.3f}")
out["signif"] = {"nominal": signif(), "exp": signif(inb_template=False), "mu_emu": signif(chans=("mujets", "emu"))}
print("prompt significance:", {k: round(v, 1) for k, v in out["signif"].items()})
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2])
