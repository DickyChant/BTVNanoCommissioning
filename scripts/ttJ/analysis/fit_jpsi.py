"""J/psi mass fit and the mass x l_xy fit of the ttbar + J/psi events (opposite sign, N_b >= 1, mu + jets, e + jets and e mu
summed), following the 2D model of jpsi_eec (scripts/prepare_fit.py): prompt, non-prompt and continuum components.
  mass   binned Poisson fit of m(mu mu), 2.6-3.6 GeV in 25 MeV bins: signal = Crystal Ball + Gaussian (common mean; the
         shape from the primary-vertex sample, b-matched and not, then fixed; jpsi_eec: two Gaussians), continuum =
         exponential, per sample.
         Samples: all, b-matched / not b-matched from the primary vertex (|z0| < 0.1 cm), and the not-b-matched J/psi in each
         lifetime region (prompt |l_xy| < 0.06 mm, non-prompt l_xy > 0.1 mm, pileup |z0| >= 0.1 cm).
  2D     binned Poisson fit of l_xy (20 um bins) in the signal window (2.95-3.25 GeV) and the two sidebands (2.60-2.90,
         3.30-3.60 GeV, summed), per primary-vertex sample:
           N_P S_r R(l) + N_NP S_r (E_lambda * R)(l) + N_B C_r B(l)
         S_r, C_r: the fraction of the signal and continuum mass shapes in the window / sidebands (from the mass fit);
         the not-b-matched b decays take the b-matched J/psi l_xy shape (both b quarks are from the ttbar decays);
         variation: their own exponential;
         R: prompt resolution, a Gaussian shared by the samples (jpsi_eec: three Gaussians; too few J/psi here);
         B: continuum l_xy per sample, a narrow part with the prompt resolution (pairs of prompt muons), a wide Gaussian
         (muons from different vertices) and a decay part (jpsi_eec: a decay term only).
         Significance of the prompt component from the likelihood ratio with N_P = 0.
The unbinned version (per candidate, with the l_xy error) waits for the per-candidate table.
-> <out.json>   usage: python3 fit_jpsi.py <results.json> <out.json>"""
import json
import os
import sys

import numpy as np
from iminuit import Minuit
from scipy.special import log_ndtr, ndtr, xlogy

R = json.load(open(sys.argv[1]))
CH = ("mujets", "ejets", "emu")
L = {c: R["channels"][c]["life"] for c in CH}
ME = np.asarray(R["channels"]["mujets"]["mass_edges"], float)
M_LO, M_HI = 2.6, 3.6
IM = (ME[:-1] >= M_LO - 1e-9) & (ME[1:] <= M_HI + 1e-9)
MEF = ME[np.concatenate([IM, [False]]) | np.concatenate([[False], IM])]  # fit-range edges
WIN, SBS = (2.95, 3.25), ((2.60, 2.90), (3.30, 3.60))
LE = np.asarray(L["mujets"]["lxy_edges"], float)
PV = ("prompt", "nonprompt", "between")


def lim(mi, d):
    for k, v in d.items():  # one by one: assigning a dict to Minuit.limits hangs in iminuit 2.32
        mi.limits[k] = v


def spectrum(regions, cat, chans=CH):
    return sum(np.asarray(L[c][f"spectrum_{r}_{cat}"], float) for c in chans for r in regions)[IM]


def cb_cdf(m, mu, s, a, n):
    """CDF of a Crystal Ball (power-law tail below mu - a s: final-state radiation), normalised on the real line"""
    t = (np.asarray(m, float) - mu) / s
    A, B = (n / a) ** n * np.exp(-0.5 * a * a), n / a - a
    tail = lambda u: A * (B - u) ** (1 - n) / (n - 1)  # noqa: E731
    core = lambda u: np.sqrt(2 * np.pi) * (ndtr(u) - ndtr(-a))  # noqa: E731
    tot = tail(-a) + np.sqrt(2 * np.pi) * (1 - ndtr(-a))
    return np.where(t <= -a, tail(np.minimum(t, -a)), tail(-a) + core(np.maximum(t, -a))) / tot


def sig_cdf(m, mu, s1, s2, f1, a, n):
    """J/psi mass: Crystal Ball (s1) + Gaussian (s2), common mean"""
    return f1 * cb_cdf(m, mu, s1, a, n) + (1 - f1) * ndtr((m - mu) / s2)


def bkg_cdf(m, k):
    """CDF of exp(-k (m - M_LO)) on [M_LO, M_HI]"""
    if abs(k) < 1e-6:
        return (m - M_LO) / (M_HI - M_LO)
    return (1 - np.exp(-k * (m - M_LO))) / (1 - np.exp(-k * (M_HI - M_LO)))


def frac(cdf, ranges):
    return sum(cdf(hi) - cdf(lo) for lo, hi in ranges)


def pnll(mu, n):
    mu = np.maximum(mu, 1e-12)
    return -(xlogy(n, mu) - mu).sum()


# ---------------- mass fits
def mass_model(p, Ns, Nb, k):
    sn = sig_cdf(MEF, *p)
    sn = np.diff(sn) / (sig_cdf(M_HI, *p) - sig_cdf(M_LO, *p))
    return Ns * sn, Nb * np.diff(bkg_cdf(MEF, k))


n_pv = spectrum(PV, "all")


def nll_shape(mu, s1, s2, f1, a, n, Ns, Nb, k):
    s, b = mass_model((mu, s1, s2, f1, a, n), Ns, Nb, k)
    return pnll(s + b, n_pv)


ms = Minuit(nll_shape, mu=3.09, s1=0.022, s2=0.045, f1=0.6, a=1.5, n=3.0, Ns=0.6 * n_pv.sum(), Nb=0.4 * n_pv.sum(), k=1.0)
ms.errordef = Minuit.LIKELIHOOD
lim(ms, {"mu": (3.05, 3.13), "s1": (0.008, 0.06), "s2": (0.02, 0.2), "f1": (0, 1), "a": (0.3, 5), "n": (1.1, 30), "Ns": (0, np.inf), "Nb": (0, np.inf),
         "k": (-10, 10)})
ms.migrad(); ms.migrad(); ms.hesse()
SHAPE = tuple(float(ms.values[k]) for k in ("mu", "s1", "s2", "f1", "a", "n"))
print(f"mass shape (primary-vertex J/psi): mean {1e3 * SHAPE[0]:.1f} MeV, CB sigma {1e3 * SHAPE[1]:.1f} MeV (alpha {SHAPE[4]:.2f}, n {SHAPE[5]:.1f}),"
      f" Gaussian sigma {1e3 * SHAPE[2]:.1f} MeV, CB fraction {SHAPE[3]:.2f}; valid {ms.valid}")
out = {"mass_edges": MEF.tolist(), "shape": dict(zip(("mu", "s1", "s2", "f1", "a", "n"), SHAPE)), "mass": {}, "lt2d": {}}
def mass_nll(n):
    def nll(Ns, Nb, k):
        s, b = mass_model(SHAPE, Ns, Nb, k)
        return pnll(s + b, n)
    return nll


SAMPLES = {"all": (("prompt", "nonprompt", "between", "pileup", "novtx"), "all"), "inb_pv": (PV, "inb"), "notb_pv": (PV, "notb"),
           "notb_prompt": (("prompt",), "notb"), "notb_nonprompt": (("nonprompt",), "notb"), "notb_between": (("between",), "notb"),
           "notb_pileup": (("pileup",), "notb")}
for name, (regs, cat) in SAMPLES.items():
    n = spectrum(regs, cat)
    nll = mass_nll(n)

    mi = Minuit(nll, Ns=max(0.3 * n.sum(), 1.0), Nb=max(0.7 * n.sum(), 1.0), k=1.0)
    mi.errordef = Minuit.LIKELIHOOD
    lim(mi, {"Ns": (0, np.inf), "Nb": (0, np.inf), "k": (-15, 15)})
    mi.migrad(); mi.hesse(); mi.minos("Ns")
    m0 = Minuit(nll, Ns=0.0, Nb=n.sum(), k=float(mi.values["k"]))
    m0.errordef = Minuit.LIKELIHOOD
    lim(m0, {"Nb": (0, np.inf), "k": (-15, 15)})
    m0.fixed["Ns"] = True
    m0.migrad()
    z = float(np.sqrt(max(2 * (m0.fval - mi.fval), 0)))
    s, b = mass_model(SHAPE, mi.values["Ns"], mi.values["Nb"], mi.values["k"])
    # the same window - sidebands subtraction as the yields, for comparison (window and sidebands on 25 MeV bins)
    c = 0.5 * (MEF[1:] + MEF[:-1])
    w = (c > WIN[0]) & (c < WIN[1])
    sb = np.zeros_like(w)
    for lo, hi in SBS:
        sb |= (c > lo) & (c < hi)
    swin = frac(lambda x: (sig_cdf(x, *SHAPE)), [WIN]) / (sig_cdf(M_HI, *SHAPE) - sig_cdf(M_LO, *SHAPE))
    ok = n > 0
    out["mass"][name] = {"data": n.tolist(), "sig": s.tolist(), "bkg": b.tolist(), "Ns": float(mi.values["Ns"]), "Ns_err": float(mi.errors["Ns"]),
                         "Ns_minos": [float(mi.merrors["Ns"].lower), float(mi.merrors["Ns"].upper)], "Nb": float(mi.values["Nb"]), "k": float(mi.values["k"]),
                         "signif": z, "sbsub": float(n[w].sum() - 0.5 * n[sb].sum()), "ns_window": float(mi.values["Ns"] * swin),
                         "chi2": float((((n - s - b) ** 2) / np.maximum(s + b, 1e-9))[ok].sum()), "nbins": int(len(n)), "valid": bool(mi.valid)}
    o = out["mass"][name]
    print(f"mass {name:15s}: J/psi {o['Ns']:7.1f} +- {o['Ns_err']:5.1f} (in the window {o['ns_window']:6.1f}; window - sidebands {o['sbsub']:6.1f}),"
          f" {z:4.1f} sigma, chi2 {o['chi2']:.1f} / {o['nbins']}, slope {o['k']:+.2f} /GeV")

# ---------------- mass x l_xy (window and sidebands x l_xy bins)
SNORM = sig_cdf(M_HI, *SHAPE) - sig_cdf(M_LO, *SHAPE)
S_W = frac(lambda x: sig_cdf(x, *SHAPE), [WIN]) / SNORM
S_SB = frac(lambda x: sig_cdf(x, *SHAPE), SBS) / SNORM
D2 = {cat: (sum(np.asarray(L[c][f"lxy_{cat}_pv_win"], float) for c in CH), sum(np.asarray(L[c][f"lxy_{cat}_pv_sb"], float) for c in CH))
      for cat in ("inb", "notb")}
K = {cat: out["mass"][f"{cat}_pv"]["k"] for cat in ("inb", "notb")}
C_W = {cat: frac(lambda x: bkg_cdf(x, K[cat]), [WIN]) for cat in K}
C_SB = {cat: frac(lambda x: bkg_cdf(x, K[cat]), SBS) for cat in K}


def cdf_decay(x, lam, s):
    return ndtr(x / s) - np.exp(s * s / (2 * lam * lam) - x / lam + log_ndtr(x / s - s / lam))


def lshapes(s, lam):
    return np.diff(ndtr(LE / s)), np.diff(cdf_decay(LE, lam, s))


def cont(s1, fn, fw, sw, lb):
    """continuum l_xy: a narrow part with the prompt resolution (pairs of prompt muons), a wide symmetric part
    (muons from different vertices), and a decay part (pairs from b decays)"""
    pn, d = lshapes(s1, lb)
    pw = np.diff(ndtr(LE / sw))
    return fn * pn + (1 - fn) * (fw * pw + (1 - fw) * d)


def model2d(cat, s1, NP, NNP, lam, NB, fn, fw, sw, lb, tmpl=None):
    """tmpl: the l_xy shape of the not-b-matched b decays (the b-matched J/psi shape), instead of an own exponential"""
    p, d = lshapes(s1, lam)
    if tmpl is not None:
        d = tmpl
    sig = NP * p + NNP * d
    bl = NB * cont(s1, fn, fw, sw, lb)
    return (S_W * sig, C_W[cat] * bl), (S_SB * sig, C_SB[cat] * bl), (NP * p, NNP * d)


PAR = ["NP", "NNP", "lam", "NB", "fn", "fw", "sw", "lb"]
NPAR = len(PAR)


TEMPLATE = {"on": True}  # nominal: the b-matched J/psi shape for the not-b-matched b decays (both ttbar b quarks)


BTJ = os.environ.get("TTJ_BTEMPLATE")  # btemplate.json: the signal's b-decay shape from the simulation (btemplate_mc.py)
BT = json.load(open(BTJ)) if BTJ else None


def inb_shape(s1, a):
    """the b-decay template of the not-b-matched J/psi: the b-matched J/psi shape (its prompt-like part included), or with
    TTJ_BTEMPLATE the simulated shape of the signal's b decays, lambda scaled by the fitted b-jet lambda / the simulation's"""
    if BT:
        c = BT["cats"]["signal"]
        p, d = lshapes(s1, c["lam"] * a[2] / BT["lam_ref_2d"])
        return c["f"] * p + (1 - c["f"]) * d
    p, d = lshapes(s1, a[2])
    return (a[0] * p + a[1] * d) / max(a[0] + a[1], 1e-9)


def nll2d(s1, *a):
    t = 0.0
    for j, cat in enumerate(("inb", "notb")):
        tm = inb_shape(s1, a[:NPAR]) if (cat == "notb" and TEMPLATE["on"]) else None
        (sw, bw), (ss, bs), _ = model2d(cat, s1, *a[NPAR * j: NPAR * j + NPAR], tmpl=tm)
        t += pnll(sw + bw, D2[cat][0]) + pnll(ss + bs, D2[cat][1])
    return t


names = ["s1"] + [f"{p}_{c}" for c in ("inb", "notb") for p in PAR]


def run2d(fix=None):
    st = {"s1": 0.018}
    for c in ("inb", "notb"):
        nw, ns = D2[c]
        nb = ns.sum() / C_SB[c]
        nsig = max(nw.sum() - C_W[c] * nb, 10.0) / S_W
        st.update({f"NP_{c}": 0.1 * nsig, f"NNP_{c}": 0.9 * nsig, f"lam_{c}": 0.4, f"NB_{c}": nb, f"fn_{c}": 0.2, f"fw_{c}": 0.4, f"sw_{c}": 0.1,
                   f"lb_{c}": 0.3})
    mi = Minuit(nll2d, *[st[k] for k in names], name=names)
    mi.errordef = Minuit.LIKELIHOOD
    d = {"s1": (0.003, 0.1)}
    for c in ("inb", "notb"):
        d.update({f"NP_{c}": (0, np.inf), f"NNP_{c}": (0, np.inf), f"lam_{c}": (0.02, 3), f"NB_{c}": (0, np.inf), f"fn_{c}": (0, 1),
                  f"fw_{c}": (0, 1), f"sw_{c}": (0.03, 1.0), f"lb_{c}": (0.01, 5)})
    lim(mi, d)
    if TEMPLATE["on"]:
        mi.fixed["lam_notb"] = True
    if fix:
        for k, v in fix.items():
            mi.values[k] = v
            mi.fixed[k] = True
    mi.migrad(); mi.migrad(); mi.hesse()
    return mi


# variation: the not-b-matched b decays as their own exponential (free lambda)
TEMPLATE["on"] = False
mx = run2d()
mx0 = run2d({"NP_notb": 0.0})
VX = {"N_P": float(mx.values["NP_notb"]), "fp": float(mx.values["NP_notb"] / (mx.values["NP_notb"] + mx.values["NNP_notb"])),
      "lam": float(mx.values["lam_notb"]), "signif": float(np.sqrt(max(2 * (mx0.fval - mx.fval), 0)))}
print(f"2D, own exponential for the not-b-matched b decays: N_P {VX['N_P']:.1f}, f_prompt {VX['fp']:.3f}, {VX['signif']:.1f} sigma")
TEMPLATE["on"] = True
m2 = run2d()
m2.minos("NP_notb", "NP_inb")
m20 = run2d({"NP_notb": 0.0})
z2 = float(np.sqrt(max(2 * (m20.fval - m2.fval), 0)))
P2 = {k: float(m2.values[k]) for k in names}
out["lt2d"] = {"lxy_edges": LE.tolist(), "params": P2, "errors": {k: float(m2.errors[k]) for k in names}, "valid": bool(m2.valid),
               "minos": {k: [float(m2.merrors[k].lower), float(m2.merrors[k].upper)] for k in ("NP_notb", "NP_inb")}, "signif_prompt_notb": z2,
               "S_W": S_W, "S_SB": S_SB, "C_W": C_W, "C_SB": C_SB, "samples": {}, "var_exp": VX, "btemplate": BTJ or "b jets"}
print(f"2D: sigma {1e3 * P2['s1']:.1f} um; valid {m2.valid}; prompt (not b-matched) significance {z2:.1f} sigma")
for c in ("inb", "notb"):
    a = [P2[f"{p}_{c}"] for p in PAR]
    tm = inb_shape(P2["s1"], [P2[f"{p}_inb"] for p in PAR]) if c == "notb" else None
    (sw, bw), (ss, bs), (pp, dd) = model2d(c, P2["s1"], *a, tmpl=tm)
    Np, Nnp = a[0], a[1]
    o = {"win": D2[c][0].tolist(), "sb": D2[c][1].tolist(), "win_prompt": (S_W * pp).tolist(), "win_nonprompt": (S_W * dd).tolist(), "win_cont": bw.tolist(),
         "sb_sig": ss.tolist(), "sb_cont": bs.tolist(), "fp": Np / (Np + Nnp), "N_P": Np, "N_NP": Nnp, "N_B": a[3],
         "fp_err": float(m2.errors[f"NP_{c}"]) / (Np + Nnp)}
    out["lt2d"]["samples"][c] = o
    print(f"  {c}: N_P {Np:6.1f} {m2.merrors[f'NP_{c}'].lower:+.1f} {m2.merrors[f'NP_{c}'].upper:+.1f}, N_NP {Nnp:6.1f} +- {m2.errors[f'NNP_{c}']:.1f},"
          f" f_prompt {o['fp']:.3f}, lambda {a[2]:.3f} mm; continuum N {a[3]:.0f}, narrow {a[4]:.2f}, wide {a[5]:.2f} of the rest (sigma {1e3 * a[6]:.0f} um),"
          f" lambda {a[7]:.2f} mm")
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2])
