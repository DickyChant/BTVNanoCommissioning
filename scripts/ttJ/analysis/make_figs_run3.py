"""Figures of the 2022-2026 prompt-J/psi deck (ttbar + J/psi -> mu mu, with the decay length l_xy of the two-muon vertex):
  figs/lxy_fit_{notb,inb}.json  l_xy (|z0| < 0.1 cm), window - sidebands, with the fitted prompt and b-decay components
                                (fit_lxy.py), mu + jets, e + jets and e mu summed, N_b >= 1; pull panel
  figs/z0.json                  |z0| of the J/psi line at the primary vertex: fraction per bin, outside the ttbar jets vs b-matched;
                                ratio panel
  figs/mass_life.json           m(mu mu) of the J/psi outside the ttbar jets in the three lifetime regions
  figs/pt_life.json             pT of the J/psi outside the ttbar jets in the three lifetime regions (unit area, N_b inclusive);
                                ratio to the non-prompt region
  figs/mfit_<sample>.json       m(mu mu) with the mass fit (fit_jpsi.py: Crystal Ball + Gaussian, exponential continuum), pull panel
  figs/lt2d_{notb,inb}.json     l_xy in the signal window with the mass x l_xy fit (fit_jpsi.py, as in jpsi_eec): prompt,
                                non-prompt and continuum, log scale, pull panel
  figs/years_fp.json            prompt fraction of the J/psi outside the ttbar jets per year and for all years, both fits;
                                (year - all years) / sigma
  figs/years_rate.json          J/psi per ttbar event per year: b-matched, outside the ttbar jets from the primary vertex, pileup;
                                ratio to all years
  figs/splot_*.json             the prompt J/psi of the sPlot (splot_prompt.py -> splot.json) against the other J/psi and all ttbar
                                events, ratio panel to a reference sample; splot_proj_{m,l} the unbinned fit projections with pulls
Inputs: results_run3.json (build_ttJ.py on out_run3), fit_lxy.json (fit_lxy.py), fit_jpsi.json (fit_jpsi.py), and the same per
year in years/ (TTJ_YEARS=<year> build_ttJ.py, then the fits). Style: cmsdeck.py (the cms-plotting skill's cmsplot_plotly.py
and the project additions; rules in hadW_ntrack_ttbar/.claude/cms-plotting/).
usage: python3 make_figs_run3.py   (FIG_PREVIEW=<dir> also writes PNG previews there)"""
import json
import os
import sys
import warnings

import numpy as np
import plotly.graph_objects as go
from scipy.special import ndtr

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cmsdeck import (CH3, NB1, RSIZE, SERIES, TT, Z0, bin_label, canvas, data_points, draw_pull, draw_ratio, fill_between,  # noqa: E402
                     label, legend, line, logx, ratio_canvas, region, save, step, xlabel, ylabel, zero_line)

R = json.load(open(os.path.join(HERE, "results_run3.json")))
FIT = json.load(open(os.path.join(HERE, "fit_lxy.json")))
FJ = json.load(open(os.path.join(HERE, "fit_jpsi.json")))
CH = [R["channels"][c] for c in ("mujets", "ejets", "emu")]
COL = {"prompt": SERIES["s2"], "nonprompt": SERIES["s1"], "pileup": SERIES["comb"]}
SYM = {"prompt": "circle", "nonprompt": "square", "pileup": "diamond"}
NAME = {"prompt": "prompt, |ℓ<sub>xy</sub>| &lt; 0.06 mm", "nonprompt": "non-prompt, ℓ<sub>xy</sub> &gt; 0.1 mm", "pileup": "pileup, |z<sub>0</sub>| ≥ 0.1 cm"}
XL = "ℓ<sub>xy</sub> [mm]"
NOTB = f"outside the {TT} jets"
WIN, SBS = (2.95, 3.25), ((2.60, 2.90), (3.30, 3.60))


def rb(a, k):
    a = np.asarray(a, float)
    n = len(a) // k * k
    return a[:n].reshape(-1, k).sum(axis=1)


def ext(y):
    """values per bin -> the y of a step drawn on the edges"""
    y = np.asarray(y, float)
    return np.append(y, y[-1])


def ylim(f, top, low=0.0, factor=1.5, ax="yaxis"):
    """headroom for the legend above the highest point and, for subtracted data, a lower limit below the lowest one"""
    lo = min(0.0, low)
    f.update_layout(**{ax: dict(range=[lo - 0.06 * top if lo < 0 else 0.0, factor * top])})


def logy(f, lo, hi, ax="yaxis"):
    """log y with decade labels and minor ticks at 2-9 (explicit: plotly's own labels read '5, 2, 1, 5' on log axes)"""
    d0, d1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    major = [10.0 ** k for k in range(d0, d1 + 1) if lo <= 10.0 ** k <= hi]
    minor = [m * 10.0 ** k for k in range(d0, d1 + 1) for m in range(2, 10) if lo <= m * 10.0 ** k <= hi]
    txt = [f"{v:g}" if 1e-3 <= v < 1e4 else f"10<sup>{int(round(np.log10(v)))}</sup>" for v in major]
    f.update_layout(**{ax: dict(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=major, ticktext=txt,
                                minor=dict(tickvals=minor, ticks="inside", showgrid=False))})


def vline(f, x, panels=(("x", "y"), ("x2", "y2"))):
    for xa, ya in panels:
        f.add_shape(type="line", xref=xa, yref=f"{ya} domain", x0=x, x1=x, y0=0, y1=1, line=dict(color="#0b0b0b", width=1, dash="dot"))


# ---- l_xy with the fit, per category: window - sidebands, pull
E = np.asarray(FIT["edges"])
for cat, title in (("notb", NOTB), ("inb", "b-matched")):
    S = FIT["samples"][cat]
    K = 2 if cat == "notb" else 1
    e = E[::K]
    c, bw = 0.5 * (e[1:] + e[:-1]), float(e[1] - e[0])
    d, de = rb(S["data"], K), np.sqrt(rb(np.asarray(S["data_err"]) ** 2, K))
    pr, npr = rb(S["prompt"], K), rb(S["nonprompt"], K)
    f, ax, axp = ratio_canvas()
    fp, mn = FIT["params"][f"fp_{cat}"], FIT["minos"][f"fp_{cat}"]
    zero_line(f)
    step(f, e, npr, SERIES["s1"], f"b decays, λ = {FIT['params']['lam_inb']:.2f} mm" if cat == "inb" else "b decays, b-matched shape", dash="dash")
    step(f, e, pr, SERIES["s2"], f"prompt, f = {100 * fp:.1f}<sup>+{100 * mn[1]:.1f}</sup><sub>−{-100 * mn[0]:.1f}</sub>%", dash="dot", width=2.0)
    step(f, e, pr + npr, SERIES["total"], "fit", width=2.0)
    data_points(f, c, d, de, "data, window − sidebands", ex=np.full(len(c), bw / 2))
    ylim(f, max((d + de).max(), (pr + npr).max()), (d - de).min(), 1.5)
    label(f)
    ylabel(f, bin_label(bw, "mm", "J/ψ"), chars=4)
    legend(f, f"{CH3}; {NB1}<br>{title}, {Z0}")
    draw_pull(f, axp, c, d, pr + npr, -0.3, 1.2, XL, err=de)
    save(f, f"lxy_fit_{cat}")

# ---- |z0|: fraction per bin, outside the ttbar jets against b-matched
L0 = CH[0]["life"]
ze = L0["z0_edges"]
zl = [f"{lo:g}–{hi:g}" if hi < 30 else f"&gt; {lo:g}" for lo, hi in zip(ze[:-1], ze[1:])]
xz = np.arange(len(zl), dtype=float)
f, ax, axr = ratio_canvas()
zv = {}
for cat, nm, col, sym, dx in (("notb", NOTB, SERIES["s2"], "circle", -0.12), ("inb", "b-matched", SERIES["s1"], "square", 0.12)):
    v = sum(np.asarray(C["life"][f"z0_{cat}"]) for C in CH)
    er = np.sqrt(sum(np.asarray(C["life"][f"z0_{cat}_err"]) ** 2 for C in CH))
    zv[cat] = (v / v.sum(), er / v.sum())
    data_points(f, xz + dx, v / v.sum(), er / v.sum(), f"{nm} ({v.sum():.0f} J/ψ)", ex=np.full(len(xz), 0.12), color=col, symbol=sym)
logy(f, 1e-4, 60)
label(f)
ylabel(f, "fraction of J/ψ", chars=5)
legend(f, f"{CH3}; {NB1}<br>window − sidebands")
draw_ratio(f, axr, xz, zv["notb"][0], zv["notb"][1], zv["inb"][0], zv["inb"][1], edges=np.append(xz - 0.5, xz[-1] + 0.5), ylab="/ b-matched",
           color=SERIES["s2"])
logy(f, 0.03, 500, ax="yaxis2")
f.update_xaxes(range=[-0.6, len(zl) - 0.4], tickvals=list(xz), ticktext=zl, minor=dict(showgrid=False, ticks=""))
xlabel(f, "|z<sub>0</sub>| of the J/ψ line at the primary vertex [cm]")
save(f, "z0")

# ---- mass per lifetime region, outside the ttbar jets (single panel: three regions of different yields, no prediction; the
# mass fits with pulls below compare each region with its fit)
K = 2
me = np.asarray(CH[0]["mass_edges"], float)[::K]
mc, mw = 0.5 * (me[1:] + me[:-1]), float(me[1] - me[0])
f = canvas()
region(f, *WIN, f"signal window {WIN[0]:.2f}–{WIN[1]:.2f} GeV", color="rgba(11,11,11,0.13)")
for k_, (lo, hi) in enumerate(SBS):
    region(f, lo, hi, "sidebands 2.60–2.90, 3.30–3.60 GeV" if k_ == 0 else None, color="rgba(11,11,11,0.06)")
top = 0
for ln in ("nonprompt", "prompt", "pileup"):
    v = sum(rb(C["life"][f"spectrum_{ln}_notb"], K) for C in CH)
    er = np.sqrt(sum(rb(np.asarray(C["life"][f"spectrum_{ln}_notb_err"]) ** 2, K) for C in CH))
    n = sum(C["life"][f"yield_{ln}_notb"]["n"] for C in CH)
    ne = np.sqrt(sum(C["life"][f"yield_{ln}_notb"]["err"] ** 2 for C in CH))
    data_points(f, mc, v, er, f"{NAME[ln]} (J/ψ: {n:.0f} ± {ne:.0f})", ex=np.full(len(mc), mw / 2), color=COL[ln], symbol=SYM[ln])
    top = max(top, (v + er).max())
f.update_yaxes(range=[0, 2.0 * top])
f.update_xaxes(range=[2.0, 4.5])
label(f)
xlabel(f, "m(μμ) [GeV]")
ylabel(f, bin_label(mw), chars=4)
legend(f, f"{CH3}; {NB1}<br>{NOTB}, opposite sign")
save(f, "mass_life")

# ---- pT per lifetime region, outside the ttbar jets (unit area, per GeV); ratio to the non-prompt region
pe = np.asarray(L0["pt_edges"], float)
pc, pw = 0.5 * (pe[1:] + pe[:-1]), np.diff(pe)
k = (pe[:-1] >= 3) & (pe[1:] <= 60)
f, ax, axr = ratio_canvas()
sh = {}
for j, ln in enumerate(("nonprompt", "prompt", "pileup")):
    v = sum(np.asarray(C["life"][f"pt_{ln}_notb"]) for C in CH)
    er = np.sqrt(sum(np.asarray(C["life"][f"pt_{ln}_notb_err"]) ** 2 for C in CH))
    s = v[k].sum()
    sh[ln] = (v[k] / s / pw[k], er[k] / s / pw[k])
    xs = pc[k] * (1 + 0.04 * (j - 1))
    data_points(f, xs, *sh[ln], NAME[ln], ex=None, color=COL[ln], symbol=SYM[ln])
logy(f, 5e-4, 2)
label(f)
ylabel(f, "fraction / GeV", chars=5)
legend(f, f"{CH3}; all N<sub>b</sub><br>{NOTB}, window − sidebands")
ek = np.append(pe[:-1][k], pe[1:][k][-1])
for j, ln in ((0, "prompt"), (2, "pileup")):
    draw_ratio(f, axr, pc[k] * (1 + 0.04 * (j - 1)), *sh[ln], sh["nonprompt"][0], sh["nonprompt"][1], edges=ek if j == 0 else None,
               ylab="/ non-prompt", rng=(0, 3.5), color=COL[ln], symbol=SYM[ln], band=j == 0)
logx(f, 3, 60, ticks=(3, 5, 8, 12, 20, 30, 60))
xlabel(f, "p<sub>T</sub>(J/ψ) [GeV]")
save(f, "pt_life")

# ---- mass fits (fit_jpsi.py), pull
MF = FJ["mass_edges"]
mfe = np.asarray(MF)
mfc, mfw = 0.5 * (mfe[1:] + mfe[:-1]), float(mfe[1] - mfe[0])
MTITLE = {"notb_pv": f"{NOTB}, {Z0}", "inb_pv": f"b-matched, {Z0}", "notb_prompt": f"{NOTB}, |ℓ<sub>xy</sub>| &lt; 0.06 mm",
          "notb_nonprompt": f"{NOTB}, ℓ<sub>xy</sub> &gt; 0.1 mm", "notb_pileup": f"{NOTB}, |z<sub>0</sub>| ≥ 0.1 cm", "all": "all J/ψ"}
for name, tt in MTITLE.items():
    o = FJ["mass"][name]
    n, sg, bg = np.asarray(o["data"]), np.asarray(o["sig"]), np.asarray(o["bkg"])
    tot = sg + bg
    f, ax, axp = ratio_canvas()
    step(f, mfe, bg, SERIES["comb"], "continuum", dash="dash", width=1.5)
    fill_between(f, mfe, ext(bg), ext(tot), SERIES["s2"] if name.startswith("notb") else SERIES["s1"], f"J/ψ: {o['Ns']:.0f} ± {o['Ns_err']:.0f}", shape="hv")
    step(f, mfe, tot, SERIES["total"], "fit", width=2.0)
    data_points(f, mfc, n, np.sqrt(n), "data", ex=np.full(len(mfc), mfw / 2))
    ylim(f, max((n + np.sqrt(n)).max(), tot.max()), 0.0, 1.55)
    label(f)
    ylabel(f, bin_label(mfw), chars=5)
    legend(f, f"{CH3}; {NB1}<br>{tt}")
    draw_pull(f, axp, mfc, n, tot, mfe[0], mfe[-1], "m(μμ) [GeV]")
    save(f, f"mfit_{name}")

# ---- l_xy in the signal window with the mass x l_xy fit (fit_jpsi.py), log scale, pull
T2 = FJ["lt2d"]
LE2 = np.asarray(T2["lxy_edges"])
for cat, title in (("notb", NOTB), ("inb", "b-matched")):
    o = T2["samples"][cat]
    K = 2 if cat == "notb" else 1
    e = LE2[::K]
    c, bw = 0.5 * (e[1:] + e[:-1]), float(e[1] - e[0])
    n, pr, npr, cb = (rb(o[k], K) for k in ("win", "win_prompt", "win_nonprompt", "win_cont"))
    tot = pr + npr + cb
    f, ax, axp = ratio_canvas()
    step(f, e, cb, SERIES["comb"], f"continuum: {o['N_B'] * T2['C_W'][cat]:.0f}", dash="dash", width=1.5)
    step(f, e, npr, SERIES["s1"], f"non-prompt J/ψ: {o['N_NP'] * T2['S_W']:.0f}" + (", b-matched shape" if cat == "notb" else ""), dash="dashdot")
    step(f, e, pr, SERIES["s2"], f"prompt J/ψ: {o['N_P'] * T2['S_W']:.0f}", dash="dot", width=2.0)
    step(f, e, tot, SERIES["total"], "fit", width=2.0)
    kk = n > 0
    data_points(f, c[kk], n[kk], np.sqrt(n[kk]), "data, signal window", ex=np.full(kk.sum(), bw / 2))
    logy(f, 0.3, max(n.max(), tot.max()) * 10 ** 1.9)
    label(f)
    ylabel(f, bin_label(bw, "mm"), chars=5)
    legend(f, f"{CH3}; {NB1}<br>{title}, {Z0}, 2.95–3.25 GeV")
    draw_pull(f, axp, c, n, tot, -0.3, 1.2, XL)
    save(f, f"lt2d_{cat}")

# ---- per year
YEARS = ["2022", "2023", "2024", "2025", "2026"]
YR = {y: json.load(open(os.path.join(HERE, "years", f"results_{y}.json"))) for y in YEARS}
YF = {y: json.load(open(os.path.join(HERE, "years", f"fit_lxy_{y}.json"))) for y in YEARS}
YJ = {y: json.load(open(os.path.join(HERE, "years", f"fit_jpsi_{y}.json"))) for y in YEARS}
XT = YEARS + ["2022–2026"]
XP = np.array([0, 1, 2, 3, 4, 5.4])  # the combined point set apart


def axis_years(f):
    f.update_xaxes(tickvals=list(XP), ticktext=XT, range=[-0.6, 6.0], minor=dict(showgrid=False, ticks=""))
    vline(f, 4.7)
    xlabel(f, "year")


# prompt fraction, outside the ttbar jets, both fits; lower panel (year - all years) / sigma(year)
f, ax, axr = ratio_canvas()
fl = [YF[y] for y in YEARS] + [FIT]
fj = [YJ[y]["lt2d"]["samples"]["notb"] for y in YEARS] + [FJ["lt2d"]["samples"]["notb"]]
v = np.array([100 * o["params"]["fp_notb"] for o in fl])
lo = np.array([-100 * o["minos"]["fp_notb"][0] for o in fl])
hi = np.array([100 * o["minos"]["fp_notb"][1] for o in fl])
f.add_trace(go.Scatter(x=XP - 0.12, y=v, mode="markers", name="ℓ<sub>xy</sub> fit, background from the sidebands",
                       marker=dict(symbol="circle", size=6, color=SERIES["s2"]),
                       error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo, thickness=1.3, width=0, color=SERIES["s2"]),
                       hovertemplate="%{y:.1f}%<extra></extra>"))
v2 = np.array([100 * o["fp"] for o in fj])
e2 = np.array([100 * o["fp_err"] for o in fj])
data_points(f, XP + 0.12, v2, e2, "mass × ℓ<sub>xy</sub> fit", color=SERIES["s1"], symbol="square")
zero_line(f)
ytop = max((v + hi).max(), (v2 + e2).max())
f.update_layout(yaxis=dict(range=[-0.1 * ytop, 1.5 * ytop]))
label(f)
ylabel(f, "prompt fraction [%]", chars=3)
legend(f, f"{CH3}; {NB1}<br>{NOTB}, {Z0}", x=0.03, xanchor="left")
# the error towards the combined value
pull1 = (v[:-1] - v[-1]) / np.where(v[:-1] > v[-1], lo[:-1], hi[:-1])
pull2 = (v2[:-1] - v2[-1]) / e2[:-1]
zero_line(f, axr)
data_points(f, XP[:-1] - 0.12, pull1, np.zeros(len(pull1)), None, ax=axr, color=SERIES["s2"], size=3)
data_points(f, XP[:-1] + 0.12, pull2, np.zeros(len(pull2)), None, ax=axr, color=SERIES["s1"], symbol="square", size=3)
f.update_layout(yaxis2=dict(range=[-3.5, 3.5], tickvals=[-3, 0, 3]))
ylabel(f, "(year − all) / σ", ax=axr)
axis_years(f)
save(f, "years_fp")


# J/psi per ttbar event, N_b >= 1, window - sidebands; lower panel: ratio to all years
def per_tt(res, kind):
    CC = [res["channels"][c] for c in ("mujets", "ejets", "emu")]
    ntt = sum(C["yields"]["data_os_all_ge1b"]["n_tt"] for C in CC)
    if kind == "inb":
        n = sum(C["yields"]["data_os_inb_ge1b"]["n"] for C in CC); e2_ = sum(C["yields"]["data_os_inb_ge1b"]["err"] ** 2 for C in CC)
    elif kind == "pileup":
        n = sum(C["life"]["yield_pileup_notb"]["n"] for C in CC); e2_ = sum(C["life"]["yield_pileup_notb"]["err"] ** 2 for C in CC)
    else:  # outside the ttbar jets from the primary vertex: all regions but pileup
        ks = ("prompt", "nonprompt", "between")
        n = sum(C["life"][f"yield_{k}_notb"]["n"] for C in CC for k in ks)
        e2_ = sum(C["life"][f"yield_{k}_notb"]["err"] ** 2 for C in CC for k in ks)
    return 1e5 * n / ntt, 1e5 * e2_ ** 0.5 / ntt


f, ax, axr = ratio_canvas()
rs = [YR[y] for y in YEARS] + [R]
first = True
for kind, nm, col, sym, dx in (("inb", "b-matched", SERIES["s1"], "square", -0.14), ("notb", f"{NOTB}, {Z0}", SERIES["s2"], "circle", 0.0),
                               ("pileup", f"{NOTB}, |z<sub>0</sub>| ≥ 0.1 cm", SERIES["comb"], "diamond", 0.14)):
    vv = np.array([per_tt(r_, kind) for r_ in rs])
    data_points(f, XP + dx, vv[:, 0], vv[:, 1], nm, color=col, symbol=sym)
    draw_ratio(f, axr, XP[:-1] + dx, vv[:-1, 0], vv[:-1, 1], np.full(len(YEARS), vv[-1, 0]), ylab="/ all years", rng=(0.4, 1.6), color=col,
               symbol=sym, band=first)
    first = False
logy(f, 0.5, 3000)
label(f)
ylabel(f, f"J/ψ per {TT} event [10<sup>−5</sup>]", chars=4)
legend(f, f"{CH3}; {NB1}<br>window − sidebands", x=0.03, xanchor="left")
axis_years(f)
save(f, "years_rate")

# ---- sPlot (splot_prompt.py): the prompt J/psi of the unbinned m x l_xy fit, against the other J/psi and all ttbar events
if os.path.exists(os.path.join(HERE, "splot.json")):
    SPL = json.load(open(os.path.join(HERE, "splot.json")))
    HS = SPL["hists"]
    REF = {"notb_prompt": (f"prompt J/ψ {NOTB} (sPlot)", SERIES["s2"], "circle"),
           "notb_nonprompt": (f"b decays {NOTB} (sPlot)", SERIES["s1"], "square"),
           "inb_nonprompt": ("b decays, b-matched (sPlot)", SERIES["total"], "triangle-up"),
           "pileup": ("pileup J/ψ, |z<sub>0</sub>| ≥ 0.1 cm (window − sidebands)", SERIES["comb"], "diamond"),
           "notb_prompt_globalres": ("prompt J/ψ, one ℓ<sub>xy</sub> resolution for all (check)", SERIES["s2"], "circle-open"),
           "pileupfit_prompt": ("prompt pileup J/ψ, |z<sub>0</sub>| ≥ 0.1 cm (sPlot)", SERIES["comb"], "diamond")}
    SLT = f"{CH3}; {NB1}; opposite sign"
    RNAME = {"pileupfit_prompt": "/ pileup", "inb_nonprompt": "/ b-matched", "notb_nonprompt": "/ b decays"}  # short: the lower panel is ~100 px tall

    logy_ = logy

    def sp_fig(v, keys, name, den, logy=False, logx_=False, xr=None, ttbar=False, perx=True, ylab=None, yr=None, rr=(0, 3)):
        """unit-area distributions of the samples in keys; lower panel: each sample / den (a key, or 'ttbar' = all ttbar events)"""
        e = np.asarray(SPL["vars"][v]["edges"], float)
        if xr is not None:
            k = (e[:-1] >= xr[0] - 1e-9) & (e[1:] <= xr[1] + 1e-9)
        else:
            k = np.ones(len(e) - 1, bool)
        c, w = 0.5 * (e[1:] + e[:-1]), np.diff(e)
        ek = np.append(e[:-1][k], e[1:][k][-1])
        unit = SPL["vars"][v]["label"].split("[")[-1].rstrip("]") if "[" in SPL["vars"][v]["label"] else ""
        uniform = np.allclose(w[k], w[k][0])
        perx = perx or not uniform  # variable bins: always per unit of x
        wk = w[k] if perx else np.ones(int(k.sum()))
        f, ax, axr = ratio_canvas()
        sh = {}
        if ttbar:
            t = np.asarray(SPL["ttbar"][f"{v}_all"], float)
            tn = t[k].sum()
            sh["ttbar"] = (t[k] / tn / wk, np.sqrt(np.maximum(t[k], 0)) / tn / wk)
            step(f, ek, sh["ttbar"][0], "#0b0b0b", f"all {TT} events", dash="dash", width=1.2)
        top, bot = 0.0, np.inf
        nk = len(keys)
        xpos = {}
        for j, key in enumerate(keys):
            h, he = (np.asarray(a, float) for a in HS[f"{key}_{v}_all"])
            sn = h[k].sum()
            y, ye = h[k] / sn / wk, he[k] / sn / wk
            sh[key] = (y, ye)
            # side by side within the bin (log x: a constant factor)
            xs = c[k] * (1 + 0.05 * (j - (nk - 1) / 2)) if logx_ else c[k] + (j - (nk - 1) / 2) * 0.12 * w[k]
            xpos[key] = xs
            data_points(f, xs, y, ye, REF[key][0], color=REF[key][1], symbol=REF[key][2],
                        ex=(np.full(len(xs), np.nan) if j else None))
            if j == 0:
                f.data[-1].error_x = dict(type="data", symmetric=False, array=e[1:][k] - xs, arrayminus=xs - e[:-1][k], thickness=1.3, width=0,
                                          color=REF[key][1])
            top = max(top, np.nanmax(y + ye))
            bot = min(bot, np.nanmin(y - ye))
        if logy:
            logy_(f, 10 ** yr[0], 10 ** yr[1])
        elif yr is None:
            f.update_layout(yaxis=dict(range=[min(0.0, bot) - 0.05 * top, 1.6 * top]))
        if yr is not None and not logy:
            f.update_layout(yaxis=dict(range=yr))
        if min(bot, 0) < 0 and not logy:
            zero_line(f)
        label(f)
        ylabel(f, ylab or (f"fraction / {unit}" if perx and unit else ("fraction / unit" if perx else f"fraction / {w[k][0]:g}")), chars=5)
        legend(f, SLT if not ttbar else f"{SLT}<br>{NOTB}")
        dy, dye = sh[den]
        first = True
        for key in keys:
            if key == den or key.endswith("_globalres"):  # the resolution check: main panel only
                continue
            draw_ratio(f, axr, xpos[key], *sh[key], dy, dye, edges=ek if first else None, ylab=f"/ all {TT}" if den == "ttbar" else RNAME[den],
                       rng=rr, color=REF[key][1], symbol=REF[key][2], band=first)
            first = False
        if logx_:
            tv = [x for x in (3, 5, 8, 12, 20, 30, 60, 100) if ek[0] <= x <= ek[-1]]
            logx(f, ek[0], ek[-1], ticks=tv)
        else:
            f.update_xaxes(range=[ek[0], ek[-1]])
        xlabel(f, SPL["vars"][v]["label"])
        save(f, name)

    sp_fig("pt", ["notb_prompt", "notb_prompt_globalres", "pileupfit_prompt", "notb_nonprompt", "inb_nonprompt"], "splot_pt", "pileupfit_prompt",
           logy=True, logx_=True, xr=(3, 100), yr=[-3.6, 0.9], rr=(0, 4))
    sp_fig("absy", ["notb_prompt", "pileupfit_prompt", "notb_nonprompt", "inb_nonprompt"], "splot_absy", "pileupfit_prompt", perx=False, rr=(0, 2.5))
    sp_fig("tt_mjj", ["notb_prompt", "notb_nonprompt"], "splot_mjj", "ttbar", ttbar=True)
    sp_fig("tt_mjjb", ["notb_prompt", "notb_nonprompt"], "splot_mjjb", "ttbar", ttbar=True)
    sp_fig("tt_mlb", ["notb_prompt", "notb_nonprompt"], "splot_mlb", "ttbar", ttbar=True)
    sp_fig("mlj", ["notb_prompt", "pileupfit_prompt", "notb_nonprompt", "inb_nonprompt"], "splot_mlj", "pileupfit_prompt")
    sp_fig("sj_dr", ["notb_prompt", "pileupfit_prompt", "notb_nonprompt"], "splot_sjdr", "pileupfit_prompt", perx=False)
    sp_fig("z", ["notb_prompt", "pileupfit_prompt", "inb_nonprompt"], "splot_z", "inb_nonprompt", perx=False, xr=(0, 1.0))

    # prompt pileup J/psi in |z0|: the Gaussian of the distance between two collisions, extrapolated under the primary vertex; pull
    PL = SPL["pu_leak"]
    e = np.asarray(PL["z0_edges"], float)
    c, w = 0.5 * (e[1:] + e[:-1]), np.diff(e)
    h, he = np.asarray(PL["prompt_pileup_z0"], float), np.asarray(PL["prompt_pileup_z0_err"], float)
    sz = PL["sigma_dz_cm"]
    A_ = h.sum() / (2 * (ndtr(e[-1] / sz) - ndtr(e[0] / sz)))
    expb = A_ * 2 * (ndtr(e[1:] / sz) - ndtr(e[:-1] / sz))  # expected per bin
    zz = np.geomspace(0.05, 15, 301)
    dens = A_ * 2 * np.exp(-0.5 * (zz / sz) ** 2) / (np.sqrt(2 * np.pi) * sz)
    f, ax, axp = ratio_canvas()
    region(f, 0.01, 0.1, f"|z<sub>0</sub>| &lt; 0.1 cm: {PL['n_in_pv']:.1f} ± {PL['n_in_pv_err']:.1f} expected", color="rgba(27,175,122,0.10)")
    zero_line(f)
    line(f, zz, dens, SERIES["total"], f"Gaussian, σ(Δz) = {sz:.1f} cm", dash="dash", width=1.6)
    data_points(f, c, h / w, he / w, "prompt pileup J/ψ (sPlot)", ex=w / 2, color=SERIES["comb"], symbol="diamond")
    f.update_layout(yaxis=dict(range=[min(-15, 1.15 * (h / w - he / w).min()), 1.7 * max((h / w + he / w).max(), dens.max())]))
    label(f)
    ylabel(f, "J/ψ / cm", chars=4)
    legend(f, f"{SLT}<br>{NOTB}", y=0.5)
    draw_pull(f, axp, c, h, expb, 0.05, 15, "|z<sub>0</sub>| of the J/ψ line at the primary vertex [cm]", err=he)
    logx(f, 0.05, 15, ticks=(0.1, 0.3, 1, 3, 10))
    save(f, "splot_z0pu")

    # unbinned fit projections, outside the ttbar jets, pull
    PJ = SPL["proj"]
    for dim, ed, xl, nm in (("m", PJ["mass_edges"], "m(μμ) [GeV]", "splot_proj_m"), ("l", PJ["lxy_edges"], XL, "splot_proj_l")):
        e = np.asarray(ed, float)
        c, bw = 0.5 * (e[1:] + e[:-1]), float(e[1] - e[0])
        P_ = PJ["notb"]
        d = np.asarray(P_[f"data_{dim}"], float)
        comp = {s_: np.asarray(P_[f"{s_}_{dim}"], float) for s_ in ("prompt", "nonprompt", "cont")}
        tot = comp["prompt"] + comp["nonprompt"] + comp["cont"]
        f, ax, axp = ratio_canvas()
        step(f, e, comp["cont"], SERIES["comb"], "continuum", dash="dash", width=1.5)
        step(f, e, comp["nonprompt"], SERIES["s1"], "b decays (b-matched shape)", dash="dashdot")
        step(f, e, comp["prompt"], SERIES["s2"], f"prompt J/ψ, {SPL['fits']['notb']['N'][0]:.0f} ± {SPL['fits']['notb']['N_err'][0]:.0f}", dash="dot", width=2.0)
        step(f, e, tot, SERIES["total"], "fit", width=2.0)
        kk = d > 0 if dim == "l" else np.ones(len(d), bool)
        data_points(f, c[kk], d[kk], np.sqrt(d[kk]), "data", ex=np.full(int(kk.sum()), bw / 2))
        if dim == "l":
            logy(f, 0.3, 30 * d.max())
            lo_, hi_ = -0.3, 1.2
        else:
            ylim(f, max((d + np.sqrt(d)).max(), tot.max()), 0.0, 1.9)
            lo_, hi_ = e[0], e[-1]
        label(f)
        ylabel(f, bin_label(bw, "mm") if dim == "l" else bin_label(bw), chars=5)
        legend(f, f"{SLT}<br>{NOTB}, {Z0}; unbinned fit")
        draw_pull(f, axp, c, d, tot, lo_, hi_, xl)
        save(f, nm)

print("figures written to figs/", RSIZE)
