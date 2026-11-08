"""Figures of data against the CMS ttbar simulation (mc_compare.py -> mc_compare.json, mc_template_fit.py -> mc_template.json),
with cmsdeck (the cms-plotting skill's cmsplot_plotly + the deck additions; rules in hadW_ntrack_ttbar/.claude/cms-plotting):
  figs/mc_lxy_{signal,ttj,inb}.json  l_xy (window - sidebands / 2) per category: data against the simulation normalised to the
                                     luminosity, stacked by MC truth (b decay, no common J/psi parent, prompt); data / simulation
  figs/mc_rates.json                 J/psi per category: data (sWeights) against the simulation (truth); data / simulation
  figs/mc_bshape.json                l_xy of the b-decay J/psi of the simulation per category, unit area, and the prompt template;
                                     each category / the b jets
  figs/mc_tfit_{ttj,signal}.json     binned l_xy fit with the templates of the simulation; pull
  figs/mc_closure_{signal,ttj}.json  the unbinned fit of the data chain run on the simulation as pseudo-data (l_xy projection), with
                                     the MC truth prompt J/psi; pull
  figs/chan_{mass,lxy,lxyfit}_{mujets,ejets,emu}.json  per channel, J/psi outside the ttbar jets: m(mu mu) and l_xy against the
                                     simulation (data / simulation), the l_xy fit of the channel (fit_lxy.json; pull)
  figs/chan_mt.json                  m_T(l, pT^miss) of the events with a prompt J/psi outside the ttbar jets (sWeights), mu + jets and
                                     e + jets, against the b-decay ones (both channels); prompt / b decay
  figs/chan_rates.json               prompt and b-decay J/psi per channel: data (sWeights) against the simulation; data / simulation
  figs/mc_toys_{mc,datalike}.json    (toy_closure.json) toy experiments from the simulation with injected prompt J/psi: mean fitted
                                     against true prompt yield per fit; fit - truth
  figs/mc_toypulls_{mc,datalike}.json  pulls of the fits at 100 injected prompt J/psi against a unit Gaussian
  figs/mc_bshape_pt.json, mc_bshape_rew.json  the b-decay l_xy shape against pT(J/psi) per category (share of short decays, mean
                                     decay length); the b-jet shape reweighted to the signal's pT against the signal's own; ratios
  figs/mc_btmpl_shapes.json          (btemplate.json) MC truth b-decay l_xy per category against its fitted model (per-candidate
                                     resolution) and the b-jet model on the vetoed jets; truth / model
  figs/mc_btmpl_data.json            the data's prompt J/psi of each fit with the b-jet template and with the per-category template (btmpl/)
  figs/mc_toys_btmpl_*, mc_toypulls_btmpl_*  the toys with the per-category template (toy_closure_btmpl.json)
  figs/mc_toys_tmplcmp_{unbinned,fit2d}.json  fit - truth of the toys with both templates in both scenarios; data-like - as simulated
  figs/mcvar_ratio{,_emu}.json       (gen_variants.json) prompt J/psi per b-decay J/psi within each sample (normalisation-free; _emu: the e mu
                                     channel with the dilepton samples), outside the
                                     ttbar jets and in b jets, against the data; data / generator
  figs/mcvar_{signal,inb}.json       (gen_variants.json) prompt and b-decay J/psi per generator / tune (MC truth, l + jets)
                                     against the data (sWeights); data / generator
usage: python3 make_figs_mc.py"""
import json
import math
import os
import sys
import warnings

import numpy as np
import plotly.graph_objects as go
from plotlyhep._units import pt2px

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cmsdeck import (CH3, NB1, SERIES, TT, Z0, bin_label, data_points, draw_pull, draw_ratio, fill_between, headroom, label,  # noqa: E402
                     legend, line, logx, ratio_canvas, save, step, xlabel, ylabel, zero_line)

C = json.load(open(os.path.join(HERE, "mc_compare.json")))
T = json.load(open(os.path.join(HERE, "mc_template.json")))
CATN = {"signal": f"J/ψ outside the {TT} jets", "ttj": f"J/ψ in the other {TT} jets (vetoed)", "inb": "J/ψ in a b jet"}
SEL = f"{CH3}; {NB1}; {Z0}"
LJ_SEL = f"μ + jets, e + jets; {NB1}; {Z0}"
HAS_BKG = "bkg" in C["lxy"]["signal_all"]
SIMN = (f"CMS simulation: {TT} + backgrounds, 310 fb<sup>−1</sup>" if HAS_BKG else f"CMS {TT} simulation only: σ<sub>{TT}</sub> × 310 fb<sup>−1</sup>")
TRUTH = (("b decay", f"CMS {TT} sim.: b-hadron decay", SERIES["s1"]), ("prompt", f"CMS {TT} sim.: prompt J/ψ", SERIES["s2"]))  # stacked
COMB = ("no J/psi", f"CMS {TT} sim.: no common J/ψ parent", SERIES["comb"])  # what the sideband subtraction leaves: a line around 0
XL = "ℓ<sub>xy</sub> [mm]"


def sim(r, key):
    """the simulation of one species: ttbar (+ the non-ttbar backgrounds when present), [value, error]"""
    b = r.get(key.replace("mc_", "bkg_"))
    return [r[key][0] + (b[0] if b else 0.0), float(np.hypot(r[key][1], b[1] if b else 0.0))]


def stack(fig, edges, comps, colors, labels):
    """components stacked as fills between the cumulative sums (slot order), step-shaped"""
    lo = np.zeros(len(edges))
    for y, col, nm in zip(comps, colors, labels):
        hi = lo + np.append(y, y[-1])
        fill_between(fig, edges, lo, hi, col, nm, shape="hv")
        lo = hi


BKGN, BKGC = f"non-{TT} backgrounds (CMS sim.)", SERIES["s3"]


def with_bkg(L, comps, errs, cols, labs, scale):
    """add the non-ttbar background histogram (all MC truths) to a stack when mc_compare.py had them (TTJ_BKG)"""
    if "bkg" in L:
        comps.append(np.asarray(L["bkg"][0], float) / scale)
        errs.append(np.asarray(L["bkg"][1], float) / scale)
        cols.append(BKGC)
        labs.append(BKGN)
    return comps, errs, cols, labs


def yrange(fig, top, low):
    fig.update_layout(yaxis=dict(range=[low - 0.06 * top, 1.6 * top]))


# ---- l_xy per category: data against the simulation by truth
E = np.asarray(C["lxy_edges"], float)
bw = np.diff(E)
c = 0.5 * (E[1:] + E[:-1])
for cat in ("signal", "ttj", "inb"):
    L = C["lxy"][f"{cat}_all"]
    f, ax, axr = ratio_canvas()
    comps = [np.asarray(L[f"mc_{t}"][0], float) / bw for t, _, _ in TRUTH]
    errs = [np.asarray(L[f"mc_{t}"][1], float) / bw for t, _, _ in TRUTH]
    comps, errs, cols, labs = with_bkg(L, comps, errs, [col for _, _, col in TRUTH], [nm for _, nm, _ in TRUTH], bw)
    stack(f, E, comps, cols, labs)
    cb, ecb = (np.asarray(a, float) / bw for a in L[f"mc_{COMB[0]}"])
    step(f, E, cb, COMB[2], COMB[1], dash="dash", width=1.5)
    hd, ed = (np.asarray(a, float) / bw for a in L["data"])
    data_points(f, c, hd, ed, label="Data", ex=bw / 2)
    tot, etot = sum(comps) + cb, np.sqrt(sum(e ** 2 for e in errs) + ecb ** 2)
    yrange(f, max(np.max(hd + ed), np.max(tot)), min(np.min(hd - ed), np.min(tot), 0.0))
    zero_line(f)
    label(f)
    ylabel(f, "J/ψ / mm", chars=6)
    legend(f, f"{SEL}<br>{CATN[cat]}<br>window − sidebands; {SIMN}")
    draw_ratio(f, axr, c, hd, ed, tot, etot, edges=E, lo=E[0], hi=E[-1], xlab=XL, ylab="data / sim.", rng=(-1.0, 3.0), ex=bw / 2)
    save(f, f"mc_lxy_{cat}")

# ---- yields per category: the simulation as a segment per category and species, data points on it
cats = ("inb", "ttj", "signal")
R = [C["rates"][f"{k}_all"] for k in cats]
f, ax, axr = ratio_canvas()
for key, dx, col, nm in (("mc_b decay", -0.21, SERIES["s1"], f"CMS {TT} sim.: b-hadron decay"), ("mc_prompt", 0.21, SERIES["s2"], f"CMS {TT} sim.: prompt J/ψ")):
    xs, ys = [], []
    for i, r in enumerate(R):
        xs += [i + dx - 0.17, i + dx + 0.17, None]
        ys += [sim(r, key)[0]] * 2 + [None]
    f.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=nm, line=dict(color=col, width=pt2px(3)), hoverinfo="y+name"))
x0 = np.arange(len(cats), dtype=float)
for key, mk, dx, sym, nm in (("data_nonprompt", "mc_b decay", -0.21, "circle", "Data (sPlot): b-hadron decay"),
                             ("data_prompt", "mc_prompt", 0.21, "diamond", "Data (sPlot): prompt J/ψ")):
    y = np.array([r[key][0] for r in R])
    ey = np.array([r[key][1] for r in R])
    data_points(f, x0 + dx, np.maximum(y, 1e-3), np.minimum(ey, np.maximum(y, 1e-3) * 0.95), label=nm, symbol=sym)
    m, em = np.array([sim(r, mk)[0] for r in R]), np.array([sim(r, mk)[1] for r in R])
    draw_ratio(f, axr, x0 + dx, y, ey, m, em, xlab="where the J/ψ is", ylab="data / sim.", symbol=sym, band=(sym == "circle"))
# the ratio on a log axis: b decays near 1, prompt J/psi far above
f.update_layout(yaxis2=dict(type="log", range=[-1.3, 2.3], tickvals=[0.1, 1, 10, 100], ticktext=["0.1", "1", "10", "100"], minor=dict(showgrid=False, ticks="")))
f.update_xaxes(range=[-0.6, len(cats) - 0.4], tickvals=list(x0), ticktext=["b jet", f"other {TT} jet (vetoed)", f"outside the {TT} jets"])
f.update_layout(yaxis=dict(type="log", range=[0, 5.8], tickvals=[1, 10, 100, 1000, 1e4, 1e5], ticktext=["1", "10", "100", "1000", "10<sup>4</sup>", "10<sup>5</sup>"],
                           minor=dict(showgrid=False, ticks="")))
label(f)
ylabel(f, "J/ψ", chars=5)
legend(f, f"{SEL}<br>{SIMN}")
save(f, "mc_rates")

# ---- b-decay l_xy shapes of the simulation per category, and the prompt template; each / the b jets
ET = np.asarray(T["edges"], float)
bt = np.diff(ET)
ct = 0.5 * (ET[1:] + ET[:-1])
f, ax, axr = ratio_canvas()
ref, eref = (np.asarray(a, float) / bt for a in T["templates"]["bdecay_inb"])
for cat, col, nm, sym in (("inb", SERIES["total"], "b jet", None), ("ttj", SERIES["s3"], f"other {TT} jets (vetoed)", "square"),
                          ("signal", SERIES["s4"], f"outside the {TT} jets", "diamond")):
    y, ey = (np.asarray(a, float) / bt for a in T["templates"][f"bdecay_{cat}"])
    step(f, ET, y, col, f"b-hadron decay, {nm}", width=2.0)
    if sym:
        draw_ratio(f, axr, ct, y, ey, ref, eref, edges=ET, lo=ET[0], hi=ET[-1], xlab=XL, ylab="/ b jet", rng=(0, 6), color=col, symbol=sym,
                   band=(cat == "ttj"))
step(f, ET, np.asarray(T["templates"]["prompt_1"][0]) / bt, SERIES["s2"], "prompt J/ψ, all categories", dash="dot", width=2.0)
headroom(f, max(np.max(np.asarray(T["templates"][k][0]) / bt) for k in ("bdecay_ttj", "prompt_1")), factor=1.45)
label(f, rlabel="13.6 TeV", sim=True)
ylabel(f, "fraction / mm", chars=4)
legend(f, f"CMS {TT} simulation (POWHEG + Pythia 8), MC truth<br>{SEL}; J/ψ window")
save(f, "mc_bshape")


# ---- the template fits, with pulls
def tfit(cat, key, title):
    F = T["fits"][key]
    H = T["hists"][f"{cat}_all"]
    h, eh = (np.asarray(a, float) for a in H["data"])
    P, eP = (np.asarray(a, float) for a in T["templates"]["prompt_1"])
    B, eB = (np.asarray(a, float) for a in T["templates"][f"bdecay_{cat}"])
    mp, mb = F["N_p"] * P, F["N_b"] * B
    mt = mp + mb
    f, ax, axp = ratio_canvas()
    step(f, ET, mt / bt, SERIES["total"], "fit", width=2.0)
    step(f, ET, mb / bt, SERIES["s1"], f"b-hadron decay: {F['N_b']:.0f} ± {F['N_b_err']:.0f}", dash="dash")
    step(f, ET, mp / bt, SERIES["s2"], f"prompt J/ψ: {F['N_p']:.0f} ± {F['N_p_err']:.0f}", dash="dot", width=2.0)
    data_points(f, ct, h / bt, eh / bt, label="Data, window − sidebands", ex=bt / 2)
    yrange(f, max(np.max((h + eh) / bt), np.max(mt / bt)), min(np.min((h - eh) / bt), 0.0))
    zero_line(f)
    label(f)
    ylabel(f, "J/ψ / mm", chars=6)
    legend(f, f"{title}<br>χ<sup>2</sup> / ndf = {F['chi2']:.1f} / {F['ndf']}")
    sig = np.sqrt(eh ** 2 + (F["N_b"] * eB) ** 2 + (F["N_p"] * eP) ** 2)
    draw_pull(f, axp, ct, h, mt, ET[0], ET[-1], XL, err=sig)
    return f


save(tfit("ttj", "ttj_all_nbcon", f"μ + jets, e + jets; {NB1}<br>J/ψ in the other {TT} jets; b-decay yield constrained"), "mc_tfit_ttj")
save(tfit("signal", "signal_all_w1", f"{SEL}<br>J/ψ outside the {TT} jets"), "mc_tfit_signal")


# ---- closure: the unbinned fit on the simulation as pseudo-data, l_xy projection, against the truth
CLS = json.load(open(os.path.join(HERE, "mc", "splot.json")))
PJ = CLS["proj"]
EP = np.asarray(PJ["lxy_edges"], float)
cp_, bp = 0.5 * (EP[1:] + EP[:-1]), float(EP[1] - EP[0])
for cat, key, where in (("signal", "notb", f"J/ψ outside the {TT} jets"), ("ttj", "ttj", f"J/ψ in the other {TT} jets (vetoed)")):
    P_ = PJ[key]
    FI = CLS["fits"][key]
    CC = C["closure"][cat]
    d = np.asarray(P_["data_l"], float)
    comp = {s_: np.asarray(P_[f"{s_}_l"], float) for s_ in ("prompt", "nonprompt", "cont")}
    tot = comp["prompt"] + comp["nonprompt"] + comp["cont"]
    tp = np.asarray(C["closure_truth_l"][cat]["prompt"], float)
    f, ax, axp = ratio_canvas()
    fill_between(f, EP, np.zeros(len(EP)), np.append(tp, tp[-1]), SERIES["s2"], f"MC truth: prompt J/ψ, {CC['truth_prompt']:.0f}", shape="hv")
    step(f, EP, comp["cont"], SERIES["comb"], "fit: continuum", dash="dash", width=1.5)
    step(f, EP, comp["nonprompt"], SERIES["s1"], f"fit: b decays (b-jet shape), {FI['N'][1]:.0f}", dash="dashdot")
    step(f, EP, comp["prompt"], SERIES["s2"], f"fit: prompt J/ψ, {FI['N'][0]:.0f} ± {FI['N_err'][0]:.0f}", dash="dot", width=2.0)
    step(f, EP, tot, SERIES["total"], "fit", width=2.0)
    kk = d > 0
    data_points(f, cp_[kk], d[kk], np.sqrt(d[kk]), label="CMS simulation as pseudo-data", ex=np.full(int(kk.sum()), bp / 2))
    f.update_layout(yaxis=dict(type="log", range=[np.log10(0.5), np.log10(600 * d.max())], minor=dict(ticks="", showgrid=False)))
    label(f, rlabel="13.6 TeV", sim=True)
    ylabel(f, f"J/ψ candidates / {bp:g} mm", chars=5)
    legend(f, f"CMS {TT} simulation (POWHEG + Pythia 8); {SEL}<br>{where}; 2.6 &lt; m(μμ) &lt; 3.6 GeV; unbinned fit")
    draw_pull(f, axp, cp_, d, tot, -0.3, 1.2, XL)
    save(f, f"mc_closure_{cat}")

# ---- per channel: J/psi outside the ttbar jets
CHS = (("mujets", "μ + jets"), ("ejets", "e + jets"), ("emu", "e μ"))
FL = json.load(open(os.path.join(HERE, "fit_lxy.json")))
MEd = np.asarray(C["mass_edges"], float)
mbw, mc_ = float(MEd[1] - MEd[0]), 0.5 * (MEd[1:] + MEd[:-1])
TR3 = TRUTH + (COMB,)
for ch, chn in CHS:
    sel = f"{chn}; {NB1}; {Z0}<br>J/ψ outside the {TT} jets"
    # m(mu mu), all l_xy of the fit: data against the simulation by truth
    MM = C["mass"][f"signal_{ch}"]
    f, ax, axr = ratio_canvas()
    comps = [np.asarray(MM[f"mc_{t}"][0], float) for t, _, _ in TR3]
    errs = [np.asarray(MM[f"mc_{t}"][1], float) for t, _, _ in TR3]
    comps, errs, cols, labs = with_bkg(MM, comps, errs, [col for _, _, col in TR3], [nm for _, nm, _ in TR3], 1.0)
    stack(f, MEd, comps, cols, labs)
    hd, ed = (np.asarray(a, float) for a in MM["data"])
    data_points(f, mc_, hd, ed, label="Data", ex=np.full(len(mc_), mbw / 2))
    tot, etot = sum(comps), np.sqrt(sum(e ** 2 for e in errs))
    headroom(f, max(np.max(hd + ed), np.max(tot)), factor=1.7)
    label(f)
    ylabel(f, bin_label(mbw, "GeV", "J/ψ candidates"), chars=4)
    legend(f, f"{sel}; {SIMN}")
    draw_ratio(f, axr, mc_, hd, ed, tot, etot, edges=MEd, lo=MEd[0], hi=MEd[-1], xlab="m(μ<sup>+</sup>μ<sup>−</sup>) [GeV]", ylab="data / sim.", rng=(0.0, 3.0),
               ex=np.full(len(mc_), mbw / 2))
    save(f, f"chan_mass_{ch}")
    # l_xy, window - sidebands: data against the simulation by truth
    L = C["lxy"][f"signal_{ch}"]
    f, ax, axr = ratio_canvas()
    comps = [np.asarray(L[f"mc_{t}"][0], float) / bw for t, _, _ in TRUTH]
    errs = [np.asarray(L[f"mc_{t}"][1], float) / bw for t, _, _ in TRUTH]
    comps, errs, cols, labs = with_bkg(L, comps, errs, [col for _, _, col in TRUTH], [nm for _, nm, _ in TRUTH], bw)
    stack(f, E, comps, cols, labs)
    cb, ecb = (np.asarray(a, float) / bw for a in L[f"mc_{COMB[0]}"])
    step(f, E, cb, COMB[2], COMB[1], dash="dash", width=1.5)
    hd, ed = (np.asarray(a, float) / bw for a in L["data"])
    data_points(f, c, hd, ed, label="Data", ex=bw / 2)
    tot, etot = sum(comps) + cb, np.sqrt(sum(e ** 2 for e in errs) + ecb ** 2)
    yrange(f, max(np.max(hd + ed), np.max(tot)), min(np.min(hd - ed), np.min(tot), 0.0))
    zero_line(f)
    label(f)
    ylabel(f, "J/ψ / mm", chars=6)
    legend(f, f"{sel}<br>window − sidebands; {SIMN}")
    draw_ratio(f, axr, c, hd, ed, tot, etot, edges=E, lo=E[0], hi=E[-1], xlab=XL, ylab="data / sim.", rng=(-1.0, 4.0), ex=bw / 2)
    save(f, f"chan_lxy_{ch}")
    # the l_xy fit of the channel (N and f_prompt free, the shapes of the combined fit)
    FC = FL["channels"][ch]
    e_ = np.asarray(FC["edges"], float)
    w_, c2 = np.diff(e_), 0.5 * (e_[1:] + e_[:-1])
    dd, de = np.asarray(FC["data"], float), np.asarray(FC["data_err"], float)
    pr, npr = np.asarray(FC["prompt"], float), np.asarray(FC["nonprompt"], float)
    f, ax, axp = ratio_canvas()
    step(f, e_, (pr + npr) / w_, SERIES["total"], "fit", width=2.0)
    step(f, e_, npr / w_, SERIES["s1"], f"b decays, b-jet shape: {FC['N'] * (1 - FC['fp']):.0f}", dash="dash")
    step(f, e_, pr / w_, SERIES["s2"], f"prompt J/ψ: f = {100 * FC['fp']:.0f} <sup>+{100 * FC['fp_minos'][1]:.0f}</sup><sub>−{-100 * FC['fp_minos'][0]:.0f}</sub>% of {FC['N']:.0f} ± {FC['N_err']:.0f}",
         dash="dot", width=2.0)
    data_points(f, c2, dd / w_, de / w_, label="Data, window − sidebands", ex=w_ / 2)
    yrange(f, max(np.max((dd + de) / w_), np.max((pr + npr) / w_)), min(np.min((dd - de) / w_), 0.0))
    zero_line(f)
    label(f)
    ylabel(f, "J/ψ / mm", chars=6)
    legend(f, f"{sel}<br>ℓ<sub>xy</sub> fit of the channel")
    draw_pull(f, axp, c2, dd, pr + npr, e_[0], e_[-1], XL, err=de)
    save(f, f"chan_lxyfit_{ch}")

# m_T of the prompt J/psi events per l + jets channel (QCD with a fake lepton would sit at low m_T), against the b decays
SPD = json.load(open(os.path.join(HERE, "splot.json")))
EM = np.asarray(SPD["vars"]["lep_mT"]["edges"], float)
wm, cm_ = np.diff(EM), 0.5 * (EM[1:] + EM[:-1])
f, ax, axr = ratio_canvas()
ref = sum(np.asarray(SPD["hists"][f"notb_nonprompt_lep_mT_{ch}"][0], float) for ch in ("mujets", "ejets"))
eref = np.sqrt(sum(np.asarray(SPD["hists"][f"notb_nonprompt_lep_mT_{ch}"][1], float) ** 2 for ch in ("mujets", "ejets")))
nr = ref.sum()
step(f, EM, ref / nr / wm, SERIES["s1"], "b decays (sPlot), μ + jets and e + jets", width=2.0)
zero_line(f)
for k, (ch, chn, col, sym, dx) in enumerate((("mujets", "μ + jets", SERIES["data"], "circle", -0.12), ("ejets", "e + jets", SERIES["comb"], "square", 0.12))):
    h, eh = (np.asarray(a, float) for a in SPD["hists"][f"notb_prompt_lep_mT_{ch}"])
    n_ = h.sum()
    data_points(f, cm_ + dx * wm, h / n_ / wm, eh / n_ / wm, label=f"prompt J/ψ (sPlot), {chn}", ex=wm / 2, color=col, symbol=sym)
    draw_ratio(f, axr, cm_ + dx * wm, h / n_, eh / n_, ref / nr, eref / nr, edges=EM, lo=EM[0], hi=EM[-1], xlab=SPD["vars"]["lep_mT"]["label"],
               ylab="prompt / b decay", rng=(-1.0, 4.0), color=col, symbol=sym, band=(k == 0))
f.update_layout(yaxis=dict(range=[-0.004, 0.03]))
label(f)
ylabel(f, "fraction / GeV", chars=5)
legend(f, f"{LJ_SEL}<br>J/ψ outside the {TT} jets; unit area")
save(f, "chan_mt")

# prompt and b-decay J/psi per channel: data against the simulation
f, ax, axr = ratio_canvas()
RC = [C["rates"][f"signal_{ch}"] for ch, _ in CHS]
x0 = np.arange(len(CHS), dtype=float)
for key, dx, col, nm in (("mc_b decay", -0.21, SERIES["s1"], f"CMS {TT} sim.: b-hadron decay"), ("mc_prompt", 0.21, SERIES["s2"], f"CMS {TT} sim.: prompt J/ψ")):
    xs, ys = [], []
    for i, r in enumerate(RC):
        xs += [i + dx - 0.17, i + dx + 0.17, None]
        ys += [sim(r, key)[0]] * 2 + [None]
    f.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=nm, line=dict(color=col, width=pt2px(3)), hoverinfo="y+name"))
for key, mk, dx, sym, nm in (("data_nonprompt", "mc_b decay", -0.21, "circle", "Data (sPlot): b-hadron decay"),
                             ("data_prompt", "mc_prompt", 0.21, "diamond", "Data (sPlot): prompt J/ψ")):
    y, ey = np.array([r[key][0] for r in RC]), np.array([r[key][1] for r in RC])
    data_points(f, x0 + dx, np.maximum(y, 0.1), np.minimum(ey, np.maximum(y, 0.1) * 0.95), label=nm, symbol=sym)
    m, em = np.array([sim(r, mk)[0] for r in RC]), np.array([sim(r, mk)[1] for r in RC])
    draw_ratio(f, axr, x0 + dx, y, ey, m, em, xlab="channel", ylab="data / sim.", symbol=sym, band=(sym == "circle"))
f.update_layout(yaxis2=dict(type="log", range=[-1.0, 1.7], tickvals=[0.1, 0.3, 1, 3, 10, 30], ticktext=["0.1", "0.3", "1", "3", "10", "30"], minor=dict(ticks="", showgrid=False)),
                yaxis=dict(type="log", range=[0, 3.5], tickvals=[1, 10, 100, 1000], ticktext=["1", "10", "100", "1000"], minor=dict(ticks="", showgrid=False)))
f.update_xaxes(range=[-0.6, len(CHS) - 0.4], tickvals=list(x0), ticktext=[n for _, n in CHS])
label(f)
ylabel(f, "J/ψ", chars=5)
legend(f, f"{NB1}; {Z0}; J/ψ outside the {TT} jets<br>{SIMN}")
save(f, "chan_rates")

# ---- generators and tunes (gen_variants.json, when the variants have been processed)
GVJ = os.path.join(HERE, "gen_variants.json")
GV = json.load(open(GVJ)) if os.path.exists(GVJ) else {"gen": {}}
if len(GV["gen"]) > 1:
    gens = list(GV["gen"])
    xg = np.arange(len(gens), dtype=float)
    # one-lepton-charge samples (Sherpa default tune: l+ only, AHADIC tune: l- only) x 2 for the comparison with both charges in data
    QF = np.array([2.0 if len({("Lplus" in ds, "Lminus" in ds) for ds in GV["datasets"].get(g, [])}) == 1 and any(k in ds for ds in GV["datasets"].get(g, []) for k in ("Lplus", "Lminus"))
                   else 1.0 for g in gens])
    SHORTV = {"Pythia 8 CP5": "CP5", "CP5 CR1": "CR1", "CP5 CR2": "CR2", "CP5 ERD on": "ERD", "Sherpa": "Sherpa<br>def.", "Sherpa AHADIC": "Sherpa<br>AHADIC", "MG5 MLM": "MLM", "aMC@NLO FxFx": "FxFx"}
    for cat, where in (("signal", f"J/ψ outside the {TT} jets"), ("inb", "J/ψ in a b jet")):
        f, ax, axr = ratio_canvas()
        ymax = 1.0
        for sp, dsp, col, sym, dx, nm in (("bdecay", "nonprompt", SERIES["s1"], "square", -0.12, "b-hadron decay"), ("prompt", "prompt", SERIES["s2"], "circle", 0.12, "prompt J/ψ")):
            d, ed = GV["data"][cat][dsp]
            f.add_shape(type="rect", xref="x domain", yref="y", x0=0, x1=1, y0=max(d - ed, 1e-3), y1=d + ed, fillcolor="rgba(11,11,11,0.12)" if sp == "prompt" else "rgba(235,104,52,0.15)",
                        line_width=0, layer="below")
            f.add_shape(type="line", xref="x domain", yref="y", x0=0, x1=1, y0=d, y1=d, line=dict(color=SERIES["data"] if sp == "prompt" else col, width=pt2px(1.2), dash="dash"))
            f.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Data (sPlot): {nm}", line=dict(color=SERIES["data"] if sp == "prompt" else col, width=pt2px(1.2), dash="dash")))
            v = np.array([GV["gen"][g][f"{cat}_{sp}_full"] for g in gens]) * QF[:, None]
            ymax = max(ymax, d + ed, np.nanmax(v[:, 0] + v[:, 1]))
            data_points(f, xg + dx, v[:, 0], v[:, 1], label=f"CMS sim., MC truth: {nm}", color=col, symbol=sym)
            draw_ratio(f, axr, xg + dx, np.full(len(gens), d), np.full(len(gens), ed), v[:, 0], v[:, 1], xlab="", ylab="data / sim.", color=col, symbol=sym,
                       band=(sp == "prompt"))
        top = max(4.6, (np.log10(ymax) + 0.15) / 0.56)  # the 7-line legend takes the top 44% of the log axis
        tv = [10.0 ** k for k in range(int(top) + 1)]
        f.update_layout(yaxis=dict(type="log", range=[0, top], tickvals=tv, ticktext=[f"{t:g}" if t < 1e4 else f"10<sup>{int(np.log10(t))}</sup>" for t in tv], minor=dict(ticks="", showgrid=False)),
                        yaxis2=dict(type="log", range=[-0.7, 2.0], tickvals=[0.3, 1, 3, 10, 30, 100], ticktext=["0.3", "1", "3", "10", "30", "100"], minor=dict(ticks="", showgrid=False)))
        f.update_xaxes(range=[-0.6, len(gens) - 0.4], tickvals=list(xg), ticktext=[SHORTV.get(g, g) for g in gens], tickangle=0)
        label(f)
        ylabel(f, "J/ψ", chars=5)
        legend(f, f"{LJ_SEL}; {where}<br>CMS {TT} samples at σ × 310 fb<sup>−1</sup>, no backgrounds<br>Sherpa default and AHADIC tunes: one lepton charge, × 2")
        save(f, f"mcvar_{cat}")
# ---- the b-decay shape against pT: b jets, other ttbar jets, outside the ttbar jets (MC truth)
if "bshape_pt" in C:
    BP_ = C["bshape_pt"]
    pe = np.asarray(BP_["pt_edges"], float)
    pc_ = np.sqrt(pe[1:] * pe[:-1])
    CATB = (("inb", "b jets", SERIES["s1"], "circle"), ("ttj", f"other {TT} jets (vetoed)", SERIES["s3"], "square"), ("signal", f"outside the {TT} jets", SERIES["s2"], "diamond"))
    f, ax, axr = ratio_canvas()
    for j, (k, nm, col, sym) in enumerate(CATB):
        r = BP_[k]
        ok = np.array(r["n"]) >= 30
        x = pc_ * (1 + 0.03 * (j - 1))
        data_points(f, x[ok], np.array(r["short"])[ok], np.array(r["short_err"])[ok], label=f"{nm} (median p<sub>T</sub> {r['median_pt']:.0f} GeV)", color=col, symbol=sym)
        if k != "inb":
            rb = BP_["inb"]
            okb = ok & (np.array(rb["n"]) >= 30)
            draw_ratio(f, axr, x[okb], np.array(r["short"])[okb], np.array(r["short_err"])[okb], np.array(rb["short"])[okb], np.array(rb["short_err"])[okb],
                       ylab="/ b jets", rng=(0, 6), color=col, symbol=sym, band=(k == "signal"))
    logx(f, 6, 100, ticks=(6, 8, 10, 15, 20, 30, 50, 100))
    f.update_layout(yaxis=dict(range=[0, 0.95]))
    label(f, rlabel="13.6 TeV", sim=True)
    ylabel(f, "share of b decays at |ℓ<sub>xy</sub>| &lt; 0.06 mm", chars=4)
    xlabel(f, "p<sub>T</sub>(J/ψ) [GeV]")
    legend(f, f"CMS {TT} simulation, MC truth b-hadron decays, J/ψ window<br>{SEL}")
    save(f, "mc_bshape_pt")
    # the b-jet shape reweighted to the signal's pT spectrum, against the signal's own
    LB = np.asarray(BP_["lxy_edges"], float)
    lw, lc = np.diff(LB), 0.5 * (LB[1:] + LB[:-1])
    ps_, es_ = (np.asarray(BP_["signal"][k], float) for k in ("shape", "shape_err"))
    f, ax, axr = ratio_canvas()
    for k, nm, col, dash in (("inb", "b jets", SERIES["s1"], None), ("inb_rew", "b jets, reweighted to the signal's p<sub>T</sub>", SERIES["comb"], "dash")):
        pk = np.asarray(BP_[k]["shape"], float)
        step(f, LB, pk / lw, col, f"{nm} (χ<sup>2</sup> {BP_[k]['chi2_vs_signal']:.0f} / {len(lc)})", dash=dash, width=2.0)
        draw_ratio(f, axr, lc, ps_, es_, pk, None, lo=LB[0], hi=LB[-1], xlab=XL, ylab=f"outside {TT} jets / b jets", rng=(0, 4), color=col,
                   symbol="circle" if k == "inb" else "square", band=(k == "inb"))
    data_points(f, lc, ps_ / lw, es_ / lw, label=f"outside the {TT} jets", ex=lw / 2, color=SERIES["s2"])
    headroom(f, float(np.max(ps_ / lw)), factor=1.5)
    label(f, rlabel="13.6 TeV", sim=True)
    ylabel(f, "fraction / mm", chars=4)
    legend(f, f"CMS {TT} simulation, MC truth b-hadron decays, J/ψ window<br>{SEL}")
    save(f, "mc_bshape_rew")

# ---- the per-category b-decay templates (btemplate_mc.py -> btemplate.json): MC truth b decays against the fitted model of
# each category, and the b-jet model on the vetoed jets' candidates
BTJ_ = os.path.join(HERE, "btemplate.json")
BT = json.load(open(BTJ_)) if os.path.exists(BTJ_) else None
CATB2 = (("inb", "b jets", SERIES["s1"], "circle"), ("ttj", f"other {TT} jets (vetoed)", SERIES["s3"], "square"), ("signal", f"outside the {TT} jets", SERIES["s2"], "diamond"))
if BT and "edges" in BT:
    EB = np.asarray(BT["edges"], float)
    wb, cbt = np.diff(EB), 0.5 * (EB[1:] + EB[:-1])
    f, ax, axr = ratio_canvas()
    for j, (k, nm, col, sym) in enumerate(CATB2):
        q = BT["cats"][k]
        h, mo = np.asarray(q["hist"], float), np.asarray(q["model"], float)
        n = h.sum()
        step(f, EB, np.maximum(mo / n / wb, 1e-6), col, None, width=1.6)
        if k == "ttj":
            step(f, EB, np.maximum(np.asarray(q["model_bjet"], float) / n / wb, 1e-6), col, "b-jet model, vetoed jets' ℓ<sub>xy</sub> errors", dash="dash", width=1.6)
        x = cbt + (j - 1) * 0.15 * wb
        ok = h > 0
        data_points(f, x[ok], (h / n / wb)[ok], (np.sqrt(h) / n / wb)[ok],
                    label=f"{nm}: f<sub>short</sub> {q['f']:.2f}, λ {q['lam']:.2f} mm", color=col, symbol=sym)
        draw_ratio(f, axr, x[ok], h[ok], np.sqrt(h)[ok], mo[ok], None, lo=EB[0], hi=EB[-1], xlab=XL, ylab="truth / model", rng=(0, 2.5), color=col,
                   symbol=sym, band=(j == 0))
    line(f, [None], [None], "grey", "lines: the fitted model of each category", width=1.6)
    f.update_layout(yaxis=dict(type="log", range=[-4.3, 4.3], tickvals=[1e-3, 1e-2, 0.1, 1, 10, 100], ticktext=["0.001", "0.01", "0.1", "1", "10", "100"],
                               minor=dict(showgrid=False, ticks="")))
    label(f, rlabel="13.6 TeV", sim=True)
    ylabel(f, "fraction / mm", chars=4)
    legend(f, f"CMS {TT} simulation, MC truth b-hadron decays, J/ψ window<br>{SEL}; model: f G + (1 − f) e<sup>−ℓ/λ</sup> ⊗ G, G per candidate")
    save(f, "mc_btmpl_shapes")

# ---- the data fits with the b-jet template and with the per-category template (btmpl/: the fits with TTJ_BTEMPLATE)
BTD = os.path.join(HERE, "btmpl")


def prompt_yields(d):
    """prompt J/psi of each fit, [value, error]: l_xy, binned 2D, unbinned, unbinned with one resolution, vetoed jets (unbinned)"""
    L_, J_, S_ = (json.load(open(os.path.join(d, n))) for n in ("fit_lxy.json", "fit_jpsi.json", "splot.json"))
    p_, e_, s2 = L_["params"], L_["errors"], J_["lt2d"]["samples"]["notb"]
    return [[p_["fp_notb"] * p_["N_notb"], e_["fp_notb"] * p_["N_notb"]], [s2["N_P"], s2["fp_err"] * (s2["N_P"] + s2["N_NP"])],
            [S_["fits"]["notb"]["N"][0], S_["fits"]["notb"]["N_err"][0]], [S_["fits"]["notb_globalres"]["N"][0], S_["fits"]["notb_globalres"]["N_err"][0]],
            [S_["fits"]["ttj"]["N"][0], S_["fits"]["ttj"]["N_err"][0]]]


if os.path.exists(os.path.join(BTD, "splot.json")):
    Y0, Y1 = np.array(prompt_yields(HERE)), np.array(prompt_yields(BTD))
    xf = np.arange(len(Y0), dtype=float)
    f, ax, axr = ratio_canvas()
    for Y, dx, col, sym, nm in ((Y0, -0.1, SERIES["data"], "circle", "b-decay template: the data b jets"),
                                (Y1, 0.1, SERIES["s1"], "square", "per category from the simulation, λ from the data b jets")):
        data_points(f, xf + dx, Y[:, 0], Y[:, 1], label=nm, color=col, symbol=sym)
    draw_ratio(f, axr, xf + 0.1, Y1[:, 0], Y1[:, 1], Y0[:, 0], Y0[:, 1], edges=np.append(xf - 0.5, xf[-1] + 0.5), xlab="", ylab="per cat. / b jets",
               rng=(0.4, 1.6), color=SERIES["s1"], symbol="square")
    f.update_xaxes(range=[-0.6, len(xf) - 0.4], tickvals=list(xf), tickangle=0,
                   ticktext=["ℓ<sub>xy</sub> fit", "binned 2D", "unbinned", "unbinned,<br>1 resolution", "vetoed jets,<br>unbinned"])
    headroom(f, float(np.max(Y0[:, 0] + Y0[:, 1])), factor=1.7)
    label(f)
    ylabel(f, "prompt J/ψ", chars=4)
    legend(f, f"{SEL}<br>prompt J/ψ outside the {TT} jets (last: in the vetoed {TT} jets)", x=0.03, xanchor="left")
    save(f, "mc_btmpl_data")

# ---- the toy closure of the fits: mean fitted against true prompt yield, fit - truth; with the b-jet template (toy_closure.json)
# and with the per-category template (toy_closure_btmpl.json)
FITC = (("lxy", "ℓ<sub>xy</sub> fit", SERIES["s3"], "square"), ("fit2d", "binned 2D fit", SERIES["comb"], "diamond"),
        ("unbinned", "unbinned fit", SERIES["data"], "circle"), ("unbinned_global", "unbinned fit, one resolution", SERIES["s4"], "triangle-up"))
TOYV = {"": ("toy_closure.json", "b-decay template: data b jets"), "_btmpl": ("toy_closure_btmpl.json", "b-decay template per category")}
TCV = {v: json.load(open(os.path.join(HERE, fn))) for v, (fn, _) in TOYV.items() if os.path.exists(os.path.join(HERE, fn))}
for var, TC in TCV.items():
    tmpl = TOYV[var][1]
    for sc, what in (("mc", "b jets as simulated (no prompt J/ψ)"), ("datalike", "prompt J/ψ in b jets as in data (5% of b decays)")):
        pts = [p_ for p_ in TC["points"] if p_["scenario"] == sc]
        if not pts:
            continue
        f, ax, axr = ratio_canvas()
        top = max(max(p_[k]["fit"] + p_[k]["rms"] for k, _, _, _ in FITC) for p_ in pts)
        line(f, [0, 1.15 * top], [0, 1.15 * top], "grey", "fitted = true", dash="dash", width=1.2)
        for j, (k, nm, col, sym) in enumerate(FITC):
            dx = (j - 1.5) * 1.5
            x = np.array([p_[k]["truth"] for p_ in pts]) + dx
            y = np.array([p_[k]["fit"] for p_ in pts])
            rms = np.array([p_[k]["rms"] for p_ in pts])
            data_points(f, x, y, rms, label=nm, color=col, symbol=sym)
            data_points(f, x, [p_[k]["bias"] for p_ in pts], [p_[k]["bias_err"] for p_ in pts], label=None, ax=axr, color=col, symbol=sym, size=3)
        zero_line(f, axr)
        f.update_layout(xaxis=dict(range=[0, 1.1 * top]), xaxis2=dict(range=[0, 1.1 * top]), yaxis=dict(range=[0, 1.8 * top]))
        label(f, rlabel="13.6 TeV", sim=True)
        ylabel(f, "mean fitted prompt J/ψ", chars=4)
        ylabel(f, "fit − truth", ax=axr)
        xlabel(f, "true prompt J/ψ outside the t̅t jets (mean per point)")
        legend(f, f"toys from the CMS {TT} simulation, data-sized, injected prompt J/ψ<br>{what}; {tmpl}<br>{pts[0]['n']} toys per point; bars: toy spread",
               x=0.03, xanchor="left")
        save(f, f"mc_toys{var}_{sc}")
        # the pulls at the injected 100 J/psi against a unit Gaussian
        P100 = next((p_ for p_ in pts if p_["inject"] == 100), None)
        if P100 and "pull_hist" in P100["unbinned"]:
            pe = np.asarray(TC["pull_edges"], float)
            pc, pw = 0.5 * (pe[1:] + pe[:-1]), float(pe[1] - pe[0])
            f, ax, axr = ratio_canvas()
            gx = np.linspace(pe[0], pe[-1], 300)
            n_ = P100["n"]
            line(f, gx, n_ * pw * np.exp(-0.5 * gx ** 2) / np.sqrt(2 * np.pi), "grey", "unit Gaussian", dash="dash", width=1.5)
            gb = n_ * (0.5 * (1 + np.vectorize(math.erf)(pe[1:] / np.sqrt(2))) - 0.5 * (1 + np.vectorize(math.erf)(pe[:-1] / np.sqrt(2))))
            for j, (k, nm, col, sym) in enumerate(FITC):
                h = np.asarray(P100[k]["pull_hist"], float)
                step(f, pe, h, col, f"{nm}: mean {P100[k]['pull_mean']:+.2f}, width {P100[k]['pull_width']:.2f}", width=2.0)
                draw_ratio(f, axr, pc + (j - 1.5) * 0.06, h, np.sqrt(h), gb, None, lo=pe[0], hi=pe[-1], xlab="pull = (fitted − true) / fitted error",
                           ylab="toys / Gaussian", rng=(0, 4), color=col, symbol=sym, band=(j == 0))
            headroom(f, max(max(P100[k]["pull_hist"]) for k, _, _, _ in FITC), factor=2.4)
            label(f, rlabel="13.6 TeV", sim=True)
            ylabel(f, f"toys / {pw:g}", chars=3)
            legend(f, f"toys from the CMS {TT} simulation, {P100['inject']} prompt J/ψ injected (true {P100['unbinned']['truth']:.0f})<br>{what}; {n_} toys<br>{tmpl}")
            save(f, f"mc_toypulls{var}_{sc}")
# the two templates side by side: fit - truth in both scenarios; lower panel: the scenario dependence (data-like - as simulated)
if len(TCV) == 2:
    for k, nm in (("unbinned", "unbinned fit"), ("fit2d", "binned 2D fit")):
        f, ax, axr = ratio_canvas()
        top = 0.0
        for j, (var, col) in enumerate((("", SERIES["data"]), ("_btmpl", SERIES["s1"]))):
            P_ = {(p_["scenario"], p_["inject"]): p_[k] for p_ in TCV[var]["points"]}
            for sc, sym, scn in (("mc", "circle", "b jets as simulated"), ("datalike", "square-open", "prompt in b jets as in data")):
                pts = sorted((q for (s_, _), q in P_.items() if s_ == sc), key=lambda q: q["truth"])
                x = np.array([q["truth"] for q in pts]) + (j - 0.5) * 3 + (0 if sc == "mc" else 1.5)
                data_points(f, x, [q["bias"] for q in pts], [q["bias_err"] for q in pts], label=f"{TOYV[var][1]}; {scn}", color=col, symbol=sym)
                top = max(top, max(abs(q["bias"]) + q["bias_err"] for q in pts), max(q["truth"] for q in pts))
            inj = sorted({i for (s_, i) in P_ if (("mc", i) in P_ and ("datalike", i) in P_)})
            dd = [P_[("datalike", i)]["bias"] - P_[("mc", i)]["bias"] for i in inj]
            de = [np.hypot(P_[("datalike", i)]["bias_err"], P_[("mc", i)]["bias_err"]) for i in inj]
            data_points(f, np.array([P_[("mc", i)]["truth"] for i in inj]) + (j - 0.5) * 3, dd, de, label=None, ax=axr, color=col, symbol="diamond", size=3)
        zero_line(f)
        zero_line(f, axr)
        bmax = max(abs(q[k]["bias"]) + q[k]["bias_err"] for T_ in TCV.values() for q in T_["points"])
        f.update_layout(yaxis=dict(range=[-1.3 * bmax, 1.2 * bmax]), yaxis2=dict(range=[-60, 30]))
        f.update_xaxes(range=[0, 1.1 * top])
        label(f, rlabel="13.6 TeV", sim=True)
        ylabel(f, f"{nm}: fit − truth", chars=4)
        ylabel(f, "data-like − sim.", ax=axr)
        xlabel(f, f"true prompt J/ψ outside the {TT} jets (mean per point)")
        legend(f, f"toys from the CMS {TT} simulation, data-sized, injected prompt J/ψ; error: of the mean", x=0.03, xanchor="left")
        save(f, f"mc_toys_tmplcmp_{k}")

# ---- prompt per b-decay J/psi within each sample: does not depend on the normalisation of the sample
if "ratio" in GV and len(GV["gen"]) > 1:
    # l + jets with the l + jets samples; e mu with the dilepton samples of the same generators
    for sfx, chn, rr2 in (("", "ℓ + jets", ([0, 2.0], [1, 3, 10, 30, 100])), ("|emu", "dilepton", ([-0.7, 2.7], [1, 10, 100]))):
        gens = [g for g in GV["gen"] if f"{g}|signal{sfx}" in GV["ratio"]]
        if not gens or f"data|signal{sfx}" not in GV["ratio"]:
            continue
        xg = np.arange(len(gens), dtype=float)
        f, ax, axr = ratio_canvas()
        for cat, col, sym, dx, nm in (("signal", SERIES["s2"], "circle", -0.12, f"outside the {TT} jets"), ("inb", SERIES["s1"], "square", 0.12, "in b jets")):
            D_ = GV["ratio"][f"data|{cat}{sfx}"]
            f.add_shape(type="rect", xref="x domain", yref="y", x0=0, x1=1, y0=max(D_["r"] - D_["err"], 1e-5), y1=D_["r"] + D_["err"],
                        fillcolor="rgba(27,175,122,0.15)" if cat == "signal" else "rgba(235,104,52,0.15)", line_width=0, layer="below")
            f.add_shape(type="line", xref="x domain", yref="y", x0=0, x1=1, y0=D_["r"], y1=D_["r"], line=dict(color=col, width=pt2px(1.2), dash="dash"))
            f.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Data (sPlot), {nm}", line=dict(color=col, width=pt2px(1.2), dash="dash")))
            rr = [GV["ratio"][f"{g}|{cat}{sfx}"] for g in gens]
            meas = [i for i, r in enumerate(rr) if r["upper"] is None]
            lim = [i for i, r in enumerate(rr) if r["upper"] is not None]
            data_points(f, xg[meas] + dx, [rr[i]["r"] for i in meas], [rr[i]["err"] for i in meas], label=f"CMS sim., MC truth, {nm}", color=col, symbol=sym)
            if lim:
                f.add_trace(go.Scatter(x=xg[lim] + dx, y=[rr[i]["upper"] for i in lim], mode="markers", name=f"90% CL upper limit, {nm}", showlegend=False,
                                       marker=dict(symbol="triangle-down-open", size=pt2px(5), color=col)))
            draw_ratio(f, axr, xg[meas] + dx, np.full(len(meas), D_["r"]), np.full(len(meas), D_["err"]), [rr[i]["r"] for i in meas], [rr[i]["err"] for i in meas],
                       xlab="", ylab="data / sim.", color=col, symbol=sym, band=(cat == "signal"))
        f.update_layout(yaxis=dict(type="log", range=[-4.3, 3.0], tickvals=[1e-4, 1e-3, 1e-2, 0.1, 1], ticktext=["10<sup>−4</sup>", "0.001", "0.01", "0.1", "1"], minor=dict(ticks="", showgrid=False)),
                        yaxis2=dict(type="log", range=rr2[0], tickvals=rr2[1], ticktext=[f"{t:g}" for t in rr2[1]], minor=dict(ticks="", showgrid=False)))
        SHORT = {"Pythia 8 CP5": "CP5", "CP5 CR1": "CR1", "CP5 CR2": "CR2", "CP5 ERD on": "ERD", "Sherpa": "Sherpa<br>def.", "Sherpa AHADIC": "Sherpa<br>AHADIC", "MG5 MLM": "MLM", "aMC@NLO FxFx": "FxFx"}
        f.update_xaxes(range=[-0.6, len(gens) - 0.4], tickvals=list(xg), ticktext=[SHORT.get(g, g) for g in gens], tickangle=0)
        label(f)
        ylabel(f, "prompt J/ψ per b-decay J/ψ", chars=5)
        legend(f, f"{LJ_SEL if not sfx else f'e μ; {NB1}; {Z0}'}; CMS {TT} samples, {chn}<br>within each sample: no normalisation; open triangles: 90% CL upper limits", x=0.03, xanchor="left")
        save(f, "mcvar_ratio" + sfx.replace("|", "_"))
print("figures written to", os.path.join(HERE, "figs"))
