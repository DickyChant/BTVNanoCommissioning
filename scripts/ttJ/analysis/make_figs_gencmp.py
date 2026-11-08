"""Figures of data against our private generator-level Pythia 8 + LETO sample (genstudy/: not a CMS sample, no detector
simulation; gen_compare.py -> gen_compare.json), cms-plotting skill via cmsdeck.py. Every legend and ratio label names it as
generator level, apart from the CMS simulation of the other figures:
  figs/gencmp_yields.json  prompt J/psi in the other ttbar jets (vetoed) and outside the ttbar jets (signal), l + jets: data
                           against Pythia (default LDMEs, normalised with the b-decay J/psi) by origin, and DPS at sigma_eff = 15 mb;
                           lower panel data / (Pythia + DPS)
  figs/gencmp_pt_ttj.json  pT of the vetoed prompt J/psi; Pythia scaled to their yield; lower panel data / Pythia
  figs/gencmp_{pt,drtt,dy,dphi}.json   the signal: pT, Delta R to the nearest ttbar jet, |Delta y| and |Delta phi| to the ttbar;
                           Pythia scaled as the vetoed J/psi, plus DPS (the prompt pileup J/psi of the data); lower panel data / pred.
Stack: Pythia nominal; dashed: Pythia with the J/psi momentum share z of the data (gen_leak_z.py), plus DPS (open diamonds in
the lower panels).
usage: python3 make_figs_gencmp.py"""
import json
import os
import sys
import warnings

import numpy as np
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cmsdeck  # noqa: E402
from cmsdeck import (LJ, NB1, SERIES, TT, data_points, draw_ratio, fill_between, headroom, label, legend, logx, ratio_canvas,  # noqa: E402
                     save, step, xlabel, ylabel, zero_line)
from plotlyhep._units import pt2px  # noqa: E402

C = json.load(open(os.path.join(HERE, "gen_compare.json")))
H, Y, SC = C["hists"], C["yields"], C["scaled"]
ORIG = (("top decay", "gen. Pythia + LETO: top decay", SERIES["s1"]), ("ISR", f"gen. Pythia + LETO: {TT} production (ISR)", SERIES["s2"]))
GN = "Pythia 8 + LETO (gen level, private)"  # the legend header: our generator-level sample, not the CMS simulation
SE = C["dps"]["sigma_eff_mb"]
DPSN = f"2nd scattering, σ<sub>eff</sub> = {SE} mb (pileup J/ψ)"
DATAN = "Data, prompt J/ψ (sPlot)"


def ratios(f, axr, x, hd, ed, series, edges=None):
    """data / each prediction in the lower panel: series = [(den, den_err or None, colour, symbol, x shift)], the first with
    the grey band of its error and the line at 1. Range 0-2.2, widened in steps of 0.5 up to [-0.5, 3] to hold the central
    values; points still outside are drawn as triangles at the panel edge (no error bar)."""
    rs = []
    for den, _, _, _, _ in series:
        den = np.asarray(den, float)
        rs.append(np.asarray(hd, float) / np.where(den != 0, den, np.nan))
    allr = np.concatenate([r[np.isfinite(r)] for r in rs]) if any(np.isfinite(r).any() for r in rs) else np.array([1.0])
    lo = max(-0.5, min(0.0, np.floor(2 * allr.min()) / 2))
    hi = min(3.0, max(2.2, np.ceil(2 * allr.max()) / 2))
    span = hi - lo
    for k, ((den, den_e, col, sym, dx), r) in enumerate(zip(series, rs)):
        out = np.isfinite(r) & ((r < lo) | (r > hi))
        num = np.where(out, np.nan, hd)
        draw_ratio(f, axr, np.asarray(x, float) + dx, num, ed, den, den_e, edges=edges, ylab="data / pred.", band=k == 0,
                   color=col if k else None, symbol=sym)
        if out.any():
            xo = (np.asarray(x, float) + dx)[out]
            yo = np.where(r[out] > hi, hi - 0.06 * span, lo + 0.06 * span)
            so = np.where(r[out] > hi, "triangle-up", "triangle-down")
            f.add_trace(go.Scatter(x=xo, y=yo, mode="markers", xaxis=axr["xaxis"], yaxis=axr["yaxis"], showlegend=False, hoverinfo="skip",
                                   marker=dict(symbol=list(so), size=pt2px(5), color=col or SERIES["data"])))
    f.update_layout(yaxis2=dict(range=[lo, hi]))


# ---- yields: stacked bars per category, data points; lower panel data / (Pythia + 2nd scattering)
f, ax, axr = ratio_canvas()
cats = (("ttj", f"other {TT} jets (vetoed)"), ("signal", f"outside the {TT} jets (signal)"))
x0 = np.arange(len(cats), dtype=float)
dps_y = np.array([0.0, C["dps"]["n_lj"][SE]])
for k, (on, nm, col) in enumerate(ORIG):
    f.add_trace(go.Bar(x=x0, y=[Y[f"pythia_nominal_{c}_{on}"][0] for c, _ in cats], name=nm + ", default LDMEs", marker=dict(color=col, line=dict(width=0)),
                       width=0.5, hoverinfo="y+name", legendrank=10 + k, xaxis="x", yaxis="y"))
f.add_trace(go.Bar(x=x0, y=dps_y, name=DPSN, marker=dict(color=SERIES["comb"], line=dict(width=0)), width=0.5, hoverinfo="y+name", legendrank=12, xaxis="x", yaxis="y"))
pred = np.array([sum(Y[f"pythia_nominal_{c}_{on}"][0] for on, _, _ in ORIG) for c, _ in cats]) + dps_y
pred_e = np.array([np.sqrt(sum(Y[f"pythia_nominal_{c}_{on}"][1] ** 2 for on, _, _ in ORIG)) for c, _ in cats])
zt = np.array([sum(Y[f"pythia_z_{c}_{on}"][0] for on, _, _ in ORIG) for c, _ in cats]) + dps_y
ZN = "gen. Pythia + LETO, J/ψ z as in data (+ 2nd scattering)"
f.add_trace(go.Scatter(x=x0, y=zt, mode="markers", name=ZN, legendrank=50, xaxis="x", yaxis="y",
                       marker=dict(symbol="diamond-open", size=pt2px(7), color=SERIES["total"], line=dict(width=pt2px(1.2)))))
d, de = np.array([Y[f"data_{c}"][0] for c, _ in cats]), np.array([Y[f"data_{c}"][1] for c, _ in cats])
data_points(f, x0, d, de, label=DATAN)
f.data[-1].legendrank = 1
f.update_layout(barmode="stack", bargap=0.5)
headroom(f, max(np.max(d + de), np.max(pred), np.max(zt)), factor=1.95)
for x, p, e in zip(x0, pred, pred_e):  # the Pythia statistical error as a band per category
    f.add_shape(type="rect", xref="x2", yref="y2", x0=x - 0.25, x1=x + 0.25, y0=1 - e / p, y1=1 + e / p, fillcolor="rgba(0,0,0,0.15)", line_width=0, layer="below")
ratios(f, axr, x0, d, de, [(pred, None, None, "circle", 0.0), (zt, None, SERIES["total"], "diamond-open", 0.08)])
f.update_xaxes(range=[-0.6, len(cats) - 0.4], tickvals=list(x0), ticktext=[c[1] for c in cats])
label(f)
ylabel(f, "prompt J/ψ", chars=4)
xlabel(f, "where the prompt J/ψ is")
legend(f, f"{GN}: normalised with the b-decay J/ψ<br>{LJ}; {NB1}")
save(f, "gencmp_yields")


# ---- distributions: data points against the stack (Pythia nominal x its scale to the vetoed J/psi, + DPS for the signal);
# lower panel data / stack (points) and data / the z variant (open diamonds)
def dist(name, cat, v, xlab, ylab, xr=None, chars=5, log=False):
    e = np.asarray(C["vars"][v], float)
    bw = np.diff(e)
    c = 0.5 * (e[1:] + e[:-1]) if not log else np.sqrt(e[1:] * e[:-1])
    lo_, hi_ = xr or (e[0], e[-1])
    f, ax, axr = ratio_canvas()
    sn, sz = SC["nominal"]["scale"], SC["z"]["scale"]
    comps = [np.asarray(H[f"pythia_nominal_{cat}_{on}_{v}"][0], float) * sn / bw for on, _, _ in ORIG]
    cerr = [np.asarray(H[f"pythia_nominal_{cat}_{on}_{v}"][1], float) * sn / bw for on, _, _ in ORIG]
    labs = [f"{nm} (× {sn:.2f})" for _, nm, _ in ORIG]
    cols = [col for _, _, col in ORIG]
    ztot = sum(np.asarray(H[f"pythia_z_{cat}_{on}_{v}"][0], float) for on, _, _ in ORIG) * sz / bw
    if cat == "signal":
        comps.append(np.asarray(H[f"dps_{v}"][0], float) / bw)
        cerr.append(np.asarray(H[f"dps_{v}"][1], float) / bw)
        labs.append(DPSN)
        cols.append(SERIES["comb"])
        ztot = ztot + comps[-1]
    base = np.zeros(len(bw))
    for k, (y, nm, col) in enumerate(zip(comps, labs, cols)):  # stacked in slot order
        fill_between(f, e, np.append(base, base[-1]), np.append(base + y, (base + y)[-1]), col, nm, shape="hv")
        f.data[-1].legendrank = 10 + k
        base = base + y
    tot, tot_e = base, np.sqrt(sum(x ** 2 for x in cerr))
    zn = f"gen. Pythia + LETO, J/ψ z as in data (× {sz:.2f})" + (" + 2nd scattering" if cat == "signal" else "")
    step(f, e, ztot, SERIES["total"], zn, dash="dash")
    f.data[-1].legendrank = 50
    hd, ed = (np.asarray(a, float) / bw for a in H[f"data_{cat}_{v}"])
    zero_line(f)
    data_points(f, c, hd, ed, label=DATAN, ex=None if log else bw / 2)
    f.data[-1].legendrank = 1
    top = max(np.max(hd + ed), np.max(tot), np.max(ztot))
    low = min(0.0, np.min(hd - ed))
    headroom(f, top, bottom=low - 0.05 * top if low < 0 else -0.05 * top, factor=1.95)
    # the z variant shifted by 0.15 bin (geometric on log x) so the two ratio series do not overlap
    zx = (np.exp(np.log(c) + 0.15 * np.log(e[1:] / e[:-1])) - c) if log else 0.15 * bw
    ratios(f, axr, c, hd, ed, [(tot, tot_e, None, "circle", 0.0), (ztot, None, SERIES["total"], "diamond-open", zx)], edges=e)
    if log:
        logx(f, lo_, hi_)
    else:
        f.update_xaxes(range=[lo_, hi_])
    label(f)
    ylabel(f, ylab, chars=chars)
    xlabel(f, xlab)
    where = f"outside the {TT} jets" if cat == "signal" else f"in the other {TT} jets"
    legend(f, f"{GN}<br>{LJ}; {NB1}; J/ψ {where}; last bin: overflow")
    save(f, name)


dist("gencmp_pt_ttj", "ttj", "pt", "p<sub>T</sub>(J/ψ) [GeV]", "prompt J/ψ / GeV", log=True)
dist("gencmp_pt", "signal", "pt", "p<sub>T</sub>(J/ψ) [GeV]", "prompt J/ψ / GeV", log=True)
dist("gencmp_drtt", "signal", "drtt", f"ΔR(J/ψ, nearest {TT} jet)", "prompt J/ψ / unit ΔR")
dist("gencmp_dy", "signal", "dy", f"|Δy(J/ψ, {TT})|", "prompt J/ψ / unit |Δy|")
dist("gencmp_dphi", "signal", "dphi", f"|Δφ(J/ψ, {TT} without the J/ψ jet)| [rad]", "prompt J/ψ / rad")
print("figures written to", cmsdeck.OUT)
