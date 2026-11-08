"""Figures of the veto of the J/psi in the ttbar decay jets (cms-plotting skill via cmsdeck.py; project rules in
hadW_ntrack_ttbar/.claude/cms-plotting):
  figs/assoc_wmass.json  m(jj) of the selection's W pair when the vetoed prompt J/psi sits in one of its jets (sPlot), against
                         the same J/psi jets as an extra jet in other ttbar events and ttbar events in general (wpair_check.json);
                         lower panel: prompt J/psi / extra jet
  figs/assoc_drtt.json   Delta R(J/psi, nearest ttbar jet): prompt J/psi outside the ttbar jets against prompt pileup J/psi;
                         lower panel: signal / pileup
  figs/assoc_gen.json    our private generator-level Pythia 8 + LETO ttbar sample (genstudy/, not a CMS sample, no detector), J/psi
                         from the parton shower (gen_assoc.json): where the J/psi of each origin ends
                         up (b jet, other ttbar jet, a jet outside the ttbar, no selected jet); lower panel: top decay / ISR
usage: python3 make_figs_assoc.py"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cmsdeck  # noqa: E402
from cmsdeck import (CH3, LJ, NB1, SERIES, TT, data_points, draw_ratio, headroom, label, legend, ratio_canvas, region, step,  # noqa: E402
                     ylabel, zero_line)

W = json.load(open(os.path.join(HERE, "wpair_check.json")))
WN = f"{W['window'][0]:.0f}–{W['window'][1]:.0f} GeV"


def rlim(r, re, lo=0.0, hi=2.2):
    """ratio-panel range that keeps every point and its error bar inside"""
    ok = np.isfinite(r) & np.isfinite(re)
    if not ok.any():
        return (lo, hi)
    a, b = np.min((r - re)[ok]), np.max((r + re)[ok])
    return (min(lo, np.floor(2 * a) / 2), max(hi, np.ceil(2 * b) / 2))


# ---- m(jj) of the W pair: fraction per 20 GeV bin
E = np.asarray(W["mjj_edges"], float)
bw = float(E[1] - E[0])
c = 0.5 * (E[1:] + E[:-1])
P = W["wpair"]
f, ax, axr = ratio_canvas()
region(f, W["window"][0], W["window"][1], f"W window, {WN}")
ref = {}
for key, nm, col, dash in (("extra_jet", f"the same J/ψ jets as an extra jet (emulated): {100 * P['extra_jet']['win'][0]:.0f}% in the window", SERIES["s1"], "dash"),
                           ("all_ttbar", f"{TT} events with a b-decay J/ψ (sPlot): {100 * P['all_ttbar']['win'][0]:.0f}%", SERIES["total"], None)):
    h = np.asarray(P[key]["h"], float)
    ref[key] = h / h.sum()
    step(f, E, ref[key], col, nm, dash=dash)
h, he = np.asarray(P["prompt_in_w"]["h"], float), np.asarray(P["prompt_in_w"]["e"], float)
n = h.sum()
y, ye = h / n, he / n
f0, fe = P["prompt_in_w"]["win"]
data_points(f, c, y, ye, label=f"prompt J/ψ in a W-pair jet (sPlot): {100 * f0:.0f} ± {100 * fe:.0f}%", ex=np.full(len(c), bw / 2))
zero_line(f)
top = max(np.max(y + ye), max(r.max() for r in ref.values()))
headroom(f, top, bottom=min(0.0, np.min(y - ye)) - 0.05 * top, factor=1.95)
label(f)
ylabel(f, cmsdeck.bin_label(bw, "GeV", "fraction"), chars=5)
legend(f, f"{LJ}; {NB1}<br>last bin: overflow")
safe = np.where(ref["extra_jet"] > 0, ref["extra_jet"], np.nan)
draw_ratio(f, axr, c, y, ye, ref["extra_jet"], edges=E, lo=E[0], hi=E[-1], xlab="m(jj) of the W pair [GeV]", ylab="prompt / extra jet",
           rng=rlim(y / safe, ye / safe, -1.0, 4.0), ex=np.full(len(c), bw / 2))
cmsdeck.save(f, "assoc_wmass")

# ---- Delta R to the nearest ttbar jet: fraction per unit Delta R (variable bins)
E = np.asarray(W["dr_edges"], float)
bw = np.diff(E)
c = 0.5 * (E[1:] + E[:-1])
f, ax, axr = ratio_canvas()
sh = {}
for k, (key, nm, col, sym) in enumerate((("signal", f"prompt J/ψ outside the {TT} jets (sPlot)", SERIES["data"], "circle"),
                                         ("pileup", "prompt pileup J/ψ (sPlot), another collision", SERIES["comb"], "diamond"))):
    D = W["drtt"][key]
    h, he = np.asarray(D["h"], float), np.asarray(D["e"], float)
    n = h.sum()
    sh[key] = (h / n / bw, he / n / bw)
    xs = c + (k - 0.5) * 0.12 * bw
    data_points(f, xs, *sh[key], label=f"{nm}: {100 * D['near_0p8'][0]:.0f} ± {100 * D['near_0p8'][1]:.0f}% below 0.8", color=col, symbol=sym,
                ex=bw / 2 if k == 0 else None)
zero_line(f)
top = max(np.max(sh[k][0] + sh[k][1]) for k in sh)
headroom(f, top, bottom=min(0.0, min(np.min(sh[k][0] - sh[k][1]) for k in sh)) - 0.05 * top, factor=1.75)
label(f)
ylabel(f, "fraction / unit ΔR", chars=5)
legend(f, f"{CH3}; {NB1}<br>vetoed below 0.4; last bin: overflow")
r, re = sh["signal"][0] / sh["pileup"][0], sh["signal"][1] / sh["pileup"][0]
draw_ratio(f, axr, c, sh["signal"][0], sh["signal"][1], sh["pileup"][0], sh["pileup"][1], edges=E, lo=E[0], hi=E[-1],
           xlab=f"ΔR(J/ψ, nearest {TT} jet)", ylab="signal / pileup", rng=rlim(r, re, 0.0, 2.5), ex=bw / 2)
cmsdeck.save(f, "assoc_drtt")

# ---- generator: where the J/psi of each origin ends up
G = json.load(open(os.path.join(HERE, "gen_assoc.json")))
JC = ["b jet", "other ttbar jet", "jet outside the ttbar", "no selected jet"]
JN = {"b jet": "b jet", "other ttbar jet": f"other {TT} jet", "jet outside the ttbar": "other jet", "no selected jet": "no selected jet"}
ORG = [("top decay", "from a top decay (b or W quark shower)", SERIES["s1"], "square"), ("ISR", "initial-state radiation", SERIES["s2"], "circle"),
       ("top radiation", "radiated by a top before its decay", SERIES["total"], "triangle-up"), ("MPI", "second parton scattering", SERIES["comb"], "diamond")]
f, ax, axr = ratio_canvas()
x0 = np.arange(len(JC))
region(f, 1.5, 3.5, f"signal: outside the {TT} jets", color="rgba(27,175,122,0.10)")
k = 0
frac = {}
for key, nm, col, sym in ORG:
    tab = G["table"].get(key)
    if not tab:
        continue
    nt = sum(tab.values())
    if nt < 5:
        continue
    y = np.array([tab[j] / nt for j in JC])
    ye = np.sqrt(y * (1 - y) / nt)
    frac[key] = (y, ye)
    data_points(f, x0 + (k - 0.5) * 0.16, y, ye, label=f"{nm} ({nt})", color=col, symbol=sym)
    k += 1
headroom(f, 0.8, bottom=-0.03, factor=1.0)
label(f, None, gen=True)  # not a CMS sample: no CMS label; 13.6 TeV in the legend
ylabel(f, "fraction of the J/ψ of each origin", chars=4)
legend(f, f"not a CMS sample: no detector simulation, gen jets<br>13.6 TeV {TT} ℓ + jets, selection and veto as in data<br>prompt J/ψ → μμ (μ p<sub>T</sub> > 3 GeV) from the shower")
(yt, yte), (yi, yie) = frac["top decay"], frac["ISR"]
r = yt / yi
re = r * np.hypot(yte / yt, yie / yi)  # categories have no bin edges, so no band: the point error carries both origins
draw_ratio(f, axr, x0, yt, re * yi, yi, lo=-0.5, hi=len(JC) - 0.5, xlab="where the J/ψ is", ylab="top decay / ISR", rng=rlim(r, re, 0.0, 2.5))
f.update_xaxes(tickvals=list(x0), ticktext=[JN[j] for j in JC])
cmsdeck.save(f, "assoc_gen")
print("figures written to", cmsdeck.OUT)
