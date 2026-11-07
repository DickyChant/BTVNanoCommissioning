"""Canvases for the analysis note that the deck does not show, with cmsdeck (the cms-plotting skill's cmsplot_plotly + the
deck additions):
  figs/an_unbinned_{m,l}_vetoed.json  the unbinned m x l_xy fit of the J/psi in the other ttbar jets (vetoed, jcat 1): mass and
                                      l_xy projections (splot.json proj ttj), pull
  figs/an_samesign.json               m(mu mu) of the J/psi candidates outside the ttbar jets (all vertices): opposite-sign pairs against same-sign
                                      pairs (the combinatorial control); opposite / same sign
  figs/an_signif_years.json           significance of the prompt component per year, l_xy fit and mass x l_xy fit, and all years
  figs/an_leak_z.json                 z = pT(J/psi) / pT(jet) of the vetoed prompt J/psi: data (sWeights) against Pythia 8 + LETO
                                      (gen_leak_z.json), unit area; data / Pythia = the z weights
usage: python3 make_figs_an.py"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cmsdeck import (LJ, NB1, SERIES, TT, Z0, bin_label, canvas, data_points, draw_pull, draw_ratio, label, legend,  # noqa: E402
                     ratio_canvas, save, step, xlabel, ylabel, zero_line)

SP = json.load(open(os.path.join(HERE, "splot.json")))
GZ = json.load(open(os.path.join(HERE, "gen_leak_z.json")))
XL = "ℓ<sub>xy</sub> [mm]"

# ---- the vetoed category: projections of its unbinned fit
PJ, F = SP["proj"], SP["fits"]["ttj"]
for dim, ed, xl in (("m", PJ["mass_edges"], "m(μ<sup>+</sup>μ<sup>−</sup>) [GeV]"), ("l", PJ["lxy_edges"], XL)):
    e = np.asarray(ed, float)
    c, bw = 0.5 * (e[1:] + e[:-1]), float(e[1] - e[0])
    P_ = PJ["ttj"]
    d = np.asarray(P_[f"data_{dim}"], float)
    comp = {s_: np.asarray(P_[f"{s_}_{dim}"], float) for s_ in ("prompt", "nonprompt", "cont")}
    tot = comp["prompt"] + comp["nonprompt"] + comp["cont"]
    f, ax, axp = ratio_canvas()
    step(f, e, comp["cont"], SERIES["comb"], f"continuum: {F['N'][2]:.0f}", dash="dash", width=1.5)
    step(f, e, comp["nonprompt"], SERIES["s1"], f"b decays (b-jet shape): {F['N'][1]:.0f} ± {F['N_err'][1]:.0f}", dash="dashdot")
    step(f, e, comp["prompt"], SERIES["s2"], f"prompt J/ψ: {F['N'][0]:.0f} ± {F['N_err'][0]:.0f}", dash="dot", width=2.0)
    step(f, e, tot, SERIES["total"], "fit", width=2.0)
    kk = d > 0 if dim == "l" else np.ones(len(d), bool)
    data_points(f, c[kk], d[kk], np.sqrt(d[kk]), label="Data", ex=np.full(int(kk.sum()), bw / 2))
    if dim == "l":
        f.update_layout(yaxis=dict(type="log", range=[np.log10(0.5), np.log10(600 * d.max())], minor=dict(ticks="", showgrid=False)))
        lo_, hi_ = -0.3, 1.2
    else:
        f.update_layout(yaxis=dict(range=[0, 1.8 * max((d + np.sqrt(d)).max(), tot.max())]))
        lo_, hi_ = e[0], e[-1]
    label(f)
    ylabel(f, bin_label(bw, "mm", "J/ψ candidates") if dim == "l" else bin_label(bw, "GeV", "J/ψ candidates"), chars=5)
    legend(f, f"{LJ}; {NB1}; {Z0}<br>J/ψ in the other {TT} jets (vetoed); unbinned fit")
    draw_pull(f, axp, c, d, tot, lo_, hi_, xl)
    save(f, f"an_unbinned_{dim}_vetoed")

# ---- the J/psi momentum share in its jet: data against Pythia 8 + LETO
ze = np.asarray(GZ["z_edges"], float)
ze[-1] = 1.0
zw, zc = np.diff(ze), 0.5 * (ze[1:] + ze[:-1])
dz, edz, pz = (np.asarray(GZ[k], float) for k in ("data_ttj", "data_ttj_err", "pythia_ttj"))
f, ax, axr = ratio_canvas()
step(f, ze, pz / zw, SERIES["s1"], "Pythia 8 + LETO, all origins", width=2.0)
data_points(f, zc, dz / zw, edz / zw, label="Data, prompt J/ψ (sPlot)", ex=zw / 2)
zero_line(f)
f.update_layout(yaxis=dict(range=[-0.3, 1.8 * max(np.max((dz + edz) / zw), np.max(pz / zw))]))
label(f)
ylabel(f, "fraction / unit z", chars=4)
legend(f, f"{LJ}; {NB1}; {Z0}<br>prompt J/ψ in the other {TT} jets (vetoed); unit area; last bin: z ≥ 0.6")
draw_ratio(f, axr, zc, dz, edz, pz, None, lo=0.0, hi=1.0, xlab="z = p<sub>T</sub>(J/ψ) / p<sub>T</sub>(jet)", ylab="data / Pythia", rng=(-0.5, 4.0), ex=zw / 2)
save(f, "an_leak_z")
# ---- same-sign check: opposite- against same-sign pairs, J/psi outside the ttbar jets, all vertices, N_b >= 1
R3 = json.load(open(os.path.join(HERE, "results_run3.json")))
CHS = list(R3["channels"])
me = np.asarray(R3["channels"][CHS[0]]["mass_edges"], float)
mw, mc = np.diff(me), 0.5 * (me[1:] + me[:-1])
hos = sum(np.asarray(R3["channels"][c]["spectra"]["data_os_notb_ge1b"], float) for c in CHS)
hss = sum(np.asarray(R3["channels"][c]["spectra"]["data_ss_notb_ge1b"], float) for c in CHS)
yos = [sum(R3["channels"][c]["yields"][f"data_{k}_notb_ge1b"]["n"] for c in CHS) for k in ("os", "ss")]
eos = [float(np.sqrt(sum(R3["channels"][c]["yields"][f"data_{k}_notb_ge1b"]["err"] ** 2 for c in CHS))) for k in ("os", "ss")]
f, ax, axr = ratio_canvas()
data_points(f, mc, hos, np.sqrt(hos), label=f"opposite sign: window − sidebands {yos[0]:.0f} ± {eos[0]:.0f}", ex=mw / 2)
data_points(f, mc, hss, np.sqrt(hss), label=f"same sign: window − sidebands {yos[1]:.0f} ± {eos[1]:.0f}", ex=mw / 2, color=SERIES["comb"], symbol="square")
f.update_layout(yaxis=dict(range=[0, 1.6 * float(np.max(hos + np.sqrt(hos)))]))
label(f)
ylabel(f, bin_label(float(mw[0]), "GeV", "candidates"), chars=4)
legend(f, f"μ + jets, e + jets, e μ; {NB1}; all vertices<br>J/ψ candidates outside the {TT} jets")
draw_ratio(f, axr, mc, hos, np.sqrt(hos), hss, np.sqrt(hss), edges=me, lo=me[0], hi=me[-1], xlab="m(μμ) [GeV]", ylab="OS / SS", rng=(0.0, 4.0), ex=mw / 2)
save(f, "an_samesign")

# ---- the prompt significance per year: l_xy fit and mass x l_xy fit
YS = ("2022", "2023", "2024", "2025", "2026")
FLA, FJA = json.load(open(os.path.join(HERE, "fit_lxy.json"))), json.load(open(os.path.join(HERE, "fit_jpsi.json")))
sl = [json.load(open(os.path.join(HERE, "years", f"fit_lxy_{y}.json")))["signif"]["nominal"] for y in YS]
s2 = [json.load(open(os.path.join(HERE, "years", f"fit_jpsi_{y}.json")))["lt2d"]["signif_prompt_notb"] for y in YS]
x = np.arange(len(YS), dtype=float)
f = canvas()
data_points(f, x - 0.1, sl, np.zeros(len(YS)), label=f"ℓ<sub>xy</sub> fit; all years {FLA['signif']['nominal']:.1f}σ", size=4)
data_points(f, x + 0.1, s2, np.zeros(len(YS)), label=f"m × ℓ<sub>xy</sub> fit; all years {FJA['lt2d']['signif_prompt_notb']:.1f}σ", color=SERIES["comb"], symbol="square", size=4)
for v, col in ((FLA["signif"]["nominal"], SERIES["data"]), (FJA["lt2d"]["signif_prompt_notb"], SERIES["comb"])):
    f.add_shape(type="line", xref="x domain", yref="y", x0=0, x1=1, y0=v, y1=v, line=dict(color=col, width=1.5, dash="dash"))
f.update_xaxes(range=[-0.6, len(YS) - 0.4], tickvals=list(x), ticktext=list(YS))
f.update_layout(yaxis=dict(range=[0, 1.6 * max(max(sl), max(s2), FJA["lt2d"]["signif_prompt_notb"])]))
label(f)
ylabel(f, "significance of the prompt J/ψ [σ]", chars=3)
xlabel(f, "year")
legend(f, f"μ + jets, e + jets, e μ; {NB1}; {Z0}<br>J/ψ outside the {TT} jets; dashed: all years")
save(f, "an_signif_years")
print("figures written to", os.path.join(HERE, "figs"))
