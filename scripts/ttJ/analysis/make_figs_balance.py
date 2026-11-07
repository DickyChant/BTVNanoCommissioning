"""Figures of the pT balance of the prompt J/psi against the ttbar system (ttbar_balance.py -> ttbar_balance.json), on cmsdeck
(the cms-plotting skill's cmsplot_plotly + the deck additions); every comparison canvas has a lower panel:
  figs/ttbal_dphi.json    |Delta phi(J/psi, ttbar)|, unit area: prompt and prompt-pileup J/psi (sPlot; the pileup J/psi taken out
                          of the MET, as the recoil from their collision is not in it), the two emulations (the J/psi jet put
                          into other ttbar events: recoiling against the ttbar, or balanced by a partner jet) and the template
                          fit of the prompt J/psi; lower panel: ratio to the 2nd-scattering emulation (the hypothesis tested)
  figs/ttbal_ppar.json    pT(ttbar) . pT-hat(J/psi), unit area per GeV, same samples and lower panel
  figs/ttbal_ppar_pt.json <pT(ttbar) . pT-hat(J/psi)> in bins of pT(J/psi); lower panel: difference to the 2nd-scattering emulation
  figs/ttbal_role.json    which jet holds the J/psi in the ttbar reconstruction (l + jets, J/psi in a jet): b_had, b_lep, W jet,
                          extra; lower panel: the prompt J/psi divided by each emulation
  figs/ttbal_njets.json   selected jets (l + jets): prompt J/psi events against the prompt-pileup J/psi events (= all ttbar
                          events); lower panel: (prompt - pileup) / sigma (the ratio runs to ~ 18 in the last bins)
  figs/ttbal_dy.json      |Delta y(J/psi, ttbar)| with the full ttbar reconstruction (W mass; e mu: W and top masses); lower
                          panel: prompt / pileup
FIGS_OUT=<dir> writes elsewhere (tests), TTBAL=<json> reads another input; FIG_PREVIEW=<dir> also writes PNG previews there.
usage: python3 make_figs_balance.py"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cmsdeck  # noqa: E402
from cmsdeck import (LJ, NB1, SERIES, TT, data_points, draw_ratio, label, legend, ratio_canvas, save, step, xlabel, ylabel,  # noqa: E402
                     zero_line)
from plotlyhep._units import pt2px  # noqa: E402

cmsdeck.OUT = os.environ.get("FIGS_OUT", cmsdeck.OUT)
B = json.load(open(os.environ.get("TTBAL", os.path.join(HERE, "ttbar_balance.json"))))
H = B["hists"]
LT = f"{cmsdeck.CH3}; {NB1}<br>{TT} without the J/ψ jet"
# data: points; emulations: lines. Species colours as the other canvases of the study: prompt s2, 2nd scattering comb
# (the pileup J/psi are its data proxy, the emulation its jet-level model), recoil against the ttbar s1, the fit total
DATA = {"prompt": ("prompt J/ψ (sPlot)", SERIES["s2"], "circle"),
        "pileup": ("prompt pileup J/ψ (sPlot)", SERIES["comb"], "diamond"),
        "inb": ("b decays, b-matched (sPlot)", SERIES["s3"], "triangle-up")}
# the pileup J/psi muons enter the MET while the recoil from their collision does not: shown with the J/psi out of the MET
VAR_OF = {"pileup": "exclnoj"}
EMU = {"sps": ("J/ψ jet recoiling against the t̅t (emulated)", SERIES["s1"], None),
       "dps": ("J/ψ jet + balancing jet, 2nd scattering (emulated)", SERIES["comb"], "dash")}


def dname(s):
    return DATA[s][0] + (", J/ψ out of the MET" if s in VAR_OF else "")


def shape(v, key, perx):
    e = np.asarray(B["vars"][v]["edges"], float)
    h, he = (np.asarray(a, float) for a in H[key])
    n, w = h.sum(), (np.diff(e) if perx else np.ones(len(e) - 1))
    return e, h / n / w, he / n / w


def xbars(f, xs, e, color):
    """horizontal bars over the whole bin for points drawn off-centre"""
    f.data[-1].error_x = dict(type="data", symmetric=False, array=e[1:] - xs, arrayminus=xs - e[:-1], thickness=pt2px(1), width=0, color=color)


def ylim(f, lo, hi, top=1.75, ya="yaxis"):
    f.update_layout(**{ya: dict(range=[min(lo, 0.0) - 0.06 * hi, top * hi])})


def dist_fig(v, name, data=("prompt", "pileup"), emu=("sps", "dps"), var="excl", perx=True, ylab=None, fit=None):
    """unit-area shapes; lower panel: ratio to the 2nd-scattering emulation, or prompt / pileup without emulations"""
    f, ax, axr = ratio_canvas()
    e = np.asarray(B["vars"][v]["edges"], float)
    c, w = 0.5 * (e[1:] + e[:-1]), np.diff(e)
    sh = {s: shape(v, f"{s}_{v}_{var}_all", perx) for s in emu}
    for s in emu:
        step(f, e, sh[s][1], EMU[s][1], EMU[s][0], dash=EMU[s][2])
    model = None
    if fit:  # the template fit of the prompt J/psi: f x recoiling + (1 - f) x 2nd scattering (both at unit area)
        F = B["fit"][fit]
        model = F["f_same"] * sh["sps"][1] + (1 - F["f_same"]) * sh["dps"][1]
        step(f, e, model, SERIES["total"], f"fit: recoiling share {F['f_same']:.2f} {F['err'][0]:+.2f} / {F['err'][1]:+.2f}".replace("-", "−"), width=2.0)
    top, low = max([sh[s][1].max() for s in emu] + [0.0]), 0.0
    nd = len(data)
    dd = {}
    for k, s in enumerate(data):
        _, y, ye = shape(v, f"{s}_{v}_{VAR_OF.get(s, var)}_all", perx)
        dd[s] = (y, ye)
        xs = c + (k - (nd - 1) / 2) * 0.12 * w
        data_points(f, xs, y, ye, label=dname(s), color=DATA[s][1], symbol=DATA[s][2], size=4)
        if k == 0:
            xbars(f, xs, e, DATA[s][1])
        top, low = max(top, np.nanmax(y + ye)), min(low, np.nanmin(y - ye))
    ylim(f, low, top)
    if low < 0:
        zero_line(f)
    ylabel(f, ylab or "fraction / bin", chars=5)
    # lower panel
    if "dps" in emu:
        den, den_e = sh["dps"][1], sh["dps"][2]
        for s in emu:
            if s != "dps":
                step(f, e, sh[s][1] / den, EMU[s][1], None, dash=EMU[s][2], ax=axr)
        if model is not None:
            step(f, e, model / den, SERIES["total"], None, width=2.0, ax=axr)
        for k, s in enumerate(data):
            xs = c + (k - (nd - 1) / 2) * 0.12 * w
            draw_ratio(f, axr, xs, dd[s][0], dd[s][1], den, den_e, edges=e, ylab="ratio to 2nd scatt.", rng=(-0.6, 2.6), color=DATA[s][1],
                       symbol=DATA[s][2], band=(k == 0))
    else:  # prompt / pileup
        (yp, ep), (yu, eu) = dd["prompt"], dd["pileup"]
        draw_ratio(f, axr, c, yp, ep, yu, eu, edges=e, ylab="prompt / pileup", rng=(-0.6, 2.6), color=DATA["prompt"][1], ex=w / 2)
    f.update_xaxes(range=[e[0], e[-1]])
    xlabel(f, B["vars"][v]["label"])
    label(f)
    legend(f, LT)
    save(f, name)


PI8 = "π/8"
dist_fig("dphi", "ttbal_dphi", ylab=f"fraction / ({PI8})", perx=False, fit="prompt_dphi")
dist_fig("ppar", "ttbal_ppar", ylab="fraction / GeV", fit="prompt_ppar")
dist_fig("dy", "ttbal_dy", data=("prompt", "pileup"), emu=(), perx=False, ylab="fraction / 0.5")

# ---- <pT(ttbar) . pT-hat(J/psi)> in bins of pT(J/psi); lower panel: the difference to the 2nd-scattering emulation
PE = np.asarray(B["pt_edges"], float)
f, ax, axr = ratio_canvas()
c = np.sqrt(PE[1:] * PE[:-1])
prof = {s: np.array(B["profile"][f"{s}_excl_all"]["ppar"], float) for s in ("sps", "dps")}
for s in ("sps", "dps"):
    step(f, PE, prof[s][:, 0], EMU[s][1], EMU[s][0], dash=EMU[s][2])
step(f, PE, prof["sps"][:, 0] - prof["dps"][:, 0], EMU["sps"][1], None, dash=EMU["sps"][2], ax=axr)
lo, hi, dlo, dhi = 0.0, 0.0, 0.0, 0.0
for k, s in enumerate(("prompt", "pileup")):
    P = np.array(B["profile"][f"{s}_{VAR_OF.get(s, 'excl')}_all"]["ppar"], float)
    y, ye = P[:, 0], P[:, 1]
    xs = c * (1 + 0.05 * (k - 0.5))
    data_points(f, xs, y, ye, label=dname(s), color=DATA[s][1], symbol=DATA[s][2], size=4)
    if k == 0:
        xbars(f, xs, PE, DATA[s][1])
    d = y - prof["dps"][:, 0]
    data_points(f, xs, d, ye, label=None, ax=axr, color=DATA[s][1], symbol=DATA[s][2], size=3)
    lo, hi = min(lo, np.min(y - ye)), max(hi, np.max(y + ye))
    dlo, dhi = min(dlo, np.min(d - ye)), max(dhi, np.max(d + ye))
zero_line(f)
zero_line(f, axr)
f.update_layout(yaxis=dict(range=[lo - 0.08 * (hi - lo), hi + 0.95 * (hi - lo)]), yaxis2=dict(range=[dlo - 0.1 * (dhi - dlo), dhi + 0.1 * (dhi - dlo)]))
cmsdeck.logx(f, PE[0], PE[-1], ticks=(3, 5, 10, 20, 50, 100))
ylabel(f, f"⟨p<sub>T</sub>({TT}) · p̂<sub>T</sub>(J/ψ)⟩ [GeV]", chars=4)
ylabel(f, "− 2nd scatt. [GeV]", ax=axr)
xlabel(f, "p<sub>T</sub>(J/ψ) [GeV]")
label(f)
legend(f, LT)
save(f, "ttbal_ppar_pt")

# ---- which jet holds the J/psi (l + jets, all jets allowed in the ttbar reconstruction); lower panel: prompt / each emulation
RO = ["b_had", "b_lep", "W jet", "extra"]  # the J/psi in a selected jet
RN = {"b_had": "b of t<sub>had</sub>", "b_lep": "b of t<sub>lep</sub>", "W jet": "W jet", "extra": "extra jet"}
f, ax, axr = ratio_canvas()
groups = [("prompt", "prompt_ljets_all_j", DATA["prompt"]), ("sps", "sps_ljets_all_j", (EMU["sps"][0], EMU["sps"][1], "square-open")),
          ("dps", "dps_ljets_all_j", (EMU["dps"][0], EMU["dps"][1], "diamond-open")), ("inb", "inb_ljets_all_j", DATA["inb"])]
x0 = np.arange(len(RO))
fr = {}
for k, (s, key, (nm, col, sym)) in enumerate(groups):
    R_ = B["roles"][key]
    fr[s] = (np.array([R_[r][0] for r in RO]), np.array([R_[r][1] for r in RO]))
    data_points(f, x0 + (k - 1.5) * 0.15, *fr[s], label=nm, color=col, symbol=sym, size=4)
yp, ep = fr["prompt"]
for k, s in enumerate(("sps", "dps")):
    xs = x0 + (k - 0.5) * 0.15
    draw_ratio(f, axr, xs, yp, ep, fr[s][0], fr[s][1], edges=np.arange(len(RO) + 1) - 0.5, ylab="prompt / emul.", rng=(-0.5, 3.5), color=EMU[s][1],
               symbol=groups[k + 1][2][2], band=(k == 0))
f.update_xaxes(range=[-0.5, len(RO) - 0.5], tickvals=list(x0), ticktext=[RN[r] for r in RO])
f.update_layout(yaxis=dict(range=[-0.05, 1.6]))
ylabel(f, "fraction", chars=4)
xlabel(f, f"jet holding the J/ψ, χ<sup>2</sup> {TT} assignment")
label(f)
legend(f, f"{LJ}; {NB1}<br>J/ψ in a jet; ≥ 4 other jets", x=0.03, xanchor="left")
save(f, "ttbal_role")

# ---- jet multiplicity (l + jets: at least 4 selected jets); lower panel: (prompt - pileup) / sigma, both errors
f, ax, axr = ratio_canvas()
e = np.asarray(B["vars"]["njets"]["edges"], float)
c0 = 0.5 * (e[1:] + e[:-1])
nj, top = {}, 0.0
for k, s in enumerate(("prompt", "pileup")):
    h = sum(np.asarray(H[f"{s}_njets_{ch}"][0], float) for ch in ("mujets", "ejets"))
    he = np.sqrt(sum(np.asarray(H[f"{s}_njets_{ch}"][1], float) ** 2 for ch in ("mujets", "ejets")))
    n = h.sum()
    nj[s] = (h / n, he / n)
    data_points(f, c0 + (k - 0.5) * 0.15, h / n, he / n, label=f"{DATA[s][0]}: mean {(c0 * h).sum() / n:.2f}", color=DATA[s][1], symbol=DATA[s][2], size=4)
    top = max(top, np.nanmax((h + he) / n))
(yp, ep), (yu, eu) = nj["prompt"], nj["pileup"]
ok = (yp != 0) | (yu != 0)
sg = np.sqrt(ep ** 2 + eu ** 2)
zero_line(f, axr)
data_points(f, c0[ok], ((yp - yu) / np.where(sg > 0, sg, 1.0))[ok], np.ones(int(ok.sum())), label=None, ax=axr, color=DATA["prompt"][1], ex=np.diff(e)[ok] / 2,
            size=3)
f.update_layout(yaxis2=dict(range=[-3.5, 3.5], tickvals=[-3, 0, 3]))
ylabel(f, "Δ / σ", ax=axr)
f.update_xaxes(range=[3.5, e[-1]], dtick=1)
f.update_layout(yaxis=dict(range=[-0.05 * top, 1.6 * top]))
ylabel(f, "fraction / jet", chars=4)
xlabel(f, "selected jets (≥ 9 in the last bin)")
label(f)
legend(f, f"{LJ}; {NB1}")
save(f, "ttbal_njets")
print("figures written to", cmsdeck.OUT)
