"""Project plotting module of the ttbar + J/psi decks: the cms-plotting skill's cmsplot_plotly.py (copied next to this file)
plus what these decks add -- canvas sizes for two figures per slide, a data / prediction ratio panel, the CMS label and
selection strings of hadW_ntrack_ttbar/.claude/cms-plotting/style.toml, and save() writing figs/<name>.json (the deck embeds
it; no PNG in the web area) with the PNG preview in $FIG_PREVIEW when set (look at every canvas before it goes on a slide).

    from cmsdeck import (SERIES, canvas, ratio_canvas, label, data_points, step, line, fill_between, draw_ratio, draw_pull,
                         legend, headroom, zero_line, xlabel, ylabel, bin_label, save)
"""
import os

import numpy as np
from plotlyhep._units import pt2px
from plotlyhep.label import exp_text

from cmsplot_plotly import (SERIES, bin_label, cms_label, data_points, draw_pull, figure, fill_between, fit_figure,  # noqa: F401
                            headroom, legend, line, region, step, xlabel, ylabel, zero_line)

__all__ = ["SERIES", "bin_label", "cms_label", "data_points", "draw_pull", "fill_between", "headroom", "legend", "line", "region", "step",
           "xlabel", "ylabel", "zero_line", "canvas", "ratio_canvas", "label", "draw_ratio", "logx", "save", "SIZE", "RSIZE", "RL", "TT", "NB1",
           "Z0", "CH3", "LJ", "LUMI", "GEN", "GENR", "CMSSIM"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
SIZE = (780, 500)    # single panel, two per slide (780 px: the skill's canvas width, room for the CMS and luminosity labels)
RSIZE = (780, 600)   # main + lower panel (ratio or pull, 4 : 1), two per slide
LUMI = 310.0         # fb^-1, 2022-2026: 34.8 + 28.3 + 109.9 + 111.2 + 25.8
RL = "2022–2026, 310 fb<sup>−1</sup> (13.6 TeV)"
TT = "t̅t"
NB1 = "N<sub>b</sub> ≥ 1"
Z0 = "|z<sub>0</sub>| &lt; 0.1 cm"
CH3 = "μ + jets, e + jets, e μ"
LJ = "μ + jets, e + jets"
# two kinds of simulation, never to be mixed up: the official CMS samples (full detector simulation and reconstruction, the
# code of the data pass) and our own generator-level Pythia 8 + LETO sample (genstudy/: no detector, not a CMS sample)
CMSSIM = "CMS simulation"
GEN = "private gen-level Pythia 8 + LETO"
GENR = "gen. Pythia"  # in ratio labels: data / gen. Pythia


def canvas(size=SIZE):
    """Single-panel canvas."""
    return figure(size)


def ratio_canvas(size=RSIZE):
    """(fig, ax, axr): main panel and a lower panel (ratio or pull) sharing x, heights 4 : 1 as the skill's fit canvas."""
    f, ax, axr = fit_figure()
    f.update_layout(width=size[0], height=size[1])
    return f, ax, axr


def label(fig, rlabel=RL, sim=False, gen=False):
    """CMS label (sim=True: 'CMS Simulation', for the CMS samples). gen=True: the private generator-level Pythia 8 + LETO
    sample alone, which is not a CMS sample: no CMS label, the generator and 'private, gen level' instead"""
    if gen:
        exp_text(fig, "Pythia 8 + LETO", "private, gen level", loc=0, lumi=rlabel)
    else:
        cms_label(fig, com=13.6, data=not sim, rlabel=rlabel)


def draw_ratio(fig, axr, centers, num, num_err, den, den_err=None, edges=None, lo=None, hi=None, xlab="", ylab="data / pred.",
               rng=(0.0, 2.2), ex=None, color=None, symbol="circle", band=True, label_=None):
    """Lower panel: num / den as points (num_err / den), the relative error of den as a grey band per bin (edges given), a
    dashed line at 1. Sets the x label. Several ratios in one panel: call again with band=False and another colour / symbol."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    safe = np.where(den != 0, den, np.nan)
    r, re = num / safe, np.asarray(num_err, float) / np.abs(safe)
    xa, ya = axr["xaxis"], axr["yaxis"]
    if band and edges is not None and den_err is not None:
        rel = np.asarray(den_err, float) / np.abs(safe)
        for a, b, e in zip(edges[:-1], edges[1:], rel):
            if np.isfinite(e) and e > 0:
                fig.add_shape(type="rect", xref=xa, yref=ya, x0=a, x1=b, y0=1 - e, y1=1 + e, fillcolor="rgba(0,0,0,0.15)", line_width=0, layer="below")
    if band:
        fig.add_shape(type="line", xref=f"{xa} domain", yref=ya, x0=0, x1=1, y0=1, y1=1, line=dict(color="grey", width=pt2px(1), dash="dash"))
    data_points(fig, centers, r, re, label=label_, ax=axr, ex=ex, color=color, symbol=symbol, size=3)
    ylabel(fig, ylab, ax=axr)
    fig.update_layout(**{f"yaxis{ya[1:]}": dict(range=list(rng))})
    if lo is not None:
        fig.update_xaxes(range=[lo, hi])
    if xlab:
        xlabel(fig, xlab)
    return r, re


def logx(fig, lo, hi, ticks=(3, 5, 7, 10, 15, 20, 30, 50, 100)):
    """log x on every panel, explicit tick labels (plotly otherwise labels 20, 30 as '2, 3')."""
    tv = [t for t in ticks if lo <= t <= hi]
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=tv, ticktext=[f"{t:g}" for t in tv], minor=dict(showgrid=False, ticks=""))


def save(fig, name):
    """figs/<name>.json for the deck; the PNG preview into $FIG_PREVIEW (outside the web area) when set, only for the names
    starting with one of $FIG_ONLY (comma-separated) when that is set."""
    os.makedirs(OUT, exist_ok=True)
    fig.write_json(os.path.join(OUT, name + ".json"))
    pv = os.environ.get("FIG_PREVIEW")
    only = [o for o in os.environ.get("FIG_ONLY", "").split(",") if o]
    if pv and (not only or any(name.startswith(o) for o in only)):
        os.makedirs(pv, exist_ok=True)
        fig.write_image(os.path.join(pv, name + ".png"), scale=1.3)
    return name
