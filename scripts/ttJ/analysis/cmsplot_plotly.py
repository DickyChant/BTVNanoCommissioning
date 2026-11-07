"""(copy of github.com/DickyChant/cms-plotting cmsplot_plotly.py @ 75a342e; do not edit here: project additions go in cmsdeck.py)
cmsplot_plotly -- the Plotly counterpart of cmsplot.py, built on plotlyhep (mplhep mirrored for Plotly).

Same names, palette and rules as the matplotlib module, for canvases that live in HTML (slides that embed
interactive figures, dashboards) instead of PNG files. Import it once; it applies the CMS template.

    from cmsplot_plotly import figure, fit_figure, cms_label, data_points, line, fill_between, draw_pull
    from cmsplot_plotly import bin_label, xlabel, ylabel, legend, headroom, save, SERIES

Plotly has no axes objects: a canvas is one go.Figure, and `ax` / `axp` below are axis references
({"xaxis": "x", "yaxis": "y"}) that every drawing helper takes as `ax=` (default: the main panel).
plotlyhep is not on PyPI: pip install git+https://github.com/DickyChant/plotlyhep
"""
from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
import plotlyhep as php
from plotlyhep._units import pt2px
from plotlyhep.ratio import gridspec_domains
from plotlyhep.styles import figsize_px, template

php.style.use("CMS")

# Categorical palette: identical to cmsplot.SERIES (validated there). Fixed order, never cycled.
SERIES = {
    "total":   "#4a3aa7",   # the summed model
    "comb":    "#2a78d6",   # combinatorial / main background
    "s1":      "#eb6834",   # component slot 1 (usually the signal)
    "s2":      "#1baf7a",   # component slot 2
    "s3":      "#eda100",   # component slot 3
    "s4":      "#e87ba4",   # component slot 4
    "data":    "#0b0b0b",   # data points wear text ink, not a series hue
    "signal":  "#008300",   # background-subtracted signal overlay
}
SERIES_ORDER = [SERIES["s1"], SERIES["s2"], SERIES["s3"], SERIES["s4"]]

# CSS pixels at plotlyhep's 100 dpi: the 7.8 x 5.2 in / 7.8 x 6.6 in canvases of cmsplot.py
PLAIN_SIZE = (780, 520)
FIT_SIZE = (780, 660)
FIT3_SIZE = (780, 760)
MAIN = {"xaxis": "x", "yaxis": "y"}
LEFT_EXTRA = 46  # px added to the template's left margin: room for the rotated y label beside 4-6 character ticks
BOTTOM_EXTRA = 12  # px added to the bottom margin: room for the x label below the tick labels


def _scale(w, h):
    W, H = figsize_px("CMS")
    return min(w / W, h / H)


FONT_SCALE = _scale(*FIT_SIZE)  # fonts, ticks and line widths of plotlyhep's CMS style at the fit-canvas size


def _ref(ax):
    """axis reference -> (x name, y name), e.g. ("x2", "y2")"""
    ax = ax or MAIN
    return ax["xaxis"], ax["yaxis"]


def _lay(name):
    """"x2" -> "xaxis2", "y" -> "yaxis" (layout keys)"""
    return f"{name[0]}axis{name[1:]}"


def _margin(f):
    m = f.layout.template.layout.margin
    f.update_layout(margin=dict(l=int(m.l or 80) + LEFT_EXTRA, r=m.r, t=m.t, b=int(m.b or 60) + BOTTOM_EXTRA))


def _template():
    """The CMS template at one font scale for every canvas size (as cmsplot keeps one set of font sizes)."""
    return template("CMS", scale=FONT_SCALE)


def figure(size=PLAIN_SIZE):
    """Single-panel canvas."""
    f = go.Figure(layout=dict(template=_template(), width=size[0], height=size[1], autosize=False, legend=dict(bgcolor="rgba(255,255,255,0)")))
    _margin(f)
    return f


def fit_figure(three=False):
    """New canvas with fit panels sharing x: (fig, ax, axp) with heights 4:1 or, three=True,
    (fig, ax, axb, axp) with 4:1.6:1 (main, data - background, pull)."""
    w, h = FIT3_SIZE if three else FIT_SIZE
    t = _template()
    ratios = (4, 1.6, 1) if three else (4, 1)
    doms = gridspec_domains(ratios, 0.05 if three else 0.04)
    f = go.Figure(layout=dict(template=t, width=w, height=h, autosize=False, legend=dict(bgcolor="rgba(255,255,255,0)")))
    tx, ty = t.layout.xaxis.to_plotly_json(), t.layout.yaxis.to_plotly_json()
    n = len(ratios)
    for i, d in enumerate(doms, 1):
        s = "" if i == 1 else str(i)
        last = i == n
        f.update_layout(**{f"xaxis{s}": dict(tx, domain=[0, 1], anchor=f"y{s}", showticklabels=last, **({} if last else {"matches": f"x{n}"})),
                           f"yaxis{s}": dict(ty, domain=list(d), anchor=f"x{s}")})
    _margin(f)
    refs = [{"xaxis": "x" if i == 1 else f"x{i}", "yaxis": "y" if i == 1 else f"y{i}"} for i in range(1, n + 1)]
    return (f, *refs)


def cms_label(fig, com=13.6, data=True, rlabel=None, status="Work in Progress"):
    """'CMS <status>' above the top-left corner, rlabel (era, luminosity, energy) top-right; data=False -> Simulation."""
    php.cms.label(fig, status if data else f"Simulation {status}".strip(), data=True, com=com, rlabel=rlabel, loc=0)


def xlabel(fig, text):
    """x label at the right end of the (lowest) x axis, as mplhep."""
    php.set_xlabel(fig, text)


def ylabel(fig, text, chars=4, ax=None):
    """y label at the top of the main panel (mplhep placement); the other panels use the axis title."""
    if ax is None or ax == MAIN:
        php.set_ylabel(fig, text, ticklabel_chars=chars)
    else:  # a lower panel: its axis title, a touch smaller than the main label (cmsplot: 13 vs 15 pt)
        _, ya = _ref(ax)
        main = fig.layout.template.layout.yaxis.title.font.size or pt2px(26)
        fig.update_layout(**{_lay(ya): dict(title=dict(text=text, font=dict(size=0.87 * main)))})


def bin_label(bw, unit="GeV", what="candidates"):
    """y label with the bin width in the x-axis unit: 'candidates / 0.005 GeV'. Never a bin count."""
    w = f"{bw:.3f}" if unit == "GeV" and bw < 1 else f"{bw:g}"
    return f"{what} / {w} {unit}".rstrip()


def data_points(fig, cen, h, err=None, label="data", ax=None, ex=None, color=None, symbol="circle", size=3.4):
    """Data as black circles with vertical error bars, drawn on top (add data last)."""
    if err is None:
        err = np.sqrt(np.maximum(h, 0))
    xa, ya = _ref(ax)
    col = color or SERIES["data"]
    fig.add_trace(go.Scatter(x=cen, y=h, mode="markers", name=label, xaxis=xa, yaxis=ya, showlegend=label is not None,
                             marker=dict(symbol=symbol, size=pt2px(size) * 1.25, color=col),
                             error_y=dict(type="data", array=err, thickness=pt2px(1), width=0, color=col),
                             error_x=dict(type="data", array=ex, thickness=pt2px(1), width=0, color=col) if ex is not None else None,
                             hovertemplate=f"{label}<br>x %{{x:.4g}}<br>y %{{y:.4g}} ± %{{error_y.array:.3g}}<extra></extra>"))


def line(fig, x, y, color, label=None, dash=None, width=2.0, ax=None, shape="linear"):
    """Model curve: total SERIES['total'] solid width 2; background SERIES['comb'] dash='dash' width 1.5."""
    xa, ya = _ref(ax)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label, showlegend=label is not None, xaxis=xa, yaxis=ya,
                             line=dict(color=color, width=pt2px(width), dash=dash, shape=shape), hoverinfo="skip" if label is None else None))


def step(fig, edges, values, color, label=None, dash=None, width=1.6, ax=None):
    """Histogram outline from bin edges (len(edges) = len(values) + 1)."""
    edges, values = np.asarray(edges, float), np.asarray(values, float)
    line(fig, edges, np.append(values, values[-1]), color, label, dash, width, ax, shape="hv")


def fill_between(fig, x, lo, hi, color, label=None, alpha=0.45, ax=None, shape="linear"):
    """A component stacked on the background: the band between lo and hi, no outline (cmsplot: alpha 0.45, lw 0)."""
    xa, ya = _ref(ax)
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0, shape=shape), showlegend=False, hoverinfo="skip", xaxis=xa, yaxis=ya))
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0, shape=shape), fill="tonexty", fillcolor=f"rgba({r},{g},{b},{alpha})",
                             name=label, showlegend=label is not None, xaxis=xa, yaxis=ya, hoverinfo="skip"))


def region(fig, x0, x1, label, color="rgba(11,11,11,0.10)", ax=None):
    """Shade an x range (a signal window, sidebands, a cut) and name it in the legend -- never with text on the canvas.
    Several ranges of one kind: pass label=None for all but one."""
    xa, ya = _ref(ax)
    fig.add_shape(type="rect", xref=xa, yref=f"{ya} domain", x0=x0, x1=x1, y0=0, y1=1, fillcolor=color, line_width=0, layer="below")
    if label:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label, xaxis=xa, yaxis=ya,
                                 marker=dict(symbol="square", size=pt2px(9), color=color.replace("0.10)", "0.18)"))))


def zero_line(fig, ax=None):
    xa, ya = _ref(ax)
    fig.add_shape(type="line", xref=f"{xa} domain", yref=ya, x0=0, x1=1, y0=0, y1=0, line=dict(color="grey", width=pt2px(0.8)))


def draw_pull(fig, axp, centers, counts, expected, lo, hi, xlab, ylab="pull", err=None):
    """Pull panel (n - f) / err with a |pull| < 2 band. Pass err for weighted data (sqrt(sum w^2));
    the default sqrt(n) is only right for raw counts. Sets the x label. Returns chi2 / filled bins."""
    counts, expected = np.asarray(counts, float), np.asarray(expected, float)
    if err is None:
        err = np.sqrt(np.where(counts > 0, counts, 1.0))
    pull = (counts - expected) / err
    xa, ya = _ref(axp)
    fig.add_shape(type="rect", xref=f"{xa} domain", yref=ya, x0=0, x1=1, y0=-2, y1=2, fillcolor="rgba(0,0,0,0.15)", line_width=0, layer="below")
    fig.add_shape(type="line", xref=f"{xa} domain", yref=ya, x0=0, x1=1, y0=0, y1=0, line=dict(color="red", width=pt2px(1)))
    fig.add_trace(go.Scatter(x=centers, y=pull, mode="markers", xaxis=xa, yaxis=ya, showlegend=False, marker=dict(size=pt2px(3) * 1.25, color=SERIES["data"]),
                             hovertemplate="x %{x:.4g}<br>pull %{y:.2f}<extra></extra>"))
    fig.update_layout(**{_lay(ya): dict(range=[-5, 5], tickvals=[-3, 0, 3]), _lay(xa): dict(range=[lo, hi])})
    fig.update_xaxes(range=[lo, hi])
    ylabel(fig, ylab, ax=axp)
    xlabel(fig, xlab)
    mask = counts > 0
    return float(np.sum(pull[mask] ** 2) / max(int(mask.sum()), 1))


def draw_bkgsub(fig, axb, centers, counts, bkg_at_centers, x_curve, sig_curve, lo, hi, ylab="data − bkg", err=None):
    """Data minus fitted background (points) with the fitted signal curve (SERIES['signal'])."""
    counts = np.asarray(counts, float)
    if err is None:
        err = np.sqrt(np.where(counts > 0, counts, 1.0))
    zero_line(fig, axb)
    data_points(fig, centers, counts - np.asarray(bkg_at_centers), err, label=None, ax=axb, size=3)
    line(fig, x_curve, sig_curve, SERIES["signal"], None, width=1.6, ax=axb)
    ylabel(fig, ylab, ax=axb)


def legend(fig, title=None, ncol=1, x=0.97, y=0.97, xanchor="right"):
    """House legend: no frame, upper right, the selection as the legend title (not as text on the canvas)."""
    leg = dict(x=x, y=y, xanchor=xanchor, yanchor="top", bgcolor="rgba(255,255,255,0)", borderwidth=0,
               font=dict(size=pt2px(10)), title=dict(text=title or "", font=dict(size=pt2px(10))))
    if ncol > 1:
        leg.update(orientation="h", entrywidthmode="fraction", entrywidth=0.55 / ncol)
    fig.update_layout(legend=leg, showlegend=True, legend_traceorder="normal")  # plotly reverses legends with fills otherwise


def headroom(fig, top, bottom=0.0, factor=1.4, ax=None):
    """Leave room above the data for the legend."""
    _, ya = _ref(ax)
    fig.update_layout(**{_lay(ya): dict(range=[bottom, top * factor])})


def save(fig, path, image=True, scale=1.4):
    """Write the canvas: path '<name>.json' (the figure for HTML embedding) and, image=True, '<name>.png'
    via kaleido (scale 1.4: the 140 dpi of cmsplot.save). A '.png' / '.pdf' / '.svg' path writes that image and the JSON next to it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stem, ext = os.path.splitext(path)
    fig.write_json(stem + ".json")
    if ext in (".png", ".pdf", ".svg"):
        fig.write_image(path, scale=scale)
    elif image:
        fig.write_image(stem + ".png", scale=scale)
    return path


def embed(fig, div_id, **kw):
    """HTML fragment for a slide: plotlyhep.html.embed with the figure's own template kept (inherit_template=False),
    so the canvas looks the same as its PNG. Put head_script() once in the page <head>."""
    from plotlyhep.html import embed as _embed  # noqa: PLC0415

    kw.setdefault("inherit_template", False)
    kw.setdefault("width", "100%")
    kw.setdefault("height", f"{int(fig.layout.height or PLAIN_SIZE[1])}px")
    return _embed(fig, div_id, **kw)


def head_script():
    """plotly.js for the page head; no page-level template (each figure carries its own)."""
    from plotlyhep.html import script_tag  # noqa: PLC0415

    return script_tag(None)
