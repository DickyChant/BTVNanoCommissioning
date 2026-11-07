"""Shared deck machinery (CSS, slide helpers, rendering, viewer) for the hadronic-W-mass HTML decks.
A content script sets TITLE / DATE / FOOT / HERE / EXTRA_LINKS, appends to SLIDES via the helpers, then calls main().

Original docstring of the main deck:

  ../index.html   slide viewer (arrows / space / click, #n in the URL, notes, fullscreen) above a file browser
  slides.html     every slide in one scrolling page with its speaker notes (prints one slide per page)

Inputs next to this script: figs/<name>.json (written by make_figs.py), fonts/ (TeX Gyre Heros for the plots).
usage: python3 build_deck.py      (needs plotly + plotlyhep for the embed helper)
"""
import html as H
import json
import os
import re

import plotly.io as pio
from plotlyhep.html import SLIDE_CSS as PHP_CSS
from plotlyhep.html import embed, script_tag

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = "Hadronic W mass vs track multiplicity in semileptonic tt̅"
DATE = "1 Oct 2026"
FOOT = ""  # set by the deck builder
TB = "t̅t"
NT = "n<i>track</i>"
MJJ = "m<i>jj</i>"
NB = "N<i>b</i>"

EXTRA_LINKS = ""
BUILD = "build_deck_main.py"
FONTS = "https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap"

CSS = """
@font-face { font-family: "TeX Gyre Heros"; src: url(FONTDIR/texgyreheros-regular.otf); font-weight: 400; font-style: normal; }
@font-face { font-family: "TeX Gyre Heros"; src: url(FONTDIR/texgyreheros-bold.otf); font-weight: 700; font-style: normal; }
@font-face { font-family: "TeX Gyre Heros"; src: url(FONTDIR/texgyreheros-italic.otf); font-weight: 400; font-style: italic; }
@font-face { font-family: "TeX Gyre Heros"; src: url(FONTDIR/texgyreheros-bolditalic.otf); font-weight: 700; font-style: italic; }
section.slide { position: relative; width: 1920px; height: 1080px; box-sizing: border-box; overflow: hidden; background: #F7F6F2; color: #3A4556;
  font-family: 'IBM Plex Sans', Arial, sans-serif; padding: 104px 128px 150px; display: flex; flex-direction: column; gap: 26px; }
section.slide * { box-sizing: border-box; }
section.slide p, section.slide h1, section.slide h2, section.slide h3, section.slide ul, section.slide ol, section.slide table { margin: 0; }
section.alt { background: #EDF1F6; }
section.div { padding: 128px 128px 160px; justify-content: center; gap: 40px; color: #E9EEF5; }
section.d0 { background: #14213D; } section.d1 { background: #14213D; } section.d2 { background: #0F2A2E; } section.d3 { background: #241A3A; } section.d4 { background: #1D2A16; }
.kicker { font-size: 24px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; color: #1F5BC0; }
.k2 { color: #0F766E; } .k3 { color: #6D3FB5; } .k4 { color: #4D7C0F; }
h2.title { font-family: 'Source Serif 4', Georgia, serif; font-size: 54px; font-weight: 600; line-height: 1.1; color: #14213D; }
.tag { position: absolute; right: 128px; top: 96px; font: 500 19px/1 'IBM Plex Mono', monospace; letter-spacing: 2px; padding: 8px 14px; border-radius: 6px; border: 1.5px solid; }
.tag.re { color: #1F5BC0; border-color: #9DB8E6; background: #EEF3FB; }
.tag.new { color: #6D3FB5; border-color: #C9B3EA; background: #F4EEFB; }
.tag.part { color: #4D7C0F; border-color: #B7D59A; background: #F1F7EA; }
.c4 { border-top: 8px solid #4D7C0F; }
.callout.green { background: #EEF5E6; border-left-color: #4D7C0F; }
.row { display: flex; gap: 44px; align-items: flex-start; }
.col { display: flex; flex-direction: column; gap: 20px; flex: 1; min-width: 0; }
ul.b, ol.b { font-size: 28px; line-height: 1.42; padding-left: 1.15em; }
ul.b li + li, ol.b li + li { margin-top: .32em; }
ul.b.s, ol.b.s { font-size: 26px; }
.fig { background: #FFFFFF; border-radius: 12px; box-shadow: 0 1px 4px rgba(20, 33, 61, .14); display: flex; flex-direction: column; align-items: center; flex: none; }
.cap { font: 400 19px/1.3 'IBM Plex Mono', monospace; color: #6B7482; padding: 4px 10px 8px; text-align: center; }
.card { background: #FFFFFF; border: 1px solid #D9DEE5; border-radius: 16px; padding: 28px 34px; display: flex; flex-direction: column; gap: 12px; }
.card h3 { font-size: 30px; font-weight: 600; color: #14213D; }
.c1 { border-top: 8px solid #1F5BC0; } .c2 { border-top: 8px solid #0F766E; } .c3 { border-top: 8px solid #6D3FB5; } .c0 { border-top: 8px solid #9C9CA1; }
.ey { font-size: 22px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; }
.big { font-size: 54px; font-weight: 600; color: #14213D; line-height: 1.1; }
.callout { background: #FBEBDC; border-left: 8px solid #D9822B; border-radius: 8px; padding: 16px 28px; font-size: 26px; line-height: 1.4; }
.callout.blue { background: #EAF0FA; border-left-color: #1F5BC0; }
.callout.violet { background: #F1EBFA; border-left-color: #6D3FB5; }
.formula { background: #EDF1F6; border-radius: 14px; padding: 20px 32px; font: 400 28px/1.45 'IBM Plex Mono', monospace; color: #14213D; }
.code { background: #10192B; color: #E3E9F2; border-radius: 16px; padding: 26px 34px; font: 400 23px/1.5 'IBM Plex Mono', monospace; white-space: pre; }
.code .c { color: #8FA3BF; } .code .k { color: #F2B479; }
table.t { border-collapse: collapse; font-size: 26px; width: 100%; color: #3A4556; }
table.t th, table.t td { border-bottom: 1px solid #C9D0D9; padding: .34em .6em; text-align: left; vertical-align: top; }
table.t th { color: #14213D; font-weight: 600; }
table.t .n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.t tr.hl { background: #E3EAF5; font-weight: 600; }
table.t tr.warn { background: #FBEBDC; }
.footer { position: absolute; left: 128px; right: 128px; bottom: 52px; display: flex; justify-content: space-between; font-size: 21px; color: #6B7482; }
section.div .footer { color: #9FB3CF; }
.dv-ey { font-size: 28px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; }
.dv-t { font-family: 'Source Serif 4', Georgia, serif; font-size: 80px; font-weight: 600; line-height: 1.1; color: #F7F6F2; }
.dv-p { font-size: 34px; line-height: 1.4; color: #BFD0E6; max-width: 1560px; }
.stat { flex: 1; display: flex; flex-direction: column; gap: 8px; border-radius: 16px; padding: 30px 38px; }
.stat .v { font-size: 54px; font-weight: 600; color: #F7F6F2; line-height: 1.1; }
.stat .l { font-size: 25px; color: #BFD0E6; }
.tl { display: flex; align-items: flex-start; justify-content: space-between; position: relative; padding: 0 30px; }
.tl::before { content: ""; position: absolute; left: 60px; right: 60px; top: 58px; height: 4px; background: #14213D; }
.tl .st { position: relative; display: flex; flex-direction: column; align-items: center; gap: 10px; width: 250px; text-align: center; }
.tl .when { font: 500 22px/1.2 'IBM Plex Mono', monospace; color: #6B7482; height: 28px; }
.tl .dot { width: 30px; height: 30px; border-radius: 50%; background: #14213D; border: 4px solid #F7F6F2; z-index: 1; }
.tl .dot.now { background: #6D3FB5; width: 40px; height: 40px; margin-top: -5px; }
.tl .dot.next { background: #F7F6F2; border-color: #14213D; }
.tl .what { font-size: 22px; line-height: 1.3; color: #3A4556; }
.tl .what.now { color: #6D3FB5; font-weight: 600; }
.outline { display: grid; grid-template-columns: 70px 1fr; row-gap: 16px; font-size: 30px; align-items: baseline; }
.outline .num { font: 500 26px 'IBM Plex Mono', monospace; color: #6B7482; }
.outline .sec b { color: #14213D; } .outline .sec span { color: #6B7482; font-size: 25px; }
.refs { font-size: 24px; line-height: 1.45; }
.refs li + li { margin-top: .5em; }
.refs a { color: #1F5BC0; }
"""

# ---------------------------------------------------------------- slide helpers
SLIDES = []  # (id, html, notes)


FIGZOOM = 0.8  # the cms-plotting canvases (780 px wide) drawn at this scale on the slides: two per row with text below


def F(name, cap=None, zoom=None):
    return f"\x00FIG:{name}:{cap or ''}:{zoom or FIGZOOM}\x00"


def tag(kind):
    return {"re": '<span class="tag re">RE-DERIVED · JUL → OCT</span>', "new": '<span class="tag new">NEW · OCT 2026</span>',
            "jul": '<span class="tag re">JULY 2026</span>', "part": '<span class="tag part">NEW · DATA 4% OF 2024</span>',
            "mc": '<span class="tag part">NEW · MC</span>'}.get(kind, "")


def content(sid, kicker, title, body, *, k="", t=None, alt=False, notes=""):
    cls = "slide alt" if alt else "slide"
    h = (f'<section id="{sid}" class="{cls}">{tag(t)}<p class="kicker {k}">{kicker}</p><h2 class="title">{title}</h2>{body}'
         f'<div class="footer"><span>{FOOT}</span><span>{DATE} · PAGE</span></div></section>')
    SLIDES.append((sid, h, notes))


def divider(sid, d, eyebrow, title, text, stats=None, notes=""):
    acc = {0: "#E8A765", 1: "#E8A765", 2: "#7FD1C7", 3: "#C9A7F0", 4: "#B5D58B"}[d]
    card = {0: "#1F2E52", 1: "#1F2E52", 2: "#17403F", 3: "#3A2C5C", 4: "#2C3D22"}[d]
    st = ""
    if stats:
        st = '<div class="row" style="gap:32px">' + "".join(
            f'<div class="stat" style="background:{card}"><p class="v">{v}</p><p class="l">{l}</p></div>' for v, l in stats) + "</div>"
    h = (f'<section id="{sid}" class="slide div d{d}"><p class="dv-ey" style="color:{acc}">{eyebrow}</p><h2 class="dv-t">{title}</h2>'
         f'<p class="dv-p">{text}</p>{st}<div class="footer"><span>{FOOT}</span><span>{DATE} · PAGE</span></div></section>')
    SLIDES.append((sid, h, notes))


def ul(items, cls="b"):
    return f'<ul class="{cls}">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def ol(items, cls="b"):
    return f'<ol class="{cls}">' + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def table(head, rows, widths=None, num=(), hl=(), warn=()):
    ws = widths or [None] * len(head)
    th = "".join(f'<th{" class=n" if i in num else ""}{f" style=width:{w}%" if w else ""}>{c}</th>' for i, (c, w) in enumerate(zip(head, ws)))
    trs = ""
    for r, row in enumerate(rows):
        cls = ' class="hl"' if r in hl else (' class="warn"' if r in warn else "")
        trs += f"<tr{cls}>" + "".join(f'<td{" class=n" if i in num else ""}>{c}</td>' for i, c in enumerate(row)) + "</tr>"
    return f'<table class="t"><tr>{th}</tr>{trs}</table>'



# ---------------------------------------------------------------- rendering
def figure_html(name, cap, uid, zoom=1.0):
    fig = pio.read_json(os.path.join(HERE, "figs", name + ".json"))
    w, h = int(fig.layout.width), int(fig.layout.height)
    emb = embed(fig, uid, editable=True, persist=True, width="100%", height=f"{h}px", inherit_template=False, frozen=True, download=True)
    capd = f'<p class="cap">{H.escape(cap)}</p>' if cap else ""
    return f'<div style="width:{w}px;height:{h}px;zoom:{zoom:g}">{emb}</div>{capd}'


def render(prefix):
    out = []
    for i, (sid, h, notes) in enumerate(SLIDES, 1):
        h = h.replace("PAGE", str(i))
        h = re.sub(r"\x00FIG:([^:]*):([^\x00]*):([0-9.]+)\x00", lambda m: figure_html(m.group(1), m.group(2), f"{prefix}{sid}-{m.group(1)}", float(m.group(3))), h)
        h = h.replace("@ASSET@/", "files/" if prefix == "v-" else "")  # files next to the slides (videos): the viewer sits one level up
        out.append((sid, h, notes))
    return out


def _text(x):
    return H.escape(re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", x))).strip())


def navhtml(S):
    """Left navigation: section dividers (and the cover) as headers, other slides as items, all clickable."""
    out = []
    for i, (sid, h, _) in enumerate(S):
        ey = re.search(r'class="dv-ey"[^>]*>(.*?)</p>', h, re.S)
        tt = re.search(r'class="dv-t"[^>]*>(.*?)</h[12]>', h, re.S) or re.search(r'<h2 class="title">(.*?)</h2>', h, re.S)
        title = _text(tt.group(1)) if tt else sid
        if ey:  # cover or section divider
            label = _text(ey.group(1)) if sid != "cover" else "Title"
            out.append(f'<a class="sec" href="#{i + 1}" data-i="{i}" title="{title}">{label}</a>')
        else:
            out.append(f'<a class="it" href="#{i + 1}" data-i="{i}"><span class="num">{i + 1}</span>{title}</a>')
    return "".join(out)


def head(fontdir):
    return (f'<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{H.escape(TITLE)}</title>'
            f'<link rel="stylesheet" href="{FONTS}">{script_tag(None)}{PHP_CSS}<style>{CSS.replace("FONTDIR", fontdir)}</style>')


VIEWER_CSS = """
:root { --navy: #14213D; --bg: #E6EAF0; --panel: #F7F8FA; --ink: #1F2937; --muted: #5B6573; --line: #C9D1DC; --accent: #2A6FDB; }
html, body { height: 100%; margin: 0; }
body { display: flex; flex-direction: column; background: var(--bg); color: var(--ink); font-family: 'IBM Plex Sans', Arial, sans-serif; font-size: 14px; }
.bar { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; padding: 8px 16px; background: var(--navy); color: #E9EEF5; }
.bar .title { font-weight: 600; font-size: 15px; margin-right: auto; }
.bar a { color: #BFD0E6; text-decoration: none; } .bar a:hover { text-decoration: underline; }
#vw-top { flex: 0 0 72%; min-height: 140px; display: flex; flex-direction: column; overflow: hidden; position: relative; }
#vw-main { flex: 1; display: flex; min-height: 0; }
#vw-nav { flex: 0 0 290px; overflow-y: auto; background: var(--panel); border-right: 1px solid var(--line); padding: 6px 0 12px; font-size: 13px; line-height: 1.35; }
#vw-nav.hide { display: none; }
#vw-nav a { display: block; text-decoration: none; border-left: 3px solid transparent; }
#vw-nav a.sec { padding: 9px 12px 3px 11px; font-weight: 600; color: var(--navy); }
#vw-nav a.it { padding: 3px 12px 3px 22px; color: var(--muted); }
#vw-nav a:hover { background: #EDF1F6; color: var(--ink); }
#vw-nav a.cur { background: #E3EAF5; color: var(--navy); border-left-color: var(--accent); }
#vw-nav .num { display: inline-block; min-width: 22px; color: #8A93A0; font-variant-numeric: tabular-nums; }
#vw-col { flex: 1; min-width: 0; display: flex; flex-direction: column; }
@media (max-width: 800px) { #vw-nav { display: none; position: absolute; z-index: 5; top: 0; bottom: 0; left: 0; box-shadow: 2px 0 10px rgba(20, 33, 61, .2); } #vw-nav.show { display: block; } }
#vw-stage { flex: 1; position: relative; overflow: hidden; }
#vw-stage > section { position: absolute; left: 0; top: 0; transform-origin: top left; box-shadow: 0 2px 12px rgba(20, 33, 61, .25); }
#vw-stage > section:not(.on) { visibility: hidden; pointer-events: none; }
#vw-prog { height: 4px; background: #C9D1DC; } #vw-prog > div { height: 100%; background: #1F5BC0; width: 0; transition: width .2s; }
#vw-controls { display: flex; align-items: center; justify-content: center; gap: 8px; padding: 6px 12px; flex-wrap: wrap; }
#vw-controls button { font: inherit; padding: 4px 12px; border: 1px solid var(--line); border-radius: 6px; background: #FFFFFF; color: var(--ink); cursor: pointer; }
#vw-controls button:hover { border-color: var(--accent); }
#vw-count { min-width: 64px; text-align: center; font-variant-numeric: tabular-nums; }
#vw-notes { display: none; max-height: 30%; overflow: auto; padding: 8px 16px; background: #FFFFFF; border-top: 1px solid var(--line); line-height: 1.5; }
#vw-split { flex: 0 0 8px; cursor: row-resize; background: var(--line); touch-action: none; }
#vw-bottom { flex: 1; min-height: 100px; display: flex; flex-direction: column; background: var(--panel); }
#vw-bottom .head { display: flex; gap: 16px; align-items: center; padding: 6px 16px; border-bottom: 1px solid var(--line); color: var(--muted); }
#vw-bottom .head a { color: var(--accent); text-decoration: none; }
#vw-bottom iframe { flex: 1; width: 100%; border: 0; background: #FFFFFF; }
body.dragging iframe { pointer-events: none; }
"""

VIEWER_JS = """
if (window.top !== window.self) { window.top.location = window.location.href; }
const secs = Array.from(document.querySelectorAll('#vw-stage > section'));
const stage = document.getElementById('vw-stage'), notes = document.getElementById('vw-notes');
let cur = 0;
function fit() {
  const s = Math.min(stage.clientWidth / 1920, stage.clientHeight / 1080) * 0.97;
  const x = (stage.clientWidth - 1920 * s) / 2, y = (stage.clientHeight - 1080 * s) / 2;
  secs.forEach(e => { e.style.transform = 'translate(' + x + 'px,' + y + 'px) scale(' + s + ')'; });
}
const nav = document.getElementById('vw-nav'), navLinks = Array.from(nav.querySelectorAll('a[data-i]'));
navLinks.forEach(a => a.addEventListener('click', e => { e.preventDefault(); show(+a.dataset.i); if (window.innerWidth <= 800) nav.classList.remove('show'); }));
function toggleNav() { if (window.innerWidth <= 800) nav.classList.toggle('show'); else nav.classList.toggle('hide'); fit(); }
function show(i) {
  cur = Math.max(0, Math.min(secs.length - 1, i));
  navLinks.forEach(a => a.classList.toggle('cur', +a.dataset.i === cur));
  const ca = navLinks.find(a => +a.dataset.i === cur); if (ca) ca.scrollIntoView({ block: 'nearest' });
  secs.forEach((e, k) => e.classList.toggle('on', k === cur));
  document.getElementById('vw-count').textContent = (cur + 1) + ' / ' + secs.length;
  document.querySelector('#vw-prog > div').style.width = (100 * (cur + 1) / secs.length) + '%';
  notes.textContent = NOTES[cur] || '(no notes)';
  if (location.hash !== '#' + (cur + 1)) history.replaceState(null, '', '#' + (cur + 1));
  fit();
}
document.getElementById('vw-prev').onclick = () => show(cur - 1);
document.getElementById('vw-next').onclick = () => show(cur + 1);
document.getElementById('vw-tnav').onclick = toggleNav;
document.getElementById('vw-tnotes').onclick = () => { notes.style.display = notes.style.display === 'block' ? 'none' : 'block'; fit(); };
document.getElementById('vw-full').onclick = () => { const t = document.getElementById('vw-top'); if (!document.fullscreenElement) t.requestFullscreen(); else document.exitFullscreen(); };
document.addEventListener('fullscreenchange', () => setTimeout(fit, 50));
function onKey(e) {
  const t = e.target;
  if (t && t.closest && t.closest('input, textarea, select, [contenteditable]')) return;
  const k = e.key;
  if (['ArrowRight', 'Right', 'PageDown', ' '].includes(k)) { e.preventDefault(); show(cur + 1); }
  else if (['ArrowLeft', 'Left', 'PageUp'].includes(k)) { e.preventDefault(); show(cur - 1); }
  else if (k === 'Home') show(0); else if (k === 'End') show(secs.length - 1);
  else if (k === 'c') toggleNav();
  else if (k === 'n') document.getElementById('vw-tnotes').click(); else if (k === 'f') document.getElementById('vw-full').click();
}
document.addEventListener('keydown', onKey);
const frame = document.querySelector('#vw-bottom iframe');
frame.addEventListener('load', () => { try { frame.contentWindow.document.addEventListener('keydown', onKey); } catch (err) {} });
const topPanel = document.getElementById('vw-top'), split = document.getElementById('vw-split');
split.addEventListener('pointerdown', e => {
  split.setPointerCapture(e.pointerId); document.body.classList.add('dragging');
  const move = ev => { const h = Math.max(140, Math.min(window.innerHeight - 120, ev.clientY - topPanel.getBoundingClientRect().top)); topPanel.style.flexBasis = h + 'px'; fit(); };
  const up = () => { split.removeEventListener('pointermove', move); split.removeEventListener('pointerup', up); document.body.classList.remove('dragging'); };
  split.addEventListener('pointermove', move); split.addEventListener('pointerup', up);
});
window.addEventListener('resize', fit);
window.addEventListener('hashchange', () => show(parseInt(location.hash.slice(1), 10) - 1 || 0));
show((parseInt(location.hash.slice(1), 10) || 1) - 1);
document.fonts && document.fonts.ready.then(fit);
"""


def main():
    S = render("v-")
    notes = json.dumps([n for _, _, n in S])
    viewer = f"""<!doctype html>
<html lang="en"><head>{head("files/fonts")}<style>{VIEWER_CSS}</style></head><body>
<div class="bar"><span class="title">{H.escape(TITLE)}</span><a href="files/slides.html">all slides + notes</a><a href="../">all hadronic W folders</a>{EXTRA_LINKS}</div>
<div id="vw-top"><div id="vw-main"><nav id="vw-nav" aria-label="slides">{navhtml(S)}</nav><div id="vw-col"><div id="vw-stage">
{"".join(h for _, h, _ in S)}
</div><div id="vw-prog"><div></div></div>
<div id="vw-controls"><button id="vw-prev" title="previous (←)">&#8592;</button><span id="vw-count"></span><button id="vw-next" title="next (→)">&#8594;</button>
<button id="vw-tnav" title="contents (c)">contents</button><button id="vw-tnotes" title="speaker notes (n)">notes</button><button id="vw-full" title="fullscreen (f)">fullscreen</button>
<span style="color:#5B6573">keys ← → c n f · plots: hover for values, “edit” to move labels, “save” for PNG/SVG/PDF</span></div>
<div id="vw-notes"></div></div></div></div>
<div id="vw-split" title="drag to resize"></div>
<div id="vw-bottom"><div class="head"><span>Files: slides, figure sources (Plotly JSON), analysis scripts, results.json</span><a href="files/" target="_blank">open in a new tab</a></div>
<iframe src="files/" title="file browser"></iframe></div>
<script>const NOTES = {notes};
{VIEWER_JS}</script></body></html>"""
    open(os.path.join(HERE, "..", "index.html"), "w").write(viewer)

    S2 = render("s-")
    parts = "".join(f'<div class="frame"><div class="wrap">{h}</div></div>\n' + (f'<div class="notes"><b>Notes, slide {i}:</b> {H.escape(n)}</div>\n' if n else "")
                    for i, (_, h, n) in enumerate(S2, 1))
    page = f"""<!doctype html>
<html lang="en"><head>{head("fonts")}<style>
body {{ margin: 0; background: #d9dde3; font-family: 'IBM Plex Sans', Arial, sans-serif; color: #1a1a1a; }}
header {{ max-width: 1200px; margin: 24px auto 8px; padding: 0 16px; font-size: 15px; color: #3a4556; }} header a {{ color: #1F5BC0; }}
.frame {{ max-width: 1200px; margin: 16px auto 0; padding: 0 16px; }}
.wrap {{ position: relative; width: 100%; aspect-ratio: 16 / 9; overflow: hidden; box-shadow: 0 2px 10px rgba(20, 33, 61, .18); }}
.wrap > section {{ position: absolute; left: 0; top: 0; transform-origin: top left; }}
.notes {{ max-width: 1168px; margin: 8px auto 0; padding: 10px 16px; font-size: 14px; line-height: 1.45; color: #3a4556; background: #f4f5f7; border-left: 4px solid #2a6fdb; }}
footer {{ max-width: 1200px; margin: 24px auto 40px; padding: 0 16px; font-size: 14px; color: #5b6573; }}
@page {{ size: 1920px 1080px; margin: 0; }}
@media print {{ body {{ background: #fff; }} header, .notes, footer, .plotlyhep-chip {{ display: none; }} .frame {{ max-width: none; margin: 0; padding: 0; break-after: page; }}
  .wrap {{ width: 1920px; height: 1080px; aspect-ratio: auto; box-shadow: none; }} .wrap > section {{ transform: none !important; }} }}
</style></head><body>
<header>{H.escape(TITLE)} · {len(S2)} slides with speaker notes · print to PDF for one slide per page · <a href="../">slide viewer and files</a></header>
{parts}
<footer>Working draft of {DATE}. Rebuild with {BUILD} (figure sources in figs/, made by make_figs.py).</footer>
<script>
function fit() {{ document.querySelectorAll('.wrap').forEach(w => {{ w.firstElementChild.style.transform = 'scale(' + (w.clientWidth / 1920) + ')'; }}); }}
window.addEventListener('resize', fit); window.addEventListener('load', fit); fit();
window.addEventListener('beforeprint', () => document.querySelectorAll('.wrap > section').forEach(s => s.style.transform = 'none'));
window.addEventListener('afterprint', fit);
</script></body></html>"""
    open(os.path.join(HERE, "slides.html"), "w").write(page)
    print(f"wrote ../index.html and slides.html ({len(S)} slides)")


