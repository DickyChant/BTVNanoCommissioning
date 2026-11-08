"""Quarkonium production and heavy-flavour fragmentation in top quark pair events (CMS Run 3, 2022-2026, 13.6 TeV,
310 fb^-1, work in progress): the tt̄ + J/ψ analysis in one animation, the project's stand-alone video. Every distribution is
redrawn from the analysis figures (<files>/figs/*.json, plotly: data points with their errors, fits and simulation as lines)
and every number is read from the fit outputs (<files>/*.json); <files> is the analysis outputs directory, env TTJ_FILES
(default: DEFAULT_FILES below). Manim Community 0.19, Text / MarkupText only (no LaTeX).
Chapters:
   1 title                       2 motivation: QCD, quarkonium, J/ψ in jets, tt̄ as a laboratory, prompt vs b-hadron J/ψ
   3 data and selection          4 the J/ψ vertex: ℓ_xy, and |z0| (pileup)
   5 three categories: b jets, additional jets, outside the top-decay jets; the W-pair veto; z of the J/ψ; our private
     generator-level Pythia 8 + LETO study (not a CMS sample; the CMS simulation is chapter 7, always named as such)
   6 signal extraction: mass, decay length, three fit models, per year
   7 the CMS tt̄ simulation: closure, short b decays in untagged jets, data against simulation
   8 which fit model and b-decay shape: toy closure          9 associated production and multiple parton scattering (interpretation)
  10 generators and tunes (preliminary)   11 summary and next steps
render (the anim environment of pixi.toml; from this directory; CHAPTERS=1,5 renders a subset):
    pixi run -e anim manim render -r 1920,1080 --fps 30 --format mp4 -o ttjpsi ttjpsi.py TtJpsi
    (preview: pixi run -e anim manim render -ql ttjpsi.py TtJpsi)
the result is copied over <www>/hadW_ntrack_ttbar/animation/ttjpsi.mp4 (one fixed location, overwritten on every update)."""
import base64
import json
import os
import re
import sys

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Axes, Circle, Create, DashedLine, DashedVMobject, Dot, FadeIn, FadeOut,
                   GrowFromCenter, Indicate, LaggedStart, Line, MarkupText, Polygon, Rectangle, Scene, Square, Text,
                   Triangle, VGroup, VMobject, Write, config)

DEFAULT_FILES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the scripts copied into the outputs directory (README)
FILES = os.environ.get("TTJ_FILES", DEFAULT_FILES)
CHAPTERS = {int(c) for c in os.environ.get("CHAPTERS", "1,2,3,4,5,6,7,8,9,10,11").split(",")}


def J(name):
    return json.load(open(os.path.join(FILES, name)))


BG, INK, INK2, GREY, LTC = "#F7F6F2", "#14213D", "#3A4556", "#9AA3AD", "#5D6673"
B, P, BLUE, TOT = "#eb6834", "#1baf7a", "#2a78d6", "#4a3aa7"
FONT = "DejaVu Sans"
config.background_color = BG
O3 = np.array([0.0, 0.0, 0.0])

# ------------------------------------------------------------------------------------------------ the numbers
R, FJ, FL, SPL = J("results_run3.json"), J("fit_jpsi.json"), J("fit_lxy.json"), J("splot.json")
MCC, MTF, GX, GEN, GZ = J("mc_compare.json"), J("mc_template.json"), J("gen_compare.json"), J("gen_assoc.json"), J("gen_leak_z.json")
BAL, WP, GV, TC = J("ttbar_balance.json"), J("wpair_check.json"), J("gen_variants.json"), J("toy_closure.json")
# the per-category b-decay template (btemplate_mc.py), the data fits with it (btmpl/) and the toys with it
BTM, BS, TCB = J("btemplate.json"), J(os.path.join("btmpl", "splot.json")), J("toy_closure_btmpl.json")
YEARS = ["2022", "2023", "2024", "2025", "2026"]
# the integrated luminosity per year, as the figures of the outputs directory use it
LUMI = {k: float(v) for k, v in re.findall(r'"(20\d\d)": ([0-9.]+)', re.search(r"^LUMI = \{.*\}$", open(os.path.join(FILES, "build_deck_assoc.py")).read(), re.M).group(0))}
LUMI_ALL = sum(LUMI.values())
CHN = {"mujets": "μ + jets", "ejets": "e + jets", "emu": "e μ"}
NTT = {c: R["channels"][c]["yields"]["data_os_all_ge1b"]["n_tt"] for c in CHN}
NTT_ALL = sum(NTT.values())
NTT_Y = {y: sum(J(f"years/results_{y}.json")["channels"][c]["yields"]["data_os_all_ge1b"]["n_tt"] for c in CHN) for y in YEARS}
MASS, SH = FJ["mass"], FJ["shape"]
F_U = SPL["fits"]["notb"]  # the unbinned m x l_xy fit of the signal (nominal)
NP_U, NPE_U = F_U["N"][0], 0.5 * (F_U["NP_minos"][1] - F_U["NP_minos"][0])
RATE, RATE_E = NP_U / NTT_ALL, NPE_U / NTT_ALL
F_LXY = FL["params"]["fp_notb"] * FL["params"]["N_notb"]
L2 = FJ["lt2d"]["samples"]["notb"]
CATN = {k: sum(R["channels"][c]["yields"][f"data_os_{k}_ge1b"]["n"] for c in CHN) for k in ("inb", "ttj", "notb")}
INJET, NOJET = SPL["hists"]["notb_prompt_cat_all"][0]
TTJ_U = SPL["fits"]["ttj"]["N"][0]
TTJ_C = MTF["fits"]["ttj_ljets_nbcon"]
W1 = MTF["fits"]["signal_all_w1"]
CL = MCC["closure"]
SHORT = {c: MCC["truth"][f"{c}_all_prompt_region"]["b decay"] / MCC["truth"][f"{c}_all_all"]["b decay"] for c in ("ttj", "inb")}
RS, RI, EV = MCC["rates"]["signal_all"], MCC["rates"]["inb_all"], MCC["events"]
GL = GEN["leak"]
_t, _i = GEN["table"]["top decay"], GEN["table"]["ISR"]
TOPSHARE = _t["other ttbar jet"] / (_t["other ttbar jet"] + _i["other ttbar jet"])
RES = [GL["leak_per_ttj"] * TTJ_C["N_p"] * TOPSHARE, GZ["variants"]["nominal"]["leak_per_vetoed_ttjet"] * TTJ_C["N_p"], GL["signal_top_decay_share"] * NP_U]
DPS = SPL["dps"]
PRED = {float(k): v for k, v in DPS["pred_prompt_notb_by_sigma_eff_mb"].items()}
SNC = GX["scaled"]["nominal_bcorr"]
BP = {k: BAL["profile"][f"{k}_excl_all"]["ppar_all_pt"] for k in ("prompt", "sps", "dps", "pileup")}
TCP = {(p["scenario"], p["inject"]): p for p in TC["points"]}
TCPB = {(p["scenario"], p["inject"]): p for p in TCB["points"]}
F_B = BS["fits"]["notb"]  # the unbinned fit with the per-category b-decay template (proposed nominal)
NP_B, NPB_LO, NPB_HI = F_B["N"][0], -F_B["NP_minos"][0], F_B["NP_minos"][1]
RATE_B, RATE_BE = NP_B / NTT_ALL, 0.5 * (NPB_LO + NPB_HI) / NTT_ALL
HAS_BKG = bool(EV["mujets"].get("bkg"))


def simp(r, sp):
    """the simulation of one species: ttbar + the non-ttbar backgrounds (QCD not stacked) when processed"""
    return r[f"mc_{sp}"][0] + (r[f"bkg_{sp}"][0] if HAS_BKG and f"bkg_{sp}" in r else 0.0)
GR = GV["ratio"]
PL = SPL["pu_leak"]
WPP = WP["wpair"]


def s0(x, d=0, sign=False):
    """number with a true minus sign (0, not −0, when it rounds to zero)"""
    if round(x, d) == 0:
        return f"{0:.{d}f}"
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace("-", "−")


# ------------------------------------------------------------------------------------------------ the narration
# The spoken text of each segment, in written form (it is also the subtitle text): anim/voice.py respells it for the speech
# synthesis and writes voice/<key>.wav and voice/durations.json; the scene starts a segment with say(key), logs its start
# time ("voice <key> at <t> s") and holds the frame with hold() until the segment has been spoken; anim/mix.py lays the voice
# and the subtitles on the joined video. Without voice/durations.json the scene keeps its own timing.
VOICE_DIR = os.environ.get("TTJ_VOICE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice"))
VOICE_LEAD, VOICE_TAIL = 0.3, 0.7  # s of silence before and after each segment
_vd = os.path.join(VOICE_DIR, "durations.json")
VDUR = {k: v["dur"] for k, v in json.load(open(_vd)).items()} if os.path.exists(_vd) else {}


def narration():
    a, b = TCPB[("mc", 100)], TCPB[("datalike", 100)]
    d = TCP[("datalike", 100)]
    lx = [q["lxy"]["bias"] for q in TCPB.values()]
    cs, ct = CL["signal"], CL["ttj"]
    sig = f"{F_B['signif_prompt']:.1f} standard deviations"
    g = {k: GR[f"{k}|signal"]["r"] for k in ("data", "Pythia 8 CP5", "CP5 CR1")}
    n_all, n_inb = (float(f"{MASS[k]['Ns']:.2g}") for k in ("all", "inb_pv"))
    rb = (RI["data_nonprompt"][0] / simp(RI, "b decay"), RS["data_nonprompt"][0] / simp(RS, "b decay"))
    return {
        "title": f"Quarkonium production and heavy-flavour fragmentation in top quark pair events: a CMS Run 3 analysis with {LUMI_ALL:.0f} "
                 "inverse femtobarns at 13.6 TeV. Work in progress.",
        "mot1": "Heavy-quark production tests QCD from start to finish: a perturbative hard scattering and parton radiation, then "
                "non-perturbative hadron formation. Quarkonium, a bound state of a heavy quark and its antiquark, is the hardest case. "
                "How it is produced, and how parton showers should model it, is still open.",
        "mot2": "A J/ψ inside a jet gives extra handles: the fraction z of the jet momentum it carries, and the hadrons around it. CMS has "
                "shown that such observables are sensitive to quarkonium production in non-relativistic QCD, and Pythia 8 with the LETO "
                "module now puts this production into the parton shower.",
        "mot3": "Top quark pairs make a distinctive laboratory. Each top quark decays to a W boson and an energetic b quark, and extra "
                "radiation adds light, charm, bottom and gluon jets: several different environments for quarkonium in one "
                "well-understood process.",
        "mot4": "J/ψ mesons from b-hadron decays are the reference: they measure b-quark fragmentation. Prompt J/ψ probe different "
                "physics, from gluon-initiated processes to bound-state formation, in a topology where the top-decay jets are separated "
                "from the extra radiation.",
        "data": f"We use the CMS Run 3 data of 2022 to 2026: {LUMI_ALL:.0f} inverse femtobarns. Top quark pairs are selected with a muon "
                f"or an electron plus jets, or with an electron and a muon, and at least one b-tagged jet: {NTT_ALL / 1e6:.1f} million "
                "events. The J/ψ is reconstructed from two muons.",
        "vtx1": "The decay length separates the two sources. A prompt J/ψ starts at the primary vertex. A b hadron flies a few "
                "millimetres before it decays, so its J/ψ is displaced.",
        "vtx2": "J/ψ from other proton collisions in the same bunch crossing sit away from the primary vertex along the beam. They are "
                "removed, and they measure the J/ψ rate per collision, which we use later.",
        "cat1": f"Where the J/ψ sits defines three categories. First, J/ψ from b-hadron decays in the b jets of the top decays: about "
                f"{n_inb:.0f} of them, the fragmentation reference.",
        "cat2": "Second, prompt J/ψ in additional jets from QCD radiation.",
        "cat3": f"Third, prompt J/ψ away from every top-decay jet: inclusive associated production. We find {NP_B:.0f} of them, {sig}, "
                "statistical only.",
        "cat4": "J/ψ in the jets of the W decay are top-decay products too, so they are vetoed from the signal.",
        "wpeak": f"Why veto the W jets? {100 * WPP['share_in_w_of_ttj']:.0f} percent of the vetoed prompt J/ψ sit in a jet of the W pair, "
                 "and that pair shows a W mass peak: these J/ψ come from the top decay.",
        "leakz": f"In those jets the J/ψ carries on average {GZ['mean_z_data']:.2f} of the jet momentum in data, against "
                 f"{GZ['mean_z_pythia']:.2f} in our private generator-level Pythia with LETO: a direct handle on quarkonium fragmentation.",
        "genleak": f"Our private generator-level study, which is not a CMS sample, shows that {100 * GL['eff']:.0f} percent of the prompt "
                   f"J/ψ from top decays still pass the veto. In data, {min(RES):.0f} to {max(RES):.0f} of the signal J/ψ may still be "
                   "top-decay products.",
        "mass": f"Now the signal extraction. The dimuon mass shows a clear J/ψ peak: {MASS['notb_pv']['Ns']:.0f} J/ψ outside the "
                "top-pair jets, from the primary vertex.",
        "fit2d": f"A fit of mass and decay length together separates prompt J/ψ from b decays: {NP_U:.0f} prompt J/ψ with the "
                 "decay-length shape of the b jets.",
        "models": f"Four fit models are compared. The proposed nominal, an unbinned fit with a b-decay shape for each category, finds "
                  f"{NP_B:.0f} prompt J/ψ, {sig}. The fit of the decay length alone runs low.",
        "years": "The prompt fraction is compatible in every year from 2022 to 2026.",
        "sim1": f"The official CMS top-pair simulation, POWHEG with Pythia 8, goes through the same chain. The fit closes: it finds "
                f"{cs['fit_prompt']:.0f} prompt J/ψ for {cs['truth_prompt']:.0f} true ones.",
        "sim2": "In untagged b jets the b hadrons fly less, so a single b-jet template would take them for prompt J/ψ.",
        "sim3": f"With one template, the fit finds {ct['fit_prompt']:.0f} prompt J/ψ for {ct['truth_prompt']:.0f} true ones in the vetoed "
                "jets. The fix is a b-decay shape for each category.",
        "sim4": "So every fit uses a b-decay shape per category, from the CMS simulation and calibrated on the data b jets. Outside the "
                "top-pair jets, the b decays look like those in b jets.",
        "sim5": f"Against the data, the b decays: data over simulation is {rb[0]:.2f} in the b jets and {rb[1]:.2f} outside. But with "
                f"prompt J/ψ from hadronisation only, the simulation predicts {simp(RS, 'prompt'):.0f} prompt J/ψ outside the top-pair "
                f"jets, {RS['data_prompt'][0] / simp(RS, 'prompt'):.0f} to {NP_B / simp(RS, 'prompt'):.0f} times below the data.",
        "sim6": "The same pattern holds in each channel.",
        "toy1": f"Which fit should be the nominal? Toys of the data's size, drawn from the CMS simulation with a chosen number of prompt "
                f"J/ψ injected, run through every fit. The fit of the decay length alone runs low, by about {-0.5 * (max(lx) + min(lx)):.0f}.",
        "toy2": f"If the b jets hold some prompt J/ψ, as the data suggest, the b-jet shape copies that share onto the signal: it finds "
                f"{-d['unbinned']['bias']:.0f} too few of {d['unbinned']['truth']:.0f}. The per-category shape stays within "
                f"{max(abs(a['unbinned']['bias']), abs(b['unbinned']['bias'])):.0f}. Hence the proposed nominal: {NP_B:.0f} prompt "
                f"J/ψ, {sig}.",
        "dps1": f"Could a second parton scattering make these J/ψ? The rate measured with the pileup J/ψ predicts about {PRED[15.0]:.0f} "
                "at an effective cross section of 15 millibarns. This comparison depends on the model; it is not a measurement.",
        "dps2": "The J/ψ are soft, so the transverse-momentum balance of the event cannot separate single from double parton scattering.",
        "dps3": f"In lepton plus jets, our private generator-level Pythia with LETO, scaled to the vetoed prompt J/ψ, accounts for "
                f"{SNC['sps_signal']:.0f} of the signal. The remaining {SNC['dps_left']:.0f} are compatible with a second scattering, within these models.",
        "gen": f"Finally, other generators and tunes in the CMS simulation. The ratio of prompt to b-decay J/ψ needs no normalisation: "
               f"{g['data']:.2f} in data outside the top-pair jets, against {g['Pythia 8 CP5']:.3f} for POWHEG with Pythia 8 and "
               f"{g['CP5 CR1']:.3f} with the CR1 colour reconnection. Both Sherpa tunes make none. The dilepton samples show the same pattern: "
               "every sample falls short.",
        "summary": f"To summarise: about {n_all:.0f} J/ψ in {NTT_ALL / 1e6:.1f} million top-pair events, mostly from b-hadron decays. "
                   f"Away from every top-decay jet we find {NP_B:.0f} prompt J/ψ, {sig}, statistical only. Some may still be top-decay "
                   f"products, and the CMS simulation predicts about {simp(RS, 'prompt'):.0f}. Top quark pair events are a common "
                   "laboratory for b fragmentation and for how prompt quarkonium forms. Next come the differential z distribution, the "
                   "systematic uncertainties and the residual top-decay J/ψ.",
    }


# ------------------------------------------------------------------------------------------------ text
SUPU = str.maketrans("0123456789−+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁻")


def markup(s, size):
    """plotly / html-like label -> Pango markup of this Pango (rise in absolute units: scaled with the font size)"""
    s = (s or "").replace("<br>", "\n").replace("t̅t", "tt̄")
    s = re.sub(r"<sup>([0-9−+\-]+)</sup>", lambda m: m.group(1).translate(SUPU), s)
    s = s.replace("<sub>", f"<span size='smaller' rise='{-int(size * 75)}'>").replace("</sub>", "</span>")
    return s.replace("<sup>", f"<span size='smaller' rise='{int(size * 170)}'>").replace("</sup>", "</span>")


def T(s, size=26, color=INK, bold=False):
    return Text(s.replace("t̅t", "tt̄"), font=FONT, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def M(s, size=26, color=INK):
    return MarkupText(markup(s, size), font=FONT, font_size=size, color=color)


def _plain(s):
    return re.sub(r"&[a-z]+;", "x", re.sub(r"<[^>]*>", "", s))


def para(s, size=22, width=12.4, color=INK2):
    """markup text broken into lines that fit the width (manim units)"""
    cfac = 0.97
    for _ in range(6):
        lines = []
        for p in s.split("\n"):
            full = M(p, size, color)
            if full.width <= width:
                lines.append(p)
                continue
            cmax = max(int(width / (full.width / max(len(_plain(p)), 1)) * cfac), 8)
            cur, n = [], 0
            for t in re.findall(r"(?:<[^>]*>|[^\s<])+", p):
                k = len(_plain(t))
                if cur and n + 1 + k > cmax:
                    lines.append(" ".join(cur))
                    cur, n = [], 0
                cur.append(t)
                n += k + (1 if len(cur) > 1 else 0)
            lines.append(" ".join(cur))
        m = M("\n".join(lines), size, color)
        if m.width <= width * 1.01:
            return m
        cfac *= 0.92
    return m.scale_to_fit_width(width)


# ------------------------------------------------------------------------------------------------ the analysis figures
def arr(v):
    if v is None:
        return None
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"])).astype(float)
    return np.array([np.nan if x is None else x for x in v], float)


def colour(c, default=INK):
    """'#rrggbb' or 'rgba(r,g,b,a)' -> (hex, opacity)"""
    if not c or not isinstance(c, str):
        return default, 1.0
    m = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([0-9.]+))?\)", c)
    if m:
        r, g, b = (int(m.group(i)) for i in (1, 2, 3))
        return f"#{r:02x}{g:02x}{b:02x}", float(m.group(4) or 1)
    return c, 1.0


def fig(name):
    f = json.load(open(os.path.join(FILES, "figs", name + ".json")))
    out = []
    for t in f["data"]:
        if t.get("xaxis", "x") != "x":
            continue  # the main panel only
        ey = t.get("error_y") if isinstance(t.get("error_y"), dict) else {}
        ex = t.get("error_x") if isinstance(t.get("error_x"), dict) else {}
        out.append(dict(name=t.get("name"), type=t.get("type", "scatter"), mode=t.get("mode") or "", x=arr(t.get("x")), y=arr(t.get("y")),
                        ey=arr(ey.get("array")), ex=arr(ex.get("array")), line=t.get("line") or {}, marker=t.get("marker") or {},
                        fill=t.get("fill"), fillcolor=t.get("fillcolor"), width=t.get("width"), showlegend=t.get("showlegend", True)))
    return out, f["layout"]


def nice_ticks(a, b, n=5):
    span = b - a
    p = 10 ** np.floor(np.log10(span / n))
    step = min((s * p for s in (1, 2, 2.5, 5, 10)), key=lambda s: abs(span / s - n))
    return [round(v, 10) + 0.0 for v in np.arange(np.ceil(a / step) * step, b + 1e-9 * step, step)]


def fmt_tick(v, log=False):
    if log:
        e = int(round(v))
        return {-3: "0.001", -2: "0.01", -1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1000"}.get(e, "10" + str(e).replace("-", "−").translate(SUPU))
    if abs(v) >= 1000:
        return f"{v / 1000:g}k"
    return f"{v if abs(v) > 1e-12 else 0.0:g}".replace("-", "−")


def marker(sym, col):
    s = (sym or "circle").replace("-open", "")
    if s == "square":
        m = Square(0.13)
    elif s == "diamond":
        m = Square(0.13).rotate(np.pi / 4)
    elif s.startswith("triangle"):
        m = Triangle().scale(0.09)
        if "down" in s:
            m.rotate(np.pi)
    else:
        m = Dot(radius=0.06)
    m.set_color(col)
    if "open" in (sym or ""):
        m.set_fill(opacity=0).set_stroke(col, width=2.5)
    else:
        m.set_fill(col, opacity=1)
    return m


PW, PH = 6.3, 4.3  # the plot of a page
PC = np.array([-2.75, 0.2, 0.0])  # its frame centre
RC_L, RC_T, RC_B, RC_W = 1.0, 2.85, -3.75, 5.85  # the right column of a page: left, top, bottom, width


class Plot:
    """The main panel of one analysis figure (figs/<name>.json) redrawn with Manim: frame, ticks, the figure's own axis titles and
    CMS label, data points with error bars, fits and simulation as lines, filled components, bars and bands. name=None: an
    empty frame (xr, yr, labels given) to draw into. A figure of our private gen-level Pythia 8 + LETO sample has its own label
    (bold generator name, 'private, gen level') instead of the CMS one, and keeps it."""

    def __init__(self, name, w=PW, h=PH, xlabel=None, ylabel=None, xr=None, yr=None, xticks=None, yticks=None, skip=()):
        self.tr, L = fig(name) if name else ([], {"xaxis": {}, "yaxis": {}})
        xa, ya = L["xaxis"], L["yaxis"]
        lt = (L.get("legend") or {}).get("title")
        self.ltitle = lt.get("text") if isinstance(lt, dict) else ""
        exp, extra, lumi, xl_, yl_ = "CMS", "Work in Progress", "" if name else f"2022–2026, {LUMI_ALL:.0f} fb<sup>−1</sup> (13.6 TeV)", None, None
        for a in L.get("annotations") or []:
            t = a.get("text") or ""
            if t == "<b>CMS</b>":
                continue
            if t.startswith("<b>") and t.endswith("</b>") and a.get("y") == 1 and a.get("yanchor") == "bottom":
                exp = t[3:-4]  # not a CMS figure: the private gen-level sample (cmsdeck.label(gen=True))
                continue
            if t.startswith("<i>"):
                extra = t[3:-4]
            elif "TeV" in t:
                lumi = t
            elif a.get("textangle") in (-90, 270):
                yl_ = yl_ or t
            else:
                xl_ = xl_ or t
        xlabel = xl_ if xlabel is None else xlabel
        ylabel = yl_ if ylabel is None else ylabel
        self.logx, self.logy = xa.get("type") == "log", ya.get("type") == "log"
        self.xr, self.yr = list(xr or xa["range"]), list(yr or ya["range"])
        self.ax = ax = Axes(x_range=[self.xr[0], self.xr[1], self.xr[1] - self.xr[0]], y_range=[self.yr[0], self.yr[1], self.yr[1] - self.yr[0]],
                            x_length=w, y_length=h, tips=False, axis_config=dict(stroke_width=0, include_ticks=False))
        c0, c1 = ax.c2p(self.xr[0], self.yr[0]), ax.c2p(self.xr[1], self.yr[1])
        frame = Rectangle(width=c1[0] - c0[0], height=c1[1] - c0[1], color=INK, stroke_width=2.5).move_to((c0 + c1) / 2)
        ticks, xtl, ytl = VGroup(), VGroup(), VGroup()
        cat = False  # (tick values in axis units: log10 on a log axis)
        if xticks is None and xa.get("tickvals") is not None and xa.get("ticktext") is not None:
            xticks = [(np.log10(v) if self.logx else v, s) for v, s in zip(xa["tickvals"], xa["ticktext"])]
            cat = True
        if xticks is None:
            xticks = [(v, fmt_tick(v, self.logx)) for v in (range(int(np.ceil(self.xr[0])), int(np.floor(self.xr[1])) + 1) if self.logx else nice_ticks(*self.xr))]
        xticks = [(u, s) for u, s in xticks if self.xr[0] - 1e-9 <= u <= self.xr[1] + 1e-9]
        for u, s in xticks:
            p = ax.c2p(u, self.yr[0])
            ticks.add(Line(p, p + 0.12 * UP, color=INK, stroke_width=2))
            if cat and "<br>" in s:  # lines given by the figure: centred, with a gap to the neighbouring labels
                lab = VGroup(*[M(t, 16, INK2) for t in s.split("<br>")]).arrange(DOWN, buff=0.06)
                lab.scale_to_fit_width(min(lab.width, 0.8 * w / len(xticks)))
            else:
                lab = para(s, 16, 0.92 * w / len(xticks), INK2) if cat else M(s, 17, INK2)
            xtl.add(lab.next_to(p, DOWN, buff=0.1))
        if yticks is None:
            yticks = [(v, fmt_tick(v, self.logy)) for v in (range(int(np.ceil(self.yr[0])), int(np.floor(self.yr[1])) + 1) if self.logy else nice_ticks(*self.yr))]
        for v, s in yticks:
            if not (self.yr[0] - 1e-9 <= v <= self.yr[1] + 1e-9):
                continue
            p = ax.c2p(self.xr[0], v)
            ticks.add(Line(p, p + 0.12 * RIGHT, color=INK, stroke_width=2))
            ytl.add(M(s, 17, INK2).next_to(p, LEFT, buff=0.1))
        labs = VGroup()
        if xlabel:
            labs.add(M(xlabel, 20, INK).next_to(VGroup(frame, xtl), DOWN, buff=0.12).align_to(frame, RIGHT))
        if ylabel:
            labs.add(M(ylabel, 20, INK).rotate(np.pi / 2).next_to(VGroup(frame, ytl) if len(ytl) else frame, LEFT, buff=0.15).align_to(frame, UP))
        top = VGroup(M(f"<b>{exp}</b> <i>{extra}</i>", 18, INK).next_to(frame, UP, buff=0.08).align_to(frame, LEFT))
        if lumi:
            lm = M(lumi, 15, INK2)
            if lm.width > frame.width - top.width - 0.25:
                lm.scale_to_fit_width(frame.width - top.width - 0.25)
            top.add(lm.next_to(frame, UP, buff=0.08).align_to(frame, RIGHT))
        self.frame = VGroup(frame, ticks, xtl, ytl, labs, top)
        self.box = frame
        self.shapes, self.lines, self.fills, self.points, self.leg = VGroup(), VGroup(), VGroup(), VGroup(), []
        for s in L.get("shapes", []) or []:
            self._shape(s)
        prev = None
        for t in self.tr:
            nm, x, y = t["name"], t["x"], t["y"]
            if nm in skip or t["type"] == "bar":
                prev = t
                continue
            if x is None or y is None or not np.isfinite(x).any():  # a legend entry only (a band, or lines drawn as shapes)
                if nm and t["showlegend"] is not False:
                    col, op = colour(t["marker"].get("color") or t["line"].get("color"), INK)
                    if "lines" in t["mode"]:
                        kind = "line"
                    elif op < 0.9 or t["marker"].get("symbol") == "square":
                        kind = "fill" if op >= 0.9 else f"band{max(op, 0.3):.2f}"
                    else:
                        kind = t["marker"].get("symbol") or "circle"
                    self.leg.append((nm, kind, col))
                prev = t
                continue
            col, op = colour(t["line"].get("color") or t["marker"].get("color"), INK)
            if t["fill"] == "tonexty" and prev is not None:
                fc, fo = colour(t["fillcolor"], col)
                self.fills.add(self._fill(prev, t, fc, fo))
                if nm and t["showlegend"] is not False:
                    self.leg.append((nm, "fill", fc))
            elif "lines" in t["mode"]:
                dashed = t["line"].get("dash") in ("dash", "dot", "dashdot")
                ln = self._polyline(x, y, t["line"].get("shape") == "hv", col, 3.2 if (t["line"].get("width") or 2) >= 2 else 2.4)
                if dashed:
                    ln = VGroup(*[DashedVMobject(m, num_dashes=max(int(m.get_arc_length() / 0.12), 2), dashed_ratio=0.55) for m in ln])
                self.lines.add(ln)
                if nm and t["showlegend"] is not False:
                    self.leg.append((nm, "dash" if dashed else "line", col))
            elif "markers" in t["mode"]:
                mc, _ = colour(t["marker"].get("color"), col)
                sym = t["marker"].get("symbol") or "circle"
                self.points.add(self._markers(x, y, t["ey"], t["ex"], mc, sym))
                if nm and t["showlegend"] is not False:
                    self.leg.append((nm, sym, mc))
            prev = t
        self.barsg = self._bars()
        self.group = VGroup(self.frame, self.shapes, self.barsg, self.fills, self.lines, self.points)

    def place(self, centre):
        self.group.shift(np.array(centre) - self.box.get_center())
        return self

    # coordinates, clipped to the frame
    def p(self, x, y):
        u = np.log10(x) if (self.logx and x > 0) else (self.xr[0] if self.logx else x)
        v = np.log10(y) if (self.logy and y > 0) else (self.yr[0] if self.logy else y)
        return self.ax.c2p(min(max(u, self.xr[0]), self.xr[1]), min(max(v, self.yr[0]), self.yr[1]))

    def _axis(self, x, y):
        """data -> axis units (log10 on a log axis; NaN where a log axis cannot show the value)"""
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), np.nan) if self.logx else np.asarray(x, float)
            v = np.where(y > 0, np.log10(np.where(y > 0, y, 1.0)), np.nan) if self.logy else np.asarray(y, float)
        return u, v

    def _segments(self, x, y, hv, clamp=False):
        """polylines in scene coordinates: broken at missing points, clipped to the x range; parts above or below the frame
        left out (clamp: pressed onto the frame edge, for the outline of a filled area)"""
        u, v = self._axis(np.asarray(x, float), np.asarray(y, float))
        if clamp:
            v = np.where(np.isfinite(u) & ~np.isfinite(v), self.yr[0], v)
        runs, cur = [], []
        for i in range(len(u)):
            if not (np.isfinite(u[i]) and np.isfinite(v[i])):
                if len(cur) > 1:
                    runs.append(cur)
                cur = []
                continue
            if hv and cur:
                cur.append((u[i], cur[-1][1]))
            cur.append((u[i], v[i]))
        if len(cur) > 1:
            runs.append(cur)
        x0, x1 = self.xr
        segs = []
        for run in runs:
            pts = []
            for (ua, va), (ub, vb) in zip(run[:-1], run[1:]):
                y0, y1 = self.yr
                if (ua < x0 and ub < x0) or (ua > x1 and ub > x1) or (not clamp and ((va < y0 and vb < y0) or (va > y1 and vb > y1))):
                    if len(pts) > 1:
                        segs.append(pts)
                    pts = []
                    continue
                ca, cb = (ua, va), (ub, vb)
                if ub != ua:
                    for xe in (x0, x1):
                        t = (xe - ua) / (ub - ua)
                        if 0 < t < 1 and (min(ua, ub) < xe < max(ua, ub)):
                            pe = (xe, va + t * (vb - va))
                            if (ua < xe) == (xe == x0):  # a enters later: a lies outside on this side
                                ca = pe
                            else:
                                cb = pe
                q = [self.ax.c2p(a, min(max(b, self.yr[0]), self.yr[1])) for a, b in (ca, cb)]
                if not pts:
                    pts.append(q[0])
                pts.append(q[1])
            if len(pts) > 1:
                segs.append(pts)
        return segs

    def _polyline(self, x, y, hv, col, sw):
        return VGroup(*[VMobject(color=col, stroke_width=sw).set_points_as_corners(s) for s in self._segments(x, y, hv)])

    def _fill(self, lo, hi, col, op):
        a = self._segments(hi["x"], hi["y"], hi["line"].get("shape") == "hv", clamp=True)
        b = self._segments(lo["x"], lo["y"], lo["line"].get("shape") == "hv", clamp=True)
        if not a or not b:
            return VGroup()
        return Polygon(*(a[0] + b[0][::-1]), stroke_width=0, fill_color=col, fill_opacity=max(op, 0.35))

    def _markers(self, x, y, ey, ex, col, sym):
        g = VGroup()
        for i in range(len(x)):
            u, v = self._axis(np.array([x[i]]), np.array([y[i]]))
            if not (np.isfinite(u[0]) and np.isfinite(v[0]) and self.xr[0] <= u[0] <= self.xr[1] and self.yr[0] <= v[0] <= self.yr[1]):
                continue
            if ey is not None and i < len(ey) and np.isfinite(ey[i]) and ey[i] > 0:
                g.add(Line(self.p(x[i], y[i] - ey[i]), self.p(x[i], y[i] + ey[i]), color=col, stroke_width=2.2))
            if ex is not None and i < len(ex) and np.isfinite(ex[i]) and ex[i] > 0:
                g.add(Line(self.p(x[i] - ex[i], y[i]), self.p(x[i] + ex[i], y[i]), color=col, stroke_width=2.2))
            g.add(marker(sym, col).move_to(self.p(x[i], y[i])))
        return g

    def _shape(self, s):
        if s.get("type") not in ("rect", "line") or str(s.get("yref", "y")).startswith("y2") or str(s.get("xref", "x")).startswith("x2"):
            return
        col, op = colour(s.get("fillcolor") or (s.get("line") or {}).get("color"), GREY)
        xd, yd = "domain" in str(s.get("xref")), "domain" in str(s.get("yref"))

        def X(v):
            return self.xr[0] + v * (self.xr[1] - self.xr[0]) if xd else (np.log10(v) if self.logx and v > 0 else v)

        def Y(v):
            return self.yr[0] + v * (self.yr[1] - self.yr[0]) if yd else (np.log10(v) if self.logy and v > 0 else v)

        def clip(u, r):
            return min(max(u, r[0]), r[1])

        a = self.ax.c2p(clip(X(s["x0"]), self.xr), clip(Y(s["y0"]), self.yr))
        b = self.ax.c2p(clip(X(s["x1"]), self.xr), clip(Y(s["y1"]), self.yr))
        if s["type"] == "rect":
            self.shapes.add(Polygon(a, [b[0], a[1], 0], b, [a[0], b[1], 0], stroke_width=0, fill_color=col, fill_opacity=max(op, 0.12)))
        else:
            lc, _ = colour((s.get("line") or {}).get("color"), GREY)
            self.shapes.add(DashedLine(a, b, color=lc, stroke_width=2))

    def _bars(self):
        g, base = VGroup(), {}
        for t in self.tr:
            if t["type"] != "bar":
                continue
            col, _ = colour(t["marker"].get("color"), GREY)
            w = (t["width"] or 0.5) / 2
            for x, y in zip(t["x"], t["y"]):
                y0 = base.get(x, 0.0)
                a, b = self.p(x - w, y0), self.p(x + w, y0 + y)
                g.add(Polygon(a, [b[0], a[1], 0], b, [a[0], b[1], 0], stroke_width=0, fill_color=col, fill_opacity=0.85))
                base[x] = y0 + y
            self.leg.insert(0, (t["name"], "fill", col))
        return g

    def legend(self, size=15, width=5.6, title=True):
        rows = VGroup()
        for nm, kind, col in self.leg:
            if kind == "line":
                sw = Line(O3, 0.45 * RIGHT, color=col, stroke_width=4)
            elif kind == "dash":
                sw = DashedLine(O3, 0.45 * RIGHT, color=col, stroke_width=4, dash_length=0.08)
            elif kind == "fill" or kind.startswith("band"):
                sw = Rectangle(width=0.42, height=0.2, stroke_width=0, fill_color=col, fill_opacity=0.8 if kind == "fill" else float(kind[4:]))
            else:
                sw = marker(kind, col)
            slot = Rectangle(width=0.45, height=0.22, stroke_width=0, fill_opacity=0)
            sw.move_to(slot)
            rows.add(VGroup(VGroup(slot, sw), para(nm, size, width - 0.65, INK2)).arrange(RIGHT, buff=0.18))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        if self.ltitle and title:
            rows = VGroup(para(self.ltitle, size - 1, width, LTC), rows).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        return rows


# ------------------------------------------------------------------------------------------------ the scene
class TtJpsi(Scene):
    def construct(self):
        self._head = None
        self._vend = 0.0
        for i, ch in enumerate((self.title, self.motivation, self.data, self.vertex, self.categories, self.extraction, self.simulation,
                                self.toys, self.dps, self.generators, self.summary), 1):
            if i in CHAPTERS:
                print(f"chapter {i} {ch.__name__} starts at {getattr(self.renderer, 'time', 0.0):.1f} s", flush=True)
                ch()

    # -- helpers
    def say(self, key):
        """start the narration segment `key` here (the log line is read by mix.py); hold() keeps the frame until it is spoken"""
        t = self.renderer.time
        print(f"voice {key} at {t:.4f} s", flush=True)
        self._vend = t + VOICE_LEAD + VDUR.get(key, 0.0) + VOICE_TAIL

    def hold(self, wait):
        """wait `wait` s, longer if the current narration segment is still being spoken; whole frames"""
        self.wait(max(round(30 * max(wait, self._vend - self.renderer.time)), 1) / 30)

    def set_head(self, num, title):
        t = M(f"<span foreground='{GREY}'>{num}</span>  <b>{title}</b>", 32, INK)
        if t.width > 12.9:
            t.scale_to_fit_width(12.9)
        t.to_corner(UP + LEFT, buff=0.4)
        h = VGroup(t, Line([-6.75, 0, 0], [6.75, 0, 0], color=GREY, stroke_width=2).next_to(t, DOWN, buff=0.12).set_x(0))
        if self._head is None:
            self.play(FadeIn(h), run_time=0.7)
        else:
            self.play(FadeOut(self._head), FadeIn(h), run_time=0.7)
        self._head = h

    def body_out(self):
        rest = [m for m in self.mobjects if m is not self._head]
        if rest:
            self.play(*[FadeOut(m) for m in rest], run_time=0.6)

    def clear(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)
        self._head = None

    def bullets(self, items, size=21, width=RC_W, buff=0.28):
        g = VGroup()
        for s in items:
            m = para(s, size, width - 0.25)
            d = Dot(radius=0.045, color=INK2)
            row = VGroup(d, m).arrange(RIGHT, buff=0.15)
            d.align_to(m, UP).shift(0.09 * size / 21 * DOWN)
            g.add(row)
        return g.arrange(DOWN, aligned_edge=LEFT, buff=buff)

    def draw(self, pl, run=1.4):
        self.play(FadeIn(pl.frame), run_time=0.8)
        anims = [FadeIn(g) for g in (pl.shapes, pl.barsg, pl.fills) if len(g)]
        if len(pl.lines):
            anims.append(Create(pl.lines))
        if anims:
            self.play(*anims, run_time=run)
        if len(pl.points):
            self.play(LaggedStart(*[FadeIn(p) for p in pl.points], lag_ratio=0.25), run_time=1.0)

    def column(self, parts, buff=0.35, top=RC_T):
        col = VGroup(*parts).arrange(DOWN, aligned_edge=LEFT, buff=buff)
        if col.height > top - RC_B:
            col.scale_to_fit_height(top - RC_B)
        col.move_to([RC_L + col.width / 2, top - col.height / 2, 0])
        return col

    def page(self, pl, items=(), wait=7.0, size=20, legend=True, lsize=15, out=True, key=None):
        """one plot on the left, its legend and the text in the right column; `key`: its narration segment"""
        if key:
            self.say(key)
        pl.place(PC)
        self.draw(pl)
        parts = []
        if legend and pl.leg:
            parts.append(pl.legend(lsize, RC_W))
        if items:
            parts.append(self.bullets(items, size, RC_W))
        if parts:
            self.column(parts)
            for m in parts:
                self.play(FadeIn(m), run_time=0.8)
        self.hold(wait)
        if out:
            self.body_out()

    # ------------------------------------------------------------------------------------------------ 1. title
    def title(self):
        t1 = M("<b>Quarkonium production and heavy-flavour\nfragmentation in top quark pair events</b>", 46, INK)
        if t1.width > 12.6:
            t1.scale_to_fit_width(12.6)
        t2 = M(f"tt̄ + J/ψ → μμ  ·  CMS Run 3, 2022–2026  ·  {LUMI_ALL:.0f} fb<sup>−1</sup>  ·  √s = 13.6 TeV", 26, INK2)
        t3 = M("<i>Work in progress</i>", 24, GREY)
        g = VGroup(t1, t2, t3).arrange(DOWN, buff=0.5)
        self.say("title")
        self.play(FadeIn(t1, shift=0.3 * UP), run_time=1.3)
        self.play(FadeIn(t2), FadeIn(t3), run_time=1.0)
        self.hold(4.0)
        self.play(FadeOut(g), run_time=0.8)

    # ------------------------------------------------------------------------------------------------ 2. motivation
    def motivation(self):
        self.set_head(1, "Why quarkonium in top quark pair events")
        # beat 1: from the hard scattering to hadrons
        boxes = VGroup(*[VGroup(Rectangle(width=2.6, height=1.05, color=c, stroke_width=3, fill_color=c, fill_opacity=0.12), M(s, 22, INK))
                         for s, c in (("hard\nscattering", BLUE), ("parton\nradiation", P), ("hadron\nformation", B))]).arrange(RIGHT, buff=0.8)
        for b in boxes:
            b[1].move_to(b[0])
        arrows = VGroup(*[Line(boxes[i].get_right(), boxes[i + 1].get_left(), color=INK2, stroke_width=3).add_tip(tip_length=0.18) for i in range(2)])
        VGroup(boxes, arrows).move_to(1.3 * UP)
        lab = VGroup(T("perturbative", 19, LTC).next_to(VGroup(boxes[0], boxes[1]), DOWN, buff=0.15),
                     T("non-perturbative", 19, LTC).next_to(boxes[2], DOWN, buff=0.15))
        b1 = self.bullets(["Heavy-quark production and hadronisation test QCD: perturbative hard scattering and radiation, then non-perturbative hadron formation",
                           "Quarkonium — a heavy quark–antiquark pair forming a bound state — is the hardest case: the roles of the production mechanisms, and how parton showers implement them, are not settled"],
                          23, 11.0, buff=0.35).move_to(1.5 * DOWN)
        self.say("mot1")
        self.play(LaggedStart(*[FadeIn(s) for s in (boxes[0], arrows[0], boxes[1], arrows[1], boxes[2])], lag_ratio=0.25), FadeIn(lab), run_time=2.0)
        self.play(FadeIn(b1[0]), run_time=0.9)
        self.wait(2.5)
        self.play(FadeIn(b1[1]), run_time=0.9)
        self.hold(5.5)
        self.body_out()
        # beat 2: J/psi inside jets
        jet = VGroup(*[Line(O3, 3.0 * np.array([np.cos(a), np.sin(a), 0]), color=GREY, stroke_width=2) for a in np.linspace(-0.3, 0.3, 9)])
        jp = Line(O3, 2.4 * np.array([np.cos(0.05), np.sin(0.05), 0]), color=P, stroke_width=6)
        cone = VGroup(jet, jp, T("J/ψ", 26, P, bold=True).next_to(jp.get_end(), UP, buff=0.12), T("jet", 24, GREY).next_to(jet, RIGHT, buff=0.1))
        cone.move_to(3.6 * LEFT + 0.9 * UP)
        z = M("z = p<sub>T</sub>(J/ψ) / p<sub>T</sub>(jet)", 30, INK).next_to(cone, DOWN, buff=0.6)
        b2 = self.bullets(["J/ψ inside jets give extra handles: the momentum fraction z and the correlation with the surrounding hadrons",
                           "CMS showed that J/ψ-in-jet observables are sensitive to NRQCD with fragmenting jet functions",
                           "NRQCD quarkonium production is now in parton showers (Pythia 8 + LETO): predictions for complex final states"],
                          22, 6.3, buff=0.35)
        b2.move_to([0.4 + b2.width / 2, 0.3, 0])
        self.say("mot2")
        self.play(FadeIn(cone), Write(z), run_time=1.5)
        for b in b2:
            self.play(FadeIn(b), run_time=0.8)
            self.wait(1.6)
        self.hold(4.0)
        self.body_out()
        # beat 3: ttbar as a laboratory
        chain = VGroup(M("t → W<sup>+</sup> <span foreground='#eb6834'><b>b</b></span>", 30), M("t̄ → W<sup>−</sup> <span foreground='#eb6834'><b>b̄</b></span>", 30),
                       M("+ QCD radiation:", 26), M("<span foreground='#2a78d6'>light</span>, <span foreground='#1baf7a'>charm</span>, <span foreground='#eb6834'>bottom</span>, gluon jets", 26)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        chain.move_to([-6.2 + chain.width / 2, 0.6, 0])
        b3 = self.bullets(["Top quark pairs are a distinctive laboratory: each t → W b gives an identifiable energetic b quark",
                           "extra radiation gives light, charm, bottom and gluon jets: quarkonium production in several physically different environments inside one well-characterised hard process"],
                          22, 6.6, buff=0.35)
        b3.move_to([0.0 + b3.width / 2, 0.6, 0])
        self.say("mot3")
        self.play(FadeIn(chain, shift=0.2 * RIGHT), run_time=1.4)
        self.play(FadeIn(b3), run_time=1.0)
        self.hold(7.0)
        self.body_out()
        # beat 4: non-prompt reference and prompt J/psi
        left = VGroup(M("<b>non-prompt J/ψ</b>  (b-hadron decays)", 26, B),
                      self.bullets(["the reference: b-quark fragmentation, the mix of weakly decaying b hadrons and their decays",
                                    "CMS used J/ψ + the W lepton for a top-quark mass measurement at 8 TeV"], 21, 5.9)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        right = VGroup(M("<b>prompt J/ψ</b>", 26, P),
                       self.bullets(["different QCD: heavy-quark fragmentation, gluon-initiated processes, short-distance quarkonium production, then bound-state formation",
                                     "in tt̄: the jets of the top decays separated from the extra radiation, a controlled topology",
                                     "heavy-flavour radiation in tt̄ also matters for precision top and Higgs (tt̄bb̄)"], 21, 6.2)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        VGroup(left, right).arrange(RIGHT, buff=0.6, aligned_edge=UP).move_to(0.4 * DOWN)
        self.say("mot4")
        self.play(FadeIn(left, shift=0.2 * UP), run_time=1.2)
        self.wait(4.5)
        self.play(FadeIn(right, shift=0.2 * UP), run_time=1.2)
        self.hold(8.0)
        self.clear()

    # ------------------------------------------------------------------------------------------------ 3. data and selection
    def data(self):
        self.set_head(2, "Data and selection")
        self.say("data")
        ax = Axes(x_range=[-0.6, 4.6, 1], y_range=[0, 7, 1], x_length=PW, y_length=4.0, tips=False, axis_config=dict(stroke_width=0, include_ticks=False))
        ax.move_to(PC + 0.1 * UP)
        fr = Rectangle(width=PW, height=4.0, color=INK, stroke_width=2.5).move_to(ax.get_center())
        bars, labs = VGroup(), VGroup()
        for i, y in enumerate(YEARS):
            v = NTT_Y[y] / 1e6
            a, b = ax.c2p(i - 0.32, 0), ax.c2p(i + 0.32, v)
            bars.add(Polygon(a, [b[0], a[1], 0], b, [a[0], b[1], 0], stroke_width=0, fill_color=BLUE, fill_opacity=0.8))
            labs.add(T(y, 18, INK2).next_to(ax.c2p(i, 0), DOWN, buff=0.12))
            labs.add(M(f"{LUMI[y]:.0f} fb<sup>−1</sup>", 16, LTC).next_to(ax.c2p(i, 0), DOWN, buff=0.45))
            labs.add(T(f"{v:.2f}M", 17, INK).next_to(b, UP, buff=0.06).set_x(ax.c2p(i, 0)[0]))
        yl = M("selected tt̄ events [millions]", 20, INK).rotate(np.pi / 2).next_to(fr, LEFT, buff=0.3).align_to(fr, UP)
        cms = VGroup(M("<b>CMS</b> <i>Work in Progress</i>", 18, INK).next_to(fr, UP, buff=0.08).align_to(fr, LEFT),
                     M(f"2022–2026, {LUMI_ALL:.0f} fb<sup>−1</sup> (13.6 TeV)", 15, INK2).next_to(fr, UP, buff=0.08).align_to(fr, RIGHT))
        self.play(FadeIn(fr), FadeIn(yl), FadeIn(cms), run_time=0.8)
        self.play(LaggedStart(*[GrowFromCenter(b) for b in bars], lag_ratio=0.2), FadeIn(labs), run_time=2.0)
        b = self.bullets([f"CMS Run 3, 2022–2026: <b>{LUMI_ALL:.0f} fb<sup>−1</sup></b> of pp collisions at 13.6 TeV",
                          "tt̄ selections: μ + jets, e + jets and e μ, at least one b-tagged jet",
                          "J/ψ → μμ: an opposite-sign muon pair, 2.6–3.6 GeV; its muons are never the tt̄ lepton candidates"], 21, RC_W)
        rows = VGroup(*[M(f"{n}:  <b>{NTT[c] / 1e6:.2f}M</b> tt̄ events", 21, INK2) for c, n in CHN.items()],
                      M(f"all:  <b>{NTT_ALL / 1e6:.1f}M</b> tt̄ events", 21, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        self.column([b, rows], buff=0.5, top=2.3)
        self.play(FadeIn(b), run_time=1.2)
        self.play(FadeIn(rows), run_time=1.0)
        self.hold(7.0)
        self.clear()

    # ------------------------------------------------------------------------------------------------ 4. the J/psi vertex
    def vertex(self):
        self.set_head(3, "The J/ψ vertex: decay length and collision")
        self.say("vtx1")
        o = np.array([-4.2, 0.6, 0])
        pv = Dot(o, radius=0.11, color=INK)
        pvl = T("primary vertex", 20, INK2).next_to(pv, LEFT, buff=0.2)
        pr = VGroup(*[Line(o, o + d, color=P, stroke_width=4).add_tip(tip_length=0.18) for d in (np.array([2.6, 1.5, 0]), np.array([2.9, 0.8, 0]))])
        prl = M("prompt J/ψ: ℓ<sub>xy</sub> ≈ 0", 24, P).next_to(pr, RIGHT, buff=0.25).shift(0.4 * UP)
        sv = o + np.array([1.8, -1.2, 0])
        fl = DashedLine(o, sv, color=B, stroke_width=3)
        svd = Dot(sv, radius=0.08, color=B)
        bd = VGroup(*[Line(sv, sv + d, color=B, stroke_width=4).add_tip(tip_length=0.18) for d in (np.array([2.4, 0.3, 0]), np.array([2.2, -0.7, 0]))])
        bdl = M("b hadron flies, then → J/ψ X: ℓ<sub>xy</sub> &gt; 0", 24, B).next_to(bd, RIGHT, buff=0.25)
        lxy = M("L<sub>xy</sub>", 22, B).next_to(fl.get_center(), DOWN + LEFT, buff=0.1)
        form = M("ℓ<sub>xy</sub> = m(J/ψ) · L<sub>xy</sub> · p<sub>T</sub> / p<sub>T</sub><sup>2</sup>   (the decay length scaled to the J/ψ boost)", 26, INK).move_to(2.3 * DOWN)
        self.play(FadeIn(pv), FadeIn(pvl), run_time=0.7)
        self.play(*[Create(m) for m in pr], FadeIn(prl), run_time=1.2)
        self.play(Create(fl), FadeIn(svd), FadeIn(lxy), *[Create(m) for m in bd], FadeIn(bdl), run_time=1.5)
        self.play(Write(form), run_time=1.2)
        self.hold(6.0)
        self.body_out()
        pl = Plot("splot_z0pu", xticks=[(-1, "0.1"), (np.log10(0.3), "0.3"), (0, "1"), (np.log10(3), "3"), (1, "10")])
        self.page(pl, ["|z<sub>0</sub>| ≥ 0.1 cm: a J/ψ from another pp collision of the same bunch crossing (pileup)",
                       f"Δz between collisions: Gaussian, σ = {PL['sigma_dz_cm']:.1f} cm → {PL['n_in_pv']:.1f} ± {PL['n_in_pv_err']:.1f} pileup J/ψ expected inside the primary-vertex window",
                       "the pileup J/ψ measure the J/ψ rate per collision with the same selection and veto"], wait=8.0, out=False, key="vtx2")
        self.clear()

    # ------------------------------------------------------------------------------------------------ 5. the three categories
    def categories(self):
        self.set_head(4, "Where the J/ψ sits: three categories")
        self.say("cat1")
        k = 0.78
        ax = Axes(x_range=[-2.5, 2.5, 1], y_range=[-np.pi, np.pi, 1], x_length=5 * k, y_length=2 * np.pi * k, tips=False, axis_config=dict(stroke_width=0, include_ticks=False))
        ax.move_to(4.45 * LEFT + 0.35 * DOWN)
        frame = Rectangle(width=5 * k, height=2 * np.pi * k, color=INK2, stroke_width=2).move_to(ax.get_center())
        lab = VGroup(T("η", 26, INK2).next_to(frame, DOWN, buff=0.1), T("φ", 26, INK2).next_to(frame, LEFT, buff=0.1))

        def jet(eta, phi, col, txt):
            c = Circle(radius=0.4 * k, color=col, stroke_width=4).move_to(ax.c2p(eta, phi)).set_fill(col, opacity=0.18)
            return VGroup(c, T(txt, 21, col, bold=True).next_to(c, UP, buff=0.04))

        jb, jbb = jet(-0.9, 1.6, B, "b"), jet(1.1, -1.2, B, "b̄")
        jq1, jq2 = jet(0.4, 2.1, BLUE, "q"), jet(1.6, 0.6, BLUE, "q̄′")
        jx = jet(-1.5, -1.9, GREY, "extra jet")
        lep = VGroup(Dot(ax.c2p(-1.7, 0.1), radius=0.08, color=TOT), T("ℓ", 24, TOT, bold=True).next_to(ax.c2p(-1.7, 0.1), RIGHT, buff=0.08))

        def jpsi(eta, phi):
            return VGroup(Dot(ax.c2p(eta, phi), radius=0.07, color=INK), Line(ax.c2p(eta, phi), ax.c2p(eta + 0.2, phi + 0.15), color=INK, stroke_width=3),
                          Line(ax.c2p(eta, phi), ax.c2p(eta + 0.18, phi - 0.16), color=INK, stroke_width=3))

        j1, j2, j3, j4 = jpsi(-0.95, 1.5), jpsi(1.55, 0.7), jpsi(-1.45, -1.85), jpsi(0.2, -0.6)
        cones = VGroup(*[DashedVMobject(Circle(radius=0.4 * k, color=INK, stroke_width=3).move_to(j[0].get_center()), num_dashes=18) for j in (jb, jbb, jq1, jq2)])
        self.play(Create(frame), FadeIn(lab), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(o, scale=0.6) for o in (jb, jbb, jq1, jq2, lep, jx)], lag_ratio=0.2), run_time=1.6)
        self.play(*[GrowFromCenter(j) for j in (j1, j2, j3, j4)], run_time=1.0)
        W_ = 8.3
        cats = VGroup(
            para(f"<b><span foreground='{B}'>1</span>  non-prompt J/ψ in the b jets of the top decays</b>\nt → W b, b → H<sub>b</sub> X, H<sub>b</sub> → J/ψ X: the fragmentation reference; <b>{MASS['inb_pv']['Ns']:.0f} ± {MASS['inb_pv']['Ns_err']:.0f}</b> J/ψ from the primary vertex", 20, W_),
            para(f"<b><span foreground='{LTC}'>2</span>  prompt J/ψ in additional jets</b>, not assigned to the top decays: quarkonium in QCD radiation (of the signal below: {INJET:.0f} in such a jet, {NOJET:.0f} in no jet)", 20, W_),
            para(f"<b><span foreground='{P}'>3</span>  prompt J/ψ separated from all top-decay jets</b> (ΔR &gt; 0.4), with or without a jet: inclusive associated production\n<b>{NP_B:.0f} <sup>+{NPB_HI:.0f}</sup><sub>−{NPB_LO:.0f}</sub></b> prompt J/ψ, <b>{F_B['signif_prompt']:.1f}σ</b> with the per-category b-decay shape (proposed nominal); R = ({RATE_B * 1e6:.1f} ± {RATE_BE * 1e6:.1f}) × 10<sup>−6</sup> per selected tt̄ event; with the b-jet template {NP_U:.0f} ± {NPE_U:.0f}, {F_U['signif_prompt']:.1f}σ, R = ({RATE * 1e6:.1f} ± {RATE_E * 1e6:.1f}) × 10<sup>−6</sup> (statistical)", 20, W_),
            para(f"J/ψ in the jets of the W decay (and of the b quarks) are top-decay products: <b>vetoed</b> from category 3 ({CATN['ttj']:.0f} J/ψ in the W-pair and other tt̄ jets)", 19, W_, LTC),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        cats.move_to([-1.8 + cats.width / 2, 2.75 - cats.height / 2, 0])
        self.play(Indicate(j1, color=B, scale_factor=1.8), FadeIn(cats[0]), run_time=1.2)
        self.hold(5.0)
        self.say("cat2")
        self.play(Indicate(j3, color=GREY, scale_factor=1.8), FadeIn(cats[1]), run_time=1.2)
        self.hold(5.0)
        self.say("cat3")
        self.play(*[Create(c) for c in cones], Indicate(j4, color=P, scale_factor=1.8), FadeIn(cats[2]), run_time=1.4)
        self.hold(6.0)
        self.say("cat4")
        self.play(Indicate(j2, color=BLUE, scale_factor=1.8), FadeIn(cats[3]), run_time=1.2)
        self.hold(5.0)
        self.body_out()
        # the W peak
        self.set_head(4, "Why every top-decay jet is vetoed")
        self.page(Plot("assoc_wmass"), [f"<b>{100 * WPP['share_in_w_of_ttj']:.0f}%</b> of the vetoed prompt J/ψ sit in a jet of the selection's W pair",
                                        f"their W pair has a W peak: <b>{100 * WPP['prompt_in_w']['win'][0]:.0f} ± {100 * WPP['prompt_in_w']['win'][1]:.0f}%</b> in 65–95 GeV, against {100 * WPP['extra_jet']['win'][0]:.0f}% for the same jets as an extra jet ({WPP['z_vs_extra']:.1f}σ)",
                                        "these J/ψ are top-decay products: every top-decay jet is vetoed in the signal"], wait=9.0, key="wpeak")
        # z of the J/psi in these jets
        self.set_head(4, "The momentum fraction z of the J/ψ in the vetoed top-decay jets")
        self.page(Plot("an_leak_z"), [f"prompt J/ψ in the vetoed top-decay jets (mostly the W pair): mean z <b>{GZ['mean_z_data']:.2f}</b> in data, <b>{GZ['mean_z_pythia']:.2f}</b> in our gen-level Pythia 8 + LETO (private, not CMS)",
                                      "in data the J/ψ carries less of the jet momentum: a direct handle on quarkonium fragmentation in a known jet"], wait=9.0, key="leakz")
        # our private gen-level Pythia 8 + LETO sample: how many top-decay J/psi leak into the signal
        self.set_head(4, "Gen-level Pythia 8 + LETO (private): top-decay J/ψ past the veto")
        self.page(Plot("assoc_gen"), [f"{100 * GL['eff']:.0f}% of the prompt J/ψ from top decays pass the veto (in no selected jet, or a jet outside the tt̄); ISR J/ψ {100 * GL['isr_eff']:.0f}%",
                                      f"they make {100 * GL['signal_top_decay_share']:.0f}% of the generated signal (gen jets, the selection and veto of the data)",
                                      f"in data, three estimates: {min(RES):.0f}–{max(RES):.0f} of the signal J/ψ may still be top-decay products"], wait=9.0, out=False, key="genleak")
        self.clear()

    # ------------------------------------------------------------------------------------------------ 6. signal extraction
    def extraction(self):
        self.set_head(5, "Signal extraction: the J/ψ mass")
        self.page(Plot("mfit_notb_pv"), [f"J/ψ outside the tt̄ jets from the primary vertex: <b>{MASS['notb_pv']['Ns']:.0f} ± {MASS['notb_pv']['Ns_err']:.0f}</b>",
                                         f"Crystal Ball (σ {1e3 * SH['s1']:.1f} MeV) + Gaussian (σ {1e3 * SH['s2']:.1f} MeV) at {SH['mu']:.3f} GeV, a smooth continuum",
                                         f"all categories: {MASS['all']['Ns']:.0f} ± {MASS['all']['Ns_err']:.0f} J/ψ"], wait=8.0, key="mass")
        self.set_head(5, "Signal extraction: mass × decay length")
        self.page(Plot("splot_proj_l"), ["2D extended maximum-likelihood fit, mass × ℓ<sub>xy</sub>, per-candidate ℓ<sub>xy</sub> resolution",
                                         "the b-decay ℓ<sub>xy</sub> shape constrained from b-jet data and simulation",
                                         f"prompt J/ψ: <b>{NP_U:.0f} ± {NPE_U:.0f}, {F_U['signif_prompt']:.1f}σ</b> (statistical) with the b-jet template; b decays {F_U['N'][1]:.0f} ± {F_U['N_err'][1]:.0f}"], wait=9.0, key="fit2d")
        # three fit models
        self.set_head(5, "Fit models")
        self.say("models")
        rows = [("unbinned fit, m × ℓ<sub>xy</sub>, b-decay shape per category (proposed nominal)", f"{NP_B:.0f} <sup>+{NPB_HI:.0f}</sup><sub>−{NPB_LO:.0f}</sub>",
                 f"{F_B['signif_prompt']:.1f}σ"),
                ("unbinned fit, m × ℓ<sub>xy</sub>, b-decay shape of the data b jets", f"{NP_U:.0f} ± {NPE_U:.0f}", f"{F_U['signif_prompt']:.1f}σ"),
                ("binned 2D fit, m × ℓ<sub>xy</sub>", f"{L2['N_P']:.0f}", f"{FJ['lt2d']['signif_prompt_notb']:.1f}σ"),
                ("ℓ<sub>xy</sub> fit, background from the mass sidebands", f"{F_LXY:.0f}", f"{FL['signif']['nominal']:.1f}σ")]
        tab = VGroup(VGroup(M("<b>fit model</b>", 22), M("<b>prompt J/ψ</b>", 22), M("<b>significance</b>", 22)))
        for a, b_, c in rows:
            tab.add(VGroup(para(a, 22, 8.9, INK2), M(f"<b>{b_}</b>", 26, P), M(c, 24, INK2)))
        for i, r in enumerate(tab):
            y = 1.9 - 0.9 * i
            r[0].move_to([-6.4 + r[0].width / 2, y, 0])
            r[1].move_to([3.7, y, 0])
            r[2].move_to([5.9, y, 0])
        self.play(FadeIn(tab[0]), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r, shift=0.2 * RIGHT) for r in tab[1:]], lag_ratio=0.4), run_time=2.0)
        note = para(f"statistical only · the toy closure (chapter 8) picks the per-category b-decay shape: the b-jet shape copies the {100 * FJ['lt2d']['params']['NP_inb'] / (FJ['lt2d']['params']['NP_inb'] + FJ['lt2d']['params']['NNP_inb']):.1f}% prompt-like share of the data b jets onto the signal's b decays · the ℓ<sub>xy</sub> fit runs low · without e + jets {FL['signif']['mu_emu']:.1f}σ (ℓ<sub>xy</sub> fit)", 21, 12.6)
        note.move_to([-6.4 + note.width / 2, -2.85, 0])
        self.play(FadeIn(note), run_time=0.8)
        self.hold(8.0)
        self.body_out()
        # per year
        self.set_head(5, "The same fits per year")
        sig = "\n".join(f"{y}: {J(f'years/fit_lxy_{y}.json')['signif']['nominal']:.1f}σ / {J(f'years/fit_jpsi_{y}.json')['lt2d']['signif_prompt_notb']:.1f}σ" for y in YEARS)
        self.page(Plot("years_fp"), ["compatible prompt fractions in every year", "significance per year (ℓ<sub>xy</sub> fit / 2D fit):\n" + sig], wait=8.0, out=False, key="years")
        self.clear()

    # ------------------------------------------------------------------------------------------------ 7. the simulation
    def simulation(self):
        cs, ct = CL["signal"], CL["ttj"]
        self.set_head(6, "The CMS tt̄ simulation through the same chain")
        self.page(Plot("mc_closure_signal"), [f"POWHEG + Pythia 8 CP5, full simulation (2024 conditions), the code of the data, σ<sub>tt̄</sub> × {LUMI_ALL:.0f} fb<sup>−1</sup>; no onia in its shower: b decays and a few prompt J/ψ from hadronisation",
                                              f"<b>closure:</b> the fit finds {cs['fit_prompt']:.0f} ± {cs['fit_prompt_err']:.0f} prompt J/ψ for <b>{cs['truth_prompt']:.0f}</b> true",
                                              f"b decays taken for prompt, scaled to the data signal: {s0(cs['fake_prompt_in_data'], 1)} ± {cs['fake_prompt_in_data_err']:.1f}"], wait=9.0, key="sim1")
        self.set_head(6, "Untagged b jets: shorter decay lengths")
        self.page(Plot("mc_bshape"), [f"b hadrons in untagged jets (the other tt̄ jets) fly little: {100 * SHORT['ttj']:.0f}% at |ℓ<sub>xy</sub>| &lt; 0.06 mm, against {100 * SHORT['inb']:.0f}% in b jets",
                                      "a single b-jet template takes them for prompt"], wait=8.0, key="sim2")
        self.page(Plot("mc_tfit_ttj"), [f"one template: {ct['fit_prompt']:.0f} prompt J/ψ found for {ct['truth_prompt']:.0f} true in the vetoed jets of the CMS simulation",
                                        f"with the b-decay shape of that category: <b>{TTJ_C['N_p']:.0f} ± {TTJ_C['N_p_err']:.0f}</b> prompt of the {TTJ_U:.0f} of the unbinned fit (data)",
                                        "→ a b-decay shape for each category"], wait=8.0, key="sim3")
        bc = BTM["cats"]
        self.set_head(6, "A b-decay shape per category, from the CMS simulation")
        self.page(Plot("mc_btmpl_shapes"), [f"the fit's own b-decay model on the true b decays of each category: short-decay share f = <b>{bc['inb']['f']:.2f}</b> in b jets, <b>{bc['signal']['f']:.2f}</b> outside the tt̄ jets, <b>{bc['ttj']['f']:.2f}</b> in the vetoed jets",
                                            "outside the tt̄ jets the b decays look like those of b jets: no prompt-like part",
                                            f"λ calibrated on the data b jets (× {BS['btemplate']['lam_ratio']:.2f}); used by every fit"], wait=9.0, key="sim4")
        self.set_head(6, "Data against the CMS simulation: tt̄ + backgrounds" if HAS_BKG else "Data against the CMS tt̄ simulation")
        self.page(Plot("mc_rates"), [f"b decays: data / simulation {RI['data_nonprompt'][0] / simp(RI, 'b decay'):.2f} in the b jets, {RS['data_nonprompt'][0] / simp(RS, 'b decay'):.2f} outside the tt̄ jets",
                                     f"prompt J/ψ outside the tt̄ jets: data <b>{RS['data_prompt'][0]:.0f}</b>, simulation {simp(RS, 'prompt'):.0f} (hadronisation only)",
                                     (f"with single top, W + jets, tt̄V and dibosons (QCD too few simulated events to stack): selected events data / simulation {EV['mujets']['ratio_all']:.2f}, {EV['ejets']['ratio_all']:.2f}, {EV['emu']['ratio_all']:.2f} (μ + jets, e + jets, e μ)"
                                      if HAS_BKG else f"tt̄ simulation only (ℓ + jets events: data / simulation {EV['mujets']['ratio']:.2f}, {EV['ejets']['ratio']:.2f})")], wait=9.0, key="sim5")
        self.page(Plot("chan_rates"), ["outside the tt̄ jets, per channel: the same pattern in μ + jets, e + jets and e μ"], wait=6.0, out=False, key="sim6")
        self.clear()

    # ------------------------------------------------------------------------------------------------ 8. toy closure
    def toys(self):
        a, b = TCPB[("mc", 100)], TCPB[("datalike", 100)]  # per-category template
        c, d = TCP[("mc", 100)], TCP[("datalike", 100)]  # b-jet template
        lx = [q["lxy"]["bias"] for q in TCPB.values()]

        def biases(p):
            return " · ".join(f"{n} {s0(p[k]['bias'], sign=True)}" for k, n in (("lxy", "ℓ<sub>xy</sub>"), ("fit2d", "2D"), ("unbinned", "unbinned"), ("unbinned_global", "one resolution")))

        self.set_head(7, "Which fit model: toy closure")
        self.page(Plot("mc_toys_btmpl_mc"), [f"toys of the data's size drawn from the CMS simulation (per channel, category and sign), 0–150 prompt J/ψ injected, the whole chain on each, {a['n']} per point",
                                             f"per-category b-decay shape, fit − truth at {a['unbinned']['truth']:.0f} true: {biases(a)}",
                                             f"the ℓ<sub>xy</sub> fit runs low by {s0(-max(lx))}–{s0(-min(lx))}"], wait=9.0, key="toy1")
        self.set_head(7, "Which b-decay shape: what if the b jets hold prompt J/ψ?")
        self.page(Plot("mc_toys_tmplcmp_unbinned"), [f"prompt J/ψ added to the b jets, 5% of their b decays: the share that looks prompt in data b jets",
                                                     f"b-jet shape: unbinned fit − truth {s0(c['unbinned']['bias'], sign=True)} → <b>{s0(d['unbinned']['bias'], sign=True)}</b>: it copies that share onto the signal's b decays",
                                                     f"per-category shape: {s0(a['unbinned']['bias'], sign=True)} and {s0(b['unbinned']['bias'], sign=True)} (one resolution: {s0(a['unbinned_global']['bias'], sign=True)}, {s0(b['unbinned_global']['bias'], sign=True)}) of {a['unbinned']['truth']:.0f}",
                                                     f"→ proposed nominal: the unbinned fit with the per-category shape, <b>{NP_B:.0f} <sup>+{NPB_HI:.0f}</sup><sub>−{NPB_LO:.0f}</sub> ({F_B['signif_prompt']:.1f}σ)</b> instead of {NP_U:.0f} ± {NPE_U:.0f} (statistical)"],
                  wait=11.0, out=False, key="toy2")
        self.clear()

    # ------------------------------------------------------------------------------------------------ 9. DPS, interpretation
    def dps(self):
        self.set_head(8, "Associated production and a second parton scattering")
        pl = Plot(None, xr=[3, 30], yr=[0, 160], xlabel="σ<sub>eff</sub> [mb]", ylabel="prompt J/ψ outside the tt̄ jets",
                  xticks=[(v, f"{v}") for v in (5, 10, 15, 20, 25, 30)], yticks=[(v, f"{v}") for v in (0, 50, 100, 150)])
        k0 = PRED[15.0] * 15.0  # the prediction goes as 1 / sigma_eff
        pl.fills.add(Polygon(pl.p(3, NP_U - NPE_U), pl.p(30, NP_U - NPE_U), pl.p(30, NP_U + NPE_U), pl.p(3, NP_U + NPE_U), stroke_width=0, fill_color=P, fill_opacity=0.25))
        pl.lines.add(DashedLine(pl.p(3, NP_U), pl.p(30, NP_U), color=P, stroke_width=3))
        pl.lines.add(VMobject(color=BLUE, stroke_width=4).set_points_as_corners([pl.p(x, k0 / x) for x in np.linspace(3.0, 30.0, 120)]))
        pl.points.add(*[Dot(pl.p(x, y), radius=0.06, color=BLUE) for x, y in PRED.items()])
        pl.leg = [("2nd parton scattering, from the pileup J/ψ rate", "line", BLUE), (f"data: {NP_U:.0f} ± {NPE_U:.0f}", "fill", P)]
        self.page(pl, ["the prompt yield as an inclusive associated-production observable",
                       f"a 2nd scattering: P = r · σ<sub>vis</sub> / σ<sub>eff</sub>, r from the prompt pileup J/ψ ({DPS['pileup_prompt_notb']:.0f} ± {DPS['pileup_prompt_notb_err']:.0f}, {DPS['mean_npu']:.0f} collisions per event) → {PRED[15.0]:.0f} J/ψ at σ<sub>eff</sub> = 15 mb, {PRED[5.0]:.0f} at 5 mb",
                       "this comparison depends on the model: it is not a σ<sub>eff</sub> measurement"], wait=9.0, key="dps1")
        self.set_head(8, "The tt̄ recoil does not separate them")
        self.page(Plot("ttbal_ppar"), ["soft J/ψ: the p<sub>T</sub> balance of the event does not separate single from double parton scattering",
                                       f"⟨p<sub>T</sub>(tt̄) · p̂<sub>T</sub>(J/ψ)⟩: data {s0(BP['prompt'][0])} ± {BP['prompt'][1]:.0f} GeV; a recoiling J/ψ jet {s0(BP['sps'][0])}, a 2nd scattering {s0(BP['dps'][0])} (emulated)"], wait=8.0, key="dps2")
        self.set_head(8, "Gen-level Pythia 8 + LETO (private): a model interpretation")
        self.page(Plot("gencmp_yields"), [f"ℓ + jets: our generator-level Pythia 8 + LETO (not a CMS sample) normalised with the b-decay J/ψ, then scaled to the vetoed prompt J/ψ (× {SNC['scale']:.2f} ± {SNC['scale_err']:.2f})",
                                          f"it gives <b>{SNC['sps_signal']:.0f} ± {SNC['sps_signal_err']:.0f}</b> of the signal; the remainder, {SNC['dps_left']:.0f} ± {SNC['dps_left_err']:.0f}, is compatible with a 2nd scattering",
                                          "an interpretation within these models, not a σ<sub>eff</sub> measurement"], wait=9.0, out=False, key="dps3")
        self.clear()

    # ------------------------------------------------------------------------------------------------ 10. generators
    def generators(self):
        self.set_head(9, "Generators and tunes (preliminary)")
        g = {k: GR[f"{k}|signal"]["r"] for k in ("Pythia 8 CP5", "CP5 CR1", "CP5 CR2", "CP5 ERD on", "MG5 MLM", "aMC@NLO FxFx") if f"{k}|signal" in GR}
        mg = " · ".join(f"{n} {g[k]:.3f}" for k, n in (("MG5 MLM", "MadGraph MLM"), ("aMC@NLO FxFx", "aMC@NLO FxFx")) if k in g)
        # Sherpa: one lepton charge (l+), x 2 against POWHEG + Pythia 8 (both charges), b-decay J/psi in b jets per luminosity
        sh = 2.0 * GV["gen"]["Sherpa"]["inb_bdecay"][0] / GV["gen"]["Pythia 8 CP5"]["inb_bdecay"][0] if "Sherpa" in GV["gen"] else None
        self.page(Plot("mcvar_ratio", xlabel="tt̄ sample (ℓ + jets)"),
                  [f"prompt / b-decay J/ψ within each sample, no normalisation: data <b>{GR['data|signal']['r']:.2f}</b> outside the tt̄ jets, {GR['data|inb']['r']:.3f} in b jets",
                   f"POWHEG + Pythia 8 CP5 {g['Pythia 8 CP5']:.3f}; colour reconnection CR1 {g['CP5 CR1']:.3f}, CR2 {g['CP5 CR2']:.3f}, ERD {g['CP5 ERD on']:.3f}" + (f"; {mg}" if mg else ""),
                   "Sherpa, default and AHADIC tunes: no prompt J/ψ (upper limits)" + (f"; it selects as many tt̄ events per luminosity as POWHEG but has {100 * (1 - sh):.0f}% fewer b-decay J/ψ in b jets (its own b-hadron decays)" if sh else ""),
                   f"e μ, dilepton samples (both lepton charges): data {GR['data|signal|emu']['r']:.2f} ± {GR['data|signal|emu']['err']:.2f}, POWHEG + Pythia 8 CP5 {GR['Pythia 8 CP5|signal|emu']['r']:.3f}, CR1 {GR['CP5 CR1|signal|emu']['r']:.3f}: the same pattern",
                   "every sample falls short: the comparison is sensitive to the modelling of quarkonium production and fragmentation"], wait=11.0, out=False, key="gen")
        self.clear()

    # ------------------------------------------------------------------------------------------------ 11. summary
    def summary(self):
        self.set_head(10, "Summary and next steps")
        self.say("summary")
        n_all = float(f"{MASS['all']['Ns']:.2g}")
        b = self.bullets([f"≈ {n_all:.0f} J/ψ in {NTT_ALL / 1e6:.1f}M tt̄ events, mostly from b-hadron decays: the b-fragmentation reference ({MASS['inb_pv']['Ns']:.0f} in the b jets of the top decays)",
                          f"prompt J/ψ separated from all top-decay jets: <b>{NP_B:.0f} <sup>+{NPB_HI:.0f}</sup><sub>−{NPB_LO:.0f}</sub>, {F_B['signif_prompt']:.1f}σ</b> with the per-category b-decay shape (proposed nominal, toy closure; {NP_U:.0f} ± {NPE_U:.0f} with the b-jet shape); R = ({RATE_B * 1e6:.1f} ± {RATE_BE * 1e6:.1f}) × 10<sup>−6</sup> per selected tt̄ event (statistical)",
                          f"about {min(RES):.0f}–{max(RES):.0f} of them may still be top-decay products; the CMS simulation predicts {simp(RS, 'prompt'):.0f}" + (" (tt̄ + backgrounds)" if HAS_BKG else ""),
                          "tt̄ events: a common framework for b fragmentation and for prompt quarkonium formation in QCD radiation",
                          "generator comparisons are sensitive to the modelling of quarkonium production and fragmentation"], 22, 12.6, buff=0.3)
        nx = para("<b>Next:</b> differential z<sub>J/ψ</sub> in the top-decay b jets · prompt rates in the additional-jet categories · systematic uncertainties of the fit model · the residual top-decay J/ψ", 21, 12.6)
        col = VGroup(b, nx).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        col.move_to([-6.4 + col.width / 2, 2.75 - col.height / 2, 0])
        for x in b:
            self.play(FadeIn(x, shift=0.2 * RIGHT), run_time=0.8)
            self.wait(2.2)
        self.wait(3.0)
        self.play(FadeIn(nx), run_time=1.0)
        self.hold(8.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)


if __name__ == "__main__" and "--narration" in sys.argv:
    json.dump(narration(), open(sys.argv[sys.argv.index("--narration") + 1], "w"), indent=1, ensure_ascii=False)
