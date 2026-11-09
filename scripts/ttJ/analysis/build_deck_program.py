"""t̅t + J/psi: the whole program on one deck, for a discussion (9 Nov 2026). Every number comes from the full analysis deck
(TTJ_FULLDECK, default ../../2026Nov07_jpsi_assoc/files): its builder build_deck_assoc.py is imported for the numbers it computes
(its slides are dropped), and the figures used here are copied from its figs/ on every build. The history slide reads the fits
of the earlier decks (2026Nov04_jpsi_prompt, 2026Nov05_jpsi_run3, 2026Nov03_jpsi) next to it.
Two kinds of simulation, kept apart on every slide: the CMS samples (full detector simulation) and our private generator-level
Pythia 8 + LETO study.   usage: python3 build_deck_program.py   (the pixi "analysis" environment; see README.md of scripts/ttJ/analysis)"""
import importlib.util
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.environ.get("TTJ_FULLDECK", os.path.join(HERE, "..", "..", "2026Nov07_jpsi_assoc", "files")))
WWW = os.path.dirname(os.path.dirname(SRC))  # the folder of all the decks
sys.path.insert(1, SRC)  # the full deck's modules after our own deck_lib.py
_spec = importlib.util.spec_from_file_location("assoc", os.path.join(SRC, "build_deck_assoc.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)  # every number of the full deck (it fills D.SLIDES, emptied below)

import deck_lib as D  # noqa: E402
from deck_lib import F, content, divider, table, ul  # noqa: E402

D.SLIDES.clear()
D.HERE = HERE
D.TITLE = "t̅t + J/ψ: the program so far"
D.DATE = "9 Nov 2026"
D.FOOT = "t̅t + J/ψ, program summary" + (f" · {os.environ['TTJ_AUTHOR']}" if os.environ.get("TTJ_AUTHOR") else "")
D.EXTRA_LINKS = ('<a href="../2026Nov07_jpsi_assoc/">full analysis deck (7–8 Nov)</a><a href="../2026Nov05_jpsi_run3/">6 Nov deck</a>'
                 '<a href="../2026Nov04_jpsi_prompt/">4 Nov deck</a><a href="../2026Nov03_jpsi/">3 Nov deck</a>')
D.BUILD = "build_deck_program.py"
AUTHOR = f"{os.environ['TTJ_AUTHOR']} · " if os.environ.get("TTJ_AUTHOR") else ""


def J(*p):
    return json.load(open(os.path.join(*p)))


def tag(kind):
    return {"hist": '<span class="tag re">HISTORY · 3–8 NOV</span>', "disc": '<span class="tag new">FOR DISCUSSION</span>'}.get(kind) or A.tag(kind)


D.tag = tag
TB, SIG, mn, pm = A.TB, A.SIG, A.mn, A.pm

# --- the numbers (all from the full deck's inputs)
BN, BNM = A.BN, A.BNM  # unbinned fit, per-category b-decay template (proposed nominal)
NOM = f"{BN['N'][0]:.0f} <sup>+{BNM[1]:.0f}</sup><sub>−{-BNM[0]:.0f}</sub>"
NOM_T = f"{BN['N'][0]:.0f} +{BNM[1]:.0f} −{-BNM[0]:.0f}"  # plain text (stats, notes)
NOM_S = BN["signif_prompt"]
ALT = f"{A.SP_NP:.0f} ± {A.SP_NPE:.0f}"  # unbinned fit, b-jet template (the fit behind the comparisons)
ALT_S = A.SPF["signif_prompt"]
GR = A.GR
R_D, R_D_E = GR["data|signal"]["r"], GR["data|signal"]["err"]
R_CP5, R_CR1 = GR["Pythia 8 CP5|signal"]["r"], GR["CP5 CR1|signal"]["r"]
R_SH = max(GR[f"{g}|signal"]["upper"] for g in ("Sherpa", "Sherpa AHADIC"))
SIMP = A.simp(A.RS, "prompt")  # CMS simulation (ttbar + backgrounds), prompt J/psi outside the ttbar jets, 310 fb^-1
XSIM = A.RS["data_prompt"][0] / SIMP
MSA, MSI = A.MS["all"], A.FJ["mass"]["inb_pv"]
TOY = {k: v["unbinned"]["bias"] for k, v in A.U.items() if v}
GL, GC = A.GL, A.GC["cuts"]
CH = A.CHV
PER = {c: (1e6 * v["sp"][0] / v["ntt"], 1e6 * v["sp"][1] / v["ntt"]) for c, v in CH.items()}  # prompt per 10^6 ttbar
ZE = (PER["ejets"][0] - PER["mujets"][0]) / (PER["ejets"][1] ** 2 + PER["mujets"][1] ** 2) ** 0.5  # e + jets above mu + jets per ttbar event
RES_LO, RES_HI = A.RES_LO, A.RES_HI
SNC, SN = A.SNC, A.SN
GV_EMU = GR.get("data|signal|emu")
OVER = (1 / SN["ratio_data_to_pythia"]["ttj"][0], 1 / SNC["ratio_data_to_pythia"]["ttj"][0])  # gen. Pythia + LETO / data, vetoed J/psi (raw, without their short b decays)

# the earlier decks: the headline fit of each
H4 = J(WWW, "2026Nov04_jpsi_prompt", "files", "fit_lxy.json")
H5 = J(WWW, "2026Nov05_jpsi_run3", "files", "oldcleaning", "fit_lxy.json")
H6 = J(WWW, "2026Nov05_jpsi_run3", "files", "fit_lxy.json")
H6S = J(WWW, "2026Nov05_jpsi_run3", "files", "splot.json")["fits"]["notb"]
VTX = J(WWW, "2026Nov03_jpsi", "files", "vtx_compare.json")
BPH = J(WWW, "2026Nov03_jpsi", "files", "results_bph.json")
VTX_HW = max(0.5 * (p["res_q"][2] - p["res_q"][0]) for p in VTX["pt_bins"] if "res_q" in p)  # um, 16-84% half-width of ours - BPHNano
BPH_NEW = sum(BPH["channels"][c]["life"]["yield_no_sltrig_all"]["n"] for c in ("mujets", "ejets", "emu"))  # parking J/psi without the 1-lepton trigger


def hfp(Fj):
    lo, hi = Fj["minos"]["fp_notb"]
    return f"{100 * Fj['params']['fp_notb']:.1f} ± {50 * (hi - lo):.1f}% of {Fj['params']['N_notb']:.0f}"


def small(tbl, px=22):
    """a table at a smaller font"""
    return tbl.replace('<table class="t">', f'<table class="t" style="font-size:{px}px">', 1)


def card(cls, head, body, style=""):
    return f'<div class="card {cls}" style="flex:1;{style}"><h3>{head}</h3>{body}</div>'


def p(x, size=24):
    return f'<p style="font-size:{size}px;line-height:1.4">{x}</p>'


CODE = A.CODE

# ================================================================ cover
D.SLIDES.append(("cover", f"""<section id="cover" class="slide div d0" style="gap:34px">
<p class="dv-ey" style="color:#E8A765">CMS Run 3 · 2022–2026 data · {A.LTOT:.0f} fb<sup>−1</sup> · √s = 13.6 TeV · program summary for discussion</p>
<h1 class="dv-t" style="font-size:80px">t̅t + J/ψ: quarkonium production in top-quark pair events</h1>
<div style="display:flex;flex-direction:column;gap:14px">
<p style="font-size:34px;line-height:1.3;color:#BFD0E6"><span style="color:#7FD1C7">Question:</span> do J/ψ form in the QCD radiation of {TB} events, away from the top decays, and do the generators make them?</p>
<p style="font-size:34px;line-height:1.3;color:#BFD0E6"><span style="color:#B5D58B">Seen:</span> {NOM} prompt J/ψ outside every top-decay jet in {A.NTT / 1e6:.1f}M {TB} events, {NOM_S:.1f}σ, statistical only (with the b-jet template: {ALT}, {ALT_S:.1f}σ)</p>
<p style="font-size:34px;line-height:1.3;color:#BFD0E6"><span style="color:#C9A7F0">Compared:</span> prompt per b-decay J/ψ {R_D:.2f} ± {R_D_E:.2f} in data, {R_CP5:.3f} in the CMS POWHEG + Pythia 8 simulation, none in Sherpa; our private gen-level Pythia 8 + LETO plus a second scattering can describe it</p>
<p style="font-size:34px;line-height:1.3;color:#BFD0E6"><span style="color:#E8A765">To decide:</span> the nominal fit · the residual top-decay J/ψ · what we quote</p>
</div>
<p style="font-size:28px;color:#BFD0E6">{AUTHOR}{D.DATE} · data, CMS simulation and a private gen-level study · code <a href="{A.REPO}/commit/{A.SHA}" style="color:#7FD1C7">ttJ_pixi @ {A.SHA[:7]}</a> · full deck: <a href="../2026Nov07_jpsi_assoc/" style="color:#7FD1C7">7–8 Nov</a></p>
<div class="footer"><span></span><span>PAGE</span></div></section>""",
                 "One deck for the whole ttbar + J/psi program, 3-8 Nov: from the first J/psi peak in ttbar to prompt J/psi outside every top-decay jet. "
                 "All numbers are statistical only. The full analysis deck (link in the bar) has every check; this one has the line of argument and the open decisions."))

# ================================================================ outline
OUT = [("1", "The question and the path", "what we look for; how the result moved from 3 to 8 Nov, and what was withdrawn"),
       ("2", "Measurement", "selection, the veto of the top-decay jets, the m × ℓ<sub>xy</sub> fit, the yield and its stability"),
       ("3", "What makes them", "CMS simulation, seven generator set-ups, the private gen-level Pythia 8 + LETO study, a second scattering"),
       ("4", "For discussion", "summary, the decisions for today, what is still missing"),
       ("B", "Backup", "the two kinds of simulation, tighter selections, the b-decay templates, how to rerun")]
content("outline", "Outline", "Outline",
        '<div class="outline" style="margin-top:20px">' + "".join(f'<span class="num">{n}</span><span class="sec"><b>{t}</b><br><span>{d}</span></span>' for n, t, d in OUT) + "</div>")

# ================================================================ 1 · the question and the path
divider("d-question", 1, "1 · The question and the path", "From a J/ψ peak in t̅t to prompt J/ψ outside every top-decay jet",
        "Four decks in six days: each step changed the signal definition or fixed a problem. Here is where it stands, and what no longer holds.",
        stats=[(f"{MSA['Ns'] / 1e3:.1f}k", f"J/ψ → μμ in {TB} events (mass fit)"), (f"{A.NTT / 1e6:.1f}M", f"{TB} events: μ + jets, e + jets, e μ"),
               (f"{A.LTOT:.0f} fb<sup>−1</sup>", "2022–2026, √s = 13.6 TeV")])

content("motivation", "The question", f"{TB} events as a laboratory for quarkonium formation and heavy-flavour fragmentation",
        '<div class="row" style="gap:30px;align-items:stretch">'
        + card("c1", "The question", p("How quarkonium forms in QCD radiation is open: the roles of colour-singlet and colour-octet production, and how parton showers implement them (Pythia 8 with LETO: onia in the shower)."
                                       f" {TB} gives one well-measured hard process with b, c, light and gluon jets around it.", 25))
        + card("c2", "The setup", p(f"{TB} → ℓ + jets and e μ, 2022–2026 data. J/ψ → μμ from loose muons (p<sub>T</sub> &gt; 3 GeV, no isolation). A two-muon vertex from central NanoAOD gives the pseudo-proper decay length ℓ<sub>xy</sub>: "
                                    "prompt J/ψ at ℓ<sub>xy</sub> ≈ 0, b decays with a tail of cτ ≈ 0.4 mm.", 25))
        + card("c3", "Measure", ul([f"<b>Non-prompt J/ψ in the top b jets</b>: the fragmentation reference ({MSI['Ns'] / 1e3:.1f}k)",
                                    f"<b>Prompt J/ψ in jets</b>: quarkonium in the radiation",
                                    f"<b>Prompt J/ψ away from every top-decay jet</b>: associated production, this deck's signal ({NOM})"], "b s"))
        + "</div>"
        + '<div class="callout"><b>Caveats:</b> every number here is statistical only, no systematics yet. A second parton scattering (DPS) gives the same signature and is not separated: '
          "the draft letter quotes the two together.</div>",
        notes="Framing of the draft letter 'Quarkonium production and heavy-flavor fragmentation in top quark pair events in pp collisions at 13.6 TeV'. "
              "Non-prompt J/psi in the b jets were also used by CMS for m_t from m(J/psi, l) at 8 TeV.")

# --- where we stand: timeline + status
STEPS = [("3 Nov", "J/ψ peak in t̅t (part of 2024); parking BPHNano adds only ≈ " + f"{BPH_NEW:.0f} J/ψ", ""),
         ("4 Nov", f"2024 lifetime fit: prompt not-b-matched J/ψ at {H4['signif']['nominal']:.1f}σ", ""),
         ("5 Nov", f"2022–2026: {H5['signif']['nominal']:.1f}σ; a jet-cleaning bug found", ""),
         ("6 Nov", f"fixed rerun: {H6S['signif_prompt']:.1f}σ (sPlot); DPS rate, p<sub>T</sub> balance", ""),
         ("7 Nov", f"veto every top-decay jet: {ALT_S:.1f}σ; CMS simulation, generators", ""),
         ("8 Nov", f"per-category b-decay template, toys, backgrounds: {NOM_S:.1f}σ", ""),
         ("9 Nov", "discussion: nominal fit, residual, what to quote", "now")]
tl = '<div class="tl" style="padding:0">' + "".join(
    f'<div class="st" style="width:210px"><span class="when">{w}</span><span class="dot {c}"></span><span class="what {c}">{t}</span></div>' for w, t, c in STEPS) + "</div>"
content("status", "Where we stand", "All inputs are processed; the open points are choices",
        tl + '<div class="row" style="gap:28px;align-items:stretch;margin-top:6px">'
        + card("c2", "Done", ul([f"<b>Data:</b> 2022–2026, {A.NJOBS} jobs, 22 of 58 723 files missing",
                                 f"<b>CMS simulation:</b> POWHEG + Pythia 8 {TB}, 7 alternative set-ups, the non-{TB} backgrounds",
                                 f"<b>Private gen level:</b> Pythia 8 + LETO, {A.GEN['n_generated'] / 1e6:.0f}M {TB}",
                                 "<b>Toys</b> for the fit: 2 templates × 2 scenarios"], "b s"))
        + card("c3", "Open", ul(["<b>The nominal fit</b> (b-decay template)", f"<b>The residual top-decay J/ψ</b>: {RES_LO:.0f}–{RES_HI:.0f} of the signal",
                                 "<b>Systematics:</b> none yet", f"<b>The comparisons</b> still use the b-jet-template fit ({ALT})"], "b s"))
        + card("c1", "Code", ul([f"<b>Workflow</b> and pixi environment: {CODE}", "<b>Analysis chain</b> in git: fits, figures, decks, the note's figures",
                                 "<b>AN-26-XXX</b> draft on Overleaf takes its figures from this chain"], "b s"))
        + "</div>", t="hist",
        notes="Clusters: 884992 (2022-2026, old cleaning), 890796 (cleaning fixed), 891634 + 895659 (run5 + retry with the candidate columns, merged in run5m); "
              "CMS simulation 897589 (ttbar), 897658 / 897660 (generators, l + jets / dilepton), 897661 (backgrounds), gen study 897323. Nothing is running.")

# --- how the headline moved
hrow = [["4 Nov", "not b-matched", "2024", hfp(H4), f"{H4['signif']['nominal']:.1f}σ", "first ℓ<sub>xy</sub> fit; the b decays take the b-jet ℓ<sub>xy</sub> shape"],
        ["5 Nov", "not b-matched", "2022–2026", hfp(H5), f"{H5['signif']['nominal']:.1f}σ", "2.8× the luminosity"],
        ["6 Nov", "not b-matched", "2022–2026", f"{H6S['N'][0]:.0f} ± {0.5 * (H6S['NP_minos'][1] - H6S['NP_minos'][0]):.0f} (sPlot)", f"{H6S['signif_prompt']:.1f}σ",
         "<b>jet cleaning fixed:</b> soft muons had removed the J/ψ b jets; e + jets kept only events with an extra muon"],
        ["7 Nov", "outside every top-decay jet", "2022–2026", f"{ALT} (sPlot)", f"{ALT_S:.1f}σ", f"<b>{A.ST_NP:.0f} of the {A.OLD_NP:.0f}</b> sat in W-pair or {TB}-fit jets (a W mass peak): vetoed"],
        ["8 Nov", "outside every top-decay jet", "2022–2026", f"{NOM} (sPlot)", f"{NOM_S:.1f}σ", "b-decay ℓ<sub>xy</sub> shape per category from the CMS simulation; toys"]]
content("history", "Where we stand", f"How the headline number moved: {H4['signif']['nominal']:.1f}σ → {NOM_S:.1f}σ, with two corrections on the way",
        table(["Deck", "Signal", "Data", "Prompt J/ψ", "Signif.", "What changed"], hrow, widths=[8, 17, 11, 19, 8, 37], num=(4,), hl=(4,), warn=(2, 3))
        + '<div class="callout"><b>Withdrawn on the way:</b> “harder than inclusive J/ψ” and “recoils against the t̅t, 3.2σ from DPS” (6 Nov: these were J/ψ in W-decay jets); '
          "the first toys (drawn per ℓ<sub>xy</sub> class, which gave the simulated b decays the data's spectrum); the pileup J/ψ as a template for the z of a second scattering.</div>",
        t="hist",
        notes="Prompt fractions are of the J/psi from the primary vertex in the category, from the l_xy fit (fit_lxy.py, background per bin from the sidebands); "
              "sPlot = the unbinned m x l_xy fit (splot_prompt.py). The 4 Nov 2024 pass (out_ttJ) had also lost about 13% of its files to --skipbadfiles; the 2022-2026 passes were retried "
              "until 22 of 58723 files were missing. The cleaning fix: ttJ_pixi 104321e.")

# ================================================================ 2 · measurement
NSIG = A.N_SIG
divider("d-measure", 2, "2 · Measurement", "Select t̅t, find the J/ψ, keep it away from the top-decay jets, fit mass × ℓ<sub>xy</sub>",
        "The prompt J/ψ sit on a large b-decay background; the b-decay ℓ<sub>xy</sub> shape decides the yield.",
        stats=[(f"{MSI['Ns'] / 1e3:.1f}k", "J/ψ in b jets: the reference"), (f"{NSIG[0]:.0f}", f"J/ψ {SIG}, about half from pileup"),
               (NOM, f"prompt J/ψ among them, {NOM_S:.1f}σ")])

srow = []
for nm, C in list(A.C3.items()) + [("all", None)]:
    cs = [C] if C else list(A.C3.values())
    yy = lambda k: (sum(c["yields"][k]["n"] for c in cs), sum(c["yields"][k]["err"] ** 2 for c in cs) ** 0.5)  # noqa: E731
    srow.append([nm, f"{A.ntt(cs) / 1e6:.2f}M", A.fmt(yy("data_os_all_ge1b")), A.fmt(yy("data_os_inb_ge1b")), A.fmt(yy("data_os_ttj_ge1b")), A.fmt(yy("data_os_notb_ge1b"))])
content("selection", "Measurement · selection", f"{A.NTT / 1e6:.1f}M {TB} events in three channels; J/ψ from loose muons, ℓ<sub>xy</sub> from a two-muon vertex",
        table(["Channel", f"{TB} events", "J/ψ, all", "in a b jet", f"in another {TB} jet", f"<b>{SIG}</b>"], srow, widths=[14, 14, 18, 18, 18, 18], num=(1, 2, 3, 4, 5), hl=(3,))
        + '<div class="row" style="gap:40px"><div class="col">'
        + ul([f"<b>{TB}:</b> μ + jets, e + jets (single-lepton triggers), e μ; ≥ 4 jets in ℓ + jets, {A.NB1} (UParT medium); JEC pinned to Summer24Prompt24 V3",
              "<b>J/ψ:</b> two opposite-sign loose muons, p<sub>T</sub> &gt; 3 GeV, |η| &lt; 2.4, no isolation, never the t̅t lepton; window 2.95–3.25 GeV minus the sidebands"], "b s")
        + '</div><div class="col">'
        + ul([f"<b>Vertex:</b> central NanoAOD has no dimuon vertex; ours, from the muon dxy and dz, gives the BPHNano vertex-fit ℓ<sub>xy</sub> to ±{VTX_HW:.0f} μm ({VTX['njpsi'] / 1e3:.0f}k parking J/ψ)",
              f"<b>Pileup J/ψ</b> (from another collision): |z<sub>0</sub>| ≥ 0.1 cm, {100 * A.NPU[0] / NSIG[0]:.0f}% of those {SIG}; the fits use |z<sub>0</sub>| &lt; 0.1 cm"], "b s")
        + "</div></div>", t="data",
        notes="Counts: opposite sign, N_b >= 1, all lifetimes and z0, window - sidebands. The parking (ParkingDoubleMuonLowMass BPHNano, 5% of 2024) was dropped on 4 Nov: "
              f"it adds about {BPH_NEW:.0f} J/psi outside the single-lepton triggers, and its vertex fit is reproduced by ours.")

WPP = A.WPP
f0, fe0 = WPP["prompt_in_w"]["win"]
content("definition", "Measurement · signal", f"Signal: the J/ψ ΔR &gt; 0.4 from every jet the two top decays can be built from",
        f'<div class="row"><div class="fig">{F("assoc_wmass", "vetoed J/ψ in a W-pair jet, unit area per GeV", 0.75)}</div><div class="col">'
        + ul([f"<b>Vetoed:</b> every b-tagged jet; in ℓ + jets the four leading jets (the W pair and two b candidates) and the jets of a χ² {TB} fit; in e μ the two b candidates",
              f"<b>Why all of them:</b> with only b jets vetoed (6 Nov), {A.ST_NP:.0f} of the {A.OLD_NP:.0f} prompt J/ψ sat in a W-pair or {TB}-fit jet, and those W pairs peak at m<sub>W</sub>: "
              f"{100 * f0:.0f} ± {100 * fe0:.0f}% in 65–95 GeV against {100 * WPP['extra_jet']['win'][0]:.0f}% for an extra jet (left)",
              f"<b>Leak</b> (private gen-level Pythia 8 + LETO): {100 * GL['eff']:.0f}% of the top-decay J/ψ still pass, in soft or failed-ID jets: about {RES_LO:.0f}–{RES_HI:.0f} of the signal (decision 2)",
              f"<b>Where the signal sits:</b> {A.SP_NOJET:.0f} of the {A.SP_NP:.0f} prompt J/ψ in no selected jet, {A.SP_INJET:.0f} in a jet that is not a {TB} jet"], "b s")
        + "</div></div>", t="fit",
        notes="assoc_veto.py; the chi2 fit is ttreco.py (m_W, m_t over the eight leading jets). W-pair check: wpair_check.py, each J/psi jet put into 300 other selected events. "
              "Leak: gen_assoc.py with the same veto code on gen jets; the three residual estimates are on the gen-study slide.")

BC = A.BC
content("fit", "Measurement · fit", "Prompt or b decay: an unbinned m × ℓ<sub>xy</sub> fit, in which the b-decay ℓ<sub>xy</sub> shape is the crux",
        f'<div class="row" style="gap:40px"><div class="fig">{F("mfit_notb_pv", f"J/ψ {SIG}, primary vertex", 0.68)}</div>'
        f'<div class="fig">{F("mc_btmpl_data", "data, both b-decay templates", 0.68)}</div></div>'
        + ul([f"<b>Model:</b> prompt = the ℓ<sub>xy</sub> resolution (per-candidate error × a scale); b decays = f G + (1 − f) e<sup>−ℓ/λ</sup> ⊗ G; continuum from the sidebands. "
              f"The mass fit alone: {A.MS['notb_pv']['Ns']:.0f} ± {A.MS['notb_pv']['Ns_err']:.0f} J/ψ {SIG} (left)",
              f"<b>b-jet template:</b> the data b jets' shape, with a {100 * A.FPI_D:.1f}% prompt-like part (prompt J/ψ in b jets), so the signal's b decays inherit it: {ALT}. "
              f"<b>Per-category template:</b> the CMS simulation's b decays {SIG} have f = {BC['signal']['f']:.3f} ± {BC['signal']['f_err']:.3f}: {NOM} (right)"], "b s"),
        t="fit",
        notes="splot_prompt.py (SIGNAL=assoc): opposite sign, N_b >= 1, |z0| < 0.1 cm, 2.6-3.6 GeV, -0.5 < l_xy < 1.5 mm. Per-category template: btemplate_mc.py on MC truth, "
              "lambda scaled by data / simulation of the b jets; the b-jet category keeps its free prompt-like share. Mass fit: Crystal Ball + Gaussian, exponential continuum.")

TR = {r[0]: r[1:] for r in A.trow}
B2 = A.BJ2["lt2d"]
rrow = [["ℓ<sub>xy</sub> only, binned", f"{A.N_PR:.0f} ({A.SG['nominal']:.1f}σ)", f"{A.B_LXY:.0f} ({A.BL['signif']['nominal']:.1f}σ)", *TR["ℓ<sub>xy</sub>"]],
        ["m × ℓ<sub>xy</sub>, binned", f"{A.S2N['N_P']:.0f} ({A.T2['signif_prompt_notb']:.1f}σ)", f"{B2['samples']['notb']['N_P']:.0f} ({B2['signif_prompt_notb']:.1f}σ)", *TR["binned 2D"]],
        ["<b>m × ℓ<sub>xy</sub>, unbinned</b>", f"{ALT} ({ALT_S:.1f}σ)", f"<b>{NOM} ({NOM_S:.1f}σ)</b>", *TR["unbinned"]]]
content("result", "Measurement · yield", f"Prompt J/ψ {SIG}: {NOM} ({NOM_S:.1f}σ) with the proposed fit; the toys pick the template",
        f'<div class="row" style="gap:36px"><div class="col" style="flex:1.25">'
        + table(["Fit", "b-jet template", "per-category", "toys: b-jet tmpl.", "toys: per-cat."], rrow, widths=[33, 18, 19, 15, 15], num=(1, 2, 3, 4), hl=(2,))
        + '<div class="cap" style="text-align:left">toys: fit − truth at ≈ 135 true prompt J/ψ, b jets as simulated / with prompt J/ψ (data-like)</div>'
        + ul(["<b>Toys:</b> data-sized samples drawn from the CMS simulation per channel, category and sign, 0–150 prompt J/ψ injected; the data-like case adds prompt J/ψ to the b jets at 5% of their b decays, as the data show",
              f"<b>Reading:</b> the b-jet template copies the b jets' prompt part onto the signal's b decays and loses ≈ {-TOY[('bjet', 'datalike')]:.0f} J/ψ; the per-category one closes in both cases ({A.sgn(TOY[('cat', 'mc')])} / {A.sgn(TOY[('cat', 'datalike')])}); "
              "the binned fits are low"], "b s")
        + f'</div><div class="fig">{F("mc_toys_tmplcmp_unbinned", "toys, unbinned fit", 0.66)}</div></div>', t="fit",
        notes="All statistical only. Toy biases at 100 injected (plus the simulation's own prompt J/psi among the drawn candidates). The toys of the first night (drawn per l_xy class) "
              "were withdrawn. Significance: likelihood ratio with no prompt component, resolution and b-decay shape fixed.")

YS = {y: A.YF[y]["signif"]["nominal"] for y in A.YEARS}
content("stability", "Measurement · stability", f"Each year alone gives {min(YS.values()):.1f}–{max(YS.values()):.1f}σ; e + jets carries half of the signal, at {PER['ejets'][0] / PER['mujets'][0]:.1f}× the μ + jets rate per {TB} event",
        f'<div class="row" style="gap:40px"><div class="fig">{F("years_fp", "each year alone, b-jet template", 0.68)}</div>'
        f'<div class="fig">{F("chan_rates", "per channel, MC truth against sWeights", 0.68)}</div></div>'
        + ul([f"<b>Per year</b> (ℓ<sub>xy</sub> fit): " + ", ".join(f"{y} {YS[y]:.1f}σ" for y in A.YEARS) + f"; all {A.SG['nominal']:.1f}σ",
              f"<b>Per channel</b> (sPlot): μ + jets {CH['mujets']['sp'][0]:.0f} ± {CH['mujets']['sp'][1]:.0f}, e + jets {CH['ejets']['sp'][0]:.0f} ± {CH['ejets']['sp'][1]:.0f}, e μ {CH['emu']['sp'][0]:.0f} ± {CH['emu']['sp'][1]:.0f}; "
              f"per 10<sup>6</sup> {TB}: e + jets {PER['ejets'][0]:.1f} ± {PER['ejets'][1]:.1f} against μ + jets {PER['mujets'][0]:.1f} ± {PER['mujets'][1]:.1f}. "
              f"Without e + jets {A.SG['mu_emu']:.1f}σ ({A.BL['signif']['mu_emu']:.1f}σ, per-category template)"], "b s"),
        t="fit",
        notes=f"The e + jets excess of 4-5 Nov was the jet-cleaning bug; what is left is {ZE:.1f} sigma. Selected events data / (ttbar + backgrounds): "
              f"{A.EVA['mujets']:.2f}, {A.EVA['ejets']:.2f}, {A.EVA['emu']:.2f} (mu + jets, e + jets, e mu); no Drell-Yan and no QCD EM-enriched simulation.")

# ================================================================ 3 · what makes them
divider("d-models", 3, "3 · What makes them", "No CMS simulation makes them; onia in the shower plus a second scattering can",
        f"CMS simulation (full detector), seven generator set-ups, our private gen-level Pythia 8 + LETO study, and the rate of a second scattering. Numbers below use the b-jet-template fit ({ALT}).",
        stats=[(f"{R_D:.2f} ± {R_D_E:.2f}", f"prompt per b-decay J/ψ {SIG}, data"), (f"{R_CP5:.3f}", "POWHEG + Pythia 8 CP5 (CMS)"), (f"&lt; {R_SH:.3f}", "Sherpa, both tunes (CMS)")])

CS = A.CS
content("cms-sim", "CMS simulation · closure and rates", f"The fit closes on the CMS {TB} simulation, which has {XSIM:.1f}× fewer prompt J/ψ {SIG} than the data",
        f'<div class="row" style="gap:40px"><div class="fig">{F("mc_closure_signal", "the simulation as pseudo-data, MC truth shaded", 0.68)}</div>'
        f'<div class="fig">{F("mc_rates", "MC truth against sWeights, 310 fb⁻¹", 0.68)}</div></div>'
        + ul([f"<b>Closure:</b> the data chain on the simulation finds {CS['fit_prompt']:.0f} ± {CS['fit_prompt_err']:.0f} prompt J/ψ {SIG} for {CS['truth_prompt']:.0f} true; its b decays fake {mn(CS['fake_prompt_in_data'])} ± {CS['fake_prompt_in_data_err']:.0f} of the {A.SP_NP:.0f}",
              f"<b>Rates</b> ({TB} + backgrounds, σ × 310 fb<sup>−1</sup>): prompt {SIMP:.0f} {SIG} against {A.RS['data_prompt'][0]:.0f} in data, {A.simp(A.RI, 'prompt'):.0f} in b jets against {A.RI['data_prompt'][0]:.0f}: hadronisation only, no onia in the shower (and no EvtGen). "
              f"b decays data / simulation {A.RI['data_nonprompt'][0] / A.simp(A.RI, 'b decay'):.2f} (b jets), {A.RS['data_nonprompt'][0] / A.simp(A.RS, 'b decay'):.2f} ({SIG})"], "b s"),
        t="cms",
        notes="CMS simulation: Summer24 TTtoLNu2Q + TTto2L2Nu, POWHEG + Pythia 8 CP5, full detector simulation, through the ttJ workflow of the data pass (hadw/out_mc5, cluster 897589); "
              "backgrounds hadw/out_bkg (single top, W + jets, ttV, VV, QCD mu-enriched; no Drell-Yan, no QCD EM-enriched). The vetoed category does not close with the b-jet "
              "shape (untagged b jets with short b decays): hence the per-category template.")

vrow = [r for r in A.vrow]
content("generators", "CMS simulation · generators", f"None of seven CMS generator set-ups makes them: prompt per b-decay J/ψ {SIG}, data {R_D:.2f} ± {R_D_E:.2f}, the closest (CR1) {R_CR1:.3f}",
        f'<div class="row" style="gap:36px"><div class="fig">{F("mcvar_ratio", "within each sample: normalisation-free", 0.66)}</div><div class="col">'
        + small(table(["CMS sample, ℓ + jets", f"prompt / b decay, {SIG}", "in b jets"], vrow, widths=[34, 36, 30], num=(1, 2), hl=(len(vrow) - 1,)))
        + ul([f"<b>Ratio within each sample:</b> no normalisation, no b-tag or μμ efficiency; prompt J/ψ come from hadronisation only in all of them; "
              f"colour reconnection (CR1) doubles them; Sherpa (AHADIC clusters, own hadron decays) makes none"
              + (f"; e μ with the dilepton samples: data {GV_EMU['r']:.2f} ± {GV_EMU['err']:.2f}, the same pattern" if GV_EMU else "")], "b s")
        + "</div></div>", t="cms",
        notes="gen_variants.py (hadw/out_alt, clusters 897658 / 897660): MC truth prompt J/psi (no b-hadron ancestor) per MC truth b-decay J/psi in the selection of the fit; "
              "an empty sample gives a 90% CL upper limit. Sherpa 2.2.15 default and AHADIC tunes both use AHADIC cluster hadronisation; the l + jets Sherpa samples have one lepton charge each. "
              "No Run 3 Herwig ttbar sample exists.")

GX = A.GX
content("genstudy", "Gen study · Pythia 8 + LETO", f"Private gen-level Pythia 8 + LETO (onia in the shower) overshoots the rate {OVER[0]:.1f}–{OVER[1]:.1f}× and shows where top-decay J/ψ leak",
        f'<div class="row" style="gap:40px"><div class="fig">{F("assoc_gen", "fraction of each origin", 0.68)}</div>'
        f'<div class="fig">{F("gencmp_yields", "ℓ + jets, prompt sWeights", 0.68)}</div></div>'
        + ul([f"<b>Our sample, not CMS:</b> Pythia 8.317, {A.GEN['n_generated'] / 1e6:.0f}M {TB} ℓ + jets, LETO (colour singlet and octet), no detector; normalised to the data's b-decay J/ψ per p<sub>T</sub> bin",
              f"<b>Rate:</b> data / Pythia {SN['ratio_data_to_pythia']['ttj'][0]:.2f} ± {SN['ratio_data_to_pythia']['ttj'][1]:.2f} for the vetoed J/ψ ({SNC['ratio_data_to_pythia']['ttj'][0]:.2f} without their short b decays); "
              f"LETO's J/ψ are too hard in their jets (z {A.GZ['mean_z_pythia']:.2f} against {A.GZ['mean_z_data']:.2f})",
              f"<b>Leak:</b> {100 * GL['eff']:.0f}% of the top-decay J/ψ pass the veto, {100 * GL['signal_top_decay_share']:.0f}% of Pythia's signal; three ways to carry it to data: {RES_LO:.0f}–{RES_HI:.0f} of the {A.SP_NP:.0f}"], "b s"),
        t="gen",
        notes="genstudy/gen_ttjpsi.py (cluster 897323), CharmoniumShower:all = on, OniaShower:ldmeFac = 20 divided out; gen jets, b tag by efficiency, smeared MET; the same veto code. "
              "Residual estimates: Pythia's leak per vetoed top-decay J/psi x the data's vetoed ones (raw, or LETO reweighted to the data z), and Pythia's top-decay share x the signal.")

DPS = A.DPS
drow = [[f"{float(k):g} mb", f"{v:.0f}", f"{100 * v / A.SP_NP:.0f}%"] for k, v in DPS["pred_prompt_notb_by_sigma_eff_mb"].items()]
content("dps", "Second scattering", "A second parton scattering is not excluded, and no distribution separates it",
        '<div class="row" style="gap:40px"><div class="col">'
        + table(["σ<sub>eff</sub>", "J/ψ from DPS", f"of the {A.SP_NP:.0f}"], drow, widths=[34, 33, 33], num=(1, 2), hl=(1,))
        + ul([f"<b>Rate from data:</b> prompt pileup J/ψ per pileup collision, same selection and veto, × σ<sub>vis</sub> / σ<sub>eff</sub>: {A.DPS15:.0f} at 15 mb (W + 2 jets, γ + 3 jets), all of them at ≈ {A.SE_ALL:.0f} mb",
              f"<b>Room in the gen study:</b> data − Pythia + LETO (scaled to the vetoed J/ψ) leaves {SNC['dps_left']:.0f} ± {SNC['dps_left_err']:.0f} (ℓ + jets): σ<sub>eff</sub> ≈ {SNC['sigma_eff_mb']:.0f} mb",
              f"<b>Shapes:</b> soft (⟨p<sub>T</sub>⟩ {A.PT_SIG:.1f} GeV, inclusive J/ψ {A.PT_PU:.1f}); the p<sub>T</sub> balance against the {TB} gives a recoiling share {A.bfit(A.BFD)} (Δφ, right): a 2nd scattering alone only {A.BFD['z_vs_dps_only']:.1f}σ off"], "b s")
        + f'</div><div class="fig">{F("ttbal_dphi", f"{TB} without the J/ψ jet, unit area", 0.68)}</div></div>', t="fit",
        notes="r per pileup vertex from the prompt pileup J/psi (|z0| >= 0.1 cm) of the same selection and veto, sigma_vis = 0.7 x 80 mb; acceptance and efficiency cancel. "
              "Emulations of the balance: each J/psi (with its jet) put into other selected ttbar events, recoiling or with a balancing partner. The 6 Nov 3.2 sigma came from W-jet J/psi.")

# ================================================================ 4 · discussion
divider("d-disc", 4, "4 · For discussion", "Summary, three decisions, and what is still missing",
        "The inputs are all in; the choices below set the nominal result and its systematics.")

content("summary", "Summary", "Summary",
        ul([f"<b>Signal:</b> prompt J/ψ ΔR &gt; 0.4 from every jet of the two top decays, 2022–2026, {A.LTOT:.0f} fb<sup>−1</sup>, {A.NTT / 1e6:.1f}M {TB} events",
            f"<b>Yield:</b> {NOM} ({NOM_S:.1f}σ) with the per-category b-decay template, {ALT} ({ALT_S:.1f}σ) with the b-jet one; the toys favour the first; statistical only",
            f"<b>CMS simulation:</b> the fit closes ({CS['fit_prompt']:.0f} ± {CS['fit_prompt_err']:.0f} for {CS['truth_prompt']:.0f}); the simulation has {XSIM:.1f}× fewer prompt J/ψ; prompt per b-decay J/ψ data {R_D:.2f}, the best of seven set-ups {R_CR1:.3f}, Sherpa none",
            f"<b>Private gen-level Pythia 8 + LETO:</b> {OVER[0]:.1f}–{OVER[1]:.1f}× the data rate with its default LDMEs; {RES_LO:.0f}–{RES_HI:.0f} of the signal may still be top decay products",
            f"<b>Second scattering:</b> {A.DPS15:.0f} of the {A.SP_NP:.0f} at σ<sub>eff</sub> = 15 mb; no spectrum or balance separates it",
            "<b>Withdrawn:</b> the hard spectrum and the recoil of 6 Nov (J/ψ in W-decay jets)"])
        + f'<div class="callout blue"><b>Prompt J/ψ outside the t̅t decay jets are there at {ALT_S:.0f}–{NOM_S:.0f}σ (statistical), several times what the CMS t̅t simulation makes.</b> '
          "What is left: the fit choice, the residual top decays, the systematics.</div>", t="disc")

c = GC
CUT_DR = c.get("ΔR(J/ψ, t̅t jets) > 0.8")
dec2_cut = (f"ΔR &gt; 0.8 from the {TB} jets: top-decay share {100 * c['veto (nominal)']['top_share']:.0f}% → {100 * CUT_DR['top_share']:.0f}%, but keeps only {100 * CUT_DR['data_prompt_frac']:.0f}% of the data signal"
            if CUT_DR else "tighter cuts: backup")
DEC = [("c1", "1 · The nominal fit", [f"<b>A:</b> unbinned m × ℓ<sub>xy</sub>, per-category b-decay template: {NOM} ({NOM_S:.1f}σ); toys {A.sgn(TOY[('cat', 'mc')])} / {A.sgn(TOY[('cat', 'datalike')])}",
                                      f"<b>B:</b> the same with the b-jet template: {ALT} ({ALT_S:.1f}σ); toys {A.sgn(TOY[('bjet', 'mc')])} / {A.sgn(TOY[('bjet', 'datalike')])}",
                                      "<b>Proposed:</b> A; the template's f, λ and width as its systematic; then move every comparison to it"]),
       ("c2", "2 · The residual top-decay J/ψ", [f"<b>A:</b> subtract the gen-level estimate, {RES_LO:.0f}–{RES_HI:.0f} J/ψ, with the spread as its uncertainty",
                                                f"<b>B:</b> a tighter selection: {dec2_cut}",
                                                "<b>Proposed:</b> A, with the tighter ΔR as a cross-check"]),
       ("c3", "3 · What we quote", [f"<b>Observable:</b> prompt per b-decay J/ψ {SIG} ({R_D:.2f} ± {R_D_E:.2f}: no luminosity, {TB} selection or μμ efficiency), or a rate per {TB} event (needs the soft-muon efficiencies)",
                                    "<b>DPS:</b> quote single and double scattering together, with the DPS expectation as interpretation, or attempt a split",
                                    "<b>Proposed:</b> the ratio, inclusive; the per-event rate as a second step"])]
content("decisions", "For discussion", "Three decisions for today",
        '<div class="row" style="gap:28px;align-items:stretch">' + "".join(card(cl, h, ul(it, "b s")) for cl, h, it in DEC) + "</div>"
        + f'<div class="callout violet"><b>Also open:</b> e + jets has {PER["ejets"][0] / PER["mujets"][0]:.1f}× the μ + jets prompt rate per {TB} event ({ZE:.1f}σ apart); no Drell–Yan or QCD EM-enriched simulation for the non-{TB} J/ψ.</div>',
        t="disc",
        notes="Decision 1: the toys are the argument; the b-jet template is biased when the b jets hold prompt J/psi, which the data b jets do (5.8% prompt-like). "
              "Decision 2: gen_cutscan.py (backup) scans tighter selections: none removes the top decays without losing as much of the rest. "
              "Decision 3: the draft letter's framing quotes single and double scattering together.")

content("outlook", "Outlook", "What is not yet solid, what is needed, and the next steps",
        '<div class="row" style="gap:28px;align-items:stretch">'
        + card("c0", "Not yet solid", ul([f"<b>The template:</b> {ALT} or {NOM}", f"<b>Residual top decays:</b> {RES_LO:.0f}–{RES_HI:.0f}",
                                          f"<b>e + jets:</b> {PER['ejets'][0] / PER['mujets'][0]:.1f}× μ + jets per {TB}", "<b>Non-t̅t J/ψ:</b> Drell–Yan and QCD EM-enriched not simulated",
                                          "<b>Errors:</b> statistical only"], "b s"))
        + card("c3", "What is needed", ul(["<b>Systematics:</b> b-decay template (f, λ, width), resolution tail, continuum shape, pileup under the primary vertex",
                                           "<b>For a rate:</b> soft-muon and J/ψ efficiencies, the t̅t selection",
                                           "<b>For the models:</b> LETO with a detector (CMS sample or folded efficiencies); Herwig (no Run 3 t̅t sample)"], "b s"))
        + card("c2", "Next", ul(["<b>Fix the nominal</b> (today)", "<b>Move the comparisons</b> (simulation, generators, gen study, DPS) to it",
                                 "<b>Systematics table</b>, then the AN-26-XXX draft and the letter"], "b s"))
        + "</div>", t="next")

# ================================================================ backup
content("bk-samples", "Backup · simulation", "Two kinds of simulation, never mixed: the CMS samples (full detector) and our generator-level Pythia 8 + LETO study",
        table(["", "CMS simulation", "Private gen-level study (genstudy/)"], A.SAMP, widths=[14, 43, 43], num=(), hl=()), alt=True)

crow = [[nm, f"{100 * v['top_share']:.0f}%", f"{100 * v['isr_eff']:.0f}%", f"{v['data_prompt']:.0f}", f"{100 * v['pileup_frac']:.0f}%"] for nm, v in GC.items()]
content("bk-cuts", "Backup · decision 2", "No simple selection on top of the veto removes the top-decay J/ψ without losing as much of the rest",
        small(table(["Selection (after the veto)", "top-decay share (gen.)", "ISR kept (gen.)", "prompt J/ψ kept, data", "pileup J/ψ kept"], crow, widths=[40, 16, 14, 16, 14], num=(1, 2, 3, 4), hl=(0,)), 23)
        + ul([f"<b>Gen level</b> (private Pythia 8 + LETO): the top-decay share of the signal and the ISR J/ψ kept; <b>data:</b> the prompt sWeights kept; pileup J/ψ (random direction): the cost for J/ψ unrelated to the {TB}",
              f"<b>No pile-up of signal J/ψ just outside the 0.4 cones:</b> {100 * A.DS['near_0p8'][0]:.0f} ± {100 * A.DS['near_0p8'][1]:.0f}% at ΔR 0.4–0.8, random {100 * A.DPU['near_0p8'][0]:.0f}% (full deck)"], "b s"),
        alt=True, t="gen")

content("bk-templates", "Backup · decision 1", f"The b-decay ℓ<sub>xy</sub> shape per category in the CMS simulation: untagged b jets hold the short b decays",
        f'<div class="row" style="gap:40px"><div class="fig">{F("mc_btmpl_shapes", "MC truth b decays, unit area per mm", 0.68)}</div>'
        f'<div class="fig">{F("mc_toys_tmplcmp_fit2d", "toys, binned 2D fit", 0.68)}</div></div>'
        + ul([f"<b>Prompt-like share f of the b decays:</b> {BC['inb']['f']:.3f} (b jets), {BC['signal']['f']:.3f} ± {BC['signal']['f_err']:.3f} ({SIG}), {BC['ttj']['f']:.2f} ± {BC['ttj']['f_err']:.2f} (vetoed {TB} jets: the b tag failed because the b hadron flew little)",
              f"<b>The data b jets</b> show {100 * A.FPI_D:.1f}%: prompt J/ψ in b jets ({A.RI['data_prompt'][0]:.0f}, the simulation has {A.simp(A.RI, 'prompt'):.0f}); the b-jet template hands that share to the signal's b decays"], "b s"),
        alt=True, t="cms")

content("bk-how", "Backup · how", "How: rerun the chain from the candidate tables",
        '<div class="code">'
        '<span class="c"># environment: the repo\'s pixi "analysis" env (grid stack + iminuit, plotly, plotlyhep)</span>\n'
        'source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/v0.84.0/setup.sh\n'
        'export TTJ_STORE=&lt;pass area&gt; TTJ_REPO=&lt;ttJ_pixi checkout&gt;\n'
        '<span class="c"># full deck (files/ of 2026Nov07_jpsi_assoc): data, CMS simulation, generators, backgrounds</span>\n'
        '<span class="k">bash</span> refresh_assoc.sh   <span class="c"># veto → life → fit → splot → balance → figs</span>\n'
        '<span class="k">bash</span> refresh_mc.sh      <span class="c"># CMS t̅t: closure, data / simulation, per-category template</span>\n'
        '<span class="k">bash</span> refresh_alt.sh     <span class="c"># generator and tune variants</span>\n'
        '<span class="k">bash</span> refresh_bkg.sh     <span class="c"># non-t̅t backgrounds</span>\n'
        '<span class="c"># this deck: reads the full deck\'s outputs, copies its figures</span>\n'
        '<span class="k">python</span> build_deck_program.py</div>'
        + ul([f"<b>Code:</b> {CODE} (the pass); analysis chain in scripts/ttJ/analysis of the same branch", "<b>Batch:</b> lxbatch via eossubmit; pixi environment unpacked from an EOS cache in each job"], "b s"),
        alt=True)

content("refs", "References", "References and decks",
        '<ol class="refs">'
        '<li>Full analysis deck, 7–8 Nov (signal outside every top-decay jet; CMS simulation, generators, gen study, toys): <a href="../2026Nov07_jpsi_assoc/">2026Nov07_jpsi_assoc</a></li>'
        '<li>6 Nov: prompt J/ψ in 2022–2026, signal not b-matched: <a href="../2026Nov05_jpsi_run3/">2026Nov05_jpsi_run3</a></li>'
        '<li>4 Nov: prompt J/ψ in 2024: <a href="../2026Nov04_jpsi_prompt/">2026Nov04_jpsi_prompt</a></li>'
        '<li>3 Nov: the first J/ψ peak in t̅t, parking BPHNano, the two-muon vertex: <a href="../2026Nov03_jpsi/">2026Nov03_jpsi</a></li>'
        '<li>Onia in the Pythia 8 parton shower (LETO): N. Cooke, P. Ilten, L. Lönnblad, S. Mrenna, Eur. Phys. J. C 84 (2024) 432, arXiv:2312.05203</li>'
        '<li>BTVNanoCommissioning, <a href="https://github.com/cms-btv-pog/BTVNanoCommissioning">github.com/cms-btv-pog/BTVNanoCommissioning</a>; plotlyhep, <a href="https://github.com/DickyChant/plotlyhep">github.com/DickyChant/plotlyhep</a></li></ol>',
        alt=True)


def copy_inputs():
    """the figures this deck shows, from the full deck (refreshed on every build)"""
    names = sorted({m for _, h, _ in D.SLIDES for m in re.findall(r"\x00FIG:([^:]*):", h)})
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    for n in names:
        shutil.copy2(os.path.join(SRC, "figs", n + ".json"), os.path.join(HERE, "figs", n + ".json"))
    for f in os.listdir(os.path.join(HERE, "figs")):  # figures no slide shows any more
        if f.endswith(".json") and f[:-5] not in names:
            os.remove(os.path.join(HERE, "figs", f))
    if not os.path.isdir(os.path.join(HERE, "fonts")):
        shutil.copytree(os.path.join(SRC, "fonts"), os.path.join(HERE, "fonts"))
    print(f"{len(names)} figures from {SRC}")


if __name__ == "__main__":
    copy_inputs()
    D.main()
