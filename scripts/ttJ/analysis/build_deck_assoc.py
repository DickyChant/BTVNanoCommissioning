"""t̅t + J/psi -> mu mu, 2022-2026 data: the search for associated production, with the J/psi required outside every jet of the
two top decays. Same machinery as the 2022-2026 deck (deck_lib.py); the inputs come from refresh_assoc.sh:
results_run3.json (life_from_cands.py), fit_lxy.json, fit_jpsi.json, years/, splot.json, ttbar_balance.json, wpair_check.json,
gen_assoc.json, gen_compare.json (data against our private generator-level Pythia 8 + LETO sample).
Two kinds of simulation, kept apart on every slide (tag, kicker, wording): the CMS samples (full detector simulation, tag
'CMS SIMULATION', kicker 'CMS simulation · ...') and our gen-level Pythia 8 + LETO study (tag 'PRIVATE GEN LEVEL', kicker
'Gen study · ...').   usage: python3 build_deck_assoc.py"""
import json
import os

import deck_lib as D
from deck_lib import F, content, table, ul

D.HERE = os.path.dirname(os.path.abspath(__file__))
D.TITLE = "t̅t + J/ψ → μμ: J/ψ that are not from the top decays"
D.DATE = "7 Nov 2026"
D.FOOT = "t̅t + J/ψ, associated production" + (f" · {os.environ['TTJ_AUTHOR']}" if os.environ.get("TTJ_AUTHOR") else "")
D.EXTRA_LINKS = ('<a href="../2026Nov05_jpsi_run3/">6 Nov deck (signal: not b-matched)</a>'
                 '<a href="../2026Nov04_jpsi_prompt/">2024 prompt deck</a><a href="../2026Nov03_jpsi/">J/ψ first-look deck</a>')
D.BUILD = "build_deck_assoc.py"
AUTHOR = f"{os.environ['TTJ_AUTHOR']} · " if os.environ.get("TTJ_AUTHOR") else ""  # the name on the cover and the footer


def J(name):
    return json.load(open(os.path.join(D.HERE, name)))


R, FIT, FJ, SPL, BAL, WP, GEN, GZ, GC, GX = (J(n) for n in ("results_run3.json", "fit_lxy.json", "fit_jpsi.json", "splot.json", "ttbar_balance.json",
                                                            "wpair_check.json", "gen_assoc.json", "gen_leak_z.json", "gen_cutscan.json", "gen_compare.json"))
# the 6 Nov signal (not b-matched): the unbinned fit of the previous deck (TTJ_PREV_SPLOT overrides its place)
OLD = json.load(open(os.environ.get("TTJ_PREV_SPLOT", os.path.join(D.HERE, "..", "..", "2026Nov05_jpsi_run3", "files", "splot.json"))))
# the CMS ttbar simulation (refresh_mc.sh): data / simulation, the closure of the fits, the template refit of the vetoed J/psi;
# the generator / tune variants (refresh_alt.sh)
MCC, MTF, MFJ = J("mc_compare.json"), J("mc_template.json"), J(os.path.join("mc", "fit_jpsi.json"))
FPI_D = FJ["lt2d"]["params"]["NP_inb"] / (FJ["lt2d"]["params"]["NP_inb"] + FJ["lt2d"]["params"]["NNP_inb"])  # prompt-like share of the data b-jet J/psi
RS, RI = MCC["rates"]["signal_all"], MCC["rates"]["inb_all"]  # data (sWeights) and simulation (truth) per category, 310 fb^-1
HAS_BKG = bool(MCC["events"]["mujets"].get("bkg"))  # the non-ttbar simulation is in (refresh_bkg.sh)


def simp(r, sp):
    """the simulation of one species in a rates entry: ttbar + the non-ttbar backgrounds when present"""
    return r[f"mc_{sp}"][0] + (r[f"bkg_{sp}"][0] if HAS_BKG and f"bkg_{sp}" in r else 0.0)


SIMN_T = "the CMS t̅t + background simulation" if HAS_BKG else "the CMS t̅t simulation alone"
GV = J("gen_variants.json") if os.path.exists(os.path.join(D.HERE, "gen_variants.json")) else {"gen": {}}
NJOBS = 4872
REPO = "https://github.com/DickyChant/BTVNanoCommissioning"
SHA = "79eecfe7a7200726b34cc80bf8d39b9ed6ee4789"  # the pass: run5 (891634) + the retry run5b (895659), merged in hadw/run5m
CODE = f'<a href="{REPO}/commit/{SHA}">ttJ_pixi @ {SHA[:7]}</a>'
YEARS = ["2022", "2023", "2024", "2025", "2026"]
YR = {y: J(os.path.join("years", f"results_{y}.json")) for y in YEARS}
YF = {y: J(os.path.join("years", f"fit_lxy_{y}.json")) for y in YEARS}
YJ = {y: J(os.path.join("years", f"fit_jpsi_{y}.json")) for y in YEARS}
CHN = {"μ + jets": "mujets", "e + jets": "ejets", "e μ": "emu"}
C3 = {nm: R["channels"][c] for nm, c in CHN.items()}
TB = "t̅t"
NB1 = "N<sub>b</sub> ≥ 1"
SIG = f"outside the {TB} jets"
mn = lambda x, d=0: f"{x:.{d}f}".replace("-", "−")  # noqa: E731
LUMI = {"2022": 34.8, "2023": 28.3, "2024": 109.9, "2025": 111.2, "2026": 25.8}
LTOT = sum(LUMI.values())

# the two kinds of simulation: the CMS samples (full detector) and our private generator-level Pythia 8 + LETO sample
TAG_CMS = "color:#1F5BC0;border-color:#9DB8E6;background:#EEF3FB"
TAG_GEN = "color:#9A4A00;border-color:#E8B57E;background:#FDF3E7"


def tag(kind):
    return {"data": '<span class="tag part">DATA · 2022–2026</span>', "fit": '<span class="tag new">FIT · 2022–2026</span>',
            "cms": f'<span class="tag" style="{TAG_CMS}">CMS SIMULATION · FULL DETECTOR</span>',
            "gen": f'<span class="tag" style="{TAG_GEN}">PRIVATE GEN LEVEL · PYTHIA 8 + LETO</span>', "next": '<span class="tag new">NEXT</span>'}.get(kind, "")


D.tag = tag


def yl(C, k):
    return C["life"][f"yield_{k}"]


def ysum(k, chans=C3.values()):
    n = sum(yl(C, k)["n"] for C in chans)
    return n, sum(yl(C, k)["err"] ** 2 for C in chans) ** 0.5


def ysum_y(k, chans=C3.values()):
    """the window - sidebands of the summary yields (all lifetimes), summed over channels"""
    return sum(C["yields"][k]["n"] for C in chans), sum(C["yields"][k]["err"] ** 2 for C in chans) ** 0.5


def fmt(ne, d=0):
    return f"{mn(ne[0], d)} ± {ne[1]:.{d}f}"


def ntt(chans=C3.values()):
    return sum(C["yields"]["data_os_all_ge1b"]["n_tt"] for C in chans)


def pm(x, lo, hi, scale=1.0, d=1):
    return f"{scale * x:.{d}f}<sup>+{scale * hi:.{d}f}</sup><sub>−{-scale * lo:.{d}f}</sub>"


def pms(x, lo, hi, scale=1.0, d=1):
    return f"{scale * x:.{d}f} ± {scale * 0.5 * (hi - lo):.{d}f}"


# --- numbers
NTT = ntt()
SPF, SPT, SPI = SPL["fits"]["notb"], SPL["fits"]["ttj"], SPL["fits"]["inb"]
SP_NP, SP_NPE = SPF["N"][0], 0.5 * (SPF["NP_minos"][1] - SPF["NP_minos"][0])
ST_NP, ST_NPE = SPT["N"][0], 0.5 * (SPT["NP_minos"][1] - SPT["NP_minos"][0])
# the vetoed prompt J/psi refitted with the b-decay l_xy shape of the simulation in their category (short b decays of untagged
# b jets; b-decay yield constrained to the simulation): the yield the Pythia + LETO estimates use
TTJ_RF = MTF["fits"]["ttj_ljets_nbcon"]
ST_C, ST_CE = TTJ_RF["N_p"], TTJ_RF["N_p_err"]
OLD_NP = OLD["fits"]["notb"]["N"][0]
P, MI, V = FIT["params"], FIT["minos"], FIT["variations"]
FP, FPE = P["fp_notb"], MI["fp_notb"]
N_PV = P["N_notb"]
N_PR = FP * N_PV
D_EXP = V["exp"]["fp_notb"] - FP
T2 = FJ["lt2d"]
S2N = T2["samples"]["notb"]
MS = FJ["mass"]
SG = FIT["signif"]
WPP = WP["wpair"]
GL = GEN["leak"]
_t, _i = GEN["table"]["top decay"], GEN["table"]["ISR"]
TOPSHARE_TTJ = _t["other ttbar jet"] / (_t["other ttbar jet"] + _i["other ttbar jet"])  # top-decay share of the vetoed other-ttbar-jet J/psi (Pythia)
# the residual top-decay J/psi in the signal, three estimates: Pythia's leak per vetoed J/psi in the other ttbar jets times the data count of
# those (raw and reweighted to the z measured in data), and Pythia's top-decay share of the signal times the signal
GZN = GZ["variants"]["nominal"]
RES = {"per vetoed (raw)": GL["leak_per_ttj"] * ST_C * TOPSHARE_TTJ, "per vetoed (z reweighted)": GZN["leak_per_vetoed_ttjet"] * ST_C,
       "share of the signal": GL["signal_top_decay_share"] * SP_NP}
RES_LO, RES_HI = min(RES.values()), max(RES.values())
GO = GEN["frac_by_origin"]
DPS = SPL["dps"]
DPS15 = DPS["pred_prompt_notb_by_sigma_eff_mb"]["15"]
SE_ALL = 15.0 * DPS15 / SP_NP
# the sWeighted prompt yield per category of the signal: in a jet outside the ttbar (jcat 2) or in no selected jet (jcat 3)
import numpy as _np  # noqa: E402

_c = _np.asarray(SPL["hists"]["notb_prompt_cat_all"][0], float)  # cat: 1 in a selected jet, 2 in none
SP_INJET, SP_NOJET = _c[0], _c[1]


def smean(key, v="pt"):
    e = _np.asarray(SPL["vars"][v]["edges"], float)
    c = 0.5 * (e[1:] + e[:-1])
    h = _np.asarray(SPL["hists"][f"{key}_{v}_all"][0], float)
    return float((h * c).sum() / h.sum())


PT_SIG, PT_PU, PT_TTJ = smean("notb_prompt"), smean("pileupfit_prompt"), smean("ttj_prompt")


def srange(a, b):
    lo, hi = sorted((a, b))
    return f"{lo:.1f}σ" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f}–{hi:.1f}σ"


SR = srange(T2["signif_prompt_notb"], SG["nominal"])

# --- cover
D.SLIDES.append(("cover", f"""<section id="cover" class="slide div d0" style="gap:38px">
<p class="dv-ey" style="color:#E8A765">CMS Run 3 · 2022–2026 data · √s = 13.6 TeV · BTVNanoCommissioning</p>
<h1 class="dv-t" style="font-size:84px">t̅t + J/ψ: the J/ψ that are not from the top decays</h1>
<div style="display:flex;flex-direction:column;gap:14px">
<p style="font-size:36px;line-height:1.3;color:#BFD0E6"><span style="color:#7FD1C7">Signal:</span> the J/ψ ΔR &gt; 0.4 from every jet the two top decays can be built from: b jets, the four {TB} jets of ℓ + jets, the jets of the full {TB} fit</p>
<p style="font-size:36px;line-height:1.3;color:#BFD0E6"><span style="color:#B5D58B">Checked:</span> the vetoed J/ψ make a W peak; in our generator-level Pythia 8 + LETO study (private, no detector), {100 * GL['eff']:.0f}% of the top-decay J/ψ pass the veto</p>
<p style="font-size:36px;line-height:1.3;color:#BFD0E6"><span style="color:#C9A7F0">Result:</span> {SP_NP:.0f} ± {SP_NPE:.0f} prompt J/ψ outside the {TB} jets, {SR} (statistical); the fit closes on the CMS {TB} simulation, which alone has {RS['data_prompt'][0] / RS['mc_prompt'][0]:.0f}× fewer; the gen-level Pythia + LETO (J/ψ in the shower) plus a second scattering describes it</p>
</div>
<p style="font-size:28px;color:#BFD0E6">{AUTHOR}{D.DATE} · data, CMS simulation and a private gen-level Pythia + LETO study · {LTOT:.0f} fb<sup>−1</sup> · code <a href="{REPO}/commit/{SHA}" style="color:#7FD1C7">ttJ_pixi @ {SHA[:7]}</a></p>
<div class="footer"><span></span><span>PAGE</span></div></section>""",
                 "Associated production means a J/psi that is not one of the top decay products. The 6 Nov deck (link in the bar) required only that the "
                 "J/psi be away from the b jets, so a J/psi inside a W-decay jet counted as signal; here the J/psi must be away from every jet of the two top "
                 "decays. Same data (the merged run5 + run5b pass), same fits. The 6 Nov conclusions that the prompt J/psi are harder than inclusive J/psi and "
                 "recoil against the ttbar came from the J/psi in W-decay jets and do not hold for this signal."))

# --- motivation (the framing of the draft letter)
MB = FJ["mass"]["inb_pv"]
mrow = [["1 · non-prompt J/ψ in the top-decay b jets", "t → W b, b → H<sub>b</sub> + X, H<sub>b</sub> → J/ψ X: b-quark fragmentation, b-hadron mix and decays (the reference)",
         f"{MB['Ns']:.0f} ± {MB['Ns_err']:.0f} J/ψ in b jets"],
        ["2 · prompt J/ψ in additional jets", "quarkonium formation in the QCD radiation of the t̅t (light, c, b, gluon jets)", f"{SP_INJET:.0f} of the prompt signal (sWeights)"],
        [f"3 · prompt J/ψ away from every top-decay jet", "inclusive associated production: radiation in the hard scattering, residual top-decay products, double parton scattering",
         f"{SP_NP:.0f} ± {SP_NPE:.0f}, {SPF['signif_prompt']:.1f}σ"]]
content("motivation", "Motivation", "t̅t events as a laboratory for heavy-flavour fragmentation and quarkonium formation in QCD radiation",
        ul(["<b>Quarkonium</b> is the hardest case of heavy-quark hadronization: the roles of the production mechanisms, and how parton showers implement them (NRQCD in Pythia 8, LETO), are open",
            "<b>J/ψ in jets</b> add the momentum share z = p<sub>T</sub>(J/ψ)/p<sub>T</sub>(jet) and the surrounding activity (CMS J/ψ-in-jet measurement, NRQCD fragmenting jet functions)",
            f"<b>In t̅t</b> every top gives an identifiable energetic b quark and the extra radiation gives light, charm, b and gluon jets: one well-measured hard process, several quarkonium environments"], "b s")
        + table(["Category", "what it probes", "this deck"], mrow, widths=[28, 50, 22], num=(), hl=(2,)),
        notes="Framing of the draft letter 'Quarkonium production and heavy-flavor fragmentation in top quark pair events in pp collisions at 13.6 TeV': non-prompt J/psi from "
              "b-hadron decays in the top-decay b jets as the fragmentation reference (also used by CMS for m_t from m(J/psi, l) at 8 TeV); prompt J/psi in additional jets as "
              "quarkonium in QCD radiation; prompt J/psi away from all top-decay jets as the inclusive associated-production category, with single and double parton "
              "scattering not separated in a model-dependent way. Heavy-flavour radiation in ttbar also matters for ttH(bb) and ttbb.")

# --- the definition
crow = []
for lab, key, fit_ in (("b jet: b candidate or UParT-M jet", "inb", SPI), (f"another {TB} jet: the W pair, or a jet of the {TB} fit", "ttj", SPT)):
    crow.append([lab, "vetoed", fmt(ysum_y(f"data_os_{key}_ge1b")), f"{fit_['N'][0]:.0f} ± {0.5 * (fit_['NP_minos'][1] - fit_['NP_minos'][0]):.0f}"])
crow.append([f"<b>{SIG}: in another jet or in no jet</b>", "<b>signal</b>", fmt(ysum_y("data_os_notb_ge1b")), f"<b>{SP_NP:.0f} ± {SP_NPE:.0f}</b>"])
content("definition", "Signal · definition", f"Signal: the J/ψ {SIG}, i.e. not in any jet of the two top decays",
        table(["Where the J/ψ is (ΔR &lt; 0.4)", "", "J/ψ, window − sidebands", "prompt J/ψ (sPlot)"], crow, widths=[46, 12, 21, 21], num=(2, 3), hl=(2,))
        + '<div class="row" style="gap:40px;margin-top:8px"><div class="col">'
        + ul([f"<b>{TB} jets vetoed:</b> ℓ + jets: the four leading selected jets (W pair and two b candidates of the selection) and the four jets of the χ² {TB} fit over the eight leading jets (m<sub>W</sub>, m<sub>t</sub>); e μ: the two b candidates; every b-tagged jet",
              f"<b>6 Nov deck:</b> only the b jets were vetoed: {OLD_NP:.0f} prompt J/ψ, {ST_NP:.0f} of them inside a W-pair or {TB}-fit jet"], "b s")
        + '</div><div class="col">'
        + ul([f"<b>Signal:</b> {SP_INJET:.0f} prompt J/ψ in a jet that is not a {TB} jet, {SP_NOJET:.0f} in no selected jet (sWeights)",
              f"<b>Leptons:</b> the J/ψ muons are never the {TB} lepton candidates; the {TB} selection and the J/ψ selection are unchanged"], "b s")
        + "</div></div>", t="fit",
        notes=f"Columns: opposite sign, {NB1.replace('<sub>', '_').replace('</sub>', '')}, all lifetimes and z0, window 2.95-3.25 GeV minus half of the sidebands; prompt J/psi from "
              "the unbinned m x l_xy fit of each category (splot_prompt.py: the primary vertex, |z0| < 0.1 cm). Veto code: assoc_veto.py (columns tt_sel, tt_chi2, "
              "jcat added to the candidate table); the chi2 fit is the one of the pT-balance test (ttreco.py), with the J/psi jet allowed. A J/psi within 0.4 of "
              "two jets counts in the category of the ttbar jet. The b candidates of l + jets are the two leading-four jets outside the W pair (the non-b-tagged "
              "pair closest to m_W), those of e mu the two jets with the highest b score.")

# --- the two kinds of simulation: the CMS samples and our private generator-level Pythia 8 + LETO study
PILL = "font:500 15px/1 'IBM Plex Mono',monospace;letter-spacing:1px;padding:5px 9px;border-radius:5px;border:1.5px solid;white-space:nowrap;"
SAMP = [["What", "official CMS samples through the code of the data pass", "our own sample, generated for this study: <b>not a CMS sample</b>"],
        ["Generators", f"POWHEG + Pythia 8 CP5 (Summer24); CR1, CR2, ERD, Sherpa, AHADIC, MadGraph MLM, aMC@NLO FxFx; single top, W + jets, {TB}V, dibosons, QCD",
         f"Pythia 8.317 alone: {TB} ℓ + jets, CP5, {GEN['n_generated'] / 1e6:.0f}M events"],
        ["Prompt J/ψ", f"from hadronisation only, no onia in the shower: {RS['mc_prompt'][0]:.0f} outside the {TB} jets",
         "LETO: onia in the parton shower (colour singlet and octet), LDMEs × 20 for statistics, divided out"],
        ["Detector", "full simulation and reconstruction; selection, veto and fits as on data", "none: gen jets, b tag by efficiency (80 / 15 / 1%), p<sub>T</sub><sup>miss</sup> smeared"],
        ["Normalisation", "σ × 310 fb<sup>−1</sup>", "to the data's b-decay J/ψ per p<sub>T</sub>(J/ψ) bin"],
        ["Used for", "closure, toys, b-decay templates, data / simulation, backgrounds, generator comparison",
         "where top-decay J/ψ end up (the leak), the prompt-J/ψ rate and shapes, the room for a 2nd scattering"],
        ["On the slides", f'<span style="{PILL}{TAG_CMS}">CMS SIMULATION · FULL DETECTOR</span> &nbsp;“CMS simulation · …”',
         f'<span style="{PILL}{TAG_GEN}">PRIVATE GEN LEVEL · PYTHIA 8 + LETO</span> &nbsp;“Gen study · …”']]
content("samples", "Samples · two kinds of simulation", "Two kinds of simulation, never mixed: the CMS samples (full detector) and our own generator-level Pythia 8 + LETO study",
        table(["", "CMS simulation", "Private gen-level study (genstudy/)"], SAMP, widths=[14, 43, 43], num=(), hl=()),
        notes="CMS samples: hadw/out_mc5 (TTtoLNu2Q + TTto2L2Nu, Summer24 NanoAODv15), hadw/out_alt (generators and tunes), hadw/out_bkg (non-ttbar backgrounds), "
              "all through the ttJ workflow of the data pass. Gen study: genstudy/gen_ttjpsi.py (cluster 897323), Pythia 8.317 from the LCG_110 view, "
              "CharmoniumShower:all = on, OniaShower:ldmeFac = 20; the CMS Pythia samples have no onia in the shower (and no EvtGen).")

# --- the W pair
f0, fe0 = WPP["prompt_in_w"]["win"]
content("wjets", "Signal · why the W pair", f"The vetoed J/ψ in a W-pair jet make a W peak: {100 * f0:.0f} ± {100 * fe0:.0f}% in 65–95 GeV, {100 * WPP['extra_jet']['win'][0]:.0f}% for an extra jet",
        f'<div class="row"><div class="fig">{F("assoc_wmass", "unit area per GeV")}</div><div class="col">'
        + ul([f"<b>{100 * WPP['share_in_w_of_ttj']:.0f}% of the vetoed prompt J/ψ</b> ({WPP['prompt_in_w']['n']:.0f}) sit in a jet of the selection's W pair",
              f"<b>An extra jet</b> (the same J/ψ jets put into other {TB} events) also lands in the W pair {100 * WPP['extra_jet']['share_in_w']:.0f}% of the time, but pairs to a broad m(jj): {100 * WPP['extra_jet']['win'][0]:.0f}% in 65–95 GeV",
              f"<b>The J/ψ W-pair jets:</b> {100 * f0:.0f} ± {100 * fe0:.0f}%, {WPP['z_vs_extra']:.1f}σ above an extra jet, like {TB} events in general ({100 * WPP['all_ttbar']['win'][0]:.0f}%): W-decay jets with a J/ψ in them",
              f"<b>So the b-jet veto alone was not enough:</b> the J/ψ it kept in these jets are top decay products: in W-decay jets, or in top b jets that failed the b tag (the CMS simulation's b decays there give {100 * MCC['wpeak_bdecay']:.0f}% in 65–95 GeV; they also pass the prompt fit, CMS simulation slides)"], "b s")
        + "</div></div>", t="fit",
        notes="sWeights of the vetoed category (jcat 1). The W pair as in the workflow: the non-b-tagged pair of the leading four jets closest to 80.4 GeV. "
              "Emulation (wpair_check.py): each J/psi jet in 300 other selected events of l + jets, the selection redone (N_b >= 1 of the leading four, the J/psi jet "
              "not a b candidate). ttbar events in general: the events of the b-matched J/psi (b-decay sWeights), whose W pair the J/psi does not touch.")

# --- the generator check
GT = GEN["table"]
grow = []
for o, lab in (("top decay", "top decay: b or W-quark shower"), ("ISR", "initial-state radiation"), ("top radiation", "radiated by a top before its decay"), ("MPI", "second parton scattering")):
    t = GT.get(o)
    if not t:
        continue
    n = sum(t.values())
    grow.append([lab, f"{n}", f"{100 * (t['b jet'] + t['other ttbar jet']) / n:.0f}%", f"{100 * (t['jet outside the ttbar'] + t['no selected jet']) / n:.0f}%"])
content("pythia", "Gen study · the veto", f"Private gen-level Pythia 8 + LETO: {100 * GL['eff']:.0f}% of the top-decay J/ψ pass the veto, {100 * GL['signal_top_decay_share']:.0f}% of the signal",
        f'<div class="row"><div class="fig">{F("assoc_gen", "fraction of each origin")}</div><div class="col">'
        + table(["Origin of the J/ψ", "J/ψ", "vetoed", "signal"], grow, widths=[52, 14, 17, 17], num=(1, 2, 3), hl=(0,))
        + ul([f"<b>Sample (ours, not CMS):</b> Pythia 8.317 {TB} ℓ + jets at generator level, no detector simulation; CP5, J/ψ in the parton shower (LETO: g → J/ψ gg, c → J/ψ c, colour octet); origin from the event record",
              "<b>Same selection and veto code as on data</b> (gen jets with the jet ID muon fraction &lt; 0.8, b tag 80/15/1% for b/c/light)",
              f"<b>Leak:</b> {GL['top_decay_passing']} of {GL['top_decay_total']} top-decay J/ψ pass: {GL['leaked']['in no selected jet, inside a gen jet']} inside an unselected jet (soft or failing the ID), "
              f"{GL['leaked']['in no jet at all']} outside every jet, {GL['leaked']['in a selected jet outside the ttbar']} in a non-{TB} jet",
              f"<b>In data:</b> {GL['leak_per_ttj']:.2f} leaked per vetoed top-decay J/ψ (Pythia) × the {ST_C * TOPSHARE_TTJ:.0f} top-decay ones among the {ST_C:.0f} ± {ST_CE:.0f} prompt vetoed ({100 * TOPSHARE_TTJ:.0f}% in Pythia; the {ST_NP:.0f} less short b decays, CMS simulation slides) ≈ {GL['leak_per_ttj'] * ST_C * TOPSHARE_TTJ:.0f} of the {SP_NP:.0f}; "
              f"{100 * GL['signal_top_decay_share']:.0f}% from Pythia's composition, {GZN['leak_per_vetoed_ttjet'] * ST_C:.0f} with LETO reweighted to the data z: about {RES_LO:.0f}–{RES_HI:.0f} of them are still top decay products"], "b s")
        + "</div></div>", t="gen",
        notes=f"genstudy/gen_ttjpsi.py (cluster 897323, {GEN['n_generated'] / 1e6:.1f}M ttbar events, W+ -> l nu, W- -> q q'), gen_assoc.py. CharmoniumShower:all = on, "
              f"OniaShower:ldmeFac = 20 (the LDMEs scaled up for statistics; fractions do not depend on it), J/psi -> mu mu. Origin: the first-mother chain of the J/psi "
              "reaches the b or the W of a top decay (top decay), a top before its decay (top radiation), an MPI parton (MPI) or an initial-state emission (ISR). "
              "Selection: the W lepton (mu pT > 30, e pT > 35 GeV), gen jets pT > 20 GeV, |eta| < 2.5, muon energy fraction < 0.8, Delta R > 0.4 from the lepton, at "
              "least four, N_b >= 1 of the leading four; prompt J/psi with both muons pT > 3 GeV, |eta| < 2.4; MET the neutrino sum smeared by 20 GeV. "
              f"Pythia's ratio of vetoed to signal prompt J/psi: {GEN['ratio_ttjet_to_signal']:.2f} (data: {ST_NP / SP_NP:.2f} for the other ttbar jets alone). "
              "The rate of J/psi in the shower is a model; what matters here is where a top-decay J/psi ends up, which follows from the jets.")

# --- data checks
DS, DPU = WP["drtt"]["signal"], WP["drtt"]["pileup"]
content("vetocheck", "Signal · data check", f"No pile-up of signal J/ψ just outside the veto cones: {100 * DS['near_0p8'][0]:.0f} ± {100 * DS['near_0p8'][1]:.0f}% at ΔR 0.4–0.8, random {100 * DPU['near_0p8'][0]:.0f}%",
        f'<div class="row"><div class="fig">{F("assoc_drtt", "unit area per unit ΔR")}</div><div class="col">'
        + ul([f"<b>Random reference:</b> prompt pileup J/ψ (another collision): placed at random with respect to the {TB} jets of the event, same veto",
              f"<b>A leak from wide-angle top-decay radiation</b> would pile up right outside the 0.4 cones: {100 * DS['near_0p8'][0]:.0f} ± {100 * DS['near_0p8'][1]:.0f}% within 0.4–0.8 for the signal, {100 * DPU['near_0p8'][0]:.0f} ± {100 * DPU['near_0p8'][1]:.0f}% random",
              f"<b>In no selected jet but inside a jet above 15 GeV</b> (no ID): {100 * DS['softjet'][0]:.0f} ± {100 * DS['softjet'][1]:.0f}% of the signal, {100 * DPU['softjet'][0]:.0f}% random; "
              f"{100 * DS['softjet_muef_ge_0p8']:.0f} points of it are jets made of the J/ψ muons (muon fraction ≥ 0.8). Soft W-quark jets below 20 GeV are where the gen-level Pythia leaks (previous slide)"], "b s")
        + "</div></div>", t="fit",
        notes="tt_drmin of assoc_veto.py: Delta R to the nearest jet of the selection or of the chi2 fit; sWeights of the prompt components. The jets above 15 GeV are "
              "the nearest jet without ID or cleaning (sj_* columns of the table).")

# --- yields
rows = []
for nm, C in list(C3.items()) + [("sum", None)]:
    cs = [C] if C else list(C3.values())
    yy = lambda k: (sum(c["yields"][k]["n"] for c in cs), sum(c["yields"][k]["err"] ** 2 for c in cs) ** 0.5)  # noqa: E731
    rows.append([nm, f"{ntt(cs) / 1e3:.0f}k", fmt(yy("data_os_all_ge1b")), fmt(yy("data_os_inb_ge1b")), fmt(yy("data_os_ttj_ge1b")), fmt(yy("data_os_notb_ge1b")),
                 fmt(ysum("prompt_notb", cs)), fmt(ysum("nonprompt_notb", cs)), fmt(ysum("pileup_notb", cs))])
N_SIG = ysum_y("data_os_notb_ge1b")
NPU = ysum("pileup_notb")
content("yields", "Result · yields", f"{N_SIG[0]:.0f} J/ψ {SIG}, {100 * NPU[0] / N_SIG[0]:.0f}% of them from pileup",
        table(["Channel", f"{TB} events", "All J/ψ", "b jet", f"other {TB} jet", f"{SIG}", "… prompt region", "… non-prompt", "… pileup"], rows,
              widths=[10, 10, 12, 12, 11, 12, 11, 11, 11], num=(1, 2, 3, 4, 5, 6, 7, 8), hl=(3,))
        + ul([f"Opposite sign, window − sidebands, {NB1}; columns 7–9 split the signal into the lifetime regions (the 'between' region is left out)",
              "<b>e μ</b> has no W pair: its only vetoed jets are the two b candidates",
              "<b>Prompt region ≠ prompt:</b> it also holds b decays with ℓ<sub>xy</sub> &lt; 0.06 mm; the fits separate them"], "b s"),
        t="data",
        notes="Merged pass: run5 (cluster 891634) and the retry of the 236 jobs with skipped files (895659), 22 of 58723 input files missing; the counts are "
              "from the per-candidate table (life_from_cands.py), the ttbar event counts from the histograms of the same jobs.")

# --- mass fit
MSH = FJ["shape"]
content("mfit", "Measurement · mass fit", f"Mass fit: {MS['notb_pv']['Ns']:.0f} ± {MS['notb_pv']['Ns_err']:.0f} J/ψ {SIG} from the primary vertex, {MS['notb_pv']['signif']:.0f}σ",
        f'<div class="row" style="gap:44px"><div class="fig">{F("mfit_notb_pv", "signal")}</div>'
        f'<div class="fig">{F("mfit_inb_pv", "b-matched (control)")}</div></div>',
        t="fit", notes=f"Binned Poisson fit, 2.6-3.6 GeV in 25 MeV bins. J/psi: Crystal Ball (sigma {1e3 * MSH['s1']:.1f} MeV) + Gaussian (sigma {1e3 * MSH['s2']:.1f} MeV), "
        "the shape from all primary-vertex J/psi and fixed for the samples; continuum: an exponential per sample. Significance from the likelihood ratio with no J/psi.")

# --- mass x l_xy fit
content("lt2d", "Measurement · mass × ℓ<sub>xy</sub>", f"Mass × ℓ<sub>xy</sub> fit: ({100 * S2N['fp']:.1f} ± {100 * S2N['fp_err']:.1f})% prompt, {S2N['N_P']:.0f} J/ψ, {T2['signif_prompt_notb']:.1f}σ",
        f'<div class="row" style="gap:44px"><div class="fig">{F("lt2d_notb", "signal, signal window")}</div>'
        f'<div class="fig">{F("lt2d_inb", "b-matched, signal window")}</div></div>',
        t="fit", notes="Binned Poisson fit of l_xy in the signal window and the sidebands (fit_jpsi.py, as in jpsi_eec): prompt N_P S R(l) + non-prompt N_NP S D(l) + "
        "continuum N_B C B(l). The b decays outside the ttbar jets take the b-matched J/psi l_xy shape (both b quarks are from the ttbar decays); with their "
        f"own exponential the prompt fraction is {100 * T2['var_exp']['fp']:.1f}% ({T2['var_exp']['signif']:.1f} sigma). R: one Gaussian of {1e3 * T2['params']['s1']:.0f} um.")

# --- the prompt yield
crow = []
for nm, c in CHN.items():
    X = FIT["channels"][c]
    crow.append([nm, f"{X['N']:.0f} ± {X['N_err']:.0f}", pm(X["fp"], *X["fp_minos"], 100), f"{X['N'] * X['fp']:.0f}", f"{1e5 * X['N'] * X['fp'] / ntt([C3[nm]]):.2f}"])
crow.append(["all three", f"{N_PV:.0f} ± {FIT['errors']['N_notb']:.0f}", pm(FP, *FPE, 100), f"{N_PR:.0f}", f"{1e5 * N_PR / NTT:.2f}"])
content("prompt", "Measurement · prompt yield", f"{N_PR:.0f} prompt J/ψ {SIG} in {NTT / 1e6:.1f}M {TB} events, {SG['nominal']:.1f}σ (ℓ<sub>xy</sub> fit)",
        '<div class="row" style="gap:40px"><div class="col">'
        + table(["Channel", "J/ψ (fit)", "prompt [%]", "prompt J/ψ", f"per {TB} [10<sup>−5</sup>]"], crow, widths=[25, 17, 23, 16, 19], num=(1, 2, 3, 4), hl=(3,))
        + ul([f"<b>Sample:</b> {SIG}, |z<sub>0</sub>| &lt; 0.1 cm, {NB1}; background bin by bin from the sidebands",
              f"<b>Three fits agree:</b> ℓ<sub>xy</sub> fit {N_PR:.0f} ({SG['nominal']:.1f}σ), mass × ℓ<sub>xy</sub> {S2N['N_P']:.0f} ({T2['signif_prompt_notb']:.1f}σ), unbinned (sPlot) {SP_NP:.0f} ({SPF['signif_prompt']:.1f}σ)"], "b s")
        + '</div><div class="col">'
        + table(["Variation", "prompt [%]", "Δ"], [["nominal", f"{100 * FP:.1f}", ""]]
                + [[lab, f"{100 * V[k]['fp_notb']:.1f}", f"{mn(100 * (V[k]['fp_notb'] - FP), 1)}"] for k, lab in
                   (("tail", "resolution: 20% tail of 3σ"), ("coarse", "9 coarse bins"), ("exp", "b decays: own exponential"))],
                widths=[60, 22, 18], num=(1, 2), hl=(), warn=(3,))
        + ul([f"<b>b-decay template</b> (the b-matched J/ψ shape) is the main systematic: an own exponential moves f by {mn(100 * D_EXP, 1)} points ({SG['exp']:.1f}σ)",
              f"<b>Without e + jets:</b> {SG['mu_emu']:.1f}σ"], "b s")
        + "</div></div>", t="fit",
        notes="fit_lxy.py: b-matched N [f R + (1 - f) (E_lambda * R)] and the signal N [f_p R + (1 - f_p) (the b-matched shape)] fitted together, sigma shared; per l_xy bin "
              "the window counts Poisson(s + b), the sideband counts Poisson(2 b), b profiled. Statistical errors: MINOS.")

# --- sPlot
C2 = None
content("splot", "sPlot · prompt J/ψ", f"Unbinned fit: {SP_NP:.0f} ± {SP_NPE:.0f} prompt J/ψ {SIG}, {SPF['signif_prompt']:.1f}σ",
        f'<div class="row" style="gap:44px"><div class="fig">{F("splot_pt", "unit area per GeV", 0.7)}</div><div class="fig">{F("splot_proj_l", "signal, 2.6–3.6 GeV", 0.7)}</div></div>'
        + ul([f"<b>Unbinned m × ℓ<sub>xy</sub> fit</b> of the {SPF['n']} candidates {SIG}, per-candidate ℓ<sub>xy</sub> error; its sWeights give the prompt J/ψ distributions of the next slides",
              f"<b>The vetoed {TB}-jet J/ψ</b>, fitted the same way: {ST_NP:.0f} ± {ST_NPE:.0f} prompt ({SPT['signif_prompt']:.1f}σ), {SPT['N'][1]:.0f} ± {SPT['N_err'][1]:.0f} b decays",
              f"<b>Soft:</b> mean p<sub>T</sub> {PT_SIG:.1f} GeV, as the prompt pileup J/ψ (inclusive J/ψ from another collision, {PT_PU:.1f} GeV); the vetoed {TB}-jet J/ψ {PT_TTJ:.1f} GeV"], "b s"),
        t="fit",
        notes=f"splot_prompt.py (SIGNAL=assoc) on the candidate table with the veto columns: opposite sign, N_b >= 1, |z0| < 0.1 cm, 2.6-3.6 GeV, -0.5 < l_xy < 1.5 mm. "
              f"Shapes from the binned fit, the continuum refitted, then the yields alone for the sWeights. Resolution: S times the l_xy error per candidate, "
              f"S = {SPF['params']['S']:.2f}. Prompt yield {SP_NP:.0f} +{SPF['NP_minos'][1]:.0f} -{-SPF['NP_minos'][0]:.0f}; b decays {SPF['N'][1]:.0f}, continuum {SPF['N'][2]:.0f}. "
              f"The significance has the resolution and the b-decay shape fixed. The hard prompt J/psi of the 6 Nov deck were the ones in W-decay jets.")

# --- pT balance
BP = {s_: BAL["profile"][f"{s_}_excl_all"] for s_ in ("prompt", "pileup", "mix", "sps", "dps")}
BP["pileup_met"], BP["pileup"] = BP["pileup"], BAL["profile"]["pileup_exclnoj_all"]
BFP, BFD = BAL["fit"]["prompt_ppar"], BAL["fit"]["prompt_dphi"]
EP = BAL["emb_pass"]
EP_PASS = ", ".join(f"{c} {100 * EP[c]['sps']:.0f}% / {100 * EP[c]['dps']:.0f}%" for c in EP)  # emulated events passing the selection


def bpp(s_, d=0):
    m_, e_ = BP[s_]["ppar_all_pt"]
    return f"{mn(m_, d)} ± {e_:.{d}f}"


def bfit(F_):
    return f"{F_['f_same']:.2f} <sup>{F_['err'][1]:+.2f}</sup><sub>{F_['err'][0]:+.2f}</sub>".replace("-", "−")


# --- the CMS ttbar simulation
CS, CT, CI = MCC["closure"]["signal"], MCC["closure"]["ttj"], MCC["closure"]["inb"]
TRS = MCC["truth"]
SHORT = {c: TRS[f"{c}_all_prompt_region"]["b decay"] / TRS[f"{c}_all_all"]["b decay"] for c in ("ttj", "inb")}
TF_FREE = MTF["fits"]["ttj_ljets_w1"]
content("sim-closure", "CMS simulation · closure", f"The fit closes on the CMS {TB} simulation: {CS['fit_prompt']:.0f} ± {CS['fit_prompt_err']:.0f} prompt J/ψ found outside the {TB} jets for {CS['truth_prompt']:.0f} true; b decays fake {mn(CS['fake_prompt_in_data'])} ± {CS['fake_prompt_in_data_err']:.0f} of the {SP_NP:.0f}",
        f'<div class="row" style="gap:44px"><div class="fig">{F("mc_closure_signal", "the simulation as pseudo-data, MC truth shaded", 0.7)}</div>'
        f'<div class="fig">{F("mc_lxy_signal", "data against the simulation, 310 fb⁻¹", 0.7)}</div></div>'
        + ul([f"<b>CMS simulation:</b> official Summer24 {TB}, POWHEG + Pythia 8 CP5, full detector simulation, the code of the data pass, σ<sub>{TB}</sub> × 310 fb<sup>−1</sup>; no onia in the shower: b decays, plus a few prompt J/ψ from hadronisation",
              f"<b>Closure</b> (the data chain on it, b-decay shape from its b jets): {1000 * CS['fake_prompt_per_bdecay']:.0f} ± {1000 * CS['fake_prompt_per_bdecay_err']:.0f} fake prompt per 1000 b decays; data / simulation {RS['ratio_bdecay']:.2f} for the b decays, the data's prompt peak sits where the simulation has {RS['mc_prompt'][0]:.0f}"], "b s"),
        t="cms",
        notes="refresh_mc.sh: hadw/out_mc5 (cluster 897589, 524 jobs, TTtoLNu2Q + TTto2L2Nu Summer24 NanoAODv15, 2024 conditions), build_ttJ.py twice (by cross section; as pseudo-data, "
              "TTJ_AS_DATA=1), dump_cands.py, assoc_veto.py, life_from_cands.py, fit_lxy.py, fit_jpsi.py, splot_prompt.py on the pseudo-data (mc/), mc_compare.py. MC truth: both muons "
              f"from one J/psi; b decay when a b hadron is among its ancestors. Truth counts are window - sidebands / 2 of the unweighted pseudo-data. In its b jets the fit finds "
              f"{CI['fit_prompt']:.0f} prompt for {CI['truth_prompt']:.0f} true. The fake rate is scaled with the data b-decay yield of the same fit ({SPF['N'][1]:.0f}).")

content("sim-ttj", "CMS simulation · vetoed jets", f"In the vetoed jets the b-decay J/ψ are short-lived and pass for prompt; with their own ℓ<sub>xy</sub> shape {ST_C:.0f} ± {ST_CE:.0f} of the {ST_NP:.0f} vetoed prompt J/ψ remain",
        f'<div class="row" style="gap:44px"><div class="fig">{F("mc_bshape", "MC truth, unit area per mm", 0.7)}</div>'
        f'<div class="fig">{F("mc_tfit_ttj", "data, window − sidebands, per mm", 0.7)}</div></div>'
        + ul([f"<b>Why:</b> a b-decay J/ψ lands in a non-b {TB} jet when the b tag failed, i.e. when the b hadron flew little: {100 * SHORT['ttj']:.0f}% of them at |ℓ<sub>xy</sub>| &lt; 0.06 mm ({100 * SHORT['inb']:.0f}% in b jets); in the simulation the fit calls {CT['fit_prompt']:.0f} ± {CT['fit_prompt_err']:.0f} of them prompt, for {CT['truth_prompt']:.0f} true",
              f"<b>Refit with this shape:</b> {ST_C:.0f} ± {ST_CE:.0f} prompt ({TF_FREE['N_p']:.0f} ± {TF_FREE['N_p_err']:.0f} with the b decays free); it normalises the gen-level Pythia + LETO (×{GX['scaled']['nominal_bcorr']['scale']:.2f} instead of {GX['scaled']['nominal']['scale']:.2f}) and the leak ({RES_LO:.0f}–{RES_HI:.0f}); the signal fit is unaffected (closure)"], "b s"),
        t="cms",
        notes="mc_template_fit.py: binned fit of the window - sidebands l_xy, chi2 with the data and template errors, prompt template = MC truth prompt J/psi of all categories "
              f"(10% wider / narrower: {MTF['fits']['ttj_ljets_w1.1']['N_p']:.0f} / {MTF['fits']['ttj_ljets_w0.9']['N_p']:.0f}), b-decay template = MC truth b decays of the vetoed "
              f"category. b decays constrained (20% Gaussian) to the simulation x the mean data / simulation of the b-jet and signal categories ({TTJ_RF['scale_mean']:.2f}): "
              f"{TTJ_RF['nb_pred']:.0f} expected, {TTJ_RF['N_b']:.0f} +- {TTJ_RF['N_b_err']:.0f} fitted; free: {TF_FREE['N_b']:.0f} +- {TF_FREE['N_b_err']:.0f}. "
              f"Closure of the binned fit on the simulation: {MTF['closure']['ttj_ljets']['N_p']:.0f} +- {MTF['closure']['ttj_ljets']['N_p_err']:.0f} for {MTF['closure']['ttj_ljets']['truth_prompt']:.0f} true.")

# --- the b-decay template: pT or the b tag?
if "bshape_pt" in MCC:
    BSP = MCC["bshape_pt"]
    content("sim-bpt", "CMS simulation · b-decay template", f"The b-jet template misses the signal's short b decays because of the b tag, not pT (reweighting to the signal's pT: χ<sup>2</sup> {BSP['inb']['chi2_vs_signal']:.0f} → {BSP['inb_rew']['chi2_vs_signal']:.0f})",
            f'<div class="row" style="gap:44px"><div class="fig">{F("mc_bshape_pt", "MC truth b decays, per p<sub>T</sub> bin", 0.68)}</div>'
            f'<div class="fig">{F("mc_bshape_rew", "MC truth b decays, unit area per mm", 0.68)}</div></div>'
            + ul([f"<b>p<sub>T</sub></b> ({BSP['inb']['median_pt']:.0f} vs {BSP['signal']['median_pt']:.0f} GeV median): the pseudo-proper ℓ<sub>xy</sub> keeps the tail, and the ℓ<sub>xy</sub> error is the same at equal p<sub>T</sub> (per candidate in the unbinned fit)",
                  "<b>The b tag:</b> at equal p<sub>T</sub>, more short b decays outside the t̅t jets than in tagged b jets (4–6× in the untagged vetoed jets): hence a b-decay shape per category from the simulation (next slide)"], "b s"),
            t="cms",
            notes="mc_compare.py bshape_pt: MC truth b decays in the J/psi window per category and pT(J/psi) bin: share at |l_xy| < 0.06 mm, mean l_xy - 0.1 mm beyond 0.1 mm, "
                  "median l_xy error; the b-jet shape reweighted in pT(J/psi) to the signal's b decays against the signal's own (chi2 with the signal's statistical errors).")

# --- the b-decay template per category (btemplate_mc.py; the fits with TTJ_BTEMPLATE -> btmpl/)
if os.path.exists(os.path.join(D.HERE, "btmpl", "splot.json")) and os.path.exists(os.path.join(D.HERE, "btemplate.json")):
    BTM = J("btemplate.json")
    BS = J("btmpl/splot.json")
    BL, BJ2 = J("btmpl/fit_lxy.json"), J("btmpl/fit_jpsi.json")
    BN, BNM = BS["fits"]["notb"], BS["fits"]["notb"]["NP_minos"]
    BT_T, BT_TM = BS["fits"]["ttj"]["N"][0], BS["fits"]["ttj"]["NP_minos"]
    BC = BTM["cats"]
    B_LXY = BL["params"]["fp_notb"] * BL["params"]["N_notb"]
    content("sim-btmpl", "CMS simulation · b-decay template",
            f"Per-category b-decay shape (outside the {TB} jets: no {100 * FPI_D:.1f}% prompt-like share as in the data b jets): "
            f"{BN['N'][0]:.0f} <sup>+{BNM[1]:.0f}</sup><sub>−{-BNM[0]:.0f}</sub> prompt J/ψ ({BN['signif_prompt']:.1f}σ), was {SP_NP:.0f} ± {SP_NPE:.0f}",
            f'<div class="row" style="gap:44px"><div class="fig">{F("mc_btmpl_shapes", "MC truth b decays, unit area per mm", 0.68)}</div>'
            f'<div class="fig">{F("mc_btmpl_data", "data, both templates", 0.68)}</div></div>'
            + ul([f"<b>Template:</b> the fit's b-decay model (f G + (1 − f) e<sup>−ℓ/λ</sup> ⊗ G, G per candidate) on the MC truth b decays per category: "
                  f"f {BC['inb']['f']:.3f} (b jets), {BC['signal']['f']:.3f} ± {BC['signal']['f_err']:.3f} (outside), {BC['ttj']['f']:.2f} ± {BC['ttj']['f_err']:.2f} (vetoed); λ × data / sim. of the b jets",
                  f"<b>Data:</b> every fit up by about 20 (ℓ<sub>xy</sub> {B_LXY:.0f}, 2D {BJ2['lt2d']['samples']['notb']['N_P']:.0f}); vetoed jets {BT_T:.0f} ± {0.5 * (BT_TM[1] - BT_TM[0]):.0f} "
                  f"(simulation-template refit {ST_C:.0f} ± {ST_CE:.0f}); which template is right: the toys"], "b s"),
            t="cms",
            notes="btemplate_mc.py: unbinned fit of the MC truth b decays (opposite sign, N_b >= 1, |z0| < 0.1 cm, J/psi window, -0.5 < l_xy < 1.5 mm) per category with the model "
                  "of splot_prompt.py, S free. The fits (fit_lxy.py, fit_jpsi.py, splot_prompt.py) with TTJ_BTEMPLATE=btemplate.json use f and lambda of the category, lambda "
                  "scaled by lambda(b jets, data) / lambda(b jets, simulation) of the same fit; the b-jet category keeps its free prompt-like share. Outputs in btmpl/; "
                  "the nominal outputs are unchanged.")

# --- the toy closure of the fits (toy_closure.py -> toy_summary.py): b-jet template (toy_closure.json), per-category template
# (toy_closure_btmpl.json); the toys drawn per mass region (the earlier draws per lifetime class biased the b decays)
TOYS = {v: J(fn) for v, fn in (("bjet", "toy_closure.json"), ("cat", "toy_closure_btmpl.json")) if os.path.exists(os.path.join(D.HERE, fn))}
TFN = (("lxy", "ℓ<sub>xy</sub>"), ("fit2d", "binned 2D"), ("unbinned", "unbinned"), ("unbinned_global", "unbinned, 1 res."))
if len(TOYS) == 2:
    TCP = {(v, p_["scenario"], p_["inject"]): p_ for v, T_ in TOYS.items() for p_ in T_["points"]}

    def sgn(x):
        return "0" if round(x) == 0 else f"{x:+.0f}".replace("-", "−")

    def tb(v, k, inj=100):
        a, b = TCP.get((v, "mc", inj)), TCP.get((v, "datalike", inj))
        return f"{sgn(a[k]['bias'])} / {sgn(b[k]['bias'])}" if a and b else "–"

    trow = [[nm, tb("bjet", k), tb("cat", k)] for k, nm in TFN]
    q100 = TCP.get(("bjet", "mc", 100))
    U = {(v, sc): TCP.get((v, sc, 100)) for v in ("bjet", "cat") for sc in ("mc", "datalike")}
    def ub(v, sc, k="unbinned"):
        return sgn(U[(v, sc)][k]['bias']) if U[(v, sc)] else "–"

    ttl = (f"Toys, unbinned fit at {q100['unbinned']['truth']:.0f} true prompt J/ψ: per-category template {ub('cat', 'mc')} / {ub('cat', 'datalike')}, "
           f"b-jet template {ub('bjet', 'mc')} / {ub('bjet', 'datalike')} (b jets as simulated / with prompt J/ψ)") if q100 else "Toy closure"
    content("sim-toys", "CMS simulation · toy closure", ttl,
            f'<div class="row" style="gap:36px"><div class="fig">{F("mc_toys_tmplcmp_unbinned", "unbinned fit", 0.62)}</div>'
            f'<div class="fig">{F("mc_toys_tmplcmp_fit2d", "binned 2D fit", 0.62)}</div><div class="col">'
            + table(["fit − truth (± 2–3)", "b-jet template", "per category"], trow, widths=[36, 32, 32], num=(1, 2), hl=(2,))
            + '<div class="cap" style="margin-top:8px">b jets as simulated / with prompt J/ψ</div>'
            + "</div></div>"
            + ul([f"<b>Toys:</b> the data's candidate count per channel, category and sign drawn from the CMS {TB} simulation, 0–150 truth-prompt J/ψ added outside the {TB} jets, "
                  f"the data chain, {q100['n'] if q100 else ''} per point (table: 100 injected); data-like: prompt J/ψ in the b jets, 5% of their b decays. The earlier toys (drawn per ℓ<sub>xy</sub> class) are withdrawn",
                  f"<b>Reading:</b> the b-jet template copies the b jets' prompt-like share onto the signal's b decays, so real prompt J/ψ there cost ≈ {-U[('bjet', 'datalike')]['unbinned']['bias'] if U[('bjet', 'datalike')] else 0:.0f}; "
                  f"the per-category template closes in both cases (one resolution: {ub('cat', 'mc', 'unbinned_global')} / {ub('cat', 'datalike', 'unbinned_global')}); the ℓ<sub>xy</sub> fit is low by 20–45. "
                  "Proposed nominal: the unbinned fit with the per-category template"], "b s"),
            t="cms",
            notes="toy_closure.py (hadw/toys_closure_v2, hadw/toys_closure_v2_btmpl with TTJ_BTEMPLATE), toy_summary.py: per toy the candidates of each stratum (channel, category, "
                  "sign) drawn without replacement from the simulation in the numbers of the data; n_inj ~ Poisson truth-prompt J/psi of the simulation (all categories, "
                  "primary vertex) added outside the ttbar jets. Truth = injected + the simulation's own prompt J/psi among the drawn signal candidates (full mass range; window - "
                  "sidebands for the l_xy fit). Pull = (fit - truth) / the fit's error. The withdrawn toys (hadw/toys_closure) were drawn per lifetime class; draws per mass region "
                  "(window, sidebands) also fail: the data has more continuum per J/psi, so the sidebands over-subtract.")

EVR = {ch: MCC["events"][ch]["ratio"] for ch in ("mujets", "ejets", "emu")}
LAM_D, LAM_M = FJ["lt2d"]["params"]["lam_inb"], MFJ["lt2d"]["params"]["lam_inb"]
EVA = {ch: MCC["events"][ch]["ratio_all"] for ch in ("mujets", "ejets", "emu")}
content("sim-rates", "CMS simulation · rates", f"{SIMN_T[0].upper() + SIMN_T[1:]}: b-decay J/ψ data / sim. {RI['data_nonprompt'][0] / simp(RI, 'b decay'):.2f}, {RS['data_nonprompt'][0] / simp(RS, 'b decay'):.2f} (b jets, outside); prompt J/ψ {RS['data_prompt'][0] / simp(RS, 'prompt'):.0f}× below data outside the {TB} jets",
        f'<div class="row" style="gap:44px"><div class="fig">{F("mc_rates", "MC truth against sWeights", 0.7)}</div>'
        f'<div class="fig">{F("mc_lxy_inb", "b jets, window − sidebands, per mm", 0.7)}</div></div>'
        + ul([(f"<b>Selected events, data / ({TB} + backgrounds):</b> {EVA['mujets']:.2f}, {EVA['ejets']:.2f}, {EVA['emu']:.2f} (μ + jets, e + jets, e μ; {TB} alone {EVR['mujets']:.2f}, {EVR['ejets']:.2f}, {EVR['emu']:.2f}); "
               f"b-decay J/ψ data / simulation {RI['data_nonprompt'][0] / simp(RI, 'b decay'):.2f} (b jets), {RS['data_nonprompt'][0] / simp(RS, 'b decay'):.2f} (outside), λ {LAM_D:.2f} vs {LAM_M:.2f} mm" if HAS_BKG else
               f"<b>Selected events, data / simulation:</b> {EVR['mujets']:.2f}, {EVR['ejets']:.2f}, {EVR['emu']:.2f} (μ + jets, e + jets, e μ): about {100 * (1 - 1 / (0.5 * (EVR['mujets'] + EVR['ejets']))):.0f}% of ℓ + jets is not {TB}; per luminosity the b decays agree to 10%, λ {LAM_D:.2f} vs {LAM_M:.2f} mm"),
              f"<b>Prompt J/ψ in the simulation</b> come from hadronisation only (low-mass c c̅ strings): {RS['mc_prompt'][0]:.0f} outside the {TB} jets, {RI['mc_prompt'][0]:.0f} in b jets; its fit finds {CI['fit_prompt']:.0f} in its b jets, so the {RI['data_prompt'][0]:.0f} in data b jets are not b decays"], "b s"),
        t="cms",
        notes="Normalisation: cross section / sum of generator weights x event weight (pileup, lepton and b-tag scale factors) x 310 fb^-1, the 2024 simulation for every year. "
              "Data: the sWeights of the unbinned fit; the vetoed category's data b-decay yield is low because the fit counts its short b decays as prompt (previous slide). "
              + ("Backgrounds (hadw/out_bkg, cluster 897661): single top (tW, t-, s-channel), W + jets (MLM, 1-4 jets), ttW, ttZ, WW / WZ / ZZ, QCD mu-enriched; no Drell-Yan "
                 "(only binned slices without cross sections) and no QCD EM-enriched." if HAS_BKG else "Non-ttbar backgrounds are being processed (cluster 897661)."))

# --- per channel
CHS3 = (("mujets", "μ + jets"), ("ejets", "e + jets"), ("emu", "e μ"))
_me = _np.asarray(MCC["mass_edges"], float)
_mc = 0.5 * (_me[1:] + _me[:-1])
_sbm = ((_mc > 2.6) & (_mc < 2.9)) | ((_mc > 3.3) & (_mc < 3.6))


def chan(ch):
    """per-channel numbers of the signal category: data, the simulation (310 fb^-1), the l_xy fit of the channel"""
    r, M_, FC = MCC["rates"][f"signal_{ch}"], MCC["mass"][f"signal_{ch}"], FIT["channels"][ch]
    mct = sum(_np.asarray(M_[f"mc_{t}"][0], float) for t in ("prompt", "b decay", "no J/psi")) + (_np.asarray(M_["bkg"][0], float) if HAS_BKG and "bkg" in M_ else 0.0)
    return {"ntt": R["channels"][ch]["yields"]["data_os_all_ge1b"]["n_tt"], "ev": MCC["events"][ch]["ratio_all" if HAS_BKG else "ratio"], "sp": r["data_prompt"], "sb": r["data_nonprompt"],
            "mp": simp(r, "prompt"), "mb": simp(r, "b decay"), "rb": r["data_nonprompt"][0] / simp(r, "b decay"), "lp": FC["N"] * FC["fp"], "fp": FC["fp"], "N": FC["N"],
            "cont": float(_np.asarray(M_["data"][0], float)[_sbm].sum() / mct[_sbm].sum())}


CHV = {ch: chan(ch) for ch, _ in CHS3}
crow3 = [[nm, f"{v['ntt'] / 1e6:.2f}M", f"{v['ev']:.2f}", f"{v['cont']:.2f}", f"{v['rb']:.2f}", f"{v['sp'][0]:.0f} ± {v['sp'][1]:.0f}", f"{v['lp']:.0f}", f"{v['mp']:.1f}",
          f"{v['sp'][0] / v['mp']:.1f}"] for (ch, nm), v in zip(CHS3, CHV.values())]
content("chan-summary", "Channels · summary", f"Against {SIMN_T}: b decays {min(v['rb'] for v in CHV.values()):.1f}–{max(v['rb'] for v in CHV.values()):.1f}, prompt J/ψ {CHV['mujets']['sp'][0] / CHV['mujets']['mp']:.0f}×, {CHV['ejets']['sp'][0] / CHV['ejets']['mp']:.0f}×, {CHV['emu']['sp'][0] / CHV['emu']['mp']:.0f}× above (μ + jets, e + jets, e μ)",
        '<div class="row" style="gap:28px"><div class="fig">' + F("chan_rates", "MC truth against sWeights", 0.6) + '</div><div class="fig">' + F("chan_mt", "unit area per GeV", 0.6) + '</div><div class="col">'
        + ul([(f"<b>With the backgrounds:</b> selected events data / simulation {EVA['mujets']:.2f}, {EVA['ejets']:.2f}, {EVA['emu']:.2f}; the dimuon continuum (sidebands) "
               f"{min(CHV['mujets']['cont'], CHV['ejets']['cont']):.1f}–{max(CHV['mujets']['cont'], CHV['ejets']['cont']):.1f}× the simulation in ℓ + jets (no Drell-Yan, no QCD EM-enriched simulated)" if HAS_BKG else
               f"<b>Backgrounds missing:</b> ℓ + jets has about 20% more selected events than the {TB} simulation and {min(CHV['mujets']['cont'], CHV['ejets']['cont']):.1f}–{max(CHV['mujets']['cont'], CHV['ejets']['cont']):.1f}× the simulated dimuon continuum (sidebands): non-{TB} events (W + jets, QCD)"),
              f"<b>e + jets</b>: {1e6 * CHV['ejets']['sp'][0] / CHV['ejets']['ntt']:.1f} ± {1e6 * CHV['ejets']['sp'][1] / CHV['ejets']['ntt']:.1f} prompt J/ψ per 10<sup>6</sup> {TB} against {1e6 * CHV['mujets']['sp'][0] / CHV['mujets']['ntt']:.1f} ± {1e6 * CHV['mujets']['sp'][1] / CHV['mujets']['ntt']:.1f} in μ + jets; its prompt-J/ψ events are not at the low m<sub>T</sub> of QCD with a fake electron (right)"], "b s")
        + "</div></div>"
        + table(["Channel", f"{TB} events", "events / sim.", "continuum / sim.", "b decays / sim.", "prompt, sPlot", "prompt, ℓ<sub>xy</sub> fit", "sim. prompt", "data / sim."], crow3,
                widths=[12, 10, 11, 12, 12, 12, 11, 10, 10], num=(1, 2, 3, 4, 5, 6, 7, 8), hl=()), t="cms",
        notes="events / sim.: selected events (opposite-sign J/psi not required) data / simulation (ttbar + the non-ttbar backgrounds when processed; the J/psi columns without QCD mu-enriched, too few simulated events). continuum / sim.: J/psi candidates outside the ttbar jets in the mass sidebands "
              "(2.6-2.9, 3.3-3.6 GeV), data / simulation. b decays / sim.: b-decay sWeights / MC truth. prompt, sPlot: the prompt sWeights of the combined unbinned fit summed per "
              "channel; prompt, l_xy: N x f_prompt of the per-channel l_xy fit (fit_lxy.py, N and f free, the shapes of the combined fit).")
for ch, nm in CHS3:
    v = CHV[ch]
    content(f"chan-{ch}", f"Channels · {nm}", f"{nm}: {v['sp'][0]:.0f} ± {v['sp'][1]:.0f} prompt J/ψ outside the {TB} jets (sPlot), {v['lp']:.0f} from the channel's ℓ<sub>xy</sub> fit; the CMS simulation has {v['mp']:.1f}",
            '<div class="row" style="gap:28px">'
            + f'<div class="fig">{F(f"chan_mass_{ch}", "all ℓ_xy of the fit", 0.6)}</div>'
            + f'<div class="fig">{F(f"chan_lxyfit_{ch}", "window − sidebands, per mm", 0.6)}</div>'
            + f'<div class="fig">{F(f"chan_lxy_{ch}", "window − sidebands, per mm", 0.6)}</div></div>'
            + ul([f"<b>{v['ntt'] / 1e6:.2f}M {TB} events</b> ({v['ev']:.2f} of the simulation's); J/ψ continuum {v['cont']:.2f} × the simulation's, b decays {v['rb']:.2f} ×",
                  f"<b>ℓ<sub>xy</sub> fit:</b> f<sub>prompt</sub> = {100 * v['fp']:.0f}% of {v['N']:.0f} J/ψ; the simulation's prompt J/ψ (green) come from hadronisation only, the data's peak at ℓ<sub>xy</sub> ≈ 0 is {v['sp'][0] / v['mp']:.0f}× larger"], "b s"),
            t="cms")

if len(GV["gen"]) > 1 and "ratio" in GV:
    GR = GV["ratio"]
    gens = [g for g in GV["gen"] if f"{g}|signal" in GR]

    def rfmt(k):
        r = GR.get(k)
        if not r:
            return "–"
        if r.get("upper") is not None:
            return f"&lt; {r['upper']:.2g} (0 / {r['n_b']})"
        return f"{r['r']:.2g} ± {r['err']:.1g} <span style=\"color:#6B7482\">({r['n_p']}/{r['n_b']})</span>"

    vrow = [[{"Pythia 8 CP5": "POWHEG + Pythia 8 CP5"}.get(g, g), rfmt(f"{g}|signal"), rfmt(f"{g}|inb")] for g in gens]
    vrow.append(["<b>Data (sPlot)</b>", f"<b>{GR['data|signal']['r']:.2f} ± {GR['data|signal']['err']:.2f}</b>", f"<b>{GR['data|inb']['r']:.3f} ± {GR['data|inb']['err']:.3f}</b>"])
    CR1 = GR.get("CP5 CR1|signal")

    def SHB(g, cat):
        """b-decay J/psi per luminosity against POWHEG + Pythia 8 of a one-lepton-charge Sherpa sample (x 2): 'NN%' fewer"""
        G_ = GV["gen"]
        if g not in G_:
            return "–"
        return f"{100 * (1 - 2 * G_[g][f'{cat}_bdecay'][0] / G_['Pythia 8 CP5'][f'{cat}_bdecay'][0]):.0f}%"
    content("sim-gen", "CMS simulation · generators (preliminary)", f"Prompt per b-decay J/ψ outside the {TB} jets: data {GR['data|signal']['r']:.2f}, POWHEG + Pythia 8 CP5 {GR['Pythia 8 CP5|signal']['r']:.3f}"
            + (f", CR1 {CR1['r']:.3f}" if CR1 and CR1.get('upper') is None else "") + "; Sherpa makes none",
            f'<div class="row" style="gap:36px"><div class="fig">{F("mcvar_ratio", "within each sample: normalisation-free", 0.66)}</div><div class="col">'
            + table(["CMS sample, ℓ + jets", f"prompt / b decay, outside {TB} jets", "in b jets"], vrow, widths=[31, 37, 32], num=(1, 2), hl=(len(vrow) - 1,))
            + "</div></div>"
            + ul([f"<b>Sherpa</b> selects as many {TB} events per luminosity as POWHEG (±1%) but makes {SHB('Sherpa', 'inb')} fewer b-decay J/ψ in b jets ({SHB('Sherpa', 'signal')} outside; AHADIC {SHB('Sherpa AHADIC', 'inb')}, {SHB('Sherpa AHADIC', 'signal')}) and no prompt J/ψ; CR1 doubles the hadronisation's prompt J/ψ; the data stay {GR['data|signal']['r'] / GR['Pythia 8 CP5|signal']['r']:.0f}× above CP5 outside the {TB} jets, {GR['data|inb']['r'] / GR['Pythia 8 CP5|inb']['r']:.0f}× in b jets"], "b s"),
            t="cms",
            notes="gen_variants.py on the l + jets samples (hadw/out_alt, cluster 897658, all jobs in): MC truth prompt J/psi (no b-hadron ancestor) per MC truth "
                  "b-decay J/psi in the selection of the fit, weighted; no prompt J/psi in a sample: 90% CL upper limit 2.3 / (raw b-decay count). Data: prompt / b-decay sWeights, l + jets. "
                  "Sherpa: the MEPS@NLO sample over-generates high jet multiplicities with lower weights (mean weight 2.8x the average at 2 jets, 0.5x at 6 jets; 17% negative), "
                  "so its selected candidates carry about half the mean weight; the weighted selection matches: selected l + jets events per pb within 1% of POWHEG + Pythia "
                  "(12 Sherpa and 6 POWHEG jobs, nbjet histograms). Its b-decay J/psi per event are about 30% lower in b jets and 15% outside the ttbar jets (Sherpa's own b-hadron decay tables and fragmentation against Pythia 8's; CMS Pythia samples do not use EvtGen); the ratios within a sample are unaffected.")

BTTJ = BAL["profile"]["ttj_excl_all"]["ppar_all_pt"]
content("balance", "DPS? · p<sub>T</sub> balance", f"The p<sub>T</sub> balance cannot tell a second scattering apart for these soft J/ψ: data {bpp('prompt')} GeV, recoiling {mn(BP['sps']['ppar_all_pt'][0])}, 2nd scattering {mn(BP['dps']['ppar_all_pt'][0])}",
        f'<div class="row" style="gap:44px"><div class="fig">{F("ttbal_dphi", f"{TB} without the J/ψ jet, unit area", 0.7)}</div>'
        f'<div class="fig">{F("ttbal_ppar", "unit area per GeV", 0.7)}</div></div>'
        + ul([f"<b>{TB}:</b> ℓ + p<sub>T</sub><sup>miss</sup> + the χ² fit jets (e μ: the two b jets); emulations: the J/ψ (and its jet) put into other {TB} events, reselected",
              f"<b>⟨p<sub>T</sub>({TB}) · p̂<sub>T</sub>(J/ψ)⟩ [GeV]:</b> prompt {bpp('prompt')}; recoiling {mn(BP['sps']['ppar_all_pt'][0])}, 2nd scattering {mn(BP['dps']['ppar_all_pt'][0])} (emulated); prompt pileup J/ψ {bpp('pileup')}; mixed events {bpp('mix')}",
              f"<b>Template fit</b> (recoiling share, not bounded to 0–1): {bfit(BFD)} from Δφ, {bfit(BFP)} from p<sub>T</sub>; a 2nd scattering alone at {BFD['z_vs_dps_only']:.1f}σ and {BFP['z_vs_dps_only']:.1f}σ: no separation",
              ], "b s"),
        t="fit",
        notes=f"The 6 Nov result (-57 +- 18 GeV, 3.2 sigma against a 2nd scattering) came from the J/psi in W-decay jets: left out of the ttbar, a top decay product balances it "
              f"trivially (here {BTTJ[0]:.0f} +- {BTTJ[1]:.0f} GeV for the vetoed ttbar-jet J/psi). ttbar_balance.py with the sWeights of the signal (N_prompt = {BAL['splot']['N_prompt']:.0f}). Emulation: the J/psi of each candidate (with its jet when it is in "
              f"one) in {BAL['settings']['nemb']} other selected events, recoiling (MET - pT(holder)) or with a balancing partner (2nd scattering); the new event must have "
              f"N_b >= 1 and the J/psi outside every ttbar jet of the new event, which {EP_PASS} "
              f"(recoiling / 2nd scattering) pass. Template fit: chi2 of the prompt histogram against N [f same + (1 - f) 2nd], {BFD['chi2_min']:.1f} / {BFD['ndf']} (Delta phi), "
              f"{BFP['chi2_min']:.1f} / {BFP['ndf']} (pT). The prompt pileup J/psi are shown with the J/psi taken out of the MET (their muons are in it, their recoil is not: "
              f"{mn(BP['pileup_met']['ppar_all_pt'][0])} GeV otherwise).")

content("balance-pt", "DPS? · p<sub>T</sub> balance", f"The {TB} p<sub>T</sub> along the J/ψ per p<sub>T</sub>(J/ψ) bin, and the rapidity difference",
        f'<div class="row" style="gap:44px"><div class="fig">{F("ttbal_ppar_pt", "sWeighted means")}</div>'
        f'<div class="fig">{F("ttbal_dy", f"full {TB} reconstruction, unit area")}</div></div>'
        + ul(["<b>Left:</b> per p<sub>T</sub>(J/ψ) bin; a recoiling J/ψ (or its jet) gives about −p<sub>T</sub> of it",
              f"<b>Right:</b> |Δy(J/ψ, {TB})| with the {TB} rapidity from the W mass (ℓ + jets) and the W and top masses (e μ)"], "b s"),
        t="fit", notes=f"pT(J/psi) bins {BAL['pt_edges']}. Errors: the spread of the sWeighted values.")

# --- DPS rate
drow = [[f"{float(k):g} mb", f"{v:.0f}", f"{100 * v / SP_NP:.0f}%"] for k, v in DPS["pred_prompt_notb_by_sigma_eff_mb"].items()]
content("dps-rate", "DPS · rate", f"A second scattering gives {DPS15:.0f} of the {SP_NP:.0f} prompt J/ψ for σ<sub>eff</sub> = 15 mb; all of them for σ<sub>eff</sub> ≈ {SE_ALL:.0f} mb",
        '<div class="row" style="gap:40px"><div class="col">'
        + table(["σ<sub>eff</sub>", "prompt J/ψ from DPS", f"of the {SP_NP:.0f} measured"], drow, widths=[30, 40, 30], num=(1, 2), hl=(1,))
        + '</div><div class="col">'
        + ul([f"<b>DPS:</b> P(J/ψ | {TB}) = r · σ<sub>vis</sub> / σ<sub>eff</sub>, r = J/ψ per pileup collision with a good vertex in the same selection and veto: {DPS['pileup_prompt_notb']:.0f} ± {DPS['pileup_prompt_notb_err']:.0f} prompt pileup J/ψ / ({DPS['n_tt'] / 1e6:.1f}M events × {DPS['mean_npu']:.1f} vertices) = {1e7 * DPS['r_per_pu_vertex']:.1f}·10<sup>−7</sup>",
              f"<b>σ<sub>vis</sub></b> = {DPS['eps_vtx']:.1f} × {DPS['sigma_inel_mb']:.0f} mb; σ<sub>eff</sub> 15–20 mb from W + 2 jets and γ + 3 jets, 2–10 mb from J/ψ pairs",
              f"<b>Open:</b> {DPS15:.0f} of {SP_NP:.0f} for 15 mb, all of them for {SE_ALL:.0f} mb; the p<sub>T</sub> balance does not separate them (previous slides) and the p<sub>T</sub> spectrum is that of inclusive J/ψ"], "b s")
        + "</div></div>", t="fit",
        notes="The pileup J/psi measure the inclusive J/psi rate per collision with the selection and the veto of the signal, so acceptance and efficiency cancel.")

# --- data against Pythia 8 + LETO
GY, GS, GM, GN = GX["yields"], GX["scaled"], GX["means"], GX["norm"]
GOR = ("top decay", "ISR")
PN = {c: sum(GY[f"pythia_nominal_{c}_{o}"][0] for o in GOR) for c in ("ttj", "signal")}
DLJ = GX["dps"]["n_lj"]["15"]
GD = {c: GY[f"data_{c}"] for c in ("ttj", "signal")}
RSIG = GD["signal"][0] / (PN["signal"] + DLJ)
SN, SZ = GS["nominal"], GS["z"]
SNC, SZC = GS["nominal_bcorr"], GS["z_bcorr"]  # Pythia scaled to the vetoed J/psi refitted with the simulated b-decay shape


def gm(k, d=1):
    return f"{GM[k][0]:.{d}f}"


content("gen-rates", "Gen study · rates", f"Private gen-level Pythia 8 + LETO, default LDMEs: {PN['ttj']:.0f} vetoed J/ψ and {PN['signal']:.0f} + {DLJ:.0f} (2nd scattering) outside the {TB} jets; data {GD['ttj'][0]:.0f} and {GD['signal'][0]:.0f}",
        f'<div class="row" style="gap:44px"><div class="fig">{F("gencmp_yields", "ℓ + jets, prompt sWeights", 0.7)}</div>'
        f'<div class="fig">{F("gencmp_pt_ttj", "vetoed J/ψ, Pythia scaled to them, per GeV", 0.7)}</div></div>'
        + ul([f"<b>Normalisation:</b> the b-decay J/ψ of the same selection per p<sub>T</sub>(J/ψ) bin, data / Pythia = {GN['k_all']:.2f}, so luminosity, {TB} selection and μμ efficiency cancel; the LETO rate divided by its LDME factor 20",
              f"<b>Data / gen. Pythia:</b> {SN['ratio_data_to_pythia']['ttj'][0]:.2f} ± {SN['ratio_data_to_pythia']['ttj'][1]:.2f} for the vetoed J/ψ ({SNC['ratio_data_to_pythia']['ttj'][0]:.2f} ± {SNC['ratio_data_to_pythia']['ttj'][1]:.2f} without their short b decays), {RSIG:.2f} ± {GD['signal'][1] / (PN['signal'] + DLJ):.2f} outside with the 2nd scattering: the default LDMEs about 1.5× high"], "b s"),
        t="gen",
        notes=f"LETO puts the J/psi too hard in its jet (z {GZ['mean_z_pythia']:.2f} vs {GZ['mean_z_data']:.2f}): vetoed <pT> {gm('pythia_nominal_ttj_pt', 0)} vs {gm('data_ttj_pt', 0)} GeV (backup). gen_compare.py: Pythia {GEN['n_generated'] / 1e6:.1f}M ttbar l + jets (the gen_assoc.py table, now with the b-decay J/psi), data l + jets only (no e mu Pythia "
              f"sample; e mu has {SP_NP - GD['signal'][0]:.0f} of the {SP_NP:.0f}). Normalisation k(pT) = data / Pythia b-decay J/psi in pT(J/psi) bins {GN['pt_edges']}: "
              f"{', '.join(f'{x:.2f}' for x in GN['k'])}. Each prompt Pythia J/psi has weight k(pT) / 20 (OniaShower:ldmeFac = 20; the onia rate is linear in the LDMEs). "
              "The z variant reweights every Pythia J/psi inside a jet to the z of the vetoed J/psi in data (gen_leak_z.py, z >= 0.6 in one bin). "
              f"2nd scattering: the prompt pileup J/psi rate of the DPS slide at sigma_eff = 15 mb, l + jets share {GX['dps']['f_lj']:.2f}.")

content("gen-shapes", "Gen study · signal", f"Gen-level Pythia + LETO scaled to the vetoed J/ψ: {SN['sps_signal']:.0f} ± {SN['sps_signal_err']:.0f} J/ψ outside the {TB} jets ({SZ['sps_signal']:.0f} with the data z); with the 2nd scattering {SN['sps_signal'] + DLJ:.0f} / {SZ['sps_signal'] + DLJ:.0f}, data {GD['signal'][0]:.0f} ± {GD['signal'][1]:.0f}",
        f'<div class="row" style="gap:44px"><div class="fig">{F("gencmp_pt", "per GeV", 0.7)}</div>'
        f'<div class="fig">{F("gencmp_drtt", "per unit ΔR", 0.7)}</div></div>'
        + ul([f"<b>⟨p<sub>T</sub>(J/ψ)⟩ [GeV]:</b> data {gm('data_signal_pt')}, Pythia {gm('pythia_nominal_signal_pt')} ({gm('pythia_z_signal_pt')} with the data z), 2nd scattering {gm('dps_pt')}: Pythia's hard tail is LETO's J/ψ that carry their whole jet",
              f"<b>Top decays among them</b> (Pythia, scaled): {SN['top_decay_signal']:.0f} ± {SN['top_decay_signal_err']:.0f} ({SZ['top_decay_signal']:.0f} with the data z); {SNC['top_decay_signal']:.0f} / {SZC['top_decay_signal']:.0f} with the vetoed J/ψ without their short b decays (×{SNC['scale']:.2f}): residual estimate {RES_LO:.0f}–{RES_HI:.0f}"], "b s"),
        t="gen", notes=f"<Delta R(J/psi, nearest ttbar jet)>: data {gm('data_signal_drtt', 2)}, Pythia {gm('pythia_nominal_signal_drtt', 2)} / {gm('pythia_z_signal_drtt', 2)}, 2nd scattering {gm('dps_drtt', 2)}. "
                      "Stack: Pythia nominal times its scale to the vetoed J/psi of the data, plus the 2nd scattering (prompt pileup J/psi; for the ttbar quantities with the J/psi "
                      "out of the MET). Dashed: Pythia with the data z, scaled the same way, plus the 2nd scattering. Data: prompt sWeights.")

FITV = (("pt", "p<sub>T</sub>(J/ψ)"), ("drtt", f"ΔR(J/ψ, {TB} jet)"), ("dphi", f"|Δφ(J/ψ, {TB})|"), ("ppar", f"p<sub>T</sub>({TB}) · p̂<sub>T</sub>(J/ψ)"), ("dy", f"|Δy(J/ψ, {TB})|"))


def gfit(k):
    F_ = GX["fit"][k]
    return f"{F_['f_dps']:.2f} <sup>{F_['err'][1]:+.2f}</sup><sub>{F_['err'][0]:+.2f}</sub>".replace("-", "−")


def gz(v, which):
    return f"{GX['fit'][f'{v}_nominal'][which]:.1f} / {GX['fit'][f'{v}_z'][which]:.1f}σ"


frow = [[lab, gfit(f"{v}_nominal"), gfit(f"{v}_z"), gz(v, "z_vs_pythia_only"), gz(v, "z_vs_dps_only")] for v, lab in FITV]
content("gen-dps", "Gen study · 2nd scattering", f"What the gen-level Pythia + LETO leaves for a 2nd scattering: {SN['dps_left']:.0f} ± {SN['dps_left_err']:.0f} J/ψ ({SZ['dps_left']:.0f} ± {SZ['dps_left_err']:.0f} with the data z), σ<sub>eff</sub> ≈ {SN['sigma_eff_mb']:.0f} / {SZ['sigma_eff_mb']:.0f} mb",
        table(["Share of the 2nd scattering from", "Pythia", "Pythia, data z", "Pythia alone off by", "2nd scattering alone off by"], frow, widths=[28, 16, 16, 20, 20], num=(1, 2, 3, 4), hl=(4,))
        + ul([f"<b>Rate:</b> data − Pythia (scaled to the vetoed J/ψ) = {SN['dps_left']:.0f} ± {SN['dps_left_err']:.0f} or {SZ['dps_left']:.0f} ± {SZ['dps_left_err']:.0f}: σ<sub>eff</sub> &gt; {SN['sigma_eff_mb_95_lower']:.0f} / {SZ['sigma_eff_mb_95_lower']:.0f} mb (95% CL), compatible with 15 mb; "
              f"<b>with the vetoed J/ψ corrected for short b decays</b> (×{SNC['scale']:.2f}): {SNC['dps_left']:.0f} ± {SNC['dps_left_err']:.0f} / {SZC['dps_left']:.0f} ± {SZC['dps_left_err']:.0f}, σ<sub>eff</sub> ≈ {SNC['sigma_eff_mb']:.0f} / {SZC['sigma_eff_mb']:.0f} mb",
              f"<b>Shapes:</b> template fit data = N [f · 2nd scattering + (1 − f) · Pythia]; p<sub>T</sub> rejects Pythia alone only with LETO's own z; the J/ψ are farther from the {TB} in rapidity than Pythia's (⟨|Δy|⟩ {gm('data_signal_dy', 2)} vs {gm('pythia_nominal_signal_dy', 2)}, 2nd scattering {gm('dps_dy', 2)}; backup)"], "b s"),
        t="gen",
        notes="Template fit as the pT-balance one: chi2 of the sWeighted data against N [f DPS + (1 - f) Pythia], both at unit area, data sWeight errors and template "
              "statistics; f not bounded to 0-1; the significance columns give Pythia nominal / Pythia with the data z. Pythia: both origins (top decay and ttbar "
              "production). z = pT(J/psi)/pT(jet) is not fitted: the muons of a pileup J/psi are taken out of the jets by PUPPI, so the 2nd-scattering template has no "
              "jets around the J/psi. Errors statistical only; sigma_eff from the 2nd-scattering rate of the DPS slide (sigma_vis = 0.7 x 80 mb).")

# --- per year
yrow = []
for y in YEARS + ["all"]:
    Rr = R if y == "all" else YR[y]
    Fl = FIT if y == "all" else YF[y]
    Fj = FJ if y == "all" else YJ[y]
    CC = [Rr["channels"][c] for c in CHN.values()]
    s2 = Fj["lt2d"]["samples"]["notb"]
    yrow.append(["2022–2026" if y == "all" else y, f"{LTOT if y == 'all' else LUMI[y]:.0f}", f"{ntt(CC) / 1e6:.2f}M",
                 f"{Fl['params']['N_notb']:.0f} ± {Fl['errors']['N_notb']:.0f}", pm(Fl["params"]["fp_notb"], *Fl["minos"]["fp_notb"], 100), f"{Fl['signif']['nominal']:.1f}σ",
                 f"{100 * s2['fp']:.1f} ± {100 * s2['fp_err']:.1f}", f"{Fj['lt2d']['signif_prompt_notb']:.1f}σ"])
content("years", "Result · per year", "Per year: prompt fractions and rates",
        f'<div class="row" style="gap:36px"><div class="fig">{F("years_fp", "both fits")}</div><div class="fig">{F("years_rate", "J/ψ per t̅t event")}</div></div>',
        t="fit", notes="Each year fitted alone with the same model; all years in one fit right of the dotted line.")
content("years-tab", "Result · per year", "Per year: yields and prompt fractions",
        table(["Year", "fb<sup>−1</sup>", f"{TB} events", f"J/ψ {SIG}, PV", "prompt [%], ℓ<sub>xy</sub>", "signif.", "prompt [%], m × ℓ<sub>xy</sub>", "signif."], yrow,
              widths=[12, 7, 11, 16, 17, 9, 18, 10], num=(1, 2, 3, 4, 5, 6, 7), hl=(5,)), t="fit")

# --- summary
content("summary", "Summary", "Summary",
        ul([f"<b>Signal:</b> J/ψ ΔR &gt; 0.4 from every jet of the two top decays (b jets, the four {TB} jets of ℓ + jets, the χ² fit jets; e μ: the two b jets)",
            f"<b>Why:</b> the b-jet veto alone kept J/ψ in top-decay jets (a W-pair mass peak); {ST_NP:.0f} of the {OLD_NP:.0f} prompt J/ψ of the 6 Nov deck; private gen-level Pythia + LETO: {100 * GL['eff']:.0f}% of top-decay J/ψ pass the veto",
            f"<b>Prompt J/ψ outside the {TB} jets:</b> {SP_NP:.0f} ± {SP_NPE:.0f} (unbinned), {N_PR:.0f} (ℓ<sub>xy</sub> fit) in {NTT / 1e6:.1f}M {TB} events: {SR}, statistical only",
            *([f"<b>Fit model (toys):</b> the b-jet template loses the signal if the b jets' prompt-like share is prompt J/ψ; the per-category template closes to ≈ 5%: "
               f"{BN['N'][0]:.0f} <sup>+{BNM[1]:.0f}</sup><sub>−{-BNM[0]:.0f}</sub> ({BN['signif_prompt']:.1f}σ), proposed nominal"]
              if len(TOYS) == 2 and "BN" in globals() else []),
            f"<b>CMS simulation:</b> the fit closes ({CS['fit_prompt']:.0f} ± {CS['fit_prompt_err']:.0f} for {CS['truth_prompt']:.0f}); {RS['mc_prompt'][0]:.0f} prompt J/ψ outside the {TB} jets (hadronisation only); "
            f"b decays data / simulation {RS['data_nonprompt'][0] / simp(RS, 'b decay'):.2f}{' with the backgrounds' if HAS_BKG else ''}",
            f"<b>Still top decays:</b> about {RES_LO:.0f}–{RES_HI:.0f} (gen-level Pythia leak, normalised to the {ST_C:.0f} ± {ST_CE:.0f} vetoed prompt J/ψ left after their short b decays)",
            f"<b>Second scattering not excluded:</b> soft as inclusive J/ψ; the gen-level Pythia + LETO gives {SNC['sps_signal']:.0f} ± {SNC['sps_signal_err']:.0f} of the {GD['signal'][0]:.0f} (ℓ + jets), leaving {SNC['dps_left']:.0f} ± {SNC['dps_left_err']:.0f}: σ<sub>eff</sub> ≈ {SNC['sigma_eff_mb']:.0f} mb; no distribution separates",
            f"<b>Withdrawn (6 Nov):</b> the hard spectrum and the recoil against the {TB} came from J/ψ in top-decay jets"])
        + f'<div class="callout blue"><b>{SR} prompt J/ψ outside the t̅t decay jets (statistical); the fit closes on the CMS {TB} simulation, whose prompt J/ψ are {RS["data_prompt"][0] / simp(RS, "prompt"):.0f}× fewer{" (with the non-" + TB + " backgrounds)" if HAS_BKG else " (non-" + TB + " backgrounds still to add)"}; {RES_LO:.0f}–{RES_HI:.0f} may be top decays.</b> Next: the nominal fit and template (toys), the residual subtraction, a DPS discriminant.</div>')

# --- backup: the vetoed category, region counts
brow = []
for nm, C in C3.items():
    for cn, cl in (("notb", SIG), ("ttj", f"other {TB} jet"), ("inb", "b jet")):
        brow.append([f"{nm}, {cl}"] + [fmt((yl(C, f'{ln}_{cn}')['n'], yl(C, f'{ln}_{cn}')['err'])) for ln in ("prompt", "nonprompt", "between", "pileup")])
content("bk-regions", "Backup · region counts", "Region counts per channel and category",
        table(["Channel, category", "prompt region", "non-prompt", "between", "pileup"], brow, widths=[36, 16, 16, 16, 16], num=(1, 2, 3, 4), hl=())
        + ul([f"Opposite sign, window − sidebands, {NB1}"], "b s"), alt=True)
content("bk-z0", "Backup · pileup", f"|z<sub>0</sub>|: the pileup J/ψ {SIG}",
        f'<div class="row"><div class="fig">{F("z0", "sideband-subtracted, fraction per bin")}</div><div class="col">'
        + ul(["<b>z<sub>0</sub>:</b> where the J/ψ line passes the beam line, relative to the selected primary vertex; |z<sub>0</sub>| ≥ 0.1 cm: another collision"], "b s")
        + "</div></div>", alt=True)
content("bk-fit", "Backup · ℓ<sub>xy</sub> fit", f"The ℓ<sub>xy</sub> fit with the background bin by bin from the sidebands: ({pms(FP, *FPE, 100)})% prompt",
        f'<div class="row" style="gap:44px"><div class="fig">{F("lxy_fit_notb", "signal")}</div><div class="fig">{F("lxy_fit_inb", "b-matched (control)")}</div></div>', alt=True)
content("bk-splot", "Backup · sPlot fit", "The unbinned m × ℓ<sub>xy</sub> fit: mass projection, and the prompt J/ψ |y|",
        f'<div class="row" style="gap:44px"><div class="fig">{F("splot_proj_m", "signal, all decay lengths")}</div><div class="fig">{F("splot_absy", "unit area")}</div></div>', alt=True)
zrow = [[f"{a:.1f}–{b:.1f}", f"{100 * d:.0f} ± {100 * e:.0f}", f"{100 * p:.0f}"] for a, b, d, e, p in
        zip(GZ["z_edges"][:-1], [min(x, 1.0) for x in GZ["z_edges"][1:]], GZ["data_ttj"], GZ["data_ttj_err"], GZ["pythia_ttj"])]
crow = [[nm, f"{100 * c['top_share']:.0f}%", f"{100 * c['isr_eff']:.0f}%", f"{c['data_prompt']:.0f}", f"{100 * c['pileup_frac']:.0f}%"] for nm, c in GC["cuts"].items()]
content("bk-leak", "Backup · gen study", f"The gen-level Pythia + LETO puts the J/ψ too hard in its jet: z {GZ['mean_z_pythia']:.2f} against {GZ['mean_z_data']:.2f} in data",
        '<div class="row" style="gap:40px"><div class="col">'
        + table(["z = p<sub>T</sub>(J/ψ) / p<sub>T</sub>(jet)", "data [%]", "gen. Pythia [%]"], zrow, widths=[40, 32, 28], num=(1, 2), hl=())
        + ul([f"<b>J/ψ in the other {TB} jets</b> (vetoed), prompt sWeights in data; Pythia all origins (top decay and ISR alike)",
              f"<b>Reweighted to the data z</b>, top-decay J/ψ passing: {100 * GZN['top_decay_pass']:.0f}% (raw {100 * GL['eff']:.0f}%), ISR {100 * GZN['isr_pass']:.0f}% (raw {100 * GL['isr_eff']:.0f}%): the top-decay share of the signal stays {100 * GZN['signal_top_decay_share']:.0f}%"], "b s")
        + "</div></div>", alt=True, t="gen",
        notes="gen_leak_z.py: weights data / Pythia in the z bins of the J/psi in the other ttbar jets, applied to every Pythia J/psi in a jet (selected, or the "
              "nearest gen jet above 15 GeV within 0.4); the J/psi in no jet keep weight 1 (0.5 and 1.5 move the share by 1-2 points). gen_cutscan.py: Pythia "
              "top-decay share and ISR efficiency, and the prompt sWeights kept in data, for selections on top of the veto; the pileup J/psi (random direction "
              "with respect to the ttbar) show the cost for a J/psi unrelated to the ttbar.")

content("bk-cuts", "Backup · gen study", "No simple selection on top of the veto removes the top-decay J/ψ without losing as much of the rest",
        table(["Selection (after the veto)", "top-decay share (gen. Pythia)", "ISR kept (gen. Pythia)", "prompt J/ψ, data", "pileup J/ψ kept"], crow, widths=[36, 16, 14, 17, 17], num=(1, 2, 3, 4), hl=(0,)),
        alt=True, t="gen", notes="gen_cutscan.py: Pythia top-decay share and ISR efficiency, and the prompt sWeights kept in data, for selections on top of the veto; the pileup J/psi "
                        "(random direction with respect to the ttbar) show the cost for a J/psi unrelated to the ttbar.")

content("bk-gencmp", "Backup · gen study", f"|Δy| and |Δφ| between the J/ψ and the {TB} against the gen-level Pythia + LETO",
        f'<div class="row" style="gap:44px"><div class="fig">{F("gencmp_dy", "per unit |Δy|")}</div><div class="fig">{F("gencmp_dphi", "per rad")}</div></div>', alt=True, t="gen",
        notes="As the Pythia signal slide; the ttbar from the chi2 fit without the J/psi jet (|Delta phi|) and with the W-mass neutrino pz (|Delta y|).")

content("bk-sim", "Backup · CMS simulation", f"The vetoed category against the CMS simulation, and the signal with the simulated b-decay shape ({MTF['fits']['signal_all_w1']['N_p']:.0f} ± {MTF['fits']['signal_all_w1']['N_p_err']:.0f})",
        f'<div class="row" style="gap:44px"><div class="fig">{F("mc_lxy_ttj", "window − sidebands, per mm", 0.7)}</div>'
        f'<div class="fig">{F("mc_tfit_signal", "window − sidebands, per mm", 0.7)}</div></div>'
        + ul([f"<b>Right:</b> the binned fit of the vetoed-category slide on the signal: {MTF['fits']['signal_all_w1']['N_p']:.0f} ± {MTF['fits']['signal_all_w1']['N_p_err']:.0f} against {SP_NP:.0f} ± {SP_NPE:.0f} unbinned; "
              f"the simulated b decays have no prompt-like part, the unbinned fit gives the b decays the {100 * FPI_D:.1f}% prompt-like share of the data b jets (≈ {FPI_D * SPF['N'][1]:.0f} J/ψ); "
              f"closure of the binned fit on the simulation: {MTF['closure']['signal_all']['N_p']:.0f} ± {MTF['closure']['signal_all']['N_p_err']:.0f} for {MTF['closure']['signal_all']['truth_prompt']:.0f} true"], "b s"),
        alt=True, t="cms")

if len(TOYS) == 2:
    for v, nm_ in (("", "b-jet template"), ("_btmpl", "per-category template")):
        PF = [(f"mc_toys{v}_mc", "fitted against true, b jets as simulated"), (f"mc_toypulls{v}_mc", "pulls at 100 injected")]
        if all(os.path.exists(os.path.join(D.HERE, "figs", n + ".json")) for n, _ in PF):
            content(f"bk-toys{v}", "Backup · toy closure", f"Toys with the {nm_}: every fit against the truth, and the pulls",
                    '<div class="row" style="gap:44px">' + "".join(f'<div class="fig">{F(n, c, 0.7)}</div>' for n, c in PF) + "</div>", alt=True, t="cms")

content("refs", "References", "References",
        '<ol class="refs">'
        + '<li>6 Nov deck (signal: not b-matched): <a href="../2026Nov05_jpsi_run3/">prompt J/ψ in 2022–2026 data</a></li>'
        + '<li>Onia in the Pythia 8 parton shower (LETO): N. Cooke, P. Ilten, L. Lönnblad, S. Mrenna, Eur. Phys. J. C 84 (2024) 432, arXiv:2312.05203</li>'
        + '<li>BTVNanoCommissioning, <a href="https://github.com/cms-btv-pog/BTVNanoCommissioning">github.com/cms-btv-pog/BTVNanoCommissioning</a></li>'
        + '<li>plotlyhep, <a href="https://github.com/DickyChant/plotlyhep">github.com/DickyChant/plotlyhep</a></li></ol>', alt=True)

if __name__ == "__main__":
    D.main()
