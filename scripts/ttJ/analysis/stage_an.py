"""Stage the figures of the signal with the ttbar-jet veto (assoc_), the simulation (mc_) and the generator comparison (gen_) for the
analysis note: figs/<deck name>.json -> <an dir>/<prefix>_<name>.pdf (vector, kaleido), and their lines "<file> | <caption>" in
<an dir>/MANIFEST.txt (the lines with these prefixes are replaced; the others kept), the numbers read from the fit outputs. Also
<an dir>/MC_SAMPLES.txt: the simulated samples (DAS names, cross sections, pass).
usage: python3 stage_an.py <an dir>"""
import glob
import json
import os
import sys

import plotly.io as pio

HERE = os.path.dirname(os.path.abspath(__file__))
AN = sys.argv[1]
J = lambda n: json.load(open(os.path.join(HERE, n)))  # noqa: E731
FJ, FL, SP, MCC, MT, GX, GZ, BAL, GEN = (J(n) for n in ("fit_jpsi.json", "fit_lxy.json", "splot.json", "mc_compare.json", "mc_template.json",
                                                         "gen_compare.json", "gen_leak_z.json", "ttbar_balance.json", "gen_assoc.json"))
R = J("results_run3.json")
SEL = "mu+jets, e+jets and e mu summed, N_b >= 1, opposite sign, |z0| < 0.1 cm"
SIG = "J/psi outside every jet of the two top decays (Delta R > 0.4 from the b jets, the four ttbar jets of l+jets and the jets of the chi2 ttbar fit; e mu: the two b candidates)"
SH = FJ["shape"]
MSH = f"J/psi = Crystal Ball (sigma {1e3 * SH['s1']:.1f} MeV) + Gaussian (sigma {1e3 * SH['s2']:.1f} MeV), common mean {1e3 * SH['mu']:.1f} MeV, continuum exponential"
NTT = sum(R["channels"][c]["yields"]["data_os_all_ge1b"]["n_tt"] for c in R["channels"])
P, MI = FL["params"], FL["minos"]
F, FT = SP["fits"]["notb"], SP["fits"]["ttj"]
L2 = FJ["lt2d"]["samples"]["notb"]
CL = MCC["closure"]
EV = MCC["events"]
RS, RI = MCC["rates"]["signal_all"], MCC["rates"]["inb_all"]
TC, TF = MT["fits"]["ttj_ljets_nbcon"], MT["fits"]["ttj_ljets_w1"]
SN, SZ, SNC, SZC = (GX["scaled"][k] for k in ("nominal", "z", "nominal_bcorr", "z_bcorr"))
BP = {s_: BAL["profile"][f"{s_}_excl_all"]["ppar_all_pt"] for s_ in ("prompt", "pileup", "sps", "dps")}
BF = BAL["fit"]
MCN = "the CMS ttbar simulation (Summer24 POWHEG + Pythia 8 CP5, full simulation) at sigma(ttbar) x 310 fb-1, no non-ttbar backgrounds"


def m(k):
    o = FJ["mass"][k]
    chi = f"; chi2 = {o['chi2']:.0f} / {o['nbins']} bins" if "chi2" in o and "nbins" in o else ""
    return f"N(J/psi) = {o['Ns']:.0f} +- {o['Ns_err']:.0f}, {o['signif']:.1f} sigma{chi}"


def ch(c):
    f_ = FL["channels"][c]
    return f"f = {100 * f_['fp']:.1f} +{100 * f_['fp_minos'][1]:.1f} -{-100 * f_['fp_minos'][0]:.1f}% of {f_['N']:.0f} +- {f_['N_err']:.0f} J/psi"


CHN = {"mujets": "mu+jets", "ejets": "e+jets", "emu": "e mu"}
STAGE = [
    # ---- the signal with the ttbar-jet veto
    ("mfit_all", "assoc_mass_fit_all", f"Dimuon mass of all J/psi candidates (every category, primary vertex and pileup), mu+jets, e+jets and e mu summed, N_b >= 1, opposite sign, binned fit: {MSH}. {m('all')}. Lower panel: pulls."),
    ("mfit_inb_pv", "assoc_mass_fit_bjet", f"Dimuon mass of the J/psi in a b jet (b candidate or UParT-M jet), {SEL}, binned fit: {MSH}. {m('inb_pv')}. Lower panel: pulls."),
    ("mfit_notb_pv", "assoc_mass_fit_signal", f"Dimuon mass of the signal sample, {SIG}, {SEL}, binned fit: {MSH}. {m('notb_pv')}. Lower panel: pulls."),
    ("mfit_notb_prompt", "assoc_mass_fit_signal_prompt_region", f"Dimuon mass of the signal sample in the prompt region |l_xy| < 0.06 mm, {SEL}. {m('notb_prompt')}. Lower panel: pulls."),
    ("mfit_notb_nonprompt", "assoc_mass_fit_signal_nonprompt_region", f"Dimuon mass of the signal sample in the non-prompt region l_xy > 0.1 mm, {SEL}. {m('notb_nonprompt')}. Lower panel: pulls."),
    ("mfit_notb_pileup", "assoc_mass_fit_signal_pileup", f"Dimuon mass of the J/psi outside every ttbar jet from pileup collisions (|z0| >= 0.1 cm), N_b >= 1, opposite sign. {m('notb_pileup')}. Lower panel: pulls."),
    ("lxy_fit_notb", "assoc_lxy_fit_signal", f"l_xy of the signal sample ({SIG}), signal window minus sidebands, {SEL}, binned likelihood fit (background bin by bin from the sidebands; fitted together with the b-jet sample, shared Gaussian resolution {1e3 * P['s1']:.1f} um, b-decay shape of the b jets, lambda = {P['lam_inb']:.3f} mm). Prompt fraction f = {100 * P['fp_notb']:.1f} +{100 * MI['fp_notb'][1]:.1f} -{-100 * MI['fp_notb'][0]:.1f}% of {P['N_notb']:.0f} +- {FL['errors']['N_notb']:.0f} J/psi ({P['fp_notb'] * P['N_notb']:.0f} prompt), {FL['signif']['nominal']:.1f} sigma (statistical; {FL['signif']['mu_emu']:.1f} sigma without e+jets; {FL['signif']['exp']:.1f} sigma with an own exponential for the b decays). Lower panel: pulls."),
    ("lxy_fit_inb", "assoc_lxy_fit_bjet", f"l_xy of the J/psi in a b jet, window minus sidebands, {SEL}, the same fit: prompt-like fraction {100 * P['fp_inb']:.1f} +- {100 * FL['errors']['fp_inb']:.1f}%, lambda = {P['lam_inb']:.3f} mm. Lower panel: pulls."),
    ("lt2d_notb", "assoc_fit2d_lxy_window_signal", f"l_xy in the J/psi window of the binned two-dimensional fit (mass x l_xy) of the signal sample, {SEL}: {L2['N_P']:.0f} prompt J/psi, {L2['N_NP']:.0f} b decays, f = {100 * L2['fp']:.1f} +- {100 * L2['fp_err']:.1f}%, {FJ['lt2d']['signif_prompt_notb']:.1f} sigma. Lower panel: pulls."),
    ("lt2d_inb", "assoc_fit2d_lxy_window_bjet", f"l_xy in the J/psi window of the binned two-dimensional fit of the J/psi in a b jet, {SEL}. Lower panel: pulls."),
    ("splot_proj_m", "assoc_unbinned_mass_proj", f"Mass projection of the unbinned extended maximum-likelihood fit in mass x l_xy of the {F['n']} signal candidates ({SIG}), {SEL}, 2.6 < m < 3.6 GeV, -0.5 < l_xy < 1.5 mm, per-candidate l_xy resolution (scale factor {F['params']['S']:.2f} x the l_xy error): prompt J/psi {F['N'][0]:.0f} +{F['NP_minos'][1]:.0f} -{-F['NP_minos'][0]:.0f} ({F['signif_prompt']:.1f} sigma), b decays {F['N'][1]:.0f} +- {F['N_err'][1]:.0f}, continuum {F['N'][2]:.0f}; {F['N'][0] / NTT * 1e6:.1f} +- {F['N_err'][0] / NTT * 1e6:.1f} prompt J/psi per 10^6 ttbar events ({NTT / 1e6:.2f}M). Lower panel: pulls."),
    ("splot_proj_l", "assoc_unbinned_lxy_proj", f"l_xy projection of the same unbinned fit of the signal sample: prompt J/psi {F['N'][0]:.0f} +{F['NP_minos'][1]:.0f} -{-F['NP_minos'][0]:.0f}, b decays {F['N'][1]:.0f}, continuum {F['N'][2]:.0f}. Lower panel: pulls."),
    ("an_unbinned_m_vetoed", "assoc_unbinned_mass_proj_vetoed", f"Mass projection of the unbinned fit of the J/psi in the other ttbar jets (vetoed category: W pair or chi2-fit jet, l+jets), {FT['n']} candidates: prompt {FT['N'][0]:.0f} +- {FT['N_err'][0]:.0f}, b decays {FT['N'][1]:.0f} +- {FT['N_err'][1]:.0f}, continuum {FT['N'][2]:.0f}. The b-decay shape of the b jets does not describe this category (b decays in untagged b jets are short-lived, see mc_closure_vetoed): refitted with the simulated shape {TC['N_p']:.0f} +- {TC['N_p_err']:.0f} prompt (mc_tfit_vetoed). Lower panel: pulls."),
    ("an_unbinned_l_vetoed", "assoc_unbinned_lxy_proj_vetoed", f"l_xy projection of the unbinned fit of the vetoed category: prompt {FT['N'][0]:.0f} +- {FT['N_err'][0]:.0f}, b decays {FT['N'][1]:.0f} +- {FT['N_err'][1]:.0f}. Lower panel: pulls."),
    ("assoc_wmass", "assoc_wpair_mjj", "m(jj) of the selection's W pair for the vetoed prompt J/psi in a W-pair jet (sWeights) against the same J/psi jets put into other ttbar events (an extra jet) and all ttbar events; lower panel: ratio. A W peak; in the simulation the b-decay J/psi in these jets (untagged top b jets) also give one, both are top decay products."),
    ("assoc_drtt", "assoc_drtt_signal_vs_pileup", "Delta R between the signal prompt J/psi and the nearest ttbar decay jet against the prompt pileup J/psi (random direction), unit area per unit Delta R; lower panel: signal / pileup. No excess right outside the 0.4 veto cones."),
    ("z0", "assoc_z0", "|z0| of the J/psi outside the ttbar jets and of the b-jet J/psi (window - sidebands, fraction per bin); lower panel: ratio to the b-jet J/psi."),
    ("years_fp", "assoc_prompt_fraction_per_year", "Prompt fraction of the signal sample per year, l_xy fit and mass x l_xy fit, and all years (right of the line); lower panel: (year - all) / sigma."),
    ("years_rate", "assoc_rate_per_year", "J/psi per ttbar event per year (prompt signal, b decays, pileup); lower panel: ratio to all years."),
    ("splot_pt", "assoc_splot_jpsi_pt", "pT of the prompt J/psi outside the ttbar jets (sWeights) against the prompt pileup J/psi (inclusive J/psi from another collision) and the b decays, unit area per GeV; lower panel: ratio to the pileup J/psi."),
    ("splot_absy", "assoc_splot_jpsi_absy", "|y| of the prompt J/psi outside the ttbar jets (sWeights) against the prompt pileup J/psi, unit area; lower panel: ratio."),
    ("ttbal_dphi", "assoc_ttbal_dphi", f"|Delta phi(J/psi, ttbar)| (ttbar without the J/psi jet) of the prompt signal J/psi (sWeights) against the emulated J/psi recoiling against the ttbar and from a 2nd scattering, unit area, with the template fit (recoiling share {BF['prompt_dphi']['f_same']:.2f}; a 2nd scattering alone at {BF['prompt_dphi']['z_vs_dps_only']:.1f} sigma); lower panel: ratio to the 2nd-scattering emulation."),
    ("ttbal_ppar", "assoc_ttbal_ppar", f"pT(ttbar) projected on the J/psi direction for the prompt signal J/psi against the two emulations: mean {BP['prompt'][0]:.1f} +- {BP['prompt'][1]:.1f} GeV, recoiling {BP['sps'][0]:.0f}, 2nd scattering {BP['dps'][0]:.0f} GeV, prompt pileup J/psi {BP['pileup'][0]:.0f} +- {BP['pileup'][1]:.0f}; template fit recoiling share {BF['prompt_ppar']['f_same']:.2f} (a 2nd scattering alone at {BF['prompt_ppar']['z_vs_dps_only']:.1f} sigma): no separation for these soft J/psi. Lower panel: ratio to the 2nd-scattering emulation."),
    ("ttbal_ppar_pt", "assoc_ttbal_ppar_vs_pt", "Mean of pT(ttbar) projected on the J/psi direction per pT(J/psi) bin, prompt signal against the emulations; lower panel: difference to the 2nd-scattering emulation [GeV]."),
    ("ttbal_dy", "assoc_ttbal_dy", "|Delta y(J/psi, ttbar)| with the ttbar rapidity from the W mass (l+jets) or W and top masses (e mu), prompt signal against the prompt pileup J/psi; lower panel: ratio."),
    # ---- the simulation
    ("mc_closure_signal", "mc_closure_signal", f"The unbinned fit of the data chain run on {MCN} as pseudo-data, signal category, l_xy projection: {CL['signal']['fit_prompt']:.0f} +- {CL['signal']['fit_prompt_err']:.0f} prompt J/psi for {CL['signal']['truth_prompt']:.0f} true (MC truth, shaded), b decays {CL['signal']['fit_nonprompt']:.0f} (true {CL['signal']['truth_bdecay']:.0f}): fake prompt per b decay {CL['signal']['fake_prompt_per_bdecay']:.4f} +- {CL['signal']['fake_prompt_per_bdecay_err']:.4f}, i.e. {CL['signal']['fake_prompt_in_data']:.1f} +- {CL['signal']['fake_prompt_in_data_err']:.1f} of the {F['N'][0]:.0f} in data. Lower panel: pulls."),
    ("mc_closure_ttj", "mc_closure_vetoed", f"The same closure in the vetoed category (other ttbar jets): the fit finds {CL['ttj']['fit_prompt']:.0f} +- {CL['ttj']['fit_prompt_err']:.0f} prompt J/psi for {CL['ttj']['truth_prompt']:.0f} true: the b-decay J/psi there sit in jets that failed the b tag, i.e. with a short b-hadron flight, which the b-jet template calls prompt. Lower panel: pulls."),
    ("mc_bshape", "mc_bdecay_lxy_shapes", "l_xy of the b-decay J/psi of the simulation (MC truth) in b jets, in the other ttbar jets (vetoed) and outside the ttbar jets, and of the prompt J/psi, unit area per mm; lower panel: ratio to the b jets. 59% of the vetoed-jet b decays at |l_xy| < 0.06 mm, 14% in b jets."),
    ("mc_tfit_ttj", "mc_tfit_vetoed", f"Binned fit of the l_xy (window - sidebands) of the vetoed data category, l+jets, with the simulated templates (b decays of that category; prompt J/psi of all categories), b-decay yield constrained to the simulation ({TC['nb_pred']:.0f} +- {0.2 * TC['nb_pred']:.0f}): {TC['N_p']:.0f} +- {TC['N_p_err']:.0f} prompt J/psi, {TC['N_b']:.0f} +- {TC['N_b_err']:.0f} b decays, chi2 = {TC['chi2']:.1f} / {TC['ndf']} ({TF['N_p']:.0f} +- {TF['N_p_err']:.0f} with the b decays free). Lower panel: pulls."),
    ("mc_tfit_signal", "mc_tfit_signal", f"The same binned template fit on the signal category: {MT['fits']['signal_all_w1']['N_p']:.0f} +- {MT['fits']['signal_all_w1']['N_p_err']:.0f} prompt J/psi (against {F['N'][0]:.0f} unbinned, whose b-decay template carries the {100 * P['fp_inb']:.1f}% prompt-like share of the data b jets); closure on the simulation {MT['closure']['signal_all']['N_p']:.0f} +- {MT['closure']['signal_all']['N_p_err']:.0f} for {MT['closure']['signal_all']['truth_prompt']:.0f} true. Lower panel: pulls."),
    ("mc_lxy_signal", "mc_lxy_signal", f"l_xy (window - sidebands) of the signal category, data against {MCN}, stacked by MC truth; lower panel: data / simulation. Data prompt J/psi {RS['data_prompt'][0]:.0f} +- {RS['data_prompt'][1]:.0f} (sPlot) against {RS['mc_prompt'][0]:.0f} in the simulation (hadronisation only); b decays data / simulation {RS['ratio_bdecay']:.2f}."),
    ("mc_lxy_ttj", "mc_lxy_vetoed", f"l_xy (window - sidebands) of the vetoed category, data against {MCN}, by MC truth; lower panel: data / simulation."),
    ("mc_lxy_inb", "mc_lxy_bjet", f"l_xy (window - sidebands) of the b-jet category, data against {MCN}, by MC truth; lower panel: data / simulation. Data prompt-like J/psi {RI['data_prompt'][0]:.0f} +- {RI['data_prompt'][1]:.0f} against {RI['mc_prompt'][0]:.0f} true prompt in the simulation; b decays {RI['ratio_bdecay']:.2f}."),
    ("mc_rates", "mc_rates", f"J/psi per category, data (sPlot: b decays, prompt) against {MCN} (MC truth); lower panel: data / simulation. Selected events data / simulation {EV['mujets']['ratio']:.2f} (mu+jets), {EV['ejets']['ratio']:.2f} (e+jets), {EV['emu']['ratio']:.2f} (e mu): ~{100 * (1 - 2 / (EV['mujets']['ratio'] + EV['ejets']['ratio'])):.0f}% of l+jets is not ttbar."),
    ("chan_rates", "mc_rates_per_channel", f"Prompt and b-decay J/psi outside the ttbar jets per channel, data (sPlot) against {MCN}; lower panel: data / simulation."),
    ("chan_mt", "assoc_mt_per_channel", "m_T(l, pT^miss) of the events with a prompt J/psi outside the ttbar jets (sWeights), mu+jets and e+jets, against the b-decay ones (both channels), unit area; lower panel: prompt / b decay. The e+jets prompt-J/psi events are not at the low m_T of QCD with a fake electron."),
    # ---- the generator comparison (Pythia 8 + LETO, gen level)
    ("assoc_gen", "gen_leak_origin", f"Pythia 8.317 + LETO (J/psi in the parton shower) ttbar l+jets, gen level, the selection and veto of the data: where the prompt J/psi of each origin (top decay, ISR) end up; {100 * GEN['leak']['eff']:.0f}% of the top-decay J/psi pass the veto, {100 * GEN['leak']['signal_top_decay_share']:.0f}% of the signal is top decay. Lower panel: top decay / ISR."),
    ("an_leak_z", "gen_leak_z", f"z = pT(J/psi) / pT(jet) of the vetoed prompt J/psi, data (sWeights; mean {GZ['mean_z_data']:.2f}) against Pythia 8 + LETO (mean {GZ['mean_z_pythia']:.2f}), unit area; lower panel: data / Pythia = the z weights applied to every Pythia J/psi inside a jet."),
    ("gencmp_yields", "gen_yields", f"Prompt J/psi in the other ttbar jets (vetoed) and outside the ttbar jets, l+jets: data (sPlot) against Pythia 8 + LETO with its default LDMEs (normalised with the b-decay J/psi per pT bin, data / Pythia {GX['norm']['k_all']:.2f}) by origin, and a 2nd scattering at sigma_eff = 15 mb ({GX['dps']['n_lj']['15']:.0f}); diamonds: Pythia with the data z. Lower panel: data / prediction."),
    ("gencmp_pt_ttj", "gen_pt_vetoed", f"pT of the vetoed prompt J/psi, data against Pythia scaled to them (x {SN['scale']:.2f}); dashed: with the data z (x {SZ['scale']:.2f}). Lower panel: data / Pythia."),
    ("gencmp_pt", "gen_pt_signal", f"pT of the prompt J/psi outside the ttbar jets, data against Pythia scaled to the vetoed J/psi ({SN['sps_signal']:.0f} +- {SN['sps_signal_err']:.0f}; {SZ['sps_signal']:.0f} with the data z) plus a 2nd scattering at 15 mb; with the vetoed yield corrected for short b decays (x {SNC['scale']:.2f}): {SNC['sps_signal']:.0f} +- {SNC['sps_signal_err']:.0f} ({SZC['sps_signal']:.0f}), leaving {SNC['dps_left']:.0f} +- {SNC['dps_left_err']:.0f} ({SZC['dps_left']:.0f} +- {SZC['dps_left_err']:.0f}) for a 2nd scattering (uncorrected {SN['dps_left']:.0f} +- {SN['dps_left_err']:.0f} / {SZ['dps_left']:.0f} +- {SZ['dps_left_err']:.0f}): sigma_eff ~ {SNC['sigma_eff_mb']:.0f} / {SZC['sigma_eff_mb']:.0f} mb, > {SNC['sigma_eff_mb_95_lower']:.1f} / {SZC['sigma_eff_mb_95_lower']:.1f} mb at 95% CL (uncorrected > {SN['sigma_eff_mb_95_lower']:.1f} / {SZ['sigma_eff_mb_95_lower']:.1f}). Lower panel: data / prediction."),
    ("gencmp_drtt", "gen_drtt_signal", "Delta R(J/psi, nearest ttbar jet) of the prompt signal J/psi against Pythia (scaled to the vetoed J/psi) plus the 2nd scattering; lower panel: data / prediction."),
    ("gencmp_dy", "gen_dy_signal", "|Delta y(J/psi, ttbar)| of the prompt signal J/psi against Pythia plus the 2nd scattering; lower panel: data / prediction."),
    ("gencmp_dphi", "gen_dphi_signal", "|Delta phi(J/psi, ttbar without the J/psi jet)| of the prompt signal J/psi against Pythia plus the 2nd scattering; lower panel: data / prediction."),
]
# same sign, the pileup |z0| fit, per year, jet multiplicity and the ttbar role of the J/psi jet
SSY = [(sum(R["channels"][c]["yields"][f"data_{k}_notb_ge1b"]["n"] for c in R["channels"]),
        sum(R["channels"][c]["yields"][f"data_{k}_notb_ge1b"]["err"] ** 2 for c in R["channels"]) ** 0.5) for k in ("os", "ss")]
PL = SP["pu_leak"]
YS = ("2022", "2023", "2024", "2025", "2026")
SYL = [J(os.path.join("years", f"fit_lxy_{y}.json"))["signif"]["nominal"] for y in YS]
SY2 = [J(os.path.join("years", f"fit_jpsi_{y}.json"))["lt2d"]["signif_prompt_notb"] for y in YS]


def nj(k):
    import numpy as _np
    e = _np.asarray(BAL["vars"]["njets"]["edges"], float)
    w = _np.asarray(BAL["hists"][f"{k}_njets_all"][0], float)
    return float((w * 0.5 * (e[1:] + e[:-1])).sum() / w.sum())


RO = {k: (v[0] if isinstance(v, list) else v) for k, v in BAL["roles"]["inb_ljets_all"].items()}  # [fraction, error] -> fraction
RP = {k: (v[0] if isinstance(v, list) else v) for k, v in BAL["roles"]["prompt_ljets_all"].items()}
STAGE += [
    ("an_samesign", "assoc_mass_samesign", f"Dimuon mass of the candidates outside the ttbar jets (all vertices), mu+jets, e+jets and e mu summed, N_b >= 1: opposite-sign pairs (points) and same-sign pairs (squares), the combinatorial control; window - sidebands: opposite sign {SSY[0][0]:.1f} +- {SSY[0][1]:.1f}, same sign {SSY[1][0]:.1f} +- {SSY[1][1]:.1f}. Lower panel: opposite / same sign."),
    ("splot_z0pu", "assoc_z0_pileup_fit", f"|z0| of the prompt pileup J/psi outside the ttbar jets (sWeights of the m x l_xy fit of the |z0| >= 0.1 cm sample) with a Gaussian in the distance between two collisions, sigma(dz) = {PL['sigma_dz_cm']:.2f} +- {PL['sigma_dz_err']:.2f} cm (chi2 {PL['chi2']:.0f} / {PL['nbins']}), extrapolated into the primary-vertex window |z0| < 0.1 cm (shaded): {PL['n_in_pv']:.1f} +- {PL['n_in_pv_err']:.1f} pileup J/psi expected among the {F['N'][0]:.0f} prompt signal J/psi. Lower panel: pulls."),
    ("an_signif_years", "assoc_significance_per_year", "Significance of the prompt J/psi outside the ttbar jets per year (statistical), l_xy fit / mass x l_xy fit: " + ", ".join(f"{y} {a:.1f} / {b:.1f}" for y, a, b in zip(YS, SYL, SY2)) + f"; all years {FL['signif']['nominal']:.1f} / {FJ['lt2d']['signif_prompt_notb']:.1f} sigma (dashed)."),
    ("ttbal_njets", "assoc_ttbal_njets", f"Number of selected jets in the events with a prompt J/psi outside the ttbar jets (sWeights; mean {nj('prompt'):.2f}) against the prompt pileup J/psi ({nj('pileup'):.2f}), the b decays outside the ttbar jets ({nj('nonprompt'):.2f}) and the b-jet J/psi ({nj('inb'):.2f}); lower panel: (prompt - pileup) / sigma."),
    ("ttbal_role", "assoc_ttbal_roles", f"The ttbar role (chi2 fit) of the jet holding the J/psi, l+jets: b-jet J/psi {100 * (RO['b_had'] + RO['b_lep']):.0f}% in a b jet of the fit (hadronic {100 * RO['b_had']:.0f}%, leptonic {100 * RO['b_lep']:.0f}%), {100 * RO['W jet']:.0f}% in a W jet, {100 * RO['extra']:.0f}% in an extra jet; prompt signal J/psi {100 * RP['none']:.0f}% in no jet, {100 * RP['extra']:.0f}% in an extra jet (by construction never in a ttbar jet). Lower panel: prompt / each emulation."),
]
for c, cn in CHN.items():
    STAGE += [(f"chan_lxyfit_{c}", f"assoc_lxy_fit_{c}", f"l_xy fit of the signal sample, {cn} only (N and f free, the shapes of the combined fit): {ch(c)}. Lower panel: pulls."),
              (f"chan_mass_{c}", f"mc_mass_{c}", f"Dimuon mass of the signal category, {cn}, all l_xy of the fit, data against {MCN} by MC truth; lower panel: data / simulation (continuum: non-ttbar events)."),
              (f"chan_lxy_{c}", f"mc_lxy_{c}", f"l_xy (window - sidebands) of the signal category, {cn}, data against {MCN} by MC truth; lower panel: data / simulation.")]

os.makedirs(AN, exist_ok=True)
lines = []
for src, dst, cap in STAGE:
    p = os.path.join(HERE, "figs", src + ".json")
    if not os.path.exists(p):
        print("missing", src)
        continue
    pio.read_json(p).write_image(os.path.join(AN, dst + ".pdf"), format="pdf")
    lines.append(f"{dst}.pdf | {cap}")
man = os.path.join(AN, "MANIFEST.txt")
old = open(man).read().splitlines() if os.path.exists(man) else []
keep = [l for l in old if not l.startswith(("assoc_", "mc_", "gen_"))]
hdr = ["# --- 2026-11-07: the signal with the ttbar-jet veto (assoc_), the CMS ttbar simulation (mc_), Pythia 8 + LETO (gen_); pass run5m "
       "(79eecfe, 22 of 58723 files missing), 310 fb-1; figures from the 7 Nov deck (cms-plotting, plotlyhep); statistical uncertainties only"]
keep = [l for l in keep if not l.startswith("# --- 2026-11-07")]
open(man, "w").write("\n".join(keep + hdr + lines) + "\n")
print(len(lines), "figures staged in", AN, "| missing figure files:", [s for s, _, _ in STAGE if not os.path.exists(os.path.join(HERE, "figs", s + ".json"))])
print("pdfs now:", len(glob.glob(os.path.join(AN, "*.pdf"))))
