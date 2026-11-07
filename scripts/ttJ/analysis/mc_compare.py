"""Data against the CMS ttbar simulation (Summer24 TTtoLNu2Q + TTto2L2Nu, POWHEG + Pythia8 CP5, full simulation; pass out_mc5
with the code of the data pass) for the J/psi of the three categories of the search (b jet: cat 0; other ttbar jet: jcat 1;
outside the ttbar jets, the signal: jcat >= 2), with the selection of the unbinned fit (opposite sign, N_b >= 1, |z0| < 0.1 cm,
2.6 < m < 3.6 GeV, -0.5 < l_xy < 1.5 mm). The simulation has no onia in its parton shower: its J/psi come from b-hadron
decays (MC truth: 0 prompt J/psi, 1 from a b hadron, 2 no common J/psi parent of the two muons).
  rates     J/psi per category, data (sWeights: prompt, b decay) against MC (truth), the MC normalised to the luminosity of the
            data (cross section / sum of generator weights x event weight x 310 fb^-1; 2024 conditions for every year); also per
            selected event of each (the data events include non-ttbar backgrounds: ~19% in l + jets, see "events")
  events    selected events (N_b >= 1) per channel, data against the ttbar simulation
  lxy       window - sidebands / 2 l_xy per category: data against MC normalised to the luminosity, MC split by truth
  closure   the unbinned fit of the data chain run on the MC as pseudo-data (TTJ_AS_DATA build, same scripts; mc/splot.json):
            the prompt yield it finds where the truth has (almost) none
  backgrounds  with TTJ_BKG=<cands_assoc dir>:<sumw.json>:<summary json> (refresh_bkg.sh): the non-ttbar simulation (single top,
            W + jets, ttV, VV, QCD mu-enriched) at 310 fb^-1 next to the ttbar one, all MC truths together, and per process
-> mc_compare.json   usage: python3 mc_compare.py <MC cands_assoc dir> <MC sumw.json> <xsec_map.json> <MC summary json>
                     <data sweights.parquet> <data results json> <MC closure splot.json> <out.json>"""
import json
import os
import sys

import numpy as np
import pandas as pd

MCD, SWJ, XSJ, MCR, SWQ, DRJ, CLJ, OUTJ = sys.argv[1:9]
CH = ("mujets", "ejets", "emu")
M_LO, M_HI, L_LO, L_HI = 2.6, 3.6, -0.5, 1.5
WIN, SBS = (2.95, 3.25), ((2.60, 2.90), (3.30, 3.60))
LE = np.array([-0.5, -0.2, -0.1, -0.06, -0.03, 0.0, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5])
PTE = np.array([3, 6, 8, 10, 13, 17, 22, 30, 45, 100.])
CATS = {"inb": "b jet", "ttj": "other ttbar jet", "signal": "outside the ttbar jets"}
TRUTH = {0: "prompt", 1: "b decay", 2: "no J/psi"}
LUMI_PB = 310.0e3  # 2022-2026: 34.8 + 28.3 + 109.9 + 111.2 + 25.8 fb^-1
SW, XS = json.load(open(SWJ)), json.load(open(XSJ))
MR, DR = json.load(open(MCR)), json.load(open(DRJ))
CL = json.load(open(CLJ)) if os.path.exists(CLJ) else None


def catmask(d):
    return {"inb": (d.cat == 0).to_numpy(), "ttj": (d.jcat == 1).to_numpy(), "signal": (d.jcat >= 2).to_numpy()}


def sbw(m):
    """window - sidebands / 2 weights"""
    m = np.asarray(m, float)
    return np.where((m > WIN[0]) & (m < WIN[1]), 1.0, 0.0) - 0.5 * np.where(((m > SBS[0][0]) & (m < SBS[0][1])) | ((m > SBS[1][0]) & (m < SBS[1][1])), 1.0, 0.0)


def ntt(R, ch, key):
    return float(R["channels"][ch]["yields"][key]["n_tt"])


# ---------------- MC: the selection of the fit, cross-section weights, per ttbar event of the channel
mc = []
for ch in CH:
    d = pd.read_parquet(os.path.join(MCD, f"cands_{ch}.parquet"))
    d = d[(d.os == 1) & (d.nb >= 1) & (d.life <= 2) & (d.lxy > L_LO) & (d.lxy < L_HI) & (d.mass > M_LO) & (d.mass < M_HI)].copy()
    d["ch"] = ch
    d["wx"] = d.weight.to_numpy(float) * np.array([XS[ds] / SW[ds] for ds in d.dataset])
    mc.append(d)
MC = pd.concat(mc, ignore_index=True)
NTT_D = {ch: ntt(DR, ch, "data_os_all_ge1b") for ch in CH}
NTT_M = {ch: ntt(MR, ch, "mc_os_all_ge1b") * LUMI_PB for ch in CH}  # the MC summary is per pb^-1
MC["wd"] = MC.wx * LUMI_PB  # MC normalised to the luminosity of the data
# the non-ttbar backgrounds (optional)
PROC = (("single top", ("TW", "TbarW", "TBbar", "TbarB")), ("W + jets", ("WtoLNu",)), ("QCD", ("QCD_",)), ("ttV", ("TTL",)), ("VV", ("WW_", "WZ_", "ZZ_")))
BK, NBK = None, {ch: 0.0 for ch in CH}
if os.environ.get("TTJ_BKG"):
    bd, bsw, bres = os.environ["TTJ_BKG"].split(":")
    BSW, BR_ = json.load(open(bsw)), json.load(open(bres))
    bk = []
    for ch in CH:
        p_ = os.path.join(bd, f"cands_{ch}.parquet")
        if not os.path.exists(p_):
            continue
        d = pd.read_parquet(p_)
        d = d[(d.os == 1) & (d.nb >= 1) & (d.life <= 2) & (d.lxy > L_LO) & (d.lxy < L_HI) & (d.mass > M_LO) & (d.mass < M_HI)].copy()
        d = d[d.dataset.isin([k for k in BSW if BSW[k] > 0 and k in XS])]
        d["ch"] = ch
        d["wd"] = d.weight.to_numpy(float) * np.array([XS[ds] / BSW[ds] for ds in d.dataset]) * LUMI_PB
        d["proc"] = [next((pn for pn, pre in PROC if ds.startswith(pre)), "other") for ds in d.dataset]
        bk.append(d)
    BK = pd.concat(bk, ignore_index=True)
    NBK = {ch: ntt(BR_, ch, "mc_os_all_ge1b") * LUMI_PB for ch in CH}
S = pd.read_parquet(SWQ)
SMP = {"inb": "inb", "ttj": "ttj", "signal": "notb"}
OUT = {"lxy_edges": LE.tolist(), "pt_edges": PTE.tolist(), "lumi_pb": LUMI_PB, "events": {ch: {"data": NTT_D[ch], "mc": NTT_M[ch], "ratio": NTT_D[ch] / NTT_M[ch], "bkg": NBK[ch], "ratio_all": NTT_D[ch] / (NTT_M[ch] + NBK[ch])} for ch in CH},
       "bkg_procs": [pn for pn, _ in PROC], "rates": {}, "lxy": {},
       "truth": {}, "pt": {}, "closure": {}}
grp = {"all": CH, "ljets": ("mujets", "ejets"), "emu": ("emu",), "mujets": ("mujets",), "ejets": ("ejets",)}
ME = np.linspace(M_LO, M_HI, 41)  # m(mu mu) per category and group: data (all candidates of the fit) against MC by truth
OUT["mass_edges"] = ME.tolist()
OUT["mass"] = {}
for gn, chs in grp.items():
    q, s = MC[MC.ch.isin(chs)], S[S.ch.isin(chs)]
    cm = catmask(q)
    nd, nm = sum(NTT_D[c] for c in chs), sum(NTT_M[c] for c in chs)
    for cat, smp in SMP.items():
        m = cm[cat]
        ds = s[s["sample"] == smp]
        r = {}
        for sp in ("prompt", "nonprompt"):  # J/psi; per 1e6 selected events in "per_event"
            w = ds[f"w_{sp}"].to_numpy(float)
            r[f"data_{sp}"] = [float(w.sum()), float(np.sqrt((w ** 2).sum()))]
        # MC truth: the true J/psi in the fit range (no sideband subtraction needed for truth 0 and 1)
        for t, tn in TRUTH.items():
            w = q.wd.to_numpy(float)[m & (q.truth == t).to_numpy()]
            r[f"mc_{tn}"] = [float(w.sum()), float(np.sqrt((w ** 2).sum()))]
        r["ratio_bdecay"] = r["data_nonprompt"][0] / r["mc_b decay"][0] if r["mc_b decay"][0] > 0 else None
        r["per_event"] = {k: [1e6 * v[0] / (nm if k.startswith("mc") else nd), 1e6 * v[1] / (nm if k.startswith("mc") else nd)] for k, v in r.items() if isinstance(v, list)}
        if BK is not None:  # the backgrounds by truth and by process
            qb = BK[BK.ch.isin(chs)]
            mb_ = catmask(qb)[cat]
            for t, tn in TRUTH.items():
                w = qb.wd.to_numpy(float)[mb_ & (qb.truth == t).to_numpy()]
                r[f"bkg_{tn}"] = [float(w.sum()), float(np.sqrt((w ** 2).sum()))]
            for pn, _ in PROC:
                w = qb.wd.to_numpy(float)[mb_ & (qb.proc == pn).to_numpy() & (qb.truth != 2).to_numpy()]
                r[f"bkgproc_{pn}"] = [float(w.sum()), float(np.sqrt((w ** 2).sum()))]
        OUT["rates"][f"{cat}_{gn}"] = r
        # l_xy, window - sidebands
        ws = sbw(ds.mass) if len(ds) else np.zeros(0)
        hd = np.histogram(ds.lxy, LE, weights=ws)[0]
        hd2 = np.histogram(ds.lxy, LE, weights=ws ** 2)[0]
        L = {"data": [hd.tolist(), np.sqrt(hd2).tolist()]}
        wm = (q.wd.to_numpy(float) * sbw(q.mass))
        for t, tn in TRUTH.items():
            k = m & (q.truth == t).to_numpy()
            L[f"mc_{tn}"] = [np.histogram(q.lxy[k], LE, weights=wm[k])[0].tolist(), np.sqrt(np.histogram(q.lxy[k], LE, weights=wm[k] ** 2)[0]).tolist()]
        if BK is not None:  # all background truths together, window - sidebands
            qb = BK[BK.ch.isin(chs)]
            kb = catmask(qb)[cat]
            wb = qb.wd.to_numpy(float) * sbw(qb.mass)
            L["bkg"] = [np.histogram(qb.lxy[kb], LE, weights=wb[kb])[0].tolist(), np.sqrt(np.histogram(qb.lxy[kb], LE, weights=wb[kb] ** 2)[0]).tolist()]
        OUT["lxy"][f"{cat}_{gn}"] = L
        # m(mu mu), all l_xy of the fit
        MM = {"data": [np.histogram(ds.mass, ME)[0].tolist(), np.sqrt(np.histogram(ds.mass, ME)[0]).tolist()]}
        for t, tn in TRUTH.items():
            k = m & (q.truth == t).to_numpy()
            MM[f"mc_{tn}"] = [np.histogram(q.mass[k], ME, weights=q.wd[k])[0].tolist(), np.sqrt(np.histogram(q.mass[k], ME, weights=q.wd[k] ** 2)[0]).tolist()]
        if BK is not None:
            qb = BK[BK.ch.isin(chs)]
            kb = catmask(qb)[cat]
            MM["bkg"] = [np.histogram(qb.mass[kb], ME, weights=qb.wd[kb])[0].tolist(), np.sqrt(np.histogram(qb.mass[kb], ME, weights=qb.wd[kb] ** 2)[0]).tolist()]
        OUT["mass"][f"{cat}_{gn}"] = MM
        # pT of the b-decay J/psi: data sWeights against MC truth
        wn = ds.w_nonprompt.to_numpy(float)
        k = m & (q.truth == 1).to_numpy()
        OUT["pt"][f"{cat}_{gn}"] = {"data_nonprompt": np.histogram(ds.pt, PTE, weights=wn)[0].tolist(), "mc_bdecay": np.histogram(q.pt[k], PTE, weights=q.wd[k])[0].tolist()}
        # truth composition (unweighted MC counts, window - sidebands) in the prompt region |l_xy| < 0.06 mm and overall
        for reg, rm in (("prompt_region", (q.lxy.abs() < 0.06).to_numpy()), ("all", np.ones(len(q), bool))):
            sw_ = sbw(q.mass)
            OUT["truth"][f"{cat}_{gn}_{reg}"] = {tn: float(sw_[m & rm & (q.truth == t).to_numpy()].sum()) for t, tn in TRUTH.items()}

# the W peak of the vetoed J/psi: the share of the MC b decays in the other ttbar jets (l + jets, J/psi window) with the W-pair
# m(jj) of the selection in 65-95 GeV, as for the data prompt J/psi (wpair_check.py)
wq = MC[MC.ch.isin(("mujets", "ejets")) & (MC.jcat == 1) & (MC.truth == 1) & (MC.mass > WIN[0]) & (MC.mass < WIN[1])]
OUT["wpeak_bdecay"] = float(((wq.tt_mjj > 65) & (wq.tt_mjj < 95)).mean())

# the b-decay l_xy shape against pT(J/psi) per category (MC truth, J/psi window): the share of short decays |l_xy| < 0.06 mm,
# the mean decay length beyond 0.1 mm, the median l_xy error; and the b-jet shape reweighted to the pT spectrum of the signal's
# b decays against the signal's own shape (is the b-jet template off because of pT, or because of the b tag?)
PTB = np.array([6, 8, 10, 13, 17, 22, 30, 45, 100.])
LEB = np.array([-0.5, -0.1, -0.03, 0, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5])
bt = MC[(MC.truth == 1) & (MC.mass > WIN[0]) & (MC.mass < WIN[1])]
cmb = catmask(bt)
OUT["bshape_pt"] = {"pt_edges": PTB.tolist(), "lxy_edges": LEB.tolist()}
for cat in ("inb", "ttj", "signal"):
    q = bt[cmb[cat]]
    r = {"n": [], "short": [], "short_err": [], "tail": [], "tail_err": [], "lxyerr_um": []}
    for lo, hi in zip(PTB[:-1], PTB[1:]):
        g = q[(q.pt >= lo) & (q.pt < hi)]
        n = len(g)
        k = float((g.lxy.abs() < 0.06).mean()) if n else float("nan")
        tl = g.lxy[g.lxy > 0.1] - 0.1
        r["n"].append(n)
        r["short"].append(k)
        r["short_err"].append(float(np.sqrt(k * (1 - k) / n)) if n else float("nan"))
        r["tail"].append(float(tl.mean()) if len(tl) else float("nan"))
        r["tail_err"].append(float(tl.std() / np.sqrt(len(tl))) if len(tl) > 1 else float("nan"))
        r["lxyerr_um"].append(float(1e3 * g.lxyerr.median()) if n else float("nan"))
    r["shape"] = (np.histogram(q.lxy, LEB)[0] / max(len(q), 1)).tolist()
    r["shape_err"] = (np.sqrt(np.histogram(q.lxy, LEB)[0]) / max(len(q), 1)).tolist()
    r["median_pt"] = float(q.pt.median())
    OUT["bshape_pt"][cat] = r
qb, qs = bt[cmb["inb"]], bt[cmb["signal"]]
hb, hs = np.histogram(qb.pt, PTB)[0], np.histogram(qs.pt, PTB)[0]
wpt = np.where(hb > 0, (hs / hs.sum()) / np.maximum(hb / hb.sum(), 1e-12), 0.0)
wb_ = wpt[np.clip(np.digitize(qb.pt, PTB) - 1, 0, len(wpt) - 1)]
OUT["bshape_pt"]["inb_rew"] = {"shape": (np.histogram(qb.lxy, LEB, weights=wb_)[0] / wb_.sum()).tolist()}
ps_, es_ = np.asarray(OUT["bshape_pt"]["signal"]["shape"]), np.asarray(OUT["bshape_pt"]["signal"]["shape_err"])
for k_ in ("inb", "inb_rew"):
    pk = np.asarray(OUT["bshape_pt"][k_]["shape"])
    OUT["bshape_pt"][k_]["chi2_vs_signal"] = float((((ps_ - pk) / np.where(es_ > 0, es_, 1)) ** 2).sum())
print("b-decay shape vs pT: chi2 of the signal's b-decay l_xy shape against the b-jet one", round(OUT["bshape_pt"]["inb"]["chi2_vs_signal"], 1),
      "and the pT-reweighted b-jet one", round(OUT["bshape_pt"]["inb_rew"]["chi2_vs_signal"], 1), f"({len(LEB) - 1} bins)")

# ---------------- closure: the fit on the MC as pseudo-data against its truth
if CL:
    # the truth l_xy of the pseudo-data (unweighted, all masses of the fit, as the fit projections) for the closure canvases
    LP = np.asarray(CL["proj"]["lxy_edges"], float)
    OUT["closure_truth_l"] = {"lxy_edges": LP.tolist()}
    cmA = catmask(MC)
    for cat in ("signal", "ttj", "inb"):
        OUT["closure_truth_l"][cat] = {tn: np.histogram(MC.lxy[cmA[cat] & (MC.truth == t).to_numpy()], LP)[0].tolist() for t, tn in TRUTH.items()}
    for key, cat in (("notb", "signal"), ("ttj", "ttj"), ("inb", "inb")):
        F = CL["fits"].get(key)
        if not F:
            continue
        T = OUT["truth"][f"{cat}_all_all"]
        OUT["closure"][cat] = {"fit_prompt": F["N"][0], "fit_prompt_err": F["N_err"][0], "fit_nonprompt": F["N"][1], "fit_nonprompt_err": F["N_err"][1],
                               "truth_prompt": T["prompt"], "truth_bdecay": T["b decay"], "fake_prompt_per_bdecay": (F["N"][0] - T["prompt"]) / max(F["N"][1], 1e-9),
                               "fake_prompt_per_bdecay_err": F["N_err"][0] / max(F["N"][1], 1e-9)}
    # what the same fake rate would give in the data: x the b-decay yield of the data fit
    DSP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(OUTJ)), "splot.json")))
    for key, cat in (("notb", "signal"), ("ttj", "ttj"), ("inb", "inb")):
        C = OUT["closure"].get(cat)
        if C and key in DSP["fits"]:
            C["data_fit_prompt"] = DSP["fits"][key]["N"][0]
            C["data_fit_nonprompt"] = DSP["fits"][key]["N"][1]
            C["fake_prompt_in_data"] = C["fake_prompt_per_bdecay"] * DSP["fits"][key]["N"][1]
            C["fake_prompt_in_data_err"] = C["fake_prompt_per_bdecay_err"] * DSP["fits"][key]["N"][1]

# ---------------- printout
for ch, e in OUT["events"].items():
    print(f"{ch:7s} selected events: data {e['data']:.0f}, ttbar simulation {e['mc']:.0f} ({LUMI_PB / 1e3:.0f} fb^-1): data / simulation {e['ratio']:.3f}"
          + (f"; backgrounds {e['bkg']:.0f}, data / (ttbar + backgrounds) {e['ratio_all']:.3f}" if e["bkg"] else ""))
for k, r in OUT["rates"].items():
    if not k.endswith(("_all", "_ljets", "_emu", "_mujets", "_ejets")):
        continue
    print(f"{k:14s} J/psi: data prompt {r['data_prompt'][0]:7.1f} +- {r['data_prompt'][1]:5.1f}, b decay {r['data_nonprompt'][0]:8.1f} +- {r['data_nonprompt'][1]:5.1f}"
          f" | MC b decay {r['mc_b decay'][0]:8.1f}, prompt {r['mc_prompt'][0]:6.1f}, no J/psi {r['mc_no J/psi'][0]:8.1f} | data / MC (b decay) {r['ratio_bdecay'] or float('nan'):.2f}")
for k, r in OUT["rates"].items():
    if BK is not None and k.endswith(("_all", "_mujets", "_ejets", "_emu")):
        print(f"{k:14s} backgrounds: prompt {r['bkg_prompt'][0]:.1f}, b decay {r['bkg_b decay'][0]:.1f} |",
              ", ".join(f"{pn} {r[f'bkgproc_{pn}'][0]:.1f}" for pn in OUT["bkg_procs"]))
for k, v in OUT["truth"].items():
    if k.endswith("_all_prompt_region") or k.endswith("_all_all"):
        print(f"MC truth {k:28s}", {a: round(b, 1) for a, b in v.items()})
for cat, C in OUT["closure"].items():
    print(f"closure {cat:6s}: fit prompt {C['fit_prompt']:.1f} +- {C['fit_prompt_err']:.1f} (truth {C['truth_prompt']:.1f}), b decay {C['fit_nonprompt']:.0f} (truth {C['truth_bdecay']:.0f});"
          f" fake prompt per b decay {C['fake_prompt_per_bdecay']:.4f} +- {C['fake_prompt_per_bdecay_err']:.4f}"
          + (f" -> in data {C['fake_prompt_in_data']:.1f} +- {C['fake_prompt_in_data_err']:.1f} of the {C['data_fit_prompt']:.0f}" if "fake_prompt_in_data" in C else ""))
json.dump(OUT, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
