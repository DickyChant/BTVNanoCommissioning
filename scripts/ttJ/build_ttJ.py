"""ttbar + J/psi -> mu mu (workflow ttJ of the ttJ branch, or the older hadw_jpsi / hadw_e_jpsi outputs): the mu + jets, e + jets
and e mu selections of the hadronic-W study, plus one soft-muon pair per event (utils/jpsi.py). Channels "mujets", "ejets",
"emu" (older outputs: "sl" of a Muon / EGamma data set -> mujets / ejets, "dil" -> emu).
  spectra   m(mu mu) 2.0-4.5 GeV, opposite and same sign, data and MC (per process; MC split by truth: prompt / from b / no J/psi)
  yields    opposite-sign signal window (2.95-3.25 GeV) minus half the two sidebands (2.60-2.90, 3.30-3.60 GeV), per
            category and N_b; per ttbar-selected event, data vs MC. Signal "notb": the J/psi not matched to a b jet
            (in another jet or in no jet); "inb": matched to a b jet (control, mostly b decays);
            muons: loose ID (both), or both also medium ("medium")
  shapes    sideband-subtracted J/psi pT, z = pT(J/psi) / pT(jet), max muon SIP3D, Delta R to the nearest jet
  ttbar     the J/psi events are ttbar: hadronic W m(jj), hadronic top m(jjb) and m(l, b_lep) (l + jets), m(l, b) per lepton
            (e mu, minimax), sideband-subtracted, against the shape of all selected ttbar events of the same jobs (N_b >= 1)
  m(l, J/psi)  sideband-subtracted, by reco side (the J/psi in the b paired with the lepton or the other one) and MC truth
            side; <m(l, J/psi)> in 0-250 GeV against n_track (bins 0-96), GeV per 100 tracks, data vs MC
MC is normalised by cross section (pb) and compared with data per ttbar-selected event, so no luminosity enters.
  life      with the two-muon vertex (ttJ outputs): yields, mass spectra and ttbar kinematics per lifetime region (prompt
            |l_xy| < 0.06 mm, non-prompt l_xy > 0.1 mm, pileup-like |z0| >= 0.1 cm), and the sideband-subtracted l_xy shapes
-> results.json   usage: python3 build_ttJ.py <outdir> [<outdir> ...] <results.json>"""
import glob
import json
import os
import re
import sys
import warnings

import hist as Hist
import numpy as np

warnings.filterwarnings("ignore")
from coffea.util import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) < 3:
    sys.exit(__doc__)
OUTDIRS, RES = sys.argv[1:-1], sys.argv[-1]
# MC cross sections {dataset: pb} for the MC normalisation (not needed for data): $TTJ_XSEC or xsec_map.json next to this script
_xs = os.environ.get("TTJ_XSEC", os.path.join(HERE, "xsec_map.json"))
XSEC = json.load(open(_xs)) if os.path.exists(_xs) else {}
EXTRA = {"TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8": 2.0}  # tW -> l nu qq exists for tbar W+ only: x2 for both charges
PROCS = {
    "tt_lj": ["TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8"],
    "tt_ll": ["TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8"],
    "tt_had": ["TTto4Q_TuneCP5_13p6TeV_powheg-pythia8"],
    "wjets": ["WtoMuNu-4Jets_TuneCP5_13p6TeV_madgraphMLM-pythia8"],
    "dy": ["DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"],
    "st": [
        "TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
        "TWminusto4Q_TuneCP5_13p6TeV_powheg-pythia8",
        "TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
        "TbarWplusto4Q_TuneCP5_13p6TeV_powheg-pythia8",
        "TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8",
    ],
    "vv": ["WW_TuneCP5_13p6TeV_pythia8", "WZ_TuneCP5_13p6TeV_pythia8", "ZZ_TuneCP5_13p6TeV_pythia8"],
}
HISTS = {
    "semi": ["jpsi_lxy", "jpsi_mass", "jpsi_pt", "jpsi_sip", "jpsi_drj", "jpsi_z", "jpsi_mlj", "nbjet",
           "jpsi_tt_mjj", "jpsi_tt_mjjb", "jpsi_tt_mlb", "hadW_mass_vs_ntrack_nb", "hadW_top_mjjb", "hadW_lepb_mlb"],
    "dil": ["jpsi_lxy", "jpsi_mass", "jpsi_pt", "jpsi_sip", "jpsi_drj", "jpsi_z", "jpsi_mlj", "dil_ntrack", "jpsi_tt_mlb", "dil_mlb_fl"],
}
KIND = {"mujets": "semi", "ejets": "semi", "emu": "dil"}
LEP = {"mujets": "mu", "ejets": "e"}


def chan_name(ds, ch):
    """older outputs (hadw_jpsi / hadw_e_jpsi): "sl" of an EGamma data set -> ejets, otherwise mujets; "dil" -> emu"""
    if ch == "sl":
        return "ejets" if ds.startswith("EGamma") else "mujets"
    return {"dil": "emu"}.get(ch, ch)


is_data = lambda ds: re.match(r"(Muon|EGamma)\dRun2024", ds) is not None  # noqa: E731


def nominal(h, flow_last=False):
    """nominal values and variances; flow_last keeps the under/overflow bins of the last axis (only)."""
    i = list(h.axes["syst"]).index("nominal")
    if not flow_last:
        v = h.view(flow=False)
    else:
        u = [1 if ax.traits.underflow else 0 for ax in h.axes]
        v = h.view(flow=True)[tuple([slice(a, a + len(ax)) for a, ax in zip(u[:-1], h.axes[:-1])] + [slice(None)])]
    return np.array(v["value"][i]), np.array(v["variance"][i])


acc, sumw, edges, shapes, nfiles = {}, {}, {}, {}, 0
files_per_dir = {}
for f in sorted(sum((glob.glob(f"{o}/*/*.coffea") for o in OUTDIRS), [])):
    try:
        o = load(f)
    except Exception:  # noqa: BLE001
        continue
    nfiles += 1
    od = os.path.dirname(os.path.dirname(f))
    files_per_dir[od] = files_per_dir.get(od, 0) + 1
    for ds, d in o.items():
        sumw[ds] = sumw.get(ds, 0.0) + float(next(iter(d.values()))["sumw"])
        for chk, hd in d.items():
            ch = chan_name(ds, chk)
            for hn in HISTS[KIND[ch]]:
                h = hd.get(hn)
                if h is None:
                    continue
                names = [ax.name for ax in h.axes]
                il = names.index("life") - 1 if "life" in names else None  # lifetime-region axis (ttJ outputs)
                it = names.index("sltrig") - 1 if "sltrig" in names else None  # single-lepton trigger axis (BPHNano outputs)
                shp = tuple(len(ax) for ax in h.axes[1:])
                drop = lambda t, *ks: tuple(x for k, x in enumerate(t) if k not in ks)  # noqa: E731
                shapes.setdefault((ch, hn), drop(shp, il, it))  # also from unfilled histograms
                if il is not None:
                    shapes.setdefault((ch, hn + "@life"), drop(shp, it))
                if it is not None:
                    shapes.setdefault((ch, hn + "@sltrig"), drop(shp, il))
                if hn == "jpsi_sip":  # with the under- and overflow of the SIP axis
                    shapes.setdefault((ch, "jpsi_sip_flow"), shp[:-1] + (shp[-1] + 2,))
                if "nominal" not in list(h.axes["syst"]):
                    continue
                v, w2 = nominal(h)
                if it is not None:
                    a = acc.setdefault(ds, {}).setdefault((ch, hn + "@sltrig"), [0.0, 0.0])
                    a[0], a[1] = a[0] + (v.sum(axis=il) if il is not None else v), a[1] + (w2.sum(axis=il) if il is not None else w2)
                    v, w2 = v.sum(axis=it), w2.sum(axis=it)
                    il = il - 1 if il is not None and il > it else il
                if il is not None:
                    a = acc.setdefault(ds, {}).setdefault((ch, hn + "@life"), [0.0, 0.0])
                    a[0], a[1] = a[0] + v, a[1] + w2
                    v, w2 = v.sum(axis=il), w2.sum(axis=il)
                a = acc.setdefault(ds, {}).setdefault((ch, hn), [0.0, 0.0])
                a[0], a[1] = a[0] + v, a[1] + w2
                if hn == "jpsi_sip":  # also with the SIP3D > 100 overflow (muons not from the primary vertex: pileup)
                    v, w2 = nominal(h, flow_last=True)
                    a = acc[ds].setdefault((ch, "jpsi_sip_flow"), [0.0, 0.0])
                    a[0], a[1] = a[0] + v, a[1] + w2
                edges.setdefault((ch, hn), [list(ax) if isinstance(ax, (Hist.axis.IntCategory, Hist.axis.StrCategory)) else list(map(float, ax.edges))
                                            for ax in h.axes[1:] if ax.name not in ("life", "sltrig")])


def total(key, which):
    """data, all MC, or one MC process: (values, variances), MC in pb."""
    v, w2 = 0.0, 0.0
    for ds, d in acc.items():
        if key not in d:
            continue
        if which == "data":
            if is_data(ds):
                v, w2 = v + d[key][0], w2 + d[key][1]
            continue
        if is_data(ds) or ds not in XSEC:
            continue
        procs = PROCS if which == "mc" else {which: PROCS[which]}
        if not any(ds in L for L in procs.values()):
            continue
        w = XSEC[ds] / sumw[ds] * EXTRA.get(ds, 1.0)
        v, w2 = v + w * d[key][0], w2 + w * w * d[key][1]
    if np.isscalar(v):  # nothing filled yet (no candidate, or no MC): zeros of the histogram shape
        v, w2 = np.zeros(shapes[key]), np.zeros(shapes[key])
    return v, w2


M = np.linspace(2.0, 4.5, 101)
MC = 0.5 * (M[1:] + M[:-1])
SIG = (MC > 2.95) & (MC < 3.25)
SB = ((MC > 2.60) & (MC < 2.90)) | ((MC > 3.30) & (MC < 3.60))
NTC = np.array([-10, 0, 12, 24, 36, 48, 60, 76, 96, 110])
XS = 0.5 * (NTC[1:8] + NTC[2:9])
ML = np.linspace(0, 250, 26)
MLc = 0.5 * (ML[1:] + ML[:-1])


def sbsub(v, w2, axis_mreg=0):
    """signal-window region (mreg 1) minus half the sidebands (mreg 0 + 2), along axis `axis_mreg`."""
    t = lambda a, i: np.take(a, i, axis=axis_mreg)  # noqa: E731
    return t(v, 1) - 0.5 * (t(v, 0) + t(v, 2)), t(w2, 1) + 0.25 * (t(w2, 0) + t(w2, 2))


def linfit(x, y, ey):
    ok = np.isfinite(y) & np.isfinite(ey) & (ey > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    W = 1 / ey[ok] ** 2
    A = np.vstack([np.ones(ok.sum()), x[ok]]).T
    C = np.linalg.inv(A.T @ (A * W[:, None]))
    p = C @ (A.T @ (W * y[ok]))
    return float(p[1]), float(np.sqrt(C[1, 1])), float(p[0])


def mean_profile(v, w2):
    """v, w2: (ntrack_c, mlj) -> means and errors in ntrack bins 1-7 and the slope (GeV / 100 tracks)."""
    ys, es = [], []
    for j in range(1, 8):
        hv, he = v[j], w2[j]
        sw = hv.sum()
        if sw <= 0:
            ys.append(np.nan); es.append(np.nan); continue
        mu = (hv * MLc).sum() / sw
        ys.append(mu); es.append(np.sqrt(max((he * (MLc - mu) ** 2).sum(), 0)) / sw)
    s, e, c = linfit(XS, np.array(ys), np.array(es))
    return {"slope": 100 * s, "err": 100 * e, "icpt": c, "means": ys, "errs": es}


R = {"files": nfiles, "files_per_dir": files_per_dir, "datasets": sorted(acc), "channels": {}}
for ch in ("mujets", "ejets", "emu"):
    if not any(k[0] == ch for k in shapes):
        continue
    C = R["channels"].setdefault(ch, {})
    # ttbar-selected events per N_b (the normalisation)
    nkey = (ch, "nbjet" if KIND[ch] == "semi" else "dil_ntrack")
    nt = {w: total(nkey, w) for w in ["data", "mc", *PROCS]}
    tt_events = {w: (v.sum(axis=1) if v.ndim == 2 else v) for w, (v, _) in nt.items()}  # per N_b
    C["n_tt_events"] = {w: list(map(float, x)) for w, x in tt_events.items()}
    # mass spectra: jpsi_mass (osss, nmed, jpsicat, jpsitruth, nbj, mass); loose = all, medium = both muons medium
    mk = (ch, "jpsi_mass")
    sp_all = {w: total(mk, w) for w in ["data", "mc", *PROCS]}
    sp = {w: (v.sum(axis=1), w2.sum(axis=1)) for w, (v, w2) in sp_all.items()}
    sp_med = {w: (v[:, 2], w2[:, 2]) for w, (v, w2) in sp_all.items()}
    C["mass_edges"] = list(M)
    C["spectra"] = {}
    CATS = {"all": slice(0, 3), "notb": slice(1, 3), "inb": slice(0, 1), "injet": slice(1, 2), "nojet": slice(2, 3)}
    for w in sp:
        for idn, (v, w2) in (("", sp[w]), ("medium_", sp_med[w])):
            for io, sgn in ((0, "os"), (1, "ss")):
                for cn in ("all", "notb", "inb"):
                    for nbname, nbs in (("incl", slice(0, 5)), ("ge1b", slice(1, 5))):
                        k = f"{w}_{idn}{sgn}_{cn}_{nbname}"
                        C["spectra"][k] = list(v[io, CATS[cn], :, nbs].sum(axis=(0, 1, 2)))
                        C["spectra"][f"{k}_err"] = list(np.sqrt(w2[io, CATS[cn], :, nbs].sum(axis=(0, 1, 2))))
    vmc, _ = sp["mc"]
    for it, tn in enumerate(("prompt", "fromb", "nojpsi")):
        for cn in ("all", "notb", "inb"):
            C["spectra"][f"mc_os_{cn}_incl_{tn}"] = list(vmc[0, CATS[cn], it].sum(axis=(0, 1)))
    # yields: OS window - sidebands/2, per category and N_b (>= 1), per ttbar event
    C["yields"] = {}
    for w, idn, (v, w2) in [(w, "", x) for w, x in sp.items()] + [(w, "medium_", x) for w, x in sp_med.items()]:
        for cn, cs in CATS.items():
            for nbname, nbs in (("incl", slice(0, 5)), ("ge1b", slice(1, 5)), ("2b", slice(2, 5))):
                for io, sgn in ((0, "os"), (1, "ss")):
                    row = v[io, cs, :, nbs].sum(axis=(0, 1, 2)); e2 = w2[io, cs, :, nbs].sum(axis=(0, 1, 2))
                    y = row[SIG].sum() - 0.5 * row[SB].sum(); ey = np.sqrt(e2[SIG].sum() + 0.25 * e2[SB].sum())
                    ntt = float(np.sum(tt_events[w][nbs]))
                    C["yields"][f"{w}_{idn}{sgn}_{cn}_{nbname}"] = {"n": float(y), "err": float(ey), "per_tt": float(y / ntt) if ntt > 0 else float("nan"),
                                                               "per_tt_err": float(ey / ntt) if ntt > 0 else float("nan"), "n_tt": ntt}
    # MC truth composition of the subtracted OS signal (all categories, N_b >= 1)
    v, w2 = sp["mc"]
    C["mc_truth_fraction"] = {}
    for ic, cn in enumerate(("inb", "injet", "nojet")):
        vals = []
        for it in range(3):
            row = v[0, ic, it, 1:].sum(axis=0)
            vals.append(row[SIG].sum() - 0.5 * row[SB].sum())
        tot = sum(vals)
        C["mc_truth_fraction"][cn] = {k: float(x / tot) if tot else float("nan") for k, x in zip(("prompt", "fromb", "nojpsi"), vals)}
    # sideband-subtracted shapes, OS, all categories (cat axis summed except z, which is in-jet only), N_b inclusive
    C["shapes"] = {}
    for hn, var in (("jpsi_pt", "pt"), ("jpsi_sip", "sip"), ("jpsi_drj", "drj"), ("jpsi_z", "z")):
        key = (ch, hn)
        C["shapes"][f"{var}_edges"] = edges.get(key, [])
        for w in ("data", "mc", "tt_lj", "tt_ll"):
            v, w2 = total(key, w)
            s, s2 = sbsub(v, w2, 0)  # mreg first: (osss, [cat], truth, var[, absy])
            s, s2 = s[0], s2[0]  # OS
            while s.ndim > 1 and hn != "jpsi_pt":
                s, s2 = s.sum(axis=0), s2.sum(axis=0)
            if hn == "jpsi_pt":
                s, s2 = s.sum(axis=(0, 1, 3)), s2.sum(axis=(0, 1, 3))
            C["shapes"][f"{var}_{w}"] = list(map(float, s))
            C["shapes"][f"{var}_{w}_err"] = list(map(float, np.sqrt(np.maximum(s2, 0))))
    # per category (data, OS, sideband-subtracted, N_b inclusive): max SIP3D with the > 100 overflow as the last bin, and pT;
    # SIP3D < 3: prompt-like (from the ttbar vertex), 3-100: displaced (b decays), > 100: pileup-like
    SC = C.setdefault("shapes_cat", {})
    if (ch, "jpsi_sip") in edges:
        SC["sip_edges"] = edges[(ch, "jpsi_sip")][-1] + [1000.0]
        SC["pt_edges"] = edges[(ch, "jpsi_pt")][-2]
    v, w2 = total((ch, "jpsi_sip_flow"), "data")  # (mreg, osss, cat, truth, [under] sip [over])
    s, s2 = sbsub(v, w2, 0)
    s, s2 = s[0].sum(axis=1)[:, 1:], s2[0].sum(axis=1)[:, 1:]  # OS, truth summed, underflow (empty) dropped
    vp, wp = total((ch, "jpsi_pt"), "data")  # (mreg, osss, cat, truth, pt, absy)
    sp_, sp2 = sbsub(vp, wp, 0)
    sp_, sp2 = sp_[0].sum(axis=(1, 3)), sp2[0].sum(axis=(1, 3))
    SPLIT = {"prompt": slice(0, 3), "displaced": slice(3, -1), "pileup": slice(-1, None)}
    C["sip_split"] = {}
    for cn, cs in CATS.items():
        a, a2 = s[cs].sum(axis=0), s2[cs].sum(axis=0)
        SC[f"sip_{cn}"], SC[f"sip_{cn}_err"] = list(map(float, a)), list(map(float, np.sqrt(np.maximum(a2, 0))))
        b, b2 = sp_[cs].sum(axis=0), sp2[cs].sum(axis=0)
        SC[f"pt_{cn}"], SC[f"pt_{cn}_err"] = list(map(float, b)), list(map(float, np.sqrt(np.maximum(b2, 0))))
        ntt = float(np.sum(tt_events["data"]))
        C["sip_split"][cn] = {k: {"n": float(a[q].sum()), "err": float(np.sqrt(max(a2[q].sum(), 0))),
                                  "per_tt": float(a[q].sum() / ntt) if ntt > 0 else float("nan")} for k, q in SPLIT.items()}
        C["sip_split"][cn]["n_tt"] = ntt
    # ttbar kinematics of the J/psi events vs all ttbar events (N_b >= 1): J/psi hists (mreg, osss, cat, nbj, x),
    # jpsi_tt_mlb (lepflav, mreg, osss, cat, nbj, x)
    T = C.setdefault("ttkin", {})
    def jpsi_shapes(v, w2, name):
        for cn in ("all", "notb", "inb"):
            cs = CATS[cn]
            win = v[1, 0, cs, 1:].sum(axis=(0, 1))  # signal window, OS, raw
            sub, sub2 = sbsub(v, w2, 0)
            a, a2 = sub[0, cs, 1:].sum(axis=(0, 1)), sub2[0, cs, 1:].sum(axis=(0, 1))
            T[f"{name}_{cn}_window"] = list(map(float, win))
            T[f"{name}_{cn}_sub"] = list(map(float, a))
            T[f"{name}_{cn}_sub_err"] = list(map(float, np.sqrt(np.maximum(a2, 0))))
    if KIND[ch] == "semi":
        for var, hn in (("mjj", "jpsi_tt_mjj"), ("mjjb", "jpsi_tt_mjjb")):
            v, w2 = total((ch, hn), "data")
            jpsi_shapes(v, w2, var)
            T[f"{var}_edges"] = edges[(ch, hn)][-1] if (ch, hn) in edges else []
        v, w2 = total((ch, "jpsi_tt_mlb"), "data")
        for il, lep in ((0, "mu"), (1, "e")):
            jpsi_shapes(v[il], w2[il], f"mlb_{lep}")
        T["mlb_edges"] = list(np.linspace(0, 300, 31))
        # all ttbar events: m(jj) (50 bins -> 25), m(jjb), m(l, b_lep) (method 1: closest to the lepton)
        r, _ = total((ch, "hadW_mass_vs_ntrack_nb"), "data")
        T["mjj_ttbar"] = list(map(float, r[1:].sum(axis=(0, 1)).reshape(-1, 2).sum(axis=1)))
        r, _ = total((ch, "hadW_top_mjjb"), "data")
        T["mjjb_ttbar"] = list(map(float, r[:, 1:].sum(axis=(0, 1, 2))))
        r, _ = total((ch, "hadW_lepb_mlb"), "data")
        T["mlb_ttbar"] = list(map(float, r[1, :, :, 1:].sum(axis=(0, 1, 2, 3))))
    else:
        v, w2 = total((ch, "jpsi_tt_mlb"), "data")
        for il, lep in ((0, "mu"), (1, "e")):
            jpsi_shapes(v[il], w2[il], f"mlb_{lep}")
        T["mlb_edges"] = list(np.linspace(0, 300, 31))
        r, _ = total((ch, "dil_mlb_fl"), "data")  # (lepflav, method, btruth, nbj, ntrack, mlb); method 1: minimax
        for il, lep in ((0, "mu"), (1, "e")):
            T[f"mlb_{lep}_ttbar"] = list(map(float, r[il, 1, :, 1:].sum(axis=(0, 1, 2))))
    # m(l, J/psi): jpsi_mlj (lepflav, mreg, osss, rside, tside, nbj, ntrack_c, mlj)
    key = (ch, "jpsi_mlj")
    C["mlj"] = {"edges": list(ML)}
    for w in ("data", "mc", "tt_lj", "tt_ll"):
        v, w2 = total(key, w)
        for il, lep in ((0, "mu"), (1, "e")):
            s, s2 = sbsub(v[il], w2[il], 0)  # -> (osss, rside, tside, nbj, ntrack, mlj)
            s, s2 = s[0], s2[0]  # OS
            # same / other: the J/psi in the b candidate paired with this lepton or the other one (b-decay control);
            # notbcand: in no b candidate (mostly the signal categories)
            for sname, rs in (("bcand", slice(0, 2)), ("same", slice(1, 2)), ("other", slice(0, 1)), ("notbcand", slice(2, 3))):
                a, a2 = s[rs].sum(axis=(0, 1)), s2[rs].sum(axis=(0, 1))  # (nbj, ntrack, mlj)
                a, a2 = a[1:].sum(axis=0), a2[1:].sum(axis=0)  # N_b >= 1
                C["mlj"][f"{w}_{lep}_{sname}"] = list(map(float, a.sum(axis=0)))
                C["mlj"][f"{w}_{lep}_{sname}_err"] = list(map(float, np.sqrt(np.maximum(a2.sum(axis=0), 0))))
                C["mlj"][f"{w}_{lep}_{sname}_vs_ntrk"] = mean_profile(a, a2)
            if w != "data":
                # MC truth side of the reco "same" choice: purity of the pairing
                t = s[1].sum(axis=(1, 2, 3))  # (tside, mlj) for rside == 1
                tot = t.sum()
                C["mlj"][f"{w}_{lep}_same_truth_purity"] = float(t[1].sum() / tot) if tot else float("nan")
    # lifetime regions (ttJ outputs, two-muon vertex): jpsi_mass@life (osss, nmed, cat, truth, nbj, life, mass)
    if (ch, "jpsi_mass@life") in shapes:
        LIFE = {"prompt": 0, "nonprompt": 1, "between": 2, "pileup": 3, "novtx": 4}
        L = C.setdefault("life", {})
        v, w2 = total((ch, "jpsi_mass@life"), "data")
        v, w2 = v.sum(axis=(1, 3)), w2.sum(axis=(1, 3))  # (osss, cat, nbj, life, mass)
        ntt = float(np.sum(tt_events["data"][1:]))
        for ln, li in LIFE.items():
            for cn, cs in CATS.items():
                row, e2 = v[0, cs, 1:, li].sum(axis=(0, 1)), w2[0, cs, 1:, li].sum(axis=(0, 1))  # OS, N_b >= 1
                yv, ey = row[SIG].sum() - 0.5 * row[SB].sum(), np.sqrt(e2[SIG].sum() + 0.25 * e2[SB].sum())
                L[f"yield_{ln}_{cn}"] = {"n": float(yv), "err": float(ey), "per_tt": float(yv / ntt) if ntt else float("nan"),
                                         "per_tt_err": float(ey / ntt) if ntt else float("nan"), "n_tt": ntt}
                if cn in ("all", "notb", "inb"):
                    L[f"spectrum_{ln}_{cn}"] = list(map(float, row))
                    L[f"spectrum_{ln}_{cn}_err"] = list(map(float, np.sqrt(np.maximum(e2, 0))))
            row = v[1, :, 1:, li].sum(axis=(0, 1))
            L[f"ss_{ln}_all"] = float(row[SIG].sum() - 0.5 * row[SB].sum())
        # BPHNano: J/psi-triggered events in which the channel's single-lepton trigger did not / did fire
        if (ch, "jpsi_mass@sltrig") in shapes:
            v2, e2_ = total((ch, "jpsi_mass@sltrig"), "data")  # (osss, nmed, cat, truth, nbj, sltrig, mass)
            v2, e2_ = v2.sum(axis=(1, 3)), e2_.sum(axis=(1, 3))
            for itg, tn in ((0, "no_sltrig"), (1, "sltrig")):
                for cn, cs in CATS.items():
                    row, e2 = v2[0, cs, 1:, itg].sum(axis=(0, 1)), e2_[0, cs, 1:, itg].sum(axis=(0, 1))
                    L[f"yield_{tn}_{cn}"] = {"n": float(row[SIG].sum() - 0.5 * row[SB].sum()), "err": float(np.sqrt(e2[SIG].sum() + 0.25 * e2[SB].sum()))}
        # l_xy, sideband-subtracted: jpsi_lxy (mreg, osss, cat, truth, nbj, z0, lxy); z0 bins 0-2: |z0| < 0.1 cm
        v, w2 = total((ch, "jpsi_lxy"), "data")
        s, s2 = sbsub(v, w2, 0)
        s, s2 = s[0].sum(axis=1), s2[0].sum(axis=1)  # OS, truth summed: (cat, nbj, z0, lxy)
        if (ch, "jpsi_lxy") in edges:
            L["lxy_edges"], L["z0_edges"] = edges[(ch, "jpsi_lxy")][-1], edges[(ch, "jpsi_lxy")][-2]
        for cn in ("all", "notb", "inb"):
            for zn, zs in (("pv", slice(0, 3)), ("pileup", slice(3, None))):
                a, a2 = s[CATS[cn], 1:, zs].sum(axis=(0, 1, 2)), s2[CATS[cn], 1:, zs].sum(axis=(0, 1, 2))
                L[f"lxy_{cn}_{zn}"] = list(map(float, a))
                L[f"lxy_{cn}_{zn}_err"] = list(map(float, np.sqrt(np.maximum(a2, 0))))
        # ttbar kinematics of the prompt and non-prompt J/psi events (jpsi_tt_*@life: ..., nbj, life, x)
        for ln in ("prompt", "nonprompt"):
            li = LIFE[ln]
            if KIND[ch] == "semi":
                for var, hn in (("mjj", "jpsi_tt_mjj"), ("mjjb", "jpsi_tt_mjjb")):
                    v, w2 = total((ch, hn + "@life"), "data")
                    jpsi_shapes(v[..., li, :], w2[..., li, :], f"{var}_{ln}")
            v, w2 = total((ch, "jpsi_tt_mlb@life"), "data")
            for il, lep in ((0, "mu"), (1, "e")):
                jpsi_shapes(v[il][..., li, :], w2[il][..., li, :], f"mlb_{lep}_{ln}")
json.dump(R, open(RES, "w"), indent=1, default=float)

for ch, C in R["channels"].items():
    print(f"== {ch}: ttbar events data {sum(C['n_tt_events']['data']):.0f}, MC {sum(C['n_tt_events']['mc']):.4g} pb")
    for w, idn in (("data", ""), ("data", "medium_"), ("mc", ""), ("mc", "medium_")):
        for cn in ("notb", "injet", "nojet", "inb", "all"):
            y = C["yields"][f"{w}_{idn}os_{cn}_ge1b"]
            ys = C["yields"][f"{w}_{idn}ss_{cn}_ge1b"]
            print(f"   {w:4s} {idn or 'loose_':7s} {cn:5s} N_b>=1: OS J/psi {y['n']:9.2f} +- {y['err']:7.2f}  per tt event {1e4 * y['per_tt']:7.3f} +- {1e4 * y['per_tt_err']:.3f} e-4"
                  f"   (SS {ys['n']:8.2f})")
    print("   MC truth (N_b>=1, OS subtracted):", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in C["mc_truth_fraction"].items()})
    for lep in ("mu", "e"):
        k = f"data_{lep}_same_vs_ntrk"
        if sum(C["mlj"][f"data_{lep}_bcand"]) or sum(C["mlj"][f"data_{lep}_notbcand"]):  # the ttbar lepton(s) of the channel
            d, m = C["mlj"][k], C["mlj"][f"mc_{lep}_same_vs_ntrk"]
            print(f"   <m({lep}, J/psi)> same-side slope: data {d['slope']:+.2f} +- {d['err']:.2f}, MC {m['slope']:+.2f} +- {m['err']:.2f} GeV/100 trk;"
                  f" MC pairing purity {C['mlj'].get(f'mc_{lep}_same_truth_purity', float('nan')):.2f}")
    if "life" in C:
        for ln in ("prompt", "nonprompt", "between", "pileup", "novtx"):
            Y = {cn: C["life"][f"yield_{ln}_{cn}"] for cn in ("all", "inb", "notb")}
            print(f"   {ln:9s} N_b>=1 OS J/psi: " + "  ".join(f"{cn} {y['n']:7.1f} +- {y['err']:5.1f}" for cn, y in Y.items())
                  + f"  (SS all {C['life'][f'ss_{ln}_all']:6.1f}); not-b per tt event {1e4 * Y['notb']['per_tt']:.3f} +- {1e4 * Y['notb']['per_tt_err']:.3f} e-4")
    T = C["ttkin"]
    for var in [k[:-len("_notb_sub")] for k in T if k.endswith("_notb_sub")]:
        ref = T.get(f"{var}_ttbar") or T.get(f"{var.split('_')[0]}_ttbar")
        n_all, n_notb = sum(T[f"{var}_all_sub"]), sum(T[f"{var}_notb_sub"])
        if not (n_all or n_notb):
            continue
        e = np.array(T.get(f"{var.split('_')[0]}_edges") or T.get("mlb_edges")); c = 0.5 * (e[1:] + e[:-1])
        mean = lambda h: float((np.array(h) * c).sum() / max(np.sum(h), 1e-9))  # noqa: E731
        print(f"   {var:7s} J/psi events (sub.) all {n_all:6.1f}  not-b {n_notb:6.1f}; mean all-J/psi {mean(T[f'{var}_all_sub']):6.1f}"
              f"  not-b {mean(T[f'{var}_notb_sub']):6.1f}  all ttbar {mean(ref) if ref else float('nan'):6.1f} GeV")
print("files", nfiles, "->", RES)
