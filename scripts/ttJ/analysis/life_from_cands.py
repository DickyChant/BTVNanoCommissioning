"""The J/psi entries of a build_ttJ.py summary (yields, mass spectra, lifetime regions, l_xy, |z0|, pT, ttbar kinematics of the
J/psi events), rebuilt from the per-candidate table (dump_cands.py, with the columns of assoc_veto.py), so that the J/psi
categories can be redefined offline. The ttbar event counts and the all-ttbar distributions stay those of the summary.
Categories: all; inb = b-matched (cat 0); with --signal notb (as the batch summaries): notb = cat != 0, injet / nojet = cat 1 / 2;
with --signal assoc (the search for associated production): notb = the J/psi outside every ttbar decay jet (jcat >= 2),
injet / nojet = jcat 2 / 3, and ttj = the J/psi in the other ttbar jets (jcat 1, mostly the W pair of l + jets).
The bin edges are those of the summary; as in build_ttJ.py: OS, window (2.95-3.25 GeV) - sidebands / 2, N_b >= 1 unless noted.
usage: python3 life_from_cands.py <summary.json> <cands dir> <out.json> [--signal notb|assoc] [--year Y]"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("summary")
ap.add_argument("cands")
ap.add_argument("out")
ap.add_argument("--signal", default="assoc", choices=("notb", "assoc"))
ap.add_argument("--year", type=int, default=None)
a = ap.parse_args()
R = json.load(open(a.summary))
LIFE = {"prompt": 0, "nonprompt": 1, "between": 2, "pileup": 3, "novtx": 4}
NBS = {"incl": 0, "ge1b": 1, "2b": 2}


def cats(d):
    C = {"all": np.ones(len(d), bool), "inb": (d.cat == 0).to_numpy()}
    if a.signal == "notb":
        C.update(notb=(d.cat != 0).to_numpy(), injet=(d.cat == 1).to_numpy(), nojet=(d.cat == 2).to_numpy())
    else:
        j = d.jcat.to_numpy()
        C.update(notb=j >= 2, injet=j == 2, nojet=j == 3, ttj=j == 1)
    return C


def h(x, e, m):
    return np.histogram(np.asarray(x, float)[m], np.asarray(e, float))[0].astype(float)


R["life_from_cands"] = {"signal": a.signal, "cands": a.cands, "year": a.year}
for ch, C in R["channels"].items():
    d = pd.read_parquet(os.path.join(a.cands, f"cands_{ch}.parquet"))
    if a.year is not None:
        d = d[d.year == a.year]
    d = d.reset_index(drop=True)
    CA = cats(d)
    SH = [k for k in CA if k != "injet" and k != "nojet"]  # categories with spectra and l_xy
    ME = np.asarray(C["mass_edges"], float)
    m = d.mass.to_numpy(float)
    win, sb = (m > 2.95) & (m < 3.25), ((m > 2.60) & (m < 2.90)) | ((m > 3.30) & (m < 3.60))
    os_, ss_ = (d.os == 1).to_numpy(), (d.os == -1).to_numpy()
    nb, life = d.nb.to_numpy(), d.life.to_numpy()
    med = (d.nmed == 2).to_numpy()
    ntt_nb = np.asarray(C["n_tt_events"]["data"], float)

    def ysub(mask):
        y, e = float(win[mask].sum() - 0.5 * sb[mask].sum()), float(np.sqrt(win[mask].sum() + 0.25 * sb[mask].sum()))
        return y, e

    def entry(mask, ntt):
        y, e = ysub(mask)
        return {"n": y, "err": e, "per_tt": y / ntt if ntt > 0 else float("nan"), "per_tt_err": e / ntt if ntt > 0 else float("nan"), "n_tt": ntt}

    # spectra and yields
    for idn, sel in (("", np.ones(len(d), bool)), ("medium_", med)):
        for sgn, so in (("os", os_), ("ss", ss_)):
            for nbn, nmin in NBS.items():
                base = sel & so & (nb >= nmin)
                for cn, cm in CA.items():
                    if cn in SH and nbn != "2b":
                        k = f"data_{idn}{sgn}_{cn}_{nbn}"
                        v = h(m, ME, base & cm)
                        C["spectra"][k], C["spectra"][f"{k}_err"] = v.tolist(), np.sqrt(v).tolist()
                    C["yields"][f"data_{idn}{sgn}_{cn}_{nbn}"] = entry(base & cm, float(ntt_nb[nmin:].sum()))
    # lifetime regions
    L = C["life"]
    ntt = float(ntt_nb[1:].sum())
    base = os_ & (nb >= 1)
    for ln, li in LIFE.items():
        for cn, cm in CA.items():
            L[f"yield_{ln}_{cn}"] = entry(base & cm & (life == li), ntt)
            if cn in SH:
                v = h(m, ME, base & cm & (life == li))
                L[f"spectrum_{ln}_{cn}"], L[f"spectrum_{ln}_{cn}_err"] = v.tolist(), np.sqrt(v).tolist()
        mm = ss_ & (nb >= 1) & (life == li)
        L[f"ss_{ln}_all"] = float(win[mm].sum() - 0.5 * sb[mm].sum())
    LE, ZE = np.asarray(L["lxy_edges"], float), np.asarray(L["z0_edges"], float)
    lxy, az = d.lxy.to_numpy(float), np.abs(d.z0.to_numpy(float))
    hasv = life <= 3  # a two-muon vertex
    for cn in SH:
        cm = CA[cn]
        for zn, zm in (("pv", az < 0.1), ("pileup", az >= 0.1)):
            sel = base & cm & zm & hasv
            hw, hs = h(lxy, LE, sel & win), h(lxy, LE, sel & sb)
            L[f"lxy_{cn}_{zn}"], L[f"lxy_{cn}_{zn}_err"] = (hw - 0.5 * hs).tolist(), np.sqrt(hw + 0.25 * hs).tolist()
            L[f"lxy_{cn}_{zn}_win"], L[f"lxy_{cn}_{zn}_sb"] = hw.tolist(), hs.tolist()
        sel = base & cm & hasv
        hw, hs = h(az, ZE, sel & win), h(az, ZE, sel & sb)
        L[f"z0_{cn}"], L[f"z0_{cn}_err"] = (hw - 0.5 * hs).tolist(), np.sqrt(hw + 0.25 * hs).tolist()
    if "pt_edges" in L and len(L["pt_edges"]):
        PE = np.asarray(L["pt_edges"], float)
        for cn in SH:
            for ln, li in LIFE.items():
                sel = os_ & CA[cn] & (life == li)  # N_b inclusive
                hw, hs = h(d.pt, PE, sel & win), h(d.pt, PE, sel & sb)
                L[f"pt_{ln}_{cn}"], L[f"pt_{ln}_{cn}_err"] = (hw - 0.5 * hs).tolist(), np.sqrt(hw + 0.25 * hs).tolist()
    # ttbar kinematics of the J/psi events (N_b >= 1, OS): all lifetimes, and the prompt / non-prompt regions
    T = C["ttkin"]
    var = [("mjj", "tt_mjj", "mjj_edges"), ("mjjb", "tt_mjjb", "mjjb_edges")] if ch != "emu" else []
    var += [(f"mlb_{lep}", f"tt_mlb_{lep}", "mlb_edges") for lep in ("mu", "e")]
    for name, col, ek in var:
        if col not in d or not len(T.get(ek, [])):
            continue
        E = np.asarray(T[ek], float)
        x = d[col].to_numpy(float)
        for lsfx, lm in (("", np.ones(len(d), bool)), ("_prompt", life == 0), ("_nonprompt", life == 1)):
            for cn in SH:
                sel = base & CA[cn] & lm
                hw, hs = h(x, E, sel & win), h(x, E, sel & sb)
                T[f"{name}{lsfx}_{cn}_window"] = hw.tolist()
                T[f"{name}{lsfx}_{cn}_sub"], T[f"{name}{lsfx}_{cn}_sub_err"] = (hw - 0.5 * hs).tolist(), np.sqrt(hw + 0.25 * hs).tolist()
json.dump(R, open(a.out, "w"), indent=1, default=float)
print("wrote", a.out, "signal", a.signal, "year", a.year)
