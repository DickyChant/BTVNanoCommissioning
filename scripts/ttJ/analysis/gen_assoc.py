"""Does the veto of the J/psi in the ttbar decay jets (assoc_veto.py) remove the J/psi of the top decays? Generator-level check
on Pythia8 ttbar l + jets with J/psi from the parton shower (LETO; genstudy/gen_ttjpsi.py), where the origin of each J/psi is
known: 1 top decay (the shower of the b or of a W quark), 2 radiated by a top before its decay, 3 initial-state radiation,
4 a second parton scattering, 5 other.
Selection close to the data: the W lepton (mu pT > 30 GeV, |eta| < 2.4; e pT > 35 GeV, |eta| < 2.5); gen jets with pT > 20 GeV,
|eta| < 2.5, muon energy fraction < 0.8 (the jet ID) and Delta R > 0.4 from the lepton, at least four; b tag (UParT
medium-like): 80% for a jet with a b hadron, 15% with a c hadron, 1% otherwise, N_b >= 1 among the leading four; a J/psi with
both muons pT > 3 GeV, |eta| < 2.4 (prompt; the b-decay J/psi are kept in the table, column prompt); MET = the neutrino sum
smeared by 20 GeV per component. The J/psi categories are those of the workflow (jpsi.jet_association: b candidates, b-tagged
jets) and of assoc_veto.flags, the same code as on data: jcat 0 b jet, 1 another ttbar jet, 2 a selected jet outside the ttbar,
3 no selected jet (signal: jcat >= 2).
-> gen_assoc.json   usage: python3 gen_assoc.py <out.json> <npz files>"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assoc_veto import flags  # noqa: E402
from ttreco import DR_JET, NJ, dphi  # noqa: E402

OUTJ, FILES = sys.argv[1], sys.argv[2:]
rng = np.random.default_rng(11)
ORIG = {1: "top decay", 2: "top radiation", 3: "ISR", 4: "MPI", 5: "other"}
JCAT = {0: "b jet", 1: "other ttbar jet", 2: "jet outside the ttbar", 3: "no selected jet"}


def m2(a, b):
    """invariant mass of two jets given (pt, eta, phi, mass) rows"""
    def v(x):
        pt, eta, phi, m = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1)
    s = v(a) + v(b)
    return np.sqrt(np.maximum(s[..., 3] ** 2 - (s[..., :3] ** 2).sum(-1), 0))


rows, ngen, nsel_ev, ldme = [], 0, 0, set()
for f in FILES:
    z = np.load(f)
    ngen += int(z["nev"])
    ldme.add(float(z["ldme"]))
    J, LEP, MET, JP = z["jets"], z["lep"], z["met"], z["jpsi"]
    n = len(J)
    lid = np.abs(LEP[:, 3]).astype(int)
    lok = np.where(lid == 13, (LEP[:, 0] > 30) & (np.abs(LEP[:, 1]) < 2.4), (LEP[:, 0] > 35) & (np.abs(LEP[:, 1]) < 2.5))
    drl = np.hypot(J[:, :, 1] - LEP[:, None, 1], dphi(J[:, :, 2], LEP[:, None, 2]))
    sel = (J[:, :, 0] > 20) & (np.abs(J[:, :, 1]) <= 2.5) & (J[:, :, 4] < 0.8) & (drl > 0.4)
    pb = np.where(J[:, :, 5] > 0, 0.80, np.where(J[:, :, 6] > 0, 0.15, 0.01))
    tag = (rng.random(pb.shape) < pb) & sel
    score = np.where(tag, 0.5 + 0.5 * rng.random(pb.shape), 0.5 * rng.random(pb.shape))
    for e in range(n):
        s = np.nonzero(sel[e])[0]
        if not lok[e] or len(s) < 4:
            continue
        js, tg = J[e, s], tag[e, s]
        if tg[:4].sum() < 1:
            continue
        nsel_ev += 1
        # the workflow's W pair and b candidates (leading four)
        light = [k for k in range(4) if not tg[k]]
        best, pair = np.inf, None
        for a, b in itertools.combinations(light, 2):
            m = m2(js[a, :4], js[b, :4])
            if abs(m - 80.4) < best:
                best, pair = abs(m - 80.4), (a, b)
        others = [k for k in range(4) if pair is None or k not in pair]
        bc = others if (pair is not None and len(others) == 2) else []
        mjj_sel = float(m2(js[pair[0], :4], js[pair[1], :4])) if pair is not None else -1.0
        ntop4 = int((js[:4, 7] > 0).sum())
        mx, my = MET[e, 0] + rng.normal(0, 20), MET[e, 1] + rng.normal(0, 20)
        for q in range(JP.shape[1]):
            p = JP[e, q]
            if p[0] <= 0 or p[3] < 3 or p[5] < 3 or abs(p[4]) > 2.4 or abs(p[6]) > 2.4:
                continue
            dr = np.hypot(js[:, 1] - p[1], dphi(js[:, 2], p[2]))
            in_b = (len(bc) and dr[bc].min() < DR_JET) or (tg.any() and dr[tg].min() < DR_JET)
            in_any = dr.min() < DR_JET
            # the J/psi inside any gen jet above 15 GeV (no ID): as sj_dr of the data
            dall = np.hypot(J[e, :, 1] - p[1], dphi(J[e, :, 2], p[2]))
            dall = np.where(J[e, :, 0] > 15, dall, 99)
            # the nearest jet of pT > 15 GeV and |eta| < 2.5 without ID or cleaning (the sj_* columns of the data)
            dsj = np.where(np.abs(J[e, :, 1]) < 2.5, dall, 99)
            isj = int(np.argmin(dsj))
            jn = int(np.argmin(dr))
            r = {"origin": int(p[8]), "prompt": bool(p[7] > 0.5), "drj": float(dr[jn]), "z": float(p[0] / js[jn, 0]) if in_any else -1.0, "cat": 0 if in_b else (1 if in_any else 2), "pt": float(p[0]), "eta": float(p[1]), "phi": float(p[2]),
                 "mass": 3.0969, "puppimet": float(np.hypot(mx, my)), "puppimet_phi": float(np.arctan2(my, mx)), "njets": len(s),
                 "sj_in": bool(dall.min() < 0.4), "near_top": bool((J[e, :, 7] > 0)[np.argmin(dall)] and dall.min() < 0.4),
                 "ch": "mujets" if lid[e] == 13 else "ejets",
                 "sj_dr": float(dsj[isj]) if dsj[isj] < 90 else -1.0, "sj_pt": float(J[e, isj, 0]) if dsj[isj] < 90 else -1.0,
                 "tt_mjj": mjj_sel, "ntop4": ntop4, "sj_top": bool(J[e, isj, 7] > 0) if dsj[isj] < 90 else False}
            lep = "mu" if lid[e] == 13 else "e"
            r[f"{lep}_pt"], r[f"{lep}_eta"], r[f"{lep}_phi"] = (float(x) for x in LEP[e, :3])
            for k in range(NJ):
                ok = k < len(js)
                for fi, nm in enumerate(("pt", "eta", "phi", "mass")):
                    r[f"jet{k}_{nm}"] = float(js[k, fi]) if ok else -1.0
                r[f"jet{k}_btag"] = float(score[e, s[k]]) if ok else -1.0
                r[f"jet{k}_isb"] = int(tg[k]) if ok else 0
            rows.append(r)
D = pd.DataFrame(rows)
for ch in ("mujets", "ejets"):
    m = (D.ch == ch).to_numpy()
    d = D[m].reset_index(drop=True)
    for k, a in flags(d, ch).items():
        D.loc[m, k] = a
D["jcat"], D["jcat_x"] = D.jcat.astype(int), D.jcat_x.astype(int)
# the statistics below: prompt J/psi; the table keeps the b-decay J/psi as well (the normalisation of gen_compare.py)
DA = D
D = DA[DA.prompt].reset_index(drop=True)
OUT = {"ldme_fac": sorted(ldme), "n_nonprompt": int((~DA.prompt).sum()), "n_generated": ngen, "n_stored_events_selected": nsel_ev, "n_jpsi": len(D), "origin": ORIG, "jcat": JCAT}
print(f"{ngen} generated ttbar (l + jets); {nsel_ev} selected events with a J/psi; {len(D)} prompt J/psi -> mu mu in acceptance ({int((~DA.prompt).sum())} from b decays)")
for col, sfx in (("jcat", ""), ("jcat_x", "_x")):
    print(f"-- veto {col}")
    OUT[f"table{sfx}"], OUT[f"frac_by_origin{sfx}"], OUT[f"frac_by_jcat{sfx}"] = {}, {}, {}
    for o, on in ORIG.items():
        q = D[D.origin == o]
        if not len(q):
            continue
        c = q[col].value_counts()
        OUT[f"table{sfx}"][on] = {JCAT[j]: int(c.get(j, 0)) for j in JCAT}
        OUT[f"frac_by_origin{sfx}"][on] = {JCAT[j]: float(c.get(j, 0) / len(q)) for j in JCAT}
        print(f"{on:14s} {len(q):5d}: " + ", ".join(f"{JCAT[j]} {c.get(j, 0) / len(q):.3f}" for j in JCAT))
    for j, jn in JCAT.items():
        q = D[D[col] == j]
        if len(q):
            c = q.origin.value_counts()
            OUT[f"frac_by_jcat{sfx}"][jn] = {ORIG[o]: float(c.get(o, 0) / len(q)) for o in ORIG}
    sig = D[D[col] >= 2]
    td = D[D.origin == 1]
    k_ = int((td[col] >= 2).sum())
    isr = D[D.origin == 3]
    OUT[f"leak{sfx}"] = {"top_decay_passing": k_, "top_decay_total": len(td), "eff": k_ / max(len(td), 1),
                         "isr_eff": float((isr[col] >= 2).mean()) if len(isr) else None,
                         "signal_total": len(sig), "signal_top_decay_share": float((sig.origin == 1).mean()) if len(sig) else None,
                         "signal_by_origin": {ORIG[o]: int((sig.origin == o).sum()) for o in ORIG},
                         "leaked": {"in no selected jet, inside a gen jet": int(((td[col] == 3) & td.sj_in).sum()), "in no jet at all": int(((td[col] == 3) & ~td.sj_in).sum()),
                                    "in a selected jet outside the ttbar": int((td[col] == 2).sum())}}
    OUT[f"ratio_ttjet_to_signal{sfx}"] = float((D[col] == 1).sum() / max(len(sig), 1))
    L_ = OUT[f"leak{sfx}"]
    print(f"top-decay J/psi passing: {k_} of {len(td)} ({L_['eff']:.3f}); ISR passing {L_['isr_eff']:.3f}; top-decay share of the signal {L_['signal_top_decay_share']:.3f}"
          f" ({len(sig)}: {L_['signal_by_origin']}); leaked: {L_['leaked']}; vetoed other-ttbar-jet / signal {OUT[f'ratio_ttjet_to_signal{sfx}']:.2f}")
OUT["leak"]["leak_per_ttj"] = float(((D.origin == 1) & (D.jcat >= 2)).sum() / max(((D.origin == 1) & (D.jcat == 1)).sum(), 1))
sig = D[D.jcat >= 2]
OUT["signal_inside_unselected_jet"] = float(((sig.jcat == 3) & sig.sj_in).mean()) if len(sig) else None
if os.environ.get("GEN_TABLE"):  # the per-J/psi table, for checks
    DA.to_parquet(os.environ["GEN_TABLE"], index=False)
json.dump(OUT, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
