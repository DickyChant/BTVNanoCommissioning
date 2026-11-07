"""The signal of associated production: a J/psi that is not part of either top decay. Per candidate, the J/psi must be
Delta R > 0.4 from every jet the ttbar decay can be built from:
    the b jets          the b candidates of the workflow and every UParT-M tagged selected jet (cat != 0, as before)
    the selection       l + jets: the four leading selected jets (the W pair and the two b candidates of the workflow);
                        e mu: the two jets with the highest b score (the b candidates)
    the chi2 fit        the four jets (l + jets) or two b candidates (e mu) of the full ttbar reconstruction over the eight
                        leading jets with the J/psi jet allowed (ttreco.py, as the "all" version of ttbar_balance.py)
Columns added to the candidate table:
    tt_sel, tt_chi2     the J/psi within 0.4 of a jet of the selection / of the chi2 assignment
    ttjet               cat == 0 or tt_sel or tt_chi2: the J/psi in a ttbar decay jet (vetoed)
    jcat                0 in a b jet (cat 0), 1 in another ttbar jet (W pair, or a jet of the chi2 fit), 2 in a selected jet
                        outside the ttbar, 3 in no selected jet; the signal is jcat >= 2
    tt_drmin            Delta R(J/psi, nearest ttbar jet of the selection or the chi2 fit)
    tt_chi2_x           a J/psi in no selected jet: the chi2 fit takes the J/psi system (the jet of pT > 15 GeV around it without
                        ID or cleaning, sj_*, else the J/psi itself) for a ttbar decay product when it is offered as one more jet:
                        a J/psi from a W quark whose jet is soft or fails the ID, which the jet vetoes miss
    jcat_x              jcat with tt_chi2_x also vetoed (1)
usage: python3 assoc_veto.py <cands dir> <out dir>"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttreco import CH, DR_JET, NJ, add_jets, dphi, inputs, solve  # noqa: E402


def flags(d, ch):
    """the veto columns (see above) of a candidate table d of channel ch"""
    n = len(d)
    J, jp, jidx, L, mx, my = inputs(d, ch)
    dr = np.where(J["ok"], np.hypot(J["eta"] - d.eta.to_numpy(float)[:, None], dphi(J["phi"], d.phi.to_numpy(float)[:, None])), 99.0)
    r = np.arange(n)
    if ch == "emu":
        key = np.where(J["ok"], J["isb"] * 10.0 + J["btag"], -np.inf)
        sel = np.argsort(-key, axis=1)[:, :2]
        sel = np.where(np.isfinite(np.take_along_axis(key, sel, 1)), sel, -1)
        chi = sel  # the neutrino weighting takes the same two b candidates (ttreco.reco_emu)
    else:
        sel = np.where(J["ok"][:, :4], np.arange(4)[None, :], -1)
        R = solve(J, jp, jidx, L, mx, my, ch, False)
        chi = np.where(R["ok"][:, None], R["idx"], -1)

    def near(idx):
        return np.where(idx >= 0, dr[r[:, None], np.maximum(idx, 0)], 99.0).min(1)

    ds, dc = near(sel), near(chi)
    cat = d.cat.to_numpy()
    out = {"tt_sel": ds < DR_JET, "tt_chi2": dc < DR_JET, "tt_drmin": np.minimum(ds, dc)}
    out["ttjet"] = (cat == 0) | out["tt_sel"] | out["tt_chi2"]
    out["jcat"] = np.where(cat == 0, 0, np.where(out["ttjet"], 1, np.where(cat == 1, 2, 3)))
    # the J/psi system as one more jet for the chi2 fit (l + jets: a W quark or, with a single b tag, a b; e mu: a b candidate
    # only if it outranks the b-tagged jets, which it cannot)
    out["tt_chi2_x"] = np.zeros(n, bool)
    if ch != "emu":
        soft = (d.sj_dr.to_numpy(float) >= 0) & (d.sj_dr.to_numpy(float) < DR_JET)
        xpt = np.where(soft, np.maximum(d.sj_pt.to_numpy(float), d.pt.to_numpy(float)), d.pt.to_numpy(float))
        X = {"pt": xpt[:, None], "eta": d.eta.to_numpy(float)[:, None], "phi": d.phi.to_numpy(float)[:, None],
             "mass": np.where(soft, 0.15 * xpt, 3.0969)[:, None], "btag": np.zeros((n, 1)), "isb": np.zeros((n, 1))}
        Jx, ix = add_jets(J, X)
        Rx = solve(Jx, jp, np.full(n, -1), L, mx, my, ch, False)
        took = Rx["ok"] & (ix >= 0) & (Rx["idx"] == ix[:, None]).any(1)
        out["tt_chi2_x"] = took & (out["jcat"] == 3)
    out["jcat_x"] = np.where(out["tt_chi2_x"], 1, out["jcat"])
    return out


if __name__ == "__main__":
    assert NJ == 8
    CD, OD = sys.argv[1:3]
    os.makedirs(OD, exist_ok=True)
    for ch in CH:
        d = pd.read_parquet(os.path.join(CD, f"cands_{ch}.parquet")).reset_index(drop=True)
        n = len(d)
        for k, a in flags(d, ch).items():
            d[k] = a
        d.to_parquet(os.path.join(OD, f"cands_{ch}.parquet"), index=False)
        pv = (d.os == 1) & (d.nb >= 1) & (d.life <= 2) & (d.mass > 2.95) & (d.mass < 3.25)
        print(ch, n, "candidates; signal window, PV, OS:", {int(k): int(v) for k, v in d[pv].jcat.value_counts().sort_index().items()},
              "| cat 1 (in a selected non-b jet) vetoed:", int((pv & (d.cat == 1) & d.ttjet).sum()), "of", int((pv & (d.cat == 1)).sum()),
              "| cat 2 (in no jet) vetoed:", int((pv & (d.cat == 2) & d.ttjet).sum()))
    print("wrote", OD)
