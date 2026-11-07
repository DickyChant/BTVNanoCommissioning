"""Data checks of the veto of the J/psi in the ttbar decay jets (assoc_veto.py), on the sWeighted candidates of splot_prompt.py
(SIGNAL=assoc; l + jets unless noted):
  wpair   the vetoed prompt J/psi in a jet of the selection's W pair (the non-b-tagged pair of the leading four jets closest to
          m_W): m(jj) of that pair, against the same J/psi jets put into other selected events as an extra jet (the selection
          redone: N_b >= 1 of the leading four, the J/psi jet not a b candidate; m(jj) when the jet lands in the W pair) and
          against ttbar events in general (the events of the b-matched J/psi, b-decay sWeights)
  drtt    Delta R(J/psi, nearest ttbar jet of the selection or of the chi2 fit) of the prompt signal J/psi (outside the ttbar
          jets) against the prompt pileup J/psi (another collision: placed at random with respect to the ttbar jets); all channels
  softjet the signal J/psi in no selected jet but inside a jet of pT > 15 GeV without ID or cleaning (sj_dr < 0.4)
-> wpair_check.json   usage: python3 wpair_check.py <sweights.parquet> <out.json>"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

SWQ, OUTJ = sys.argv[1:3]
S = pd.read_parquet(SWQ)
rng = np.random.default_rng(7)
F = ("pt", "eta", "phi", "mass", "isb")
WIN = (65.0, 95.0)


def jets(q):
    return {k: np.stack([q[f"jet{i}_{k}"].to_numpy(float) for i in range(8)], 1) for k in F}


def m2(J, a, b):
    r = np.arange(len(J["pt"]))

    def v(i):
        pt, eta, phi, m = (J[k][r, i] for k in ("pt", "eta", "phi", "mass"))
        return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], 1)
    s = v(a) + v(b)
    return np.sqrt(np.maximum(s[:, 3] ** 2 - (s[:, :3] ** 2).sum(1), 0))


def select(J):
    """the workflow's ttbar jets of l + jets: N_b of the leading four, W pair (i1, i2; -1 if none), m(jj), valid b candidates"""
    n = len(J["pt"])
    ok4 = J["pt"][:, :4] > 0
    b = (J["isb"][:, :4] > 0.5) & ok4
    best, i1, i2, mjj = np.full(n, np.inf), np.full(n, -1), np.full(n, -1), np.full(n, -1.0)
    for a, c in itertools.combinations(range(4), 2):
        use = ok4[:, a] & ok4[:, c] & ~b[:, a] & ~b[:, c]
        m = m2(J, np.full(n, a), np.full(n, c))
        better = use & (np.abs(m - 80.4) < best)
        best, i1, i2, mjj = np.where(better, np.abs(m - 80.4), best), np.where(better, a, i1), np.where(better, c, i2), np.where(better, m, mjj)
    return b.sum(1), i1, i2, mjj, (ok4.sum(1) == 4) & (i1 >= 0)


def wfrac(m, w):
    """weighted fraction of m in WIN with its error (all m(jj))"""
    win = (m > WIN[0]) & (m < WIN[1])
    tot = w.sum()
    f = w[win].sum() / tot
    return float(f), float(np.sqrt((w[win] ** 2).sum() * (1 - f) ** 2 + (w[~win] ** 2).sum() * f ** 2) / abs(tot))


OUT = {"window": WIN}
L = S[S.ch.isin(["mujets", "ejets"])].reset_index(drop=True)
J = jets(L)
dr = np.where(J["pt"] > 0, np.hypot(J["eta"] - L.eta.to_numpy(float)[:, None],
                                     np.abs((J["phi"] - L.phi.to_numpy(float)[:, None] + np.pi) % (2 * np.pi) - np.pi)), 99)
jn = dr.argmin(1)
jidx = np.where(dr[np.arange(len(L)), jn] < 0.4, jn, -1)
nb, i1, i2, mjj, two = select(J)
inW = (jidx >= 0) & ((jidx == i1) | (jidx == i2))
E = np.arange(20.0, 261.0, 20.0)  # display; the last bin takes the overflow
OUT["mjj_edges"] = E.tolist()
q = (L["sample"] == "ttj").to_numpy() & inW
w = L.w_prompt.to_numpy(float)
def ov(x):
    return np.clip(x, E[0], E[-1] - 1e-6)


h = np.histogram(ov(mjj[q]), E, weights=w[q])[0]
h2 = np.histogram(ov(mjj[q]), E, weights=w[q] ** 2)[0]
OUT["wpair"] = {"prompt_in_w": {"h": h.tolist(), "e": np.sqrt(h2).tolist(), "n": float(w[q].sum()), "win": wfrac(mjj[q], w[q])},
                "share_in_w_of_ttj": float(w[q].sum() / w[(L["sample"] == "ttj").to_numpy()].sum())}
# the same J/psi jets as an extra jet in other selected events
ev = L.drop_duplicates(["run", "lumi", "event"]).index.to_numpy()
K = 300
src = np.nonzero(q)[0]
i = np.repeat(src, K)
j = rng.choice(ev, size=len(i))
keep = L.event.to_numpy()[i] != L.event.to_numpy()[j]
i, j = i[keep], j[keep]
A = {k: np.concatenate([J[k][j], J[k][i, jidx[i]][:, None]], 1) for k in F}
A["pt"] = np.where(A["pt"] > 0, A["pt"], -99)
o = np.argsort(-A["pt"], 1, kind="stable")[:, :8]
Jn = {k: np.take_along_axis(A[k], o, 1) for k in F}
inew = np.where((o == 8).any(1), (o == 8).argmax(1), -1)
nb2, a1, a2, mj2, two2 = select(Jn)
passed = (nb2 >= 1) & ~((inew >= 0) & (inew < 4) & two2 & (inew != a1) & (inew != a2))
lw = (inew >= 0) & ((inew == a1) | (inew == a2)) & passed
we = w[i] / K
h = np.histogram(ov(mj2[lw]), E, weights=we[lw])[0]
OUT["wpair"]["extra_jet"] = {"h": h.tolist(), "n": float(we[lw].sum()), "win": wfrac(mj2[lw], we[lw]), "share_in_w": float(we[lw].sum() / we[passed].sum())}
# ttbar events in general: the events of the b-matched J/psi (b decays, sWeights), whose W pair the J/psi does not touch
qb = (L["sample"] == "inb").to_numpy() & (mjj > 0)
wb = L.w_nonprompt.to_numpy(float)
h = np.histogram(ov(mjj[qb]), E, weights=wb[qb])[0]
OUT["wpair"]["all_ttbar"] = {"h": h.tolist(), "n": float(wb[qb].sum()), "win": wfrac(mjj[qb], wb[qb])}
f, fe = OUT["wpair"]["prompt_in_w"]["win"]
fx = OUT["wpair"]["extra_jet"]["win"][0]
OUT["wpair"]["z_vs_extra"] = (f - fx) / fe
print(f"W pair: {OUT['wpair']['prompt_in_w']['n']:.1f} prompt J/psi in a W-pair jet ({100 * OUT['wpair']['share_in_w_of_ttj']:.0f}% of the vetoed ones);"
      f" m(jj) in {WIN[0]:.0f}-{WIN[1]:.0f} GeV: {f:.2f} +- {fe:.2f}, an extra jet {fx:.2f} ({OUT['wpair']['z_vs_extra']:.1f} sigma), ttbar events with a b-decay J/psi {OUT['wpair']['all_ttbar']['win'][0]:.2f};"
      f" an extra jet lands in the W pair {OUT['wpair']['extra_jet']['share_in_w']:.2f} of the time")

# Delta R to the nearest ttbar jet: signal vs pileup (all channels)
DE = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0])
OUT["dr_edges"] = DE.tolist()
OUT["drtt"] = {}
for smp, nm in (("notb", "signal"), ("pileup", "pileup")):
    q = S[S["sample"] == smp]
    x, w = np.clip(q.tt_drmin.to_numpy(float), DE[0], DE[-1] - 1e-6), q.w_prompt.to_numpy(float)
    h, h2 = np.histogram(x, DE, weights=w)[0], np.histogram(x, DE, weights=w ** 2)[0]
    near = x < 0.8
    f = w[near].sum() / w.sum()
    fe = np.sqrt((w[near] ** 2).sum() * (1 - f) ** 2 + (w[~near] ** 2).sum() * f ** 2) / abs(w.sum())
    j3 = (q.jcat == 3).to_numpy()
    sj = j3 & (q.sj_dr.to_numpy(float) < 0.4)
    fs = w[sj].sum() / w.sum()
    fse = np.sqrt((w[sj] ** 2).sum() * (1 - fs) ** 2 + (w[~sj] ** 2).sum() * fs ** 2) / abs(w.sum())
    mf = q.sj_muef.to_numpy(float)
    OUT["drtt"][nm] = {"h": h.tolist(), "e": np.sqrt(h2).tolist(), "n": float(w.sum()), "near_0p8": [float(f), float(fe)],
                       "softjet": [float(fs), float(fse)], "softjet_muef_ge_0p8": float(w[sj & (mf >= 0.8)].sum() / w.sum()),
                       "by_channel": {c: float(w[(q.ch == c).to_numpy()].sum()) for c in ("mujets", "ejets", "emu")}}
    print(f"{nm}: N {w.sum():.1f}; nearest ttbar jet at 0.4-0.8: {f:.2f} +- {fe:.2f}; in no selected jet but inside a pT > 15 GeV jet: {fs:.2f} +- {fse:.2f}"
          f" (of which jets with muon fraction >= 0.8, the J/psi itself: {OUT['drtt'][nm]['softjet_muef_ge_0p8']:.2f})")
json.dump(OUT, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
