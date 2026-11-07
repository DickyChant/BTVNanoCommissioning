"""Is the prompt J/psi of the ttbar events part of the ttbar collision (ttbar + g, g -> J/psi + X), or from a second parton
scattering (DPS) or another collision? The ttbar system is rebuilt per candidate from the stored selected jets (candidate
table of ttJ_pixi 79eecfe):
    l + jets  t_lep = l + nu + b_l, nu pT = PUPPI MET, pz from m(l nu) = m_W (the real part if there is no solution);
              t_had = q q' + b_h; the four jets from the minimum of
              chi2 = ((m_qq - m_W) / 10)^2 + ((m_qqb - m_t) / 15)^2 + ((m_lnub - m_t) / 20)^2  [GeV],
              the b candidates being the b-tagged jets (UParT medium) where there are any
    e mu      b candidates: the two jets with the highest (medium tag, UParT score); both neutrinos from m_W and m_t
              (neutrino weighting: eta of each neutrino on a grid, the two solutions of the mass constraints at each eta,
              weight exp(-|MET - nu1 - nu2|^2 / 2 (25 GeV)^2) x a Gaussian prior on the etas; both l-b pairings)
Two versions: "excl", the jet holding the J/psi (Delta R < 0.4, nearest) is left out of the ttbar system (it is then part of
the recoil), and "all", every selected jet may be taken (to see which jet holds the J/psi).
The ttbar pT does not need the neutrino pz: pT(ttbar) = pT(leptons) + MET + pT(b and W jets) = -(everything else in the
event). In a single collision with ttbar + g -> J/psi + X, the J/psi and its jet carry the recoil: Delta phi(J/psi, ttbar)
near pi and pT(ttbar) . pT-hat(J/psi) ~ -pT(J/psi jet) < 0. A J/psi from a second scattering comes with its own recoil (both
in "everything else"): Delta phi flat and <pT(ttbar) . pT-hat(J/psi)> = 0. (An angle to the plane of t and tbar in the rest
frame of t + tbar + J/psi jet is zero by construction, so the transverse balance is the test.)
Samples (sWeights from splot_prompt.py): prompt and b-decay J/psi not b-matched, b-decay J/psi b-matched (the J/psi jet must
come out as a b of the ttbar), prompt pileup J/psi (|z0| >= 0.1 cm: another collision; shown with the J/psi taken out of
the MET, see VARIANTS), and event mixing (the J/psi of a
not-b-matched candidate against the ttbar system of another event of the same channel: no correlation by construction).
-> ttbar_balance.json: sWeighted histograms per sample and channel, the <pT(ttbar) . pT-hat(J/psi)> profile in pT(J/psi), the
role of the J/psi jet, reconstruction efficiencies.
usage: python3 ttbar_balance.py <sweights.parquet> <out.json>"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttreco import CH, DR_JET, MT, MW, SIG_ETA, SIG_MET, ST_H, ST_L, SW_, add_jets, dphi, inputs, observables, rapidity, reconstruct, solve  # noqa: E402

SWQ, OUTJ = sys.argv[1:3]
# (leave the J/psi jet out, MET, take the J/psi out of the MET): PUPPI MET nominal; PF MET as a check; "exclnoj" for the
# pileup J/psi, whose muons enter the MET (PUPPI and PF alike) while the recoil from their collision does not, which puts
# them at <pT(ttbar) . pT-hat(J/psi)> ~ -pT(J/psi) without any correlation
VARIANTS = {"excl": (True, "puppimet", False), "all": (False, "puppimet", False), "exclpf": (True, "met", False), "exclnoj": (True, "puppimet", True)}
# TTBAL_SEED=<n>: another seed (the spread of the emulations)
rng = np.random.default_rng(int(os.environ.get("TTBAL_SEED", 20261107)))

# ---------------- reconstruction per channel and sample
S = pd.read_parquet(SWQ)
S = S[S.ch.isin(CH)].reset_index(drop=True)
SAMPLES = {"prompt": ("notb", "w_prompt"), "nonprompt": ("notb", "w_nonprompt"), "inb": ("inb", "w_nonprompt"), "pileup": ("pileup", "w_prompt")}
# with the signal outside the ttbar decay jets (splot_prompt.py SIGNAL=assoc), the vetoed prompt J/psi of the other ttbar jets
ASSOC = "ttj" in set(S["sample"])
if ASSOC:
    SAMPLES["ttj"] = ("ttj", "w_prompt")
rows = {}
eff = {}
for ch in CH:
    d = S[S.ch == ch].reset_index(drop=True)
    if not len(d):
        continue
    jpt, jphi = d.pt.to_numpy(float), d.phi.to_numpy(float)
    R = {v: reconstruct(d, ch, ex, mc, nj) for v, (ex, mc, nj) in VARIANTS.items()}
    jy = rapidity(R["excl"]["jp"])
    for v in R:
        ob = observables(R[v], jpt, jphi, jy)
        for k, a in ob.items():
            d[f"{k}_{v}"] = a
        d[f"ok_{v}"] = R[v]["ok"]
        d[f"role_{v}"] = R[v]["role"]
    # njets: the stored count of the table (the reconstruction sees the leading NJ jets only)
    d["holder_pt"], d["injet"] = R["excl"]["holder_pt"], R["excl"]["jidx"] >= 0
    rows[ch] = (d, R)
    for s, (smp, wc) in SAMPLES.items():
        q = d[d["sample"] == smp]
        w = q[wc].to_numpy(float)
        eff[f"{s}_{ch}"] = {v: float((w * q[f"ok_{v}"]).sum() / w.sum()) for v in VARIANTS}
    print(ch, len(d), "candidates; reconstructed (excl / all):", {s: tuple(round(eff[f"{s}_{ch}"][v], 3) for v in ("excl", "all")) for s in SAMPLES})

# ---------------- event mixing: the J/psi of each not-b-matched candidate against the ttbar system ("all") of K other events
K = 100
mix = {}
for ch, (d, R) in rows.items():
    ev = d.drop_duplicates(["run", "lumi", "event"]).index.to_numpy()
    ev = ev[d.loc[ev, "ok_all"].to_numpy()]
    q = d[d["sample"] == "notb"]
    i = np.repeat(q.index.to_numpy(), K)
    j = rng.choice(ev, size=len(i))
    same = (d.loc[i, "event"].to_numpy() == d.loc[j, "event"].to_numpy()) & (d.loc[i, "run"].to_numpy() == d.loc[j, "run"].to_numpy())
    i, j = i[~same], j[~same]
    Rm = {k: R["all"][k][j] for k in ("ttx", "tty", "tt")}
    ob = observables(Rm, d.pt.to_numpy(float)[i], d.phi.to_numpy(float)[i], rapidity(R["all"]["jp"][i]))
    mix[ch] = pd.DataFrame({**{f"{k}_excl": a for k, a in ob.items()}, "pt": d.pt.to_numpy(float)[i], "holder_pt": d.holder_pt.to_numpy(float)[i],
                            "w_mix": d.w_prompt.to_numpy(float)[i] / K, "ok_excl": True, "sample": "mix", "ch": ch})

# ---------------- the two hypotheses, emulated in real ttbar events: the jet holding each not-b-matched J/psi (or the J/psi
# itself when it is in no jet) is put into KE x NREP other selected events of the channel, which are then reconstructed as the data
#   same collision ("sps"): the ttbar system recoils against it, MET -> MET - pT(holder)
#   second scattering ("dps"): with a partner that balances it (pT x exp(G(0, 0.2)), phi + pi + G(0, 0.15), the eta of the
#   holder of another candidate; a selected jet if pT > 20 GeV and |eta| < 2.5), MET unchanged
# The partner can be taken for a ttbar jet, which pulls pT(ttbar) away from the J/psi also without a common collision.
# With the signal outside the ttbar decay jets (ASSOC), the new event must pass the selection of the signal again: N_b >= 1
# (l + jets: among the leading four jets; e mu: any b tag) and the J/psi Delta R > 0.4 from every b-tagged jet, from the ttbar
# jets of the selection (l + jets: the leading four; e mu: the two highest b scores) and from the jets of the chi2 fit.
# Each embedded event brings its own ttbar pT (a spread of about 120 GeV along the J/psi), so the templates need many events
# per J/psi jet: NREP rounds of KE (the memory of one round); with 10 in all, the z values below move by about 0.4 between seeds.
KE, NREP = 100, 4
emb = {}
emb_pass = {}
for ch, (d, R) in rows.items():
    ev = d.drop_duplicates(["run", "lumi", "event"]).index.to_numpy()
    ev = ev[d.loc[ev, "ok_all"].to_numpy()]
    q = d[d["sample"] == "notb"]
    J0, jp0, jidx0, L0, mx0, my0 = inputs(d, ch)
    r0 = np.arange(len(d))
    heta0 = np.where(jidx0 >= 0, J0["eta"][r0, np.maximum(jidx0, 0)], d.eta.to_numpy(float))
    parts = {"sps": [], "dps": []}
    npass = {"sps": [0.0, 0.0], "dps": [0.0, 0.0]}
    for rep in range(NREP):
        i = np.repeat(q.index.to_numpy(), KE)
        j = rng.choice(ev, size=len(i))
        keep = ~((d.event.to_numpy()[i] == d.event.to_numpy()[j]) & (d.run.to_numpy()[i] == d.run.to_numpy()[j]))
        i, j = i[keep], j[keep]
        injet = jidx0[i] >= 0
        H = {k: np.where(injet, J0[k][i, np.maximum(jidx0[i], 0)], -99.0)[:, None] for k in ("pt", "eta", "phi", "mass", "btag", "isb")}
        hold = R["excl"]["holder"][i]
        hpt, hphi = np.hypot(hold[:, 0], hold[:, 1]), np.arctan2(hold[:, 1], hold[:, 0])
        n_ = len(i)
        ppt = hpt * np.exp(rng.normal(0, 0.2, n_))
        pphi = (hphi + rng.normal(0, 0.15, n_)) % (2 * np.pi) - np.pi
        peta = heta0[rng.choice(q.index.to_numpy(), n_)]
        pok = (ppt > 20) & (np.abs(peta) < 2.5)
        P = {"pt": np.where(pok, ppt, -99.0)[:, None], "eta": peta[:, None], "phi": pphi[:, None], "mass": (0.12 * ppt)[:, None],
             "btag": np.zeros((n_, 1)), "isb": np.zeros((n_, 1))}
        Jj, Lj, jpi = {k: a[j] for k, a in J0.items()}, [l_[j] for l_ in L0], jp0[i]
        for hyp in ("sps", "dps"):
            Jn, inew = add_jets(Jj, H if hyp == "sps" else {k: np.concatenate([H[k], P[k]], 1) for k in H})
            inew = np.where(injet, inew, -1)
            mx, my = (mx0[j] - hold[:, 0], my0[j] - hold[:, 1]) if hyp == "sps" else (mx0[j], my0[j])
            Rx, Ra = solve(Jn, jpi, inew, Lj, mx, my, ch, True), solve(Jn, jpi, inew, Lj, mx, my, ch, False)
            ob = observables(Rx, d.pt.to_numpy(float)[i], d.phi.to_numpy(float)[i], rapidity(jpi))
            w_ = d.w_prompt.to_numpy(float)[i] / (KE * NREP)
            sel = np.ones(len(i), bool)
            if ASSOC:
                drn = np.where(Jn["ok"], np.hypot(Jn["eta"] - d.eta.to_numpy(float)[i][:, None], dphi(Jn["phi"], d.phi.to_numpy(float)[i][:, None])), 99.0)
                r_ = np.arange(len(i))
                if ch == "emu":
                    key = np.where(Jn["ok"], Jn["isb"] * 10.0 + Jn["btag"], -np.inf)
                    tt = np.argsort(-key, axis=1)[:, :2]
                    tt = np.where(np.isfinite(np.take_along_axis(key, tt, 1)), tt, -1)
                    nbok = Jn["isb"].any(1)
                else:
                    tt = np.where(Jn["ok"][:, :4], np.arange(4)[None, :], -1)
                    nbok = Jn["isb"][:, :4].any(1)
                chi = np.where(Ra["ok"][:, None], Ra["idx"], -1)
                near = lambda idx: np.where(idx >= 0, drn[r_[:, None], np.maximum(idx, 0)], 99.0).min(1)  # noqa: E731
                sel = nbok & (near(tt) >= DR_JET) & (near(chi) >= DR_JET) & ~((drn < DR_JET) & Jn["isb"]).any(1)
            npass[hyp][0] += float(w_[sel].sum())
            npass[hyp][1] += float(w_.sum())
            parts[hyp].append(pd.DataFrame({**{f"{k}_excl": a for k, a in ob.items()}, "ok_excl": Rx["ok"], "ok_all": Ra["ok"], "role_all": Ra["role"],
                                            "njets": d.njets.to_numpy()[j] + injet + (pok if hyp == "dps" else 0), "pt": d.pt.to_numpy(float)[i], "holder_pt": hpt,
                                            "w_emb": w_, "ch": ch})[sel])
    emb[ch] = {h: pd.concat(parts[h], ignore_index=True) for h in parts}
    emb_pass[ch] = {h: npass[h][0] / npass[h][1] for h in npass}
    print(ch, "embedded events passing the selection of the signal:", {h: round(v, 3) for h, v in emb_pass[ch].items()})

# ---------------- histograms
VARS = {
    "dphi": (np.linspace(0, np.pi, 9), "|Δφ(J/ψ, t̅t)|"),
    "ppar": (np.array([-150, -100, -70, -50, -30, -15, 0, 15, 30, 50, 70, 100, 150.]), "p<sub>T</sub>(t̅t) · p̂<sub>T</sub>(J/ψ) [GeV]"),
    "pperp": (np.array([-150, -100, -70, -50, -30, -15, 0, 15, 30, 50, 70, 100, 150.]), "p<sub>T</sub>(t̅t) × p̂<sub>T</sub>(J/ψ) [GeV]"),
    "ptt": (np.array([0, 15, 30, 45, 60, 80, 100, 130, 170, 250.]), "p<sub>T</sub>(t̅t) [GeV]"),
    "dy": (np.arange(0, 4.01, 0.5), "|Δy(J/ψ, t̅t)|"),
    "njets": (np.arange(1.5, 10.5), "selected jets"),
}
ROLES = {"ljets": ["none", "b_had", "b_lep", "W jet", "extra"], "emu": ["none", "b", "extra"]}
PTB = np.array([3, 8.5, 10, 12, 100.])  # about equal prompt sWeights per bin (the J/psi outside the ttbar jets: mean pT 8 GeV)
OUT = {"vars": {v: {"edges": e.tolist(), "label": lab} for v, (e, lab) in VARS.items()}, "pt_edges": PTB.tolist(), "eff": eff, "hists": {}, "profile": {},
       "roles": {}, "settings": {"mW": MW, "mt": MT, "chi2_sigmas": [SW_, ST_H, ST_L], "sigma_met": SIG_MET, "eta_prior": SIG_ETA, "nmix": K, "nemb": KE * NREP, "dr_jet": DR_JET},
       "signal": "assoc" if ASSOC else "notb", "emb_pass": emb_pass}


def hist(x, w, e, clip=True):
    x = np.clip(x, e[0] + 1e-9, e[-1] - 1e-9) if clip else x
    return np.histogram(x, e, weights=w)[0].tolist(), np.sqrt(np.histogram(x, e, weights=w ** 2)[0]).tolist()


def wmean(x, w):
    sw = w.sum()
    if abs(sw) < 1e-9:
        return float("nan"), float("nan")
    m = (w * x).sum() / sw
    return float(m), float(np.sqrt((w ** 2 * (x - m) ** 2).sum()) / abs(sw))


def frame(s, ch):
    chs = [c for c in (CH if ch == "all" else (ch,)) if c in rows]
    if s == "mix":
        q = pd.concat([mix[c] for c in chs], ignore_index=True)
        return q, q.w_mix.to_numpy(float)
    if s in ("sps", "dps"):
        q = pd.concat([emb[c][s] for c in chs], ignore_index=True)
        return q, q.w_emb.to_numpy(float)
    smp, wc = SAMPLES[s]
    q = pd.concat([rows[c][0] for c in (CH if ch == "all" else (ch,)) if c in rows], ignore_index=True)
    q = q[q["sample"] == smp]
    return q, q[wc].to_numpy(float)


for s in list(SAMPLES) + ["mix", "sps", "dps"]:
    for ch in ("all",) + CH:
        q, w = frame(s, ch)
        if not len(q):
            continue
        for var in tuple(VARIANTS) if s in SAMPLES else ("excl",):
            ok = q[f"ok_{var}"].to_numpy(bool)
            for v, (e, _) in VARS.items():
                if v == "njets":
                    if s != "mix" and var == "excl":
                        OUT["hists"][f"{s}_{v}_{ch}"] = hist(q.njets.to_numpy(float), w, e)
                    continue
                x = q[f"{v}_{var}"].to_numpy(float)
                OUT["hists"][f"{s}_{v}_{var}_{ch}"] = hist(x[ok], w[ok], e)
            # <pT(ttbar) . pT-hat(J/psi)> in pT(J/psi) bins, with <pT(J/psi)> and <pT of the object holding the J/psi>
            pp, pt, hp = q[f"ppar_{var}"].to_numpy(float), q.pt.to_numpy(float), q.holder_pt.to_numpy(float)
            prof = {"ppar": [], "pt": [], "holder": [], "sumw": []}
            for lo, hi in zip(PTB[:-1], PTB[1:]):
                b = ok & (pt >= lo) & (pt < hi)
                prof["ppar"].append(wmean(pp[b], w[b]))
                prof["pt"].append(wmean(pt[b], w[b])[0])
                prof["holder"].append(wmean(hp[b], w[b])[0])
                prof["sumw"].append(float(w[b].sum()))
            prof["ppar_all_pt"] = wmean(pp[ok], w[ok])
            prof["dphi_gt_half_pi"] = wmean((q[f"dphi_{var}"].to_numpy(float)[ok] > np.pi / 2).astype(float), w[ok])
            OUT["profile"][f"{s}_{var}_{ch}"] = prof
        if s != "mix":
            # where the J/psi sits in the "all" reconstruction (sWeighted fractions); "_x": only the events that keep enough
            # jets without the J/psi jet (as the emulated events, which are selected events before the J/psi jet is added);
            # "_j": of those, the J/psi in a selected jet (the emulations take the in-no-jet share of the data by construction)
            for grp, chs in (("ljets", ("mujets", "ejets")), ("emu", ("emu",))):
                if ch not in ("all",) + chs:
                    continue
                for sfx, need in (("", "ok_all"), ("_x", "ok_excl"), ("_j", "ok_excl")):
                    sel = q.ch.isin(chs).to_numpy() & q.ok_all.to_numpy(bool) & q[need].to_numpy(bool)
                    if sfx == "_j":
                        sel &= q.role_all.to_numpy() != "none"
                    r_, ws = q.role_all.to_numpy()[sel], w[sel]
                    tot = ws.sum()
                    if abs(tot) < 1e-9:
                        continue
                    OUT["roles"][f"{s}_{grp}_{ch}{sfx}"] = {"sumw": float(tot), **{nm: [float(ws[r_ == nm].sum() / tot), float(np.sqrt((ws[r_ == nm] ** 2).sum()) / abs(tot))]
                                                                                  for nm in ROLES[grp]}}

# ---------------- the share of the prompt J/psi from the ttbar collision: the sWeighted histogram of the prompt J/psi against
# N [f x (same collision) + (1 - f) x (second scattering)], the two emulations at unit area; chi2 with the sWeight errors of
# the data and the statistical errors of the emulations
def frac_fit(v, data="prompt"):
    hd, ed = (np.asarray(a_, float) for a_ in OUT["hists"][f"{data}_{v}_excl_all"])
    (hs, es), (hp, ep) = ((np.asarray(a_, float) for a_ in OUT["hists"][f"{h}_{v}_excl_all"]) for h in ("sps", "dps"))
    N, ts, tp, ets, etp = hd.sum(), hs / hs.sum(), hp / hp.sum(), es / hs.sum(), ep / hp.sum()
    fs = np.linspace(-1.5, 2.5, 4001)
    pred = N * (fs[:, None] * ts + (1 - fs[:, None]) * tp)
    var = np.maximum(ed ** 2 + N ** 2 * (fs[:, None] ** 2 * ets ** 2 + (1 - fs[:, None]) ** 2 * etp ** 2), 1e-9)
    c2 = ((hd - pred) ** 2 / var).sum(1)
    k = c2.argmin()
    inside = fs[c2 <= c2[k] + 1]
    at = lambda f: float(np.interp(f, fs, c2))  # noqa: E731
    return {"f_same": float(fs[k]), "err": [float(inside.min() - fs[k]), float(inside.max() - fs[k])], "chi2_min": float(c2[k]), "ndf": int(len(hd) - 1),
            "chi2_dps_only": at(0.0), "chi2_same_only": at(1.0), "z_vs_dps_only": float(np.sqrt(max(at(0.0) - c2[k], 0)))}


OUT["fit"] = {f"prompt_{v}": frac_fit(v) for v in ("ppar", "dphi")}
OUT["splot"] = {f"N_{s_}": float(S.loc[S["sample"] == SAMPLES[s_][0], SAMPLES[s_][1]].sum()) for s_ in SAMPLES}  # sums of the sWeights of the input  # the emulations carry the prompt J/psi jets: for the prompt sample only
for k_, F in OUT["fit"].items():
    print(f"{k_:16s} f(same collision) {F['f_same']:.2f} {F['err'][0]:+.2f} {F['err'][1]:+.2f}; chi2 {F['chi2_min']:.1f} / {F['ndf']};"
          f" second scattering only: chi2 {F['chi2_dps_only']:.1f} ({F['z_vs_dps_only']:.1f} sigma); same collision only: chi2 {F['chi2_same_only']:.1f}")
for s, var in itertools.product(("prompt", "nonprompt", "pileup", "mix", "sps", "dps", "inb", "ttj"), ("excl", "exclpf", "exclnoj")):
    P = OUT["profile"].get(f"{s}_{var}_all")
    if P:
        print(f"{s:9s} {var:6s} <pT(tt) . pT-hat(J/psi)> {P['ppar_all_pt'][0]:+6.1f} +- {P['ppar_all_pt'][1]:4.1f} GeV; P(|dphi| > pi/2) {P['dphi_gt_half_pi'][0]:.2f} +- {P['dphi_gt_half_pi'][1]:.2f};"
              f" per pT bin {', '.join(f'{m:+.0f}+-{e:.0f}' for m, e in P['ppar'])}; <pT(J/psi)> {', '.join(f'{x:.0f}' for x in P['pt'])};"
              f" <pT(J/psi jet, or J/psi)> {', '.join(f'{x:.0f}' for x in P['holder'])}")
for k, v in OUT["roles"].items():
    if k.endswith(("_all", "_all_j")):
        print(k, {nm: round(f[0], 2) for nm, f in v.items() if nm != "sumw"}, f"sum w {v['sumw']:.0f}")
json.dump(OUT, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
