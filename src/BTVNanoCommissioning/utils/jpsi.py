"""J/psi -> mu mu candidates on top of the ttbar selections (the *_jpsi modes of ttsemilep_valid_sf and ttdilep_valid_sf).

The ttbar objects are those of the hadronic-W study. In addition, one dimuon per event is built from loose-ID
muons (pT > 3 GeV, |eta| < 2.4, no isolation) that are not ttbar-lepton candidates: the opposite-sign pair closest
to the J/psi mass in 2.0-4.5 GeV, or, without one, the closest same-sign pair (combinatorial-background control).
The number of its muons passing the medium ID is kept, so a medium-ID selection can be made afterwards.
The two muons of that pair are exempt from the jet-muon cleaning, so a jet containing the J/psi stays a jet.
Signal: a J/psi not matched to a b jet (Delta R > 0.4 from every b candidate and every b-tagged selected jet),
categories 1 and 2 below; the b-matched J/psi (category 0, mostly from b-hadron decays) are kept as a control.
"""

import awkward as ak
import numpy as np

MJPSI = 3.0969
MASS_LO, MASS_HI = 2.0, 4.5
# mass regions: 0 low sideband, 1 signal window, 2 high sideband, 3 elsewhere in 2.0-4.5 GeV
MREG = ((2.60, 2.90), (2.95, 3.25), (3.30, 3.60))
DZ_MAX = 0.5  # |dz(mu1) - dz(mu2)| [cm]: both muons from one vertex (no dimuon vertex fit in NanoAOD)
DR_JET = 0.4  # the J/psi is "in" a jet within this Delta R of the jet axis


def jpsi_muon_mask(events, exclude):
    """Loose-ID muons for the J/psi (no isolation); `exclude` masks the ttbar-lepton candidates on events.Muon."""
    mu = events.Muon
    return mu.looseId & (mu.pt > 3) & (abs(mu.eta) < 2.4) & ~exclude


def find_jpsi(events, cand):
    """Best dimuon per event from the muons in mask `cand` (on events.Muon).
    Returns a dict of per-event arrays (invalid events carry dummy values) and the mask of its two muons on events.Muon."""
    mu = events.Muon
    lidx = ak.local_index(mu.pt)
    pairs = ak.combinations(lidx[cand], 2, fields=["i1", "i2"])
    m1, m2 = mu[pairs.i1], mu[pairs.i2]
    p4 = m1 + m2
    mass = p4.mass
    os_ = (m1.charge * m2.charge) < 0
    ok = (mass > MASS_LO) & (mass < MASS_HI) & (abs(m1.dz - m2.dz) < DZ_MAX)
    # opposite sign first, then the distance to the J/psi mass
    score = ak.where(ok, abs(mass - MJPSI) + ak.where(os_, 0.0, 100.0), 1e9)
    best = ak.argmin(score, axis=1, keepdims=True)
    valid = ak.fill_none(ak.firsts(ok[best]), False)
    i1 = ak.fill_none(ak.firsts(pairs.i1[best]), -1)
    i2 = ak.fill_none(ak.firsts(pairs.i2[best]), -1)
    i1, i2 = ak.where(valid, i1, -1), ak.where(valid, i2, -1)
    in_pair = (lidx == i1) | (lidx == i2)
    jp = ak.firsts(p4[best])
    energy, pz = jp.energy, jp.z
    rap = 0.5 * np.log((energy + pz) / np.maximum(energy - pz, 1e-9))
    sip = np.maximum(ak.firsts(m1.sip3d[best]), ak.firsts(m2.sip3d[best]))
    nmed = ak.values_astype(m1.mediumId, np.int64) + ak.values_astype(m2.mediumId, np.int64)
    info = {
        "valid": valid,
        "p4": jp,
        "os": ak.fill_none(ak.where(ak.firsts(os_[best]), 1, -1), 1),
        "mass": ak.fill_none(jp.mass, -1.0),
        "pt": ak.fill_none(jp.pt, -1.0),
        "absy": ak.fill_none(abs(rap), -1.0),
        "sip": ak.fill_none(sip, -1.0),
        "nmed": ak.fill_none(ak.firsts(nmed[best]), 0),
        "i1": i1,
        "i2": i2,
    }
    return info, ak.fill_none(in_pair, False)


def mass_region(mass):
    reg = np.full(len(mass), 3, dtype=np.int64)
    m = ak.to_numpy(mass)
    for k, (lo, hi) in enumerate(MREG):
        reg[(m > lo) & (m < hi)] = k
    return reg


def _flat(arr):
    counts = ak.to_numpy(ak.num(arr, axis=1))
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return ak.to_numpy(ak.flatten(arr)), offsets, counts


def jpsi_truth(events, info):
    """MC truth of the dimuon: 0 prompt J/psi, 1 J/psi from a b-hadron decay, 2 no common J/psi parent of the
    two muons (combinatorial or misidentified) or data."""
    n = len(events)
    out = np.full(n, 2, dtype=np.int64)
    if "GenPart" not in events.fields or "genPartIdx" not in events.Muon.fields:
        return out
    pdg, goff, gcnt = _flat(events.GenPart.pdgId)
    mom, _, _ = _flat(events.GenPart.genPartIdxMother)
    gidx, moff, mcnt = _flat(events.Muon.genPartIdx)
    valid = ak.to_numpy(info["valid"])
    i1, i2 = ak.to_numpy(info["i1"]), ak.to_numpy(info["i2"])
    nmu = max(len(gidx), 1)

    def gen_of(i):
        g = np.where(valid & (i >= 0), gidx[np.clip(moff + i, 0, nmu - 1)], -1)
        return np.where(g < gcnt, g, -1)

    def mother(g):
        # local index of the mother, -1 when none; skip copies of the same particle (e.g. FSR)
        ok = g >= 0
        cur = np.where(ok, mom[np.clip(goff + g, 0, len(mom) - 1)], -1)
        for _ in range(4):
            same = (cur >= 0) & ok & (pdg[np.clip(goff + cur, 0, len(pdg) - 1)] == pdg[np.clip(goff + g, 0, len(pdg) - 1)])
            if not same.any():
                break
            cur = np.where(same, mom[np.clip(goff + cur, 0, len(mom) - 1)], cur)
        return cur

    p1, p2 = mother(gen_of(i1)), mother(gen_of(i2))
    pdg_at = lambda c: np.where(c >= 0, pdg[np.clip(goff + c, 0, len(pdg) - 1)], 0)  # noqa: E731
    matched = (p1 >= 0) & (p1 == p2) & (pdg_at(p1) == 443)
    from_b = np.zeros(n, dtype=bool)
    cur = np.where(matched, p1, -1)
    for _ in range(40):
        cur = np.where(cur >= 0, mom[np.clip(goff + cur, 0, len(mom) - 1)], -1)
        a = np.abs(pdg_at(cur))
        from_b |= (cur >= 0) & (((a >= 500) & (a < 600)) | ((a >= 5000) & (a < 6000)))
        if not (cur >= 0).any():
            break
    out[matched] = np.where(from_b[matched], 1, 0)
    return out


def jet_association(jp, jets, bcands, btagged):
    """Where the J/psi sits: cat 0 matched to a b jet (within 0.4 of a ttbar b candidate or of a b-tagged selected
    jet), 1 in another selected jet, 2 in no selected jet; with the Delta R to the nearest selected jet,
    z = pT(J/psi) / pT(jet) of that jet, the index of the containing b candidate (-1 if none) and the partonFlavour
    of the nearest jet (0 if none or data)."""
    dr_all = jets.delta_r(jp)
    inear = ak.argmin(dr_all, axis=1, keepdims=True)
    drj = ak.fill_none(ak.firsts(dr_all[inear]), 99.0)
    near = ak.firsts(jets[inear])
    dr_b = bcands.delta_r(jp)
    ib = ak.argmin(dr_b, axis=1, keepdims=True)
    in_bcand = ak.fill_none(ak.firsts(dr_b[ib]) < DR_JET, False)
    in_tagged = ak.fill_none(ak.min(btagged.delta_r(jp), axis=1) < DR_JET, False)
    in_b = in_bcand | in_tagged
    in_any = drj < DR_JET
    cat = ak.where(in_b, 0, ak.where(in_any, 1, 2))
    z = ak.fill_none(jp.pt / near.pt, -1.0)
    pf = ak.fill_none(near.partonFlavour, 0) if "partonFlavour" in jets.fields else ak.zeros_like(drj, dtype=np.int64)
    return {
        "cat": ak.values_astype(cat, np.int64),
        "drj": drj,
        "z": ak.where(in_any, z, -1.0),
        "ib": ak.fill_none(ak.where(in_bcand, ak.firsts(ib), -1), -1),
        "pf": ak.where(in_any, pf, 0),
    }


def store(pruned_ev, info, sel, assoc, leptons, nb, ntrack, ttkin):
    """Write the J/psi fields of the selected events (`sel`: the event-level mask that made pruned_ev).
    `leptons`: {name: (lepton, index of its b candidate among bcands)}; `ttkin`: {name: ttbar observable of the
    selected events, <= 0 where undefined}, stored as jpsi_tt_<name> to show that the J/psi events are ttbar."""
    sel = ak.to_numpy(sel)
    pruned_ev["jpsi_valid"] = info["valid"][sel]
    for k in ("os", "mass", "pt", "absy", "sip", "nmed"):
        pruned_ev[f"jpsi_{k}"] = info[k][sel]
    pruned_ev["jpsi_mreg"] = mass_region(info["mass"][sel])
    pruned_ev["jpsi_truth"] = info["truth"][sel]
    for k in ("cat", "drj", "z"):
        pruned_ev[f"jpsi_{k}"] = assoc[k]
    jp = info["p4"][sel]
    for name, (lep, ib_lep) in leptons.items():
        pruned_ev[f"jpsi_mlj_{name}"] = ak.fill_none((lep + jp).mass, -1.0)
        # reco side: 1 the J/psi is in the b candidate paired with this lepton, 0 in the other one, 2 in no b candidate
        pruned_ev[f"jpsi_rside_{name}"] = ak.values_astype(
            ak.where(assoc["ib"] < 0, 2, ak.where(assoc["ib"] == ib_lep, 1, 0)), np.int64
        )
        # truth side: the jet of the J/psi is the b of this lepton's top (partonFlavour 5 x charge), 0 the other b, 2 neither
        want = 5 * lep.charge
        pruned_ev[f"jpsi_tside_{name}"] = ak.values_astype(
            ak.where(assoc["pf"] == want, 1, ak.where(assoc["pf"] == -want, 0, 2)), np.int64
        )
    pruned_ev["jpsi_nb"] = nb
    pruned_ev["jpsi_ntrack"] = ntrack
    for k, v in ttkin.items():
        pruned_ev[f"jpsi_tt_{k}"] = ak.fill_none(v, -1.0)
