"""J/psi -> mu mu candidates on top of the ttbar selections (the *_jpsi modes of ttsemilep_valid_sf and ttdilep_valid_sf).

The ttbar objects are those of the hadronic-W study. In addition, one dimuon per event is built from loose-ID
muons (pT > 3 GeV, |eta| < 2.4, no isolation) that are not ttbar-lepton candidates: the opposite-sign pair closest
to the J/psi mass in 2.0-4.5 GeV, or, without one, the closest same-sign pair (combinatorial-background control).
The number of its muons passing the medium ID is kept, so a medium-ID selection can be made afterwards.
The two muons of that pair are exempt from the jet-muon cleaning, so a jet containing the J/psi stays a jet.
Signal: a J/psi not matched to a b jet (Delta R > 0.4 from every b candidate and every b-tagged selected jet),
categories 1 and 2 below; the b-matched J/psi (category 0, mostly from b-hadron decays) are kept as a control.

TTJ_ONIUM=upsilon (environment, read at import) makes the same pass for Upsilon(1S, 2S, 3S) -> mu mu: the pair closest
to the Upsilon(1S) mass in 7-14 GeV, with that mass in l_xy. The fields and histograms keep their jpsi_* names, so
every builder and fit reads either pass; an Upsilon pass is told apart by its mass range.

The onium -> e e pair (find_jpsi with coll="Electron", from the electrons of onium_electron_mask) takes the electrons of
the eta -> mu mu e e analysis (Greenberg et al., eta2mu2eSkimmer), so that its electron scale factors (EGM wp90noiso ID
and RecoBelow20 reconstruction) or their measurement carry over; its mass range (EE_RANGE) is wider below the peak for
the bremsstrahlung tail.
"""

import os

import awkward as ak
import numpy as np

# per onium: PDG mass (pair choice and l_xy), pair mass range [GeV] and histogram bins, mass regions (low sideband,
# window, high sideband, of equal widths), PDG ids for the MC truth, lifetime regions (l_xy [mm], see below)
ONIA = {
    "jpsi": dict(mass=3.0969, lo=2.0, hi=4.5, nbins=100, mreg=((2.60, 2.90), (2.95, 3.25), (3.30, 3.60)), pdg=(443,),
                 lxy=(0.06, 0.10), ee=(1.5, 4.5)),
    # the window holds the three states (9.46, 10.02, 10.36 GeV, resolution 0.07-0.15 GeV); the l_xy regions scale
    # with the mass (l_xy = m L_xy / p_T at the same resolution on L_xy)
    "upsilon": dict(mass=9.4603, lo=7.0, hi=14.0, nbins=140, mreg=((7.50, 9.00), (9.15, 10.65), (10.80, 12.30)),
                    pdg=(553, 100553, 200553), lxy=(0.18, 0.30), ee=(6.0, 14.0)),
}
ONIUM = os.environ.get("TTJ_ONIUM", "jpsi")
if ONIUM not in ONIA:
    raise ValueError(f"TTJ_ONIUM={ONIUM}: expected one of {', '.join(ONIA)}")
MJPSI = ONIA[ONIUM]["mass"]
MASS_LO, MASS_HI, MASS_NBINS = ONIA[ONIUM]["lo"], ONIA[ONIUM]["hi"], ONIA[ONIUM]["nbins"]
# mass regions: 0 low sideband, 1 signal window, 2 high sideband, 3 elsewhere in MASS_LO-MASS_HI
MREG = ONIA[ONIUM]["mreg"]
PDG_ONIUM = ONIA[ONIUM]["pdg"]
EE_RANGE = ONIA[ONIUM]["ee"]  # pair mass range of the onium -> e e pair [GeV]
DZ_MAX = 0.5  # |dz(mu1) - dz(mu2)| [cm]: both muons from one vertex (no dimuon vertex fit in NanoAOD)
DR_JET = 0.4  # the J/psi is "in" a jet within this Delta R of the jet axis
# lifetime regions (dimuon_vertex): on inclusive 2024 J/psi the prompt peak of l_xy has a core of about 20 um, so
# |l_xy| < 0.06 mm is prompt-like and l_xy > 0.1 mm is dominated by b-hadron decays; |z0| > 0.1 cm points to another vertex
# (J/psi; the Upsilon regions are scaled with the mass)
LXY_PROMPT, LXY_NONPROMPT = ONIA[ONIUM]["lxy"]  # mm
Z0_PV = 0.1  # cm


# muon ID flags of the two muons in one integer each (bit i set: passes MUON_ID_BITS[i]); a flag missing from the NanoAOD
# version leaves its bit 0. mediumPromptId = medium + |dxy| < 0.02 cm + |dz| < 0.1 cm, as the pair-level PV category.
MUON_ID_BITS = ("looseId", "mediumId", "mediumPromptId", "tightId", "softId", "isGlobal", "isTracker", "highPurity")
MUON_FIELDS = ("iso04_1", "iso04_2", "phi1", "phi2", "dxy1", "dxy2", "dz1", "dz2", "idbits1", "idbits2")  # leading, subleading muon
# the same for the electrons of an e e pair (idbits*: ELE_ID_BITS; "nmed" counts mvaNoIso_WP80), plus the MVA no-isolation
# score and the supercluster eta (eta + deltaEtaSC, the variable of the EGM scale factors) of each electron
ELE_ID_BITS = ("mvaNoIso_WP90", "mvaNoIso_WP80", "mvaIso_WP90", "mvaIso_WP80", "convVeto", "isPFcand", "isEcalDriven", "isEB")
ELE_FIELDS = ("mva1", "mva2", "sceta1", "sceta2")
EE_ETA_GAP = (1.44, 1.57)  # the barrel-endcap transition removed by the eta -> mu mu e e analysis (on eta)


def muon_id_bits(mu, names=MUON_ID_BITS):
    bits = 0
    for i, f in enumerate(names):
        if f in mu.fields:
            bits = bits + ak.values_astype(mu[f], np.int64) * (1 << i)
    return bits


def jpsi_muon_mask(events, exclude):
    """Loose-ID muons for the J/psi (no isolation); `exclude` masks the ttbar-lepton candidates on events.Muon."""
    mu = events.Muon
    return mu.looseId & (mu.pt > 3) & (abs(mu.eta) < 2.4) & ~exclude


def onium_electron_mask(events, exclude):
    """The electrons of the eta -> mu mu e e analysis (eta2mu2eSkimmer + plot_2mu2e.py, nominal): MVA no-isolation WP90
    (Run 3 NanoAOD: Winter22 V1, the ID of the EGM wp90noiso scale factors), conversion veto (passConversionVeto, the
    analysis's hasMatchedConversion with the beam spot), no missing inner hit, |eta| < 2.5 outside 1.44-1.57; no isolation.
    pT > 5 GeV is the NanoAOD Electron threshold (the analysis, on MINIAOD, went down to 2 GeV). `exclude` masks the W / Z
    electrons on events.Electron."""
    el = events.Electron
    aeta = abs(el.eta)
    gap = (aeta > EE_ETA_GAP[0]) & (aeta < EE_ETA_GAP[1])
    return el.mvaNoIso_WP90 & el.convVeto & (el.lostHits == 0) & (el.pt > 5) & (aeta < 2.5) & ~gap & ~exclude


def find_jpsi(events, cand, coll="Muon", mass_range=None):
    """Best dimuon per event from the muons in mask `cand` (on events.Muon); with coll="Electron" the best e e pair from
    the electrons in `cand` (on events.Electron) in `mass_range` (default: the onium's mu mu range).
    Returns a dict of per-event arrays (invalid events carry dummy values) and the mask of its two leptons on events[coll]."""
    mu = events[coll]
    ele = coll == "Electron"
    lo, hi = mass_range if mass_range is not None else (MASS_LO, MASS_HI)
    lidx = ak.local_index(mu.pt)
    pairs = ak.combinations(lidx[cand], 2, fields=["i1", "i2"])
    m1, m2 = mu[pairs.i1], mu[pairs.i2]
    p4 = m1 + m2
    mass = p4.mass
    os_ = (m1.charge * m2.charge) < 0
    ok = (mass > lo) & (mass < hi) & (abs(m1.dz - m2.dz) < DZ_MAX)
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
    good = "mvaNoIso_WP80" if ele else "mediumId"
    nmed = ak.values_astype(m1[good], np.int64) + ak.values_astype(m2[good], np.int64)
    lead = ak.firsts(m1[best]).pt >= ak.firsts(m2[best]).pt
    mu_a, mu_b = ak.firsts(m1[best]), ak.firsts(m2[best])
    bits = ELE_ID_BITS if ele else MUON_ID_BITS
    vtx = dimuon_vertex(ak.firsts(m1[best]), ak.firsts(m2[best]), jp, ak.to_numpy(valid))
    info = {
        "valid": valid,
        "p4": jp,
        "os": ak.fill_none(ak.where(ak.firsts(os_[best]), 1, -1), 1),
        "mass": ak.fill_none(jp.mass, -1.0),
        "pt": ak.fill_none(jp.pt, -1.0),
        "absy": ak.fill_none(abs(rap), -1.0),
        "sip": ak.fill_none(sip, -1.0),
        "nmed": ak.fill_none(ak.firsts(nmed[best]), 0),
        # the two muons, leading and subleading in pT (for the per-candidate table)
        "pt1": ak.fill_none(ak.where(lead, mu_a.pt, mu_b.pt), -1.0),
        "eta1": ak.fill_none(ak.where(lead, mu_a.eta, mu_b.eta), -9.0),
        "pt2": ak.fill_none(ak.where(lead, mu_b.pt, mu_a.pt), -1.0),
        "eta2": ak.fill_none(ak.where(lead, mu_b.eta, mu_a.eta), -9.0),
        # isolation of the two muons (b decays leave other tracks around the J/psi): PF isolation in 0.3, mini-isolation,
        # and the relative isolation with respect to the matched jet
        **{f"{k}{i}": ak.fill_none(ak.where(lead if i == 1 else ~lead, mu_a[f], mu_b[f]), -1.0)
           for i in (1, 2) for k, f in (("iso", "pfRelIso03_all"), ("miniiso", "miniPFRelIso_all"), ("jetiso", "jetRelIso"))},
        # PF isolation in 0.4 (the muon POG working points, tight < 0.15), direction and impact parameters (cm) of the two muons
        **{f"{k}{i}": ak.fill_none(ak.where(lead if i == 1 else ~lead, mu_a[f], mu_b[f]), -1.0 if k == "iso04_" else -99.0)
           for i in (1, 2) for k, f in (("iso04_", "pfRelIso04_all"), ("phi", "phi"), ("dxy", "dxy"), ("dz", "dz"))},
        **{f"idbits{i}": ak.fill_none(ak.where(lead if i == 1 else ~lead, muon_id_bits(mu_a, bits), muon_id_bits(mu_b, bits)), 0)
           for i in (1, 2)},
        "i1": i1,
        "i2": i2,
        **vtx,
    }
    if ele:
        sc_a, sc_b = mu_a.eta + mu_a.deltaEtaSC, mu_b.eta + mu_b.deltaEtaSC
        for i in (1, 2):
            pick = lead if i == 1 else ~lead
            info[f"mva{i}"] = ak.fill_none(ak.where(pick, mu_a.mvaNoIso, mu_b.mvaNoIso), -99.0)
            info[f"sceta{i}"] = ak.fill_none(ak.where(pick, sc_a, sc_b), -99.0)
    return info, ak.fill_none(in_pair, False)


def _np(x, fill):
    return ak.to_numpy(ak.fill_none(x, fill)).astype(np.float64)


def dimuon_vertex(b1, b2, jp, valid):
    """Two-muon vertex and the pseudo-proper decay length (central NanoAOD has no dimuon vertex fit).

    Each muon track is a straight line near the primary vertex (the curvature is negligible over a few mm): its
    transverse point of closest approach to the PV is (-dxy sin(phi), dxy cos(phi), dz) relative to the PV, its
    direction u from (eta, phi). The vertex V minimises the chi2 of the distances to the two lines along the two
    directions perpendicular to each track, n_T (transverse, error dxyErr) and n_L = u x n_T (error dzErr sin(theta)):
    a linear fit with 4 measurements and 3 unknowns (1 degree of freedom).
      lxy = m(J/psi) (L_xy . p_T) / p_T^2, L = V - PV          (pseudo-proper decay length, mm, PDG mass)
      z0  = z of the J/psi line at its transverse closest approach to the PV (cm): ~0 for a J/psi from the PV
            (prompt or from a b hadron), the distance to another vertex for pileup."""
    n = len(valid)
    A, b = np.zeros((n, 3, 3)), np.zeros((n, 3))
    meas = []
    for m in (b1, b2):
        phi, eta = _np(m.phi, 0.0), _np(m.eta, 0.0)
        cl, sl = 1.0 / np.cosh(eta), np.tanh(eta)
        n_t = np.stack([-np.sin(phi), np.cos(phi), np.zeros(n)], axis=1)
        n_l = np.stack([-sl * np.cos(phi), -sl * np.sin(phi), cl], axis=1)
        s_t = np.maximum(_np(m.dxyErr, 1.0), 1e-4)
        s_l = np.maximum(_np(m.dzErr, 1.0), 1e-4) * cl
        # offsets of the track's closest-approach point along n_T and n_L: dxy and dz sin(theta)
        meas += [(n_t, _np(m.dxy, 0.0), s_t), (n_l, _np(m.dz, 0.0) * cl, s_l)]
    for nv, d, sig in meas:
        w = 1.0 / sig**2
        A += w[:, None, None] * nv[:, :, None] * nv[:, None, :]
        b += (w * d)[:, None] * nv
    det = np.linalg.det(A)
    ok = valid & np.isfinite(det) & (det > 0)
    A[~ok], b[~ok] = np.eye(3), 0.0
    C = np.linalg.inv(A)
    V = np.einsum("nij,nj->ni", C, b)
    chi2 = sum(((np.einsum("ni,ni->n", nv, V) - d) / sig) ** 2 for nv, d, sig in meas)
    px, py, pz = _np(jp.px, 0.0), _np(jp.py, 0.0), _np(jp.pz, 0.0)
    pt2 = np.maximum(px**2 + py**2, 1e-6)
    g = np.stack([px, py, np.zeros(n)], axis=1) * (MJPSI / pt2)[:, None]
    lxy = np.einsum("ni,ni->n", g, V)
    lerr = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", g, C, g), 0.0))
    z0 = V[:, 2] - (V[:, 0] * px + V[:, 1] * py) / pt2 * pz
    ok &= np.isfinite(lxy) & np.isfinite(z0) & (lerr < 1.0)
    bad = lambda x: np.where(ok, x, -99.0)  # noqa: E731
    return {"lxy": bad(10.0 * lxy), "lxyerr": bad(10.0 * lerr), "z0": bad(z0), "vchi2": bad(chi2), "vok": ok}


def life_region(info):
    """0 prompt (|l_xy| < 0.06 mm), 1 non-prompt (l_xy > 0.1 mm), 2 in between or negative tail, each with
    |z0| < 0.1 cm (from the primary vertex); 3 pileup-like (|z0| >= 0.1 cm); 4 no vertex."""
    lxy, z0, ok = np.asarray(info["lxy"]), np.abs(np.asarray(info["z0"])), np.asarray(info["vok"])
    reg = np.where(lxy > LXY_NONPROMPT, 1, np.where(np.abs(lxy) < LXY_PROMPT, 0, 2))
    reg = np.where(z0 >= Z0_PV, 3, reg)
    return np.where(ok, reg, 4).astype(np.int64)


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


def jpsi_truth(events, info, coll="Muon"):
    """MC truth of the dimuon: 0 prompt J/psi, 1 J/psi from a b-hadron decay, 2 no common J/psi parent of the
    two muons (combinatorial or misidentified) or data (with TTJ_ONIUM=upsilon: an Upsilon(1S, 2S, 3S) parent;
    bottomonium states above it do not count as b hadrons). coll="Electron" for an e e pair."""
    n = len(events)
    out = np.full(n, 2, dtype=np.int64)
    if "GenPart" not in events.fields or "genPartIdx" not in events[coll].fields:
        return out
    pdg, goff, gcnt = _flat(events.GenPart.pdgId)
    mom, _, _ = _flat(events.GenPart.genPartIdxMother)
    gidx, moff, mcnt = _flat(events[coll].genPartIdx)
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
    matched = (p1 >= 0) & (p1 == p2) & np.isin(pdg_at(p1), PDG_ONIUM)
    from_b = np.zeros(n, dtype=bool)
    cur = np.where(matched, p1, -1)
    for _ in range(40):
        cur = np.where(cur >= 0, mom[np.clip(goff + cur, 0, len(mom) - 1)], -1)
        a = np.abs(pdg_at(cur))
        bbbar = (a % 1000 > 550) & (a % 1000 < 560)  # bottomonium (551, 553, 10551, 100553, ...)
        from_b |= (cur >= 0) & (((a >= 500) & (a < 600) & ~bbbar) | ((a >= 5000) & (a < 6000)))
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


def near_objects(ev, jp):
    """the secondary vertex (SV collection) and the jet (any jet with pT > 15 GeV, |eta| < 2.5, no ID or cleaning)
    closest to the J/psi direction: a b-hadron decay leaves a displaced vertex or a soft jet around the J/psi.
    sv_*: Delta R, number of tracks, flight-distance significance, mass; sj_*: Delta R, pT, UParT b score, muon energy
    fraction (a jet made of the J/psi muons alone has muEF near 1). -1 where there is none."""
    out = {}
    sv = ev.SV  # no four-vector behaviour in coffea 0.7: Delta R by hand
    dphi = (sv.phi - jp.phi + np.pi) % (2 * np.pi) - np.pi
    dr = ak.fill_none(np.sqrt((sv.eta - jp.eta) ** 2 + dphi**2), 99.0)
    i = ak.argmin(dr, axis=1, keepdims=True)
    near = ak.firsts(sv[i])
    out["sv_dr"] = ak.fill_none(ak.firsts(dr[i]), -1.0)
    for k, f in (("sv_ntrk", "ntracks"), ("sv_dlensig", "dlenSig"), ("sv_mass", "mass")):
        out[k] = ak.fill_none(near[f], -1.0)
    jets = ev.Jet[(ev.Jet.pt > 15) & (abs(ev.Jet.eta) < 2.5)]
    dr = ak.fill_none(jets.delta_r(jp), 99.0)
    i = ak.argmin(dr, axis=1, keepdims=True)
    near = ak.firsts(jets[i])
    out["sj_dr"] = ak.fill_none(ak.firsts(dr[i]), -1.0)
    btag = "btagUParTAK4B" if "btagUParTAK4B" in ev.Jet.fields else "btagDeepFlavB"
    for k, f in (("sj_pt", "pt"), ("sj_btag", btag), ("sj_muef", "muEF")):
        out[k] = ak.fill_none(near[f], -1.0)
    return out


def store(pruned_ev, info, sel, assoc, leptons, nb, ntrack, ttkin):
    """Write the J/psi fields of the selected events (`sel`: the event-level mask that made pruned_ev).
    `leptons`: {name: (lepton, index of its b candidate among bcands)}; `ttkin`: {name: ttbar observable of the
    selected events, <= 0 where undefined}, stored as jpsi_tt_<name> to show that the J/psi events are ttbar."""
    sel = ak.to_numpy(sel)
    pruned_ev["jpsi_valid"] = info["valid"][sel]
    for k in ("os", "mass", "pt", "absy", "sip", "nmed", "pt1", "eta1", "pt2", "eta2", "iso1", "iso2", "miniiso1", "miniiso2", "jetiso1", "jetiso2",
              *MUON_FIELDS):
        pruned_ev[f"jpsi_{k}"] = info[k][sel]
    pruned_ev["jpsi_mreg"] = mass_region(info["mass"][sel])
    for k in ("lxy", "lxyerr", "z0", "vchi2"):
        pruned_ev[f"jpsi_{k}"] = np.asarray(info[k])[sel]
    pruned_ev["jpsi_life"] = life_region(info)[sel]
    pruned_ev["jpsi_truth"] = info["truth"][sel]
    for k in ("cat", "drj", "z"):
        pruned_ev[f"jpsi_{k}"] = assoc[k]
    jp = info["p4"][sel]
    # the J/psi direction, for its pT balance against the ttbar system
    pruned_ev["jpsi_eta"] = ak.fill_none(jp.eta, -99.0)
    pruned_ev["jpsi_phi"] = ak.fill_none(jp.phi, -99.0)
    for k, v in near_objects(pruned_ev, jp).items():
        pruned_ev[f"jpsi_{k}"] = v
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


# one row per J/psi candidate (output "jpsi_cands"), for unbinned fits of the mass and l_xy with the per-candidate
# l_xy error, in bins of pT: all selected ttbar events with a dimuon in MASS_LO-MASS_HI, opposite and same sign
CAND_FIELDS = ("os", "mass", "pt", "absy", "sip", "nmed", "pt1", "eta1", "pt2", "eta2", "iso1", "iso2", "miniiso1", "miniiso2",
               "jetiso1", "jetiso2", "mreg", "lxy", "lxyerr", "z0", "vchi2", "life", "truth", "cat", "drj", "z", "nb", "ntrack",
               "sv_dr", "sv_ntrk", "sv_dlensig", "sv_mass", "sj_dr", "sj_pt", "sj_btag", "sj_muef", "eta", "phi", *MUON_FIELDS)
NJ = 8  # selected jets kept per candidate (pT order as in the workflow; -99 padded)
JET_FIELDS = {"pt": "pt", "eta": "eta", "phi": "phi", "mass": "mass", "btag": "btagUParTAK4B", "muef": "muEF"}


def candidate_table(pruned_ev, weight=None, jets=None, isb=None):
    """{column: column_accumulator} of the J/psi candidates of the selected events: CAND_FIELDS, the ttbar observables
    (jpsi_tt_*, jpsi_mlj_*), the ttbar lepton(s) (pT, eta, phi, charge, isolation, m_T), MET (PF and PUPPI, with phi),
    run / lumi / event, the number of good primary vertices and the event weight (1 for data).
    jets: the selected jets of the events (the workflow's order), stored as jet<i>_<field> for i < NJ with their medium
    b tag `isb`: with the J/psi direction, the ttbar system can be rebuilt offline (the workflow's jet assignment, or a
    kinematic reconstruction) to test the pT balance of ttbar and J/psi."""
    from coffea.processor import column_accumulator

    ok = ak.to_numpy(ak.fill_none(pruned_ev.jpsi_valid, False))
    cols = {"run": pruned_ev.run, "lumi": pruned_ev.luminosityBlock, "event": pruned_ev.event, "npv": pruned_ev.PV.npvsGood}
    cols.update({k: pruned_ev[f"jpsi_{k}"] for k in CAND_FIELDS})
    cols.update({f: pruned_ev[f] for f in pruned_ev.fields if f.startswith(("jpsi_tt_", "jpsi_mlj_", "jpsi_rside_"))})
    # the ttbar lepton(s) and MET, to tell ttbar from multijet events (isolation, m_T)
    met = pruned_ev.MET
    cols["met"], cols["met_phi"] = met.pt, met.phi
    if "PuppiMET" in pruned_ev.fields:
        cols["puppimet"], cols["puppimet_phi"] = pruned_ev.PuppiMET.pt, pruned_ev.PuppiMET.phi
    for f, nm, iso in (("SelMuon", "mu", "pfRelIso04_all"), ("SelElectron", "e", "pfRelIso03_all")):
        if f in pruned_ev.fields:
            lep = pruned_ev[f]
            cols.update({f"{nm}_pt": lep.pt, f"{nm}_eta": lep.eta, f"{nm}_phi": lep.phi, f"{nm}_q": lep.charge, f"{nm}_iso": lep[iso],
                         f"{nm}_mT": np.sqrt(2.0 * lep.pt * met.pt * (1.0 - np.cos(lep.phi - met.phi)))})
    if jets is not None:
        cols["njets"] = ak.num(jets.pt, axis=1)
        pad = lambda a: ak.fill_none(ak.pad_none(a, NJ, axis=1, clip=True), -99)  # noqa: E731
        for k, f in JET_FIELDS.items():
            a = pad(jets[f])
            for i in range(NJ):
                cols[f"jet{i}_{k}"] = a[:, i]
        if isb is not None:
            a = pad(ak.values_astype(isb, np.int64))
            for i in range(NJ):
                cols[f"jet{i}_isb"] = a[:, i]
    cols["weight"] = np.ones(len(ok)) if weight is None else weight
    out = {}
    for k, v in cols.items():
        a = ak.to_numpy(ak.fill_none(v, -99)) if not isinstance(v, np.ndarray) else v
        out[k.replace("jpsi_", "", 1)] = column_accumulator(np.asarray(a)[ok])
    return out


def npv_hist(pruned_ev, weight=None):
    """good primary vertices of all selected ttbar events (pileup of the sample, for the pileup J/psi)"""
    import hist as Hist

    h = Hist.Hist(Hist.axis.Integer(0, 150, name="npv", label="good primary vertices"), storage=Hist.storage.Weight())
    h.fill(npv=ak.to_numpy(pruned_ev.PV.npvsGood), weight=np.ones(len(pruned_ev)) if weight is None else weight)
    return h
