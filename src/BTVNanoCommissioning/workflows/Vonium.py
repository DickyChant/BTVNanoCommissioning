"""W / Z + onium -> mu mu (the J/psi, or with TTJ_ONIUM=upsilon the Upsilon(1S, 2S, 3S); utils/jpsi.py), four channels in one
pass over the events:
  "wmu":   W -> mu nu, IsoMu24             (Muon PD)
  "we":    W -> e nu,  Ele30_WPTight_Gsf   (EGamma PD)
  "zmumu": Z -> mu mu, IsoMu24             (Muon PD)
  "zee":   Z -> e e,   Ele30_WPTight_Gsf   (EGamma PD)
(Run 2 paths: utils/selection.py SINGLE_LEPTON_HLT). The event is triggered by the W / Z lepton, not by the onium: the
low-mass dimuon path HLT_DoubleMu4_3_LowMass stops at 8.5 GeV and the Upsilon paths ask for pT(mu mu) > 10 GeV.

Leptons are the ttbar ones: muons with tight ID, pfRelIso04 < 0.12, |eta| < 2.4; electrons with the MVA-isolation WP80,
outside the barrel-endcap gap; pT > 15 GeV, and the trigger lepton above 30 GeV (muon) / the electron threshold.
W: the leading lepton of the trigger flavour above the trigger threshold; Z: the opposite-sign pair of that flavour
closest to the Z mass in 60-120 GeV with its leading lepton above the threshold. The onium pair is built from the
loose-ID muons that are not the W / Z leptons (utils/jpsi.py, no isolation); then no other lepton above 15 GeV may be
left, the muons of the onium pair apart. The four channels are exclusive (one or two leptons, one flavour).

Output per dataset: {channel: {"cands": one row per selected W / Z event with an onium pair (opposite or same sign, in
the onium mass range), "nV": all selected W / Z events in (good primary vertices, m_T of the W or m_ll of the Z), for the
W / Z yield and the pileup (the rate of pileup onia)}, "sumw", and the processed luminosity blocks of the data}.
No histograms, no scale factors. Data runs the channels of its primary dataset, so no event enters twice; MC runs all four.
"""
import re

import awkward as ak
import hist as Hist
import numpy as np
from coffea import processor
from coffea.processor import column_accumulator

from BTVNanoCommissioning.helpers.func import dump_lumi, update
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils import jpsi as ONI
from BTVNanoCommissioning.utils.correction import common_shifts, load_lumi, load_SF
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    MET_filters,
    btag_mu_idiso,
    btag_wp,
    ele_mvatightid,
    jet_id,
    single_lepton_trigger,
)

# channel: (lepton flavour, number of W / Z leptons)
CHANNELS = {"wmu": ("mu", 1), "we": ("e", 1), "zmumu": ("mu", 2), "zee": ("e", 2)}
PD_CHANNELS = (
    (re.compile(r"(Single)?Muon\d*Run"), ("wmu", "zmumu")),
    (re.compile(r"(EGamma|SingleElectron)\d*Run"), ("we", "zee")),
)
LEP_PTMIN = 15.0  # every lepton (W / Z leptons and the extra-lepton veto)
MU_PTMIN = 30.0  # the trigger muon (IsoMu24; the ttbar muon)
MZ, MZ_LO, MZ_HI = 91.1876, 60.0, 120.0
NV_FIELDS = ("pt", "eta", "phi", "q", "iso", "dz")
# onium fields of utils/jpsi.find_jpsi kept per candidate
ONIUM_FIELDS = ("os", "mass", "pt", "absy", "sip", "nmed", "pt1", "eta1", "pt2", "eta2", "iso1", "iso2", "miniiso1", "miniiso2",
                "jetiso1", "jetiso2", "lxy", "lxyerr", "z0", "vchi2", "truth", *ONI.MUON_FIELDS)


def channels(dataset, is_data):
    if not is_data:
        return tuple(CHANNELS)
    for rx, chans in PD_CHANNELS:
        if rx.match(dataset):
            return chans
    raise ValueError(f"Vonium: no channel for data set {dataset} (expected a Muon or EGamma primary dataset)")


def _np(x, fill=-99.0):
    return np.asarray(ak.to_numpy(ak.fill_none(x, fill)))


class NanoProcessor(processor.ProcessorABC):
    def __init__(self, year="2022", campaign="Summer22Run3", name="", isSyst=False, isArray=False, noHist=False, chunksize=75000):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = False  # nominal objects only
        self.chunksize = chunksize
        self.lumiMask = load_lumi(self._campaign)
        # the corrections of the ttbar selection (JEC of the campaign, jet veto map, muon / electron scale and smearing)
        self.SF_map = load_SF(self._year, self._campaign)

    def process(self, events):
        events = missing_branch(events)
        dataset = events.metadata["dataset"]
        is_data = not hasattr(events, "genWeight")
        out = {"sumw": float(len(events)) if is_data else float(np.sum(np.asarray(events.genWeight), dtype=np.float64))}
        if is_data:
            out = dump_lumi(events[self.lumiMask(events.run, events.luminosityBlock)], out)
        vetoed, shifts = common_shifts(self, events)
        ev = update(vetoed, shifts[0][0])  # nominal: no systematic shifts
        for ch in channels(dataset, is_data):
            out[ch] = self.process_channel(ev, ch, is_data)
        return {dataset: out}

    def process_channel(self, ev, ch, is_data):
        flav, nlep = CHANNELS[ch]
        elec = flav == "e"
        trig, ele_ptmin = single_lepton_trigger(self._campaign, elec)
        req = HLT_helper(ev, trig) & MET_filters(ev, self._campaign)
        if is_data:
            req = req & self.lumiMask(ev.run, ev.luminosityBlock)
        tmu = (ev.Muon.pt > LEP_PTMIN) & btag_mu_idiso(ev, self._campaign)
        tel = (ev.Electron.pt > LEP_PTMIN) & ele_mvatightid(ev, self._campaign)
        L, tl, ptmin = (ev.Electron, tel, ele_ptmin) if elec else (ev.Muon, tmu, MU_PTMIN)
        lidx = ak.local_index(L.pt)

        # the W / Z leptons (indices on L; i2 = -1 for the W)
        if nlep == 1:
            cand = tl & (L.pt > ptmin)
            lead = ak.argmax(ak.where(cand, L.pt, -1.0), axis=1, keepdims=True)
            ok = ak.fill_none(ak.firsts(cand[lead]), False)
            i1 = ak.where(ok, ak.fill_none(ak.firsts(lidx[lead]), -1), -1)
            i2 = ak.full_like(i1, -1)
        else:
            pairs = ak.combinations(lidx[tl], 2, fields=["a", "b"])
            la, lb = L[pairs.a], L[pairs.b]
            mll = (la + lb).mass
            okp = (la.charge * lb.charge < 0) & (mll > MZ_LO) & (mll < MZ_HI) & (np.maximum(la.pt, lb.pt) > ptmin)
            best = ak.argmin(ak.where(okp, abs(mll - MZ), 1e9), axis=1, keepdims=True)
            ok = ak.fill_none(ak.firsts(okp[best]), False)
            i1 = ak.where(ok, ak.fill_none(ak.firsts(pairs.a[best]), -1), -1)
            i2 = ak.where(ok, ak.fill_none(ak.firsts(pairs.b[best]), -1), -1)
        sel = ak.to_numpy(ak.fill_none(req & ok, False))
        e, i1, i2, lidx = ev[sel], i1[sel], i2[sel], lidx[sel]
        tmu, tel = tmu[sel], tel[sel]
        vlep = (lidx == i1) | (lidx == i2)  # on L of the selected events
        vmu = ak.zeros_like(tmu) if elec else vlep

        # the onium pair from the loose muons that are not W / Z leptons, then no extra lepton outside it
        info, pair = ONI.find_jpsi(e, ONI.jpsi_muon_mask(e, vmu))
        info["truth"] = ONI.jpsi_truth(e, info)
        n_extra = ak.sum(tmu & ~vmu & ~pair, axis=1) + ak.sum(tel & ~(vlep if elec else ak.zeros_like(tel)), axis=1)
        keep = ak.to_numpy(n_extra == 0)
        e, i1, i2, vlep = e[keep], i1[keep], i2[keep], vlep[keep]
        info = {k: (v[keep] if not isinstance(v, (int, float)) else v) for k, v in info.items()}
        Lsel = e.Electron if elec else e.Muon
        l1 = ak.firsts(Lsel[ak.local_index(Lsel.pt) == i1])
        l2 = ak.firsts(Lsel[ak.local_index(Lsel.pt) == i2])
        met = e.MET  # the corrected MET of common_shifts (PUPPI)
        if nlep == 1:
            mT = np.sqrt(2.0 * l1.pt * met.pt * (1.0 - np.cos(l1.phi - met.phi)))
            vx, vy = l1.pt * np.cos(l1.phi) + met.pt * np.cos(met.phi), l1.pt * np.sin(l1.phi) + met.pt * np.sin(met.phi)
            v = {"mv": mT, "ptv": np.hypot(vx, vy), "phiv": np.arctan2(vy, vx), "yv": ak.full_like(mT, -99.0)}
        else:
            z = l1 + l2
            rap = 0.5 * np.log((z.energy + z.z) / np.maximum(z.energy - z.z, 1e-9))
            v = {"mv": z.mass, "ptv": z.pt, "phiv": z.phi, "yv": rap}
        w = np.ones(len(e)) if is_data else _np(e.genWeight, 0.0)
        npv = _np(e.PV.npvsGood, 0)

        nV = Hist.Hist(
            Hist.axis.Integer(0, 150, name="npv", label="good primary vertices"),
            Hist.axis.Regular(60, 0, 300, name="mv", label="m_T of the W / m_ll of the Z [GeV]"),
            storage=Hist.storage.Weight(),
        )
        nV.fill(npv=npv, mv=_np(v["mv"], -1.0), weight=w)

        ok = _np(info["valid"], False).astype(bool)
        cols = {"run": e.run, "lumi": e.luminosityBlock, "event": e.event, "npv": npv, "weight": w}
        cols.update({k: info[k] for k in ONIUM_FIELDS})
        cols["mreg"] = ONI.mass_region(info["mass"])
        cols["life"] = ONI.life_region(info)
        jp = info["p4"]
        cols["eta"], cols["phi"] = ak.fill_none(jp.eta, -99.0), ak.fill_none(jp.phi, -99.0)
        cols.update(ONI.near_objects(e, jp))
        # the W / Z leptons (l2_*: -99 for the W) and the W / Z system
        for k, lep in (("l1", l1), ("l2", l2)):
            for f, src in zip(NV_FIELDS, ("pt", "eta", "phi", "charge", "pfRelIso03_all" if elec else "pfRelIso04_all", "dz")):
                cols[f"{k}_{f}"] = lep[src]
        cols.update(v)
        cols["met"], cols["met_phi"] = met.pt, met.phi
        if "PuppiMET" in e.fields:
            cols["puppimet"], cols["puppimet_phi"] = e.PuppiMET.pt, e.PuppiMET.phi
        # jets away from the W / Z leptons (the onium muons stay in their jets), with the UParT medium b tag: to set ttbar apart
        Lv = Lsel[vlep]
        jets = e.Jet[ak.fill_none(jet_id(e, self._campaign) & ak.all(e.Jet.metric_table(Lv) > 0.4, axis=2, mask_identity=False), False)]
        jets = jets[jets.pt > 30]
        try:
            isb = btag_wp(jets, self._year, self._campaign, "UParTAK4", "b", "M")
        except Exception:
            isb = jets.btagUParTAK4B > 0.1272
        cols["njets"] = ak.num(jets.pt, axis=1)
        cols["nb"] = ak.sum(ak.fill_none(isb, False), axis=1)
        cols["j1_pt"] = ak.fill_none(ak.firsts(jets.pt), -1.0)
        return {
            "cands": {k: column_accumulator(np.asarray(_np(c) if not isinstance(c, np.ndarray) else c)[ok]) for k, c in cols.items()},
            "nV": nV,
        }

    def postprocess(self, accumulator):
        return accumulator
