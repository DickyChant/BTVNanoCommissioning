"""W / Z + onium -> mu mu (the J/psi, or with TTJ_ONIUM=upsilon the Upsilon(1S, 2S, 3S); utils/jpsi.py), four channels in one
pass over the events:
  "wmu":   W -> mu nu, IsoMu24             (Muon PD)
  "we":    W -> e nu,  Ele30_WPTight_Gsf   (EGamma PD)
  "zmumu": Z -> mu mu, IsoMu24             (Muon PD)
  "zee":   Z -> e e,   Ele30_WPTight_Gsf   (EGamma PD)
(Run 2 paths: utils/selection.py SINGLE_LEPTON_HLT). The event is triggered by the W / Z lepton, not by the onium: the
low-mass dimuon path HLT_DoubleMu4_3_LowMass stops at 8.5 GeV and the Upsilon paths ask for pT(mu mu) > 10 GeV.

W: the semileptonic ttbar selection (ttsemilep_valid_sf) without its jet requirements: the single-lepton trigger, exactly
one lepton of the trigger flavour above the trigger threshold (muon: tight ID, pfRelIso04 < 0.12, |eta| < 2.4, pT > 30 GeV;
electron: MVA-isolation WP80 outside the barrel-endcap gap, pT above the electron threshold), MET > 50 GeV (the corrected
PUPPI MET), the MET filters. No jet multiplicity, and the jet veto map does not veto the event (stored as a flag).
Z: the opposite-sign pair of the trigger flavour closest to the Z mass in 60-120 GeV, both leptons of the ttbar
definition above 15 GeV and the leading one above the trigger threshold; no MET requirement.
The onium -> mu mu pair is built from the loose-ID muons that are not the W / Z leptons (utils/jpsi.py, no isolation);
the onium -> e e pair, independently, from the electrons of the eta -> mu mu e e analysis that are not the W / Z leptons
(utils/jpsi.onium_electron_mask: MVA no-isolation WP90, conversion veto, no missing inner hit, |eta| < 2.5 outside
1.44-1.57, pT > 5 GeV), so that its electron scale factors carry over. Other leptons of the ttbar definitions above 15 GeV
(either flavour, the leptons of the pair of the table apart) are counted, not cut on ("nextra"): a W with nextra = 0 is
not in a Z channel, and no event is in both W channels.

Output per dataset: {channel: {"cands": one row per selected W / Z event with an onium -> mu mu pair (opposite or same
sign, in the onium mass range), "cands_ee": the same for the onium -> e e pair (utils/jpsi.EE_RANGE, with the electron
fields utils/jpsi.ELE_FIELDS), "nV": all selected W / Z events in (good primary vertices, m_T of the W or m_ll of the Z,
jet veto flag, nextra 0 / >= 1), for the W / Z yield and the pileup (the rate of pileup onia)}, "sumw", and the processed
luminosity blocks of the data}. Both tables carry the PUPPI MET (corrected and raw) and the track MET (TrkMET, charged
PF candidates of the primary vertex: blind to the pileup onia). No other histograms, no scale factors. Data runs the
channels of its primary dataset; MC runs all four.
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
from BTVNanoCommissioning.utils.correction import common_shifts, jetveto, load_lumi, load_SF
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
LEP_PTMIN = 15.0  # the Z leptons and the extra-lepton count
MU_PTMIN = 30.0  # the trigger muon (IsoMu24; the ttbar muon)
MET_MIN = 50.0  # W (the ttbar l + jets MET requirement)
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
        # the corrections of the ttbar selection (JEC of the campaign, muon / electron scale and smearing); the jet veto map
        # is taken out, so that common_shifts keeps every event, and evaluated as a flag
        self.SF_map = load_SF(self._year, self._campaign)
        self.jv_map = {k: self.SF_map.pop(k) for k in ("jetveto", "jetveto_cfg") if k in self.SF_map}

    def process(self, events):
        events = missing_branch(events)
        dataset = events.metadata["dataset"]
        is_data = not hasattr(events, "genWeight")
        out = {"sumw": float(len(events)) if is_data else float(np.sum(np.asarray(events.genWeight), dtype=np.float64))}
        if is_data:
            out = dump_lumi(events[self.lumiMask(events.run, events.luminosityBlock)], out)
        _, shifts = common_shifts(self, events)
        ev = update(events, shifts[0][0])  # nominal: no systematic shifts
        jv = ak.any(jetveto(events.Jet, self.jv_map) > 0, axis=1) if "jetveto" in self.jv_map else ak.zeros_like(events.run, dtype=bool)
        ev["jetvetoed"] = ak.values_astype(jv, np.int64)
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
            # the ttbar l + jets lepton and MET requirements
            cand = tl & (L.pt > ptmin)
            ok = (ak.sum(cand, axis=1) == 1) & (ev.MET.pt > MET_MIN)
            i1 = ak.where(ok, ak.fill_none(ak.firsts(lidx[cand]), -1), -1)
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
        vel = vlep if elec else ak.zeros_like(tel)

        # the onium pairs, mu mu from the loose muons and e e from the eta -> mu mu e e electrons that are not W / Z leptons;
        # the other leptons outside the pair are counted (per table: the leptons of its own pair apart)
        info, pair = ONI.find_jpsi(e, ONI.jpsi_muon_mask(e, vmu))
        info["truth"] = ONI.jpsi_truth(e, info)
        nextra = _np(ak.sum(tmu & ~vmu & ~pair, axis=1) + ak.sum(tel & ~vel, axis=1), 0)
        info_ee, pair_ee = ONI.find_jpsi(e, ONI.onium_electron_mask(e, vel), coll="Electron", mass_range=ONI.EE_RANGE)
        info_ee["truth"] = ONI.jpsi_truth(e, info_ee, coll="Electron")
        nextra_ee = _np(ak.sum(tmu & ~vmu, axis=1) + ak.sum(tel & ~vel & ~pair_ee, axis=1), 0)
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

        jv = _np(e.jetvetoed, 0)
        nV = Hist.Hist(
            Hist.axis.Regular(25, 0, 100, name="npv", label="good primary vertices"),
            Hist.axis.Regular(30, 0, 300, name="mv", label="m_T of the W / m_ll of the Z [GeV]"),
            Hist.axis.Integer(0, 2, name="jv", label="jet veto map vetoes the event", underflow=False, overflow=False),
            Hist.axis.Integer(0, 2, name="nx", label="extra leptons (1: >= 1)", underflow=False, overflow=False),
            storage=Hist.storage.Weight(),
        )
        nV.fill(npv=npv, mv=_np(v["mv"], -1.0), jv=jv, nx=np.minimum(nextra, 1), weight=w)

        # the event columns, common to both tables
        cols = {"run": e.run, "lumi": e.luminosityBlock, "event": e.event, "npv": npv, "weight": w, "jetvetoed": jv}
        # the W / Z leptons (l2_*: -99 for the W) and the W / Z system
        for k, lep in (("l1", l1), ("l2", l2)):
            for f, src in zip(NV_FIELDS, ("pt", "eta", "phi", "charge", "pfRelIso03_all" if elec else "pfRelIso04_all", "dz")):
                cols[f"{k}_{f}"] = lep[src]
        cols.update(v)
        cols["met"], cols["met_phi"] = met.pt, met.phi
        if "PuppiMET" in e.fields:
            cols["puppimet"], cols["puppimet_phi"] = e.PuppiMET.pt, e.PuppiMET.phi
        if "TrkMET" in e.fields:
            cols["trkmet"], cols["trkmet_phi"] = e.TrkMET.pt, e.TrkMET.phi
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
        cols = {k: c if isinstance(c, np.ndarray) else _np(c) for k, c in cols.items()}
        return {
            "cands": self.table(e, info, nextra, cols, ONIUM_FIELDS),
            "cands_ee": self.table(e, info_ee, nextra_ee, cols, ONIUM_FIELDS + ONI.ELE_FIELDS),
            "nV": nV,
        }

    @staticmethod
    def table(e, info, nextra, cols, fields):
        """{column: column_accumulator} of the events with a valid pair: the event columns `cols` and the pair's"""
        ok = _np(info["valid"], False).astype(bool)
        pc = {"nextra": nextra, **{k: info[k] for k in fields}}
        pc["mreg"] = ONI.mass_region(info["mass"])
        pc["life"] = ONI.life_region(info)
        jp = info["p4"]
        pc["eta"], pc["phi"] = ak.fill_none(jp.eta, -99.0), ak.fill_none(jp.phi, -99.0)
        pc.update(ONI.near_objects(e, jp))
        pc = {k: c if isinstance(c, np.ndarray) else _np(c) for k, c in pc.items()}
        return {k: column_accumulator(np.asarray(c)[ok]) for k, c in {**cols, **pc}.items()}

    def postprocess(self, accumulator):
        return accumulator
