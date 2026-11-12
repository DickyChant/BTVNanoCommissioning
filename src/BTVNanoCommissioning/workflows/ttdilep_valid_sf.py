from BTVNanoCommissioning.helpers.func import V15_SCHEMA
import os
import awkward as ak, numpy as np
from coffea import processor

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.upart_sf import add_upart_weights
from BTVNanoCommissioning.utils import jpsi as JPSI
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
    reweighting,
)

# user helper function
from BTVNanoCommissioning.helpers.func import update, dump_lumi, PFCand_link
from BTVNanoCommissioning.helpers.update_branch import missing_branch

## load histograms & selctions for this workflow
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writer,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    btag_mu_idiso,
    btag_wp,
    HLT_helper,
    single_lepton_trigger,
    jet_id,
    mu_promptmvaid,
    ele_promptmvaid,
    mu_idiso,
    ele_cuttightid,
)


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        selectionModifier="tt_dilep",
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        self.selMod = selectionModifier
        ## Load corrections
        # the hadW (and hadW + J/psi) mode uses the tt_dilep lepton IDs, so it takes the tt_dilep scale factors
        self.SF_map = load_SF(
            self._year,
            self._campaign,
            "tt_dilep" if self.selMod in ("tt_dilep_hadw", "tt_dilep_hadw_e", "tt_dilep_jpsi") else self.selMod,
        )

    @property
    def accumulator(self):
        return self._accumulator

    ## Apply corrections on momentum/mass on MET, Jet, Muon
    def process(self, events):
        events = missing_branch(events)
        sumws = reweighting(events, self.isSyst)
        vetoed_events, shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(vetoed_events, collections), sumws, name)
            for collections, name in shifts
        )

    def process_shift(self, events, sumws, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")

        ## Create histograms
        output = {}
        if not self.noHist:
            output = histogrammer(
                events.Jet.fields,
                obj_list=["mu", "ele", "jet0", "jet1"],
                hist_collections=(
                    ["ttdilep"]
                    if os.environ.get("HADW_STUDY_HISTS")  # study histograms only (systematics runs)
                    else ["common", "fourvec", "ttdilep"]
                )
                + (["ttjpsi"] if self.selMod == "tt_dilep_jpsi" else []),
                hadw=self.selMod in ("tt_dilep_hadw", "tt_dilep_hadw_e", "tt_dilep_jpsi"),
                elec=self.selMod == "tt_dilep_hadw_e",
            )

        if shift_name is None:
            output["sumw"] = sumws["sumw"]
            if not isRealData and self.isSyst:
                if "LHEPdfWeight" in events.fields:
                    output["PDF_sumwUp"] = sumws["PDF_sumwUp"]
                    output["PDF_sumwDown"] = sumws["PDF_sumwDown"]
                    output["aS_sumwUp"] = sumws["aS_sumwUp"]
                    output["aS_sumwDown"] = sumws["aS_sumwDown"]
                    output["PDFaS_sumwUp"] = sumws["PDFaS_sumwUp"]
                    output["PDFaS_sumwDown"] = sumws["PDFaS_sumwDown"]
                if "LHEScaleWeight" in events.fields:
                    output["muR_sumwUp"] = sumws["muR_sumwUp"]
                    output["muR_sumwDown"] = sumws["muR_sumwDown"]
                    output["muF_sumwUp"] = sumws["muF_sumwUp"]
                    output["muF_sumwDown"] = sumws["muF_sumwDown"]
                if "PSWeight" in events.fields:
                    if len(events.PSWeight[0]) == 4:
                        output["ISR_sumwUp"] = sumws["ISR_sumwUp"]
                        output["ISR_sumwDown"] = sumws["ISR_sumwDown"]
                        output["FSR_sumwUp"] = sumws["FSR_sumwUp"]
                        output["FSR_sumwDown"] = sumws["FSR_sumwDown"]

        ####################
        #    Selections    #
        ####################

        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        ## HLT
        triggers = [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        ]
        hadw = self.selMod in ("tt_dilep_hadw", "tt_dilep_hadw_e", "tt_dilep_jpsi")
        # electron mirror: the same e mu selection triggered by the single electron (EGamma PD)
        elec = self.selMod == "tt_dilep_hadw_e"
        jpsi_mode = self.selMod == "tt_dilep_jpsi"
        if hadw:
            # run on the Muon primary dataset like the semileptonic study (paths per year: utils/selection.py)
            triggers, _ = single_lepton_trigger(self._campaign, elec)
        req_trig = HLT_helper(events, triggers)
        mutrig = (
            ak.values_astype(events.HLT.IsoMu24, np.int64)
            if elec and hasattr(events.HLT, "IsoMu24")
            else None
        )

        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        if self._campaign in V15_SCHEMA:  # NanoAODv15
            mu_sel = (events.Muon.pt > (30 if hadw else 25)) & (
                # hadronic-W study: the muon ID and isolation of the semileptonic selection
                btag_mu_idiso(events, self._campaign) if hadw else mu_promptmvaid(events, self._campaign)
            )
        else:
            mu_sel = (events.Muon.pt > 25) & mu_idiso(events, self._campaign)
        # J/psi -> mu mu (tt_dilep_jpsi): one dimuon from the loose-ID muons (no isolation) that are not ttbar-muon candidates
        # (utils/jpsi.py), built before the muon collection is reduced to the selected muon. The jets are
        # cleaned against the selected muon and electron only, so a b jet containing the J/psi stays.
        if jpsi_mode:
            jinfo, _ = JPSI.find_jpsi(events, JPSI.jpsi_muon_mask(events, mu_sel))
            jinfo["truth"] = JPSI.jpsi_truth(events, jinfo)
        events.Muon = events.Muon[mu_sel]
        events.Muon = ak.pad_none(events.Muon, 1, axis=1)
        req_muon = ak.count(events.Muon.pt, axis=1) == 1

        ## Electron cuts
        # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        if self._campaign in V15_SCHEMA:  # NanoAODv15
            events.Electron = events.Electron[
                (events.Electron.pt > (35 if elec else 25))  # 5 GeV above the Ele30 threshold
                & ele_promptmvaid(events, self._campaign)
            ]
        else:
            events.Electron = events.Electron[
                (events.Electron.pt > 25) & ele_cuttightid(events, self._campaign)
            ]

        events.Electron = ak.pad_none(events.Electron, 1, axis=1)
        req_ele = ak.count(events.Electron.pt, axis=1) == 1

        ## Jet cuts
        jetsel = ak.fill_none(
            jet_id(events, self._campaign)
            & (
                ak.all(
                    events.Jet.metric_table(events.Muon) > 0.4,
                    axis=2,
                    mask_identity=True,
                )
            )
            & (
                ak.all(
                    events.Jet.metric_table(events.Electron) > 0.4,
                    axis=2,
                    mask_identity=True,
                )
            ),
            False,
        )
        event_jet = events.Jet[jetsel]
        req_jets = ak.num(event_jet.pt) >= 2

        ## Other cuts
        req_opposite_charge = (
            events.Electron[:, 0:1].charge * events.Muon[:, 0:1].charge
        ) == -1
        req_opposite_charge = ak.fill_none(req_opposite_charge, False)
        req_opposite_charge = ak.flatten(req_opposite_charge)

        ## store jet index for PFCands, create mask on the jet index
        jetindx = ak.mask(ak.local_index(events.Jet.pt), jetsel)
        jetindx = ak.pad_none(jetindx, 2)
        jetindx = jetindx[:, :2]

        event_level = (
            req_trig & req_lumi & req_muon & req_ele & req_jets & req_opposite_charge
        )
        event_level = ak.fill_none(event_level, False)
        if len(events[event_level]) == 0:
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    None,
                    ["nominal"],
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}

        ####################
        # Selected objects #
        ####################

        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = event_jet[event_level][:, :2]
        pruned_ev["SelMuon"] = events.Muon[event_level][:, 0]
        pruned_ev["SelElectron"] = events.Electron[event_level][:, 0]
        pruned_ev["dr_mujet0"] = pruned_ev.Muon.delta_r(pruned_ev.Jet[:, 0])
        pruned_ev["dr_mujet1"] = pruned_ev.Muon.delta_r(pruned_ev.Jet[:, 1])
        pruned_ev["njet"] = ak.count(event_jet[event_level].pt, axis=1)

        if hadw:
            # b candidates: the two jets with the highest UParT score
            jets = event_jet[event_level]
            jets = jets[ak.argsort(jets.btagUParTAK4B, axis=1, ascending=False)]
            try:
                bm = btag_wp(jets, self._year, self._campaign, "UParTAK4", "b", "M")
            except Exception:
                bm = jets.btagUParTAK4B > 0.1272
            if "hadronFlavour" in jets.fields:
                # per-jet inputs of the UParT SFs and of the tagging-efficiency histogram (MC)
                pruned_ev["upeff_pt"] = jets.pt
                pruned_ev["upeff_eta"] = jets.eta
                pruned_ev["upeff_flav"] = ak.values_astype(jets.hadronFlavour, np.int64)
                pruned_ev["upeff_tag"] = ak.values_astype(ak.fill_none(bm, False), np.int64)
            pruned_ev["dil_nb"] = ak.values_astype(
                ak.sum(ak.fill_none(bm[:, :2], False), axis=1), np.int64
            )
            j1, j2 = jets[:, 0], jets[:, 1]
            mu, el = pruned_ev.SelMuon, pruned_ev.SelElectron
            # pairing A = (mu, j1)(e, j2), B = (mu, j2)(e, j1)
            mA_mu, mA_e = (mu + j1).mass, (el + j2).mass
            mB_mu, mB_e = (mu + j2).mass, (el + j1).mass
            use_a = {
                "dr": (mu.delta_r(j1) + el.delta_r(j2)) < (mu.delta_r(j2) + el.delta_r(j1)),
                "mm": np.maximum(mA_mu, mA_e) < np.maximum(mB_mu, mB_e),
            }
            for tag, ua in use_a.items():
                pruned_ev[f"dil_mlb_mu_{tag}"] = ak.where(ua, mA_mu, mB_mu)
                pruned_ev[f"dil_mlb_e_{tag}"] = ak.where(ua, mA_e, mB_e)
                if "partonFlavour" in jets.fields:
                    # the b of a l+ is a b quark (partonFlavour +5), of a l- a bbar
                    pf_mu = ak.where(ua, j1.partonFlavour, j2.partonFlavour)
                    pf_e = ak.where(ua, j2.partonFlavour, j1.partonFlavour)
                    right = (pf_mu == 5 * mu.charge) & (pf_e == 5 * el.charge)
                    swapped = (pf_mu == 5 * el.charge) & (pf_e == 5 * mu.charge)
                    t = ak.where(right, 1, ak.where(swapped, 0, 2))
                else:
                    t = ak.full_like(mu.pt, 2)
                pruned_ev[f"dil_truth_{tag}"] = ak.values_astype(t, np.int64)
            chg = lambda j: j.nConstituents - j.neMultiplicity  # noqa: E731
            pruned_ev["dil_ntrack"] = (pruned_ev.PV.ndof + 3.0) / 2.0 - chg(j1) - chg(j2) - 2.0
            if elec:
                pruned_ev["dil_mutrig"] = mutrig[event_level]
                # diagnostics per lepton with the minimax pairing: lepton pT, Delta R(l, b), b pT;
                # and the two n_track terms
                ua = use_a["mm"]
                for lep, lv, ja, jb in (("mu", mu, j1, j2), ("e", el, j2, j1)):  # pairing A: (mu, j1)(e, j2)
                    pruned_ev[f"dgd_lpt_{lep}"] = lv.pt
                    pruned_ev[f"dgd_dr_{lep}"] = ak.where(ua, lv.delta_r(ja), lv.delta_r(jb))
                    pruned_ev[f"dgd_bpt_{lep}"] = ak.where(ua, ja.pt, jb.pt)
                pruned_ev["dgd_ntpv"] = (pruned_ev.PV.ndof + 3.0) / 2.0
                pruned_ev["dgd_ntj"] = chg(j1) + chg(j2)

            if jpsi_mode:
                # b candidates: the two highest-score jets; each lepton's b from the minimax pairing.
                # b jets for the J/psi matching: the b candidates and every UParT-M tagged selected jet
                assoc = JPSI.jet_association(
                    jinfo["p4"][ak.to_numpy(event_level)], jets, jets[:, :2], jets[ak.fill_none(bm, False)]
                )
                ua = use_a["mm"]
                JPSI.store(
                    pruned_ev,
                    jinfo,
                    event_level,
                    assoc,
                    {"mu": (mu, ak.where(ua, 0, 1)), "e": (el, ak.where(ua, 1, 0))},
                    pruned_ev.dil_nb,
                    pruned_ev.dil_ntrack,
                    {"mlb_mu": pruned_ev.dil_mlb_mu_mm, "mlb_e": pruned_ev.dil_mlb_e_mm},
                )

        # Find the PFCands associate with selected jets. Search from jetindex->JetPFCands->PFCand
        if "PFCands" in events.fields:
            pruned_ev.PFCands = PFCand_link(events, event_level, jetindx)

        ####################
        #     Output       #
        ####################

        # Configure SFs
        weights = weight_manager(
            pruned_ev,
            self.SF_map,
            self.isSyst,
            ttbar_reweights=getattr(self, "ttbar_reweights", "none"),
            campaign=self._campaign,
        )
        # UParT b-tag SFs, method 1a over all selected jets (N_b of the two highest-score
        # jets is min(number of tagged jets, 2), a function of the full tag configuration)
        if hadw:
            add_upart_weights(weights, pruned_ev, self._campaign, "dilep", self.isSyst)
        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        # Configure histograms
        if not self.noHist:
            output = histo_writer(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )
        # the J/psi candidates, one row each, and the pileup of all selected events
        if jpsi_mode and shift_name is None:
            w = None if isRealData else weights.weight()
            # all selected jets with their UParT medium tag, for the ttbar system offline
            alljets = event_jet[event_level]
            try:
                isb = btag_wp(alljets, self._year, self._campaign, "UParTAK4", "b", "M")
            except Exception:
                isb = alljets.btagUParTAK4B > 0.1272
            output["jpsi_cands"] = JPSI.candidate_table(pruned_ev, w, jets=alljets, isb=ak.fill_none(isb, False))
            output["npv"] = JPSI.npv_hist(pruned_ev, w)
        # Output arrays
        if self.isArray:
            array_writer(
                self, pruned_ev, events, weights, systematics, dataset, isRealData
            )
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
