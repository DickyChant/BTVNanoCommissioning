import collections, gc
import os
import uproot
import numpy as np, awkward as ak

from coffea import processor
from coffea.analysis_tools import Weights
from BTVNanoCommissioning.utils.upart_sf import add_upart_weights
from BTVNanoCommissioning.utils import jpsi as JPSI
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
    reweighting,
)

from BTVNanoCommissioning.helpers.func import update, dump_lumi, PFCand_link
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writter,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    btag_mu_idiso,
    btag_wp,
    ele_mvatightid,
    MET_filters,
)


class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        selectionModifier="tt_semilep",
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        ### Added selection for ttbar semileptonic
        self.ttaddsel = selectionModifier
        ## Load corrections
        self.SF_map = (
            load_SF(self._year, self._campaign, "tt_semilep_e")
            if selectionModifier in ("tt_semilep_e", "tt_semilep_e_jpsi")
            else load_SF(self._year, self._campaign)
        )

    @property
    def accumulator(self):
        return self._accumulator

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
        histname = (
            "ttsemilep_sf" if self.ttaddsel != "c_tt_semilep" else "c_ttsemilep_sf"
        )
        output = {}
        if not self.noHist:
            objs = ["mu", "MET", "cjet"]
            if not self.ttaddsel == "c_tt_semilep":
                for i in range(4):
                    objs.append(f"jet{i}")

            output = histogrammer(
                events.Jet.fields,
                obj_list=objs,
                hist_collections=(
                    ["ttsemilep"]
                    if os.environ.get("HADW_STUDY_HISTS")  # study histograms only (systematics runs)
                    else ["common", "fourvec", "ttsemilep"]
                )
                + (["ttjpsi"] if self.ttaddsel in ("tt_semilep_jpsi", "tt_semilep_e_jpsi") else []),
                c_ttsemilep=self.ttaddsel == "c_tt_semilep",
                elec=self.ttaddsel in ("tt_semilep_e", "tt_semilep_e_jpsi"),
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
        # electron mirror of the hadronic-W study: single-electron trigger (EGamma PD), one tight electron
        elec = self.ttaddsel in ("tt_semilep_e", "tt_semilep_e_jpsi")
        # (Ele30_WPTight_Gsf: the lowest unprescaled 2024 path, the one with EGM trigger SFs)
        triggers = ["Ele30_WPTight_Gsf"] if elec else ["IsoMu24"]
        req_trig = HLT_helper(events, triggers)

        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        if elec:
            # MVA-isolation WP80 (the EGM recommendation with 2024 SFs), pT 5 GeV above the trigger threshold
            event_muon = events.Electron[
                (events.Electron.pt > 35) & ele_mvatightid(events, self._campaign)
            ]
        else:
            event_muon = events.Muon[
                (events.Muon.pt > 30) & btag_mu_idiso(events, self._campaign)
            ]
        # event_muon = ak.pad_none(events.Muon, 1, axis=1)
        req_muon = ak.count(event_muon.pt, axis=1) == 1

        # J/psi -> mu mu (tt_semilep_jpsi, tt_semilep_e_jpsi): one dimuon from the loose-ID muons (no isolation) that
        # are not tight isolated pT > 30 GeV muons (utils/jpsi.py). The jets are then cleaned against the tight
        # isolated pT > 30 GeV muons only, as in the dilepton workflow: cleaning against every other muon (any ID)
        # removed most b jets around a J/psi, through a soft muon of the b decay or a muon object without ID in the
        # jet, so that b-decay J/psi counted as not b-matched; and with mask_identity=True an event without such a
        # muon had no jets, so the e + jets channel kept only events with an extra muon.
        jpsi_mode = self.ttaddsel in ("tt_semilep_jpsi", "tt_semilep_e_jpsi")
        clean_mu = events.Muon
        if jpsi_mode:
            tight_mu = (events.Muon.pt > 30) & btag_mu_idiso(events, self._campaign)
            jinfo, jpair = JPSI.find_jpsi(events, JPSI.jpsi_muon_mask(events, tight_mu))
            jinfo["truth"] = JPSI.jpsi_truth(events, jinfo)
            clean_mu = events.Muon[tight_mu & ~jpair]

        ## Jet cuts
        # the electron mirror keeps the muon-channel jet definition (away from the muons of clean_mu) and also
        # removes jets within 0.4 of the selected electron
        clean_e = (
            ak.all(events.Jet.metric_table(event_muon) > 0.4, axis=2, mask_identity=True)
            if elec
            else True
        )
        event_jet = events.Jet[
            ak.fill_none(
                jet_id(events, self._campaign)
                & (
                    # mask_identity=False: no muon to clean against keeps the jet (True would give None, i.e. no jets)
                    ak.all(
                        events.Jet.metric_table(clean_mu) > 0.4,
                        axis=2,
                        mask_identity=False,
                    )
                )
                & clean_e,
                False,
                axis=-1,
            )
        ]
        req_jets = ak.num(event_jet.pt) >= 4

        if self.ttaddsel == "c_tt_semilep":

            # Sort all jets by pt in descending order
            sorted_jets = event_jet[ak.argsort(event_jet.pt, axis=-1, ascending=False)]

            # Pad the sorted jets to ensure there are at least two jets in each event
            padded_jets = ak.pad_none(sorted_jets, 2, axis=1)

            # Determine the leading and subleading jets
            leading_jet = padded_jets[:, 0]
            subleading_jet = padded_jets[:, 1]

            c_selection = (
                event_jet.btagDeepFlavC > 0.6
            )  # & (event_jet.pt == leading_jet.pt)
            ### plot # c jets , DeepFlavC of the c jets
            c_selection = ak.fill_none(c_selection, False)
            req_c_jets = ak.num(event_jet[c_selection].pt) >= 1
            req_c_jets = ak.fill_none(req_c_jets, False)
            c_jets = event_jet[c_selection]
            single_c_jet = ak.firsts(c_jets)
            remaining_jets = event_jet[~c_selection]
            nob_selection = remaining_jets.btagDeepFlavB < 0.5

            nob_candidate_jets = remaining_jets[nob_selection]

            # Create combinations of single_c_jet and nob_candidate_jets
            combinations = ak.cartesian(
                {"c_jet": single_c_jet, "nob_jet": nob_candidate_jets}, axis=1
            )

            # Calculate the invariant mass of each pair using the .mass attribute
            masses = (combinations["c_jet"] + combinations["nob_jet"]).mass

            # Create a mask for pairs with mass between 55 and 110
            mass_mask = (masses > 50) & (masses < 110)

            # Apply the mask to get the valid combinations
            valid_combinations = combinations[mass_mask]
            # Create an event-level mask
            event_mask = ak.any(mass_mask, axis=1)

            req_w_mass = ak.fill_none(event_mask, False)

            nob_w_selection = (
                event_jet.btagDeepFlavB < 0.5
            )  # & (event_jet.pt == subleading_jet.pt)
            nob_w_selection = ak.fill_none(nob_w_selection, False)
            w_cand_b_jet = ak.firsts(event_jet[nob_w_selection])

            w_mass_lead_sublead = (single_c_jet + w_cand_b_jet).mass

            w_mask = (w_mass_lead_sublead > 50) & (w_mass_lead_sublead < 110)
            w_mask = ak.fill_none(w_mask, False)

            # req_w_jets = ak.fill_none(req_w_jets, False)

        ## store jet index for PFCands, create mask on the jet index
        jetindx = ak.mask(
            ak.local_index(events.Jet.pt),
            (
                jet_id(events, self._campaign)
                & (
                    ak.all(
                        events.Jet.metric_table(clean_mu) > 0.4,
                        axis=2,
                        mask_identity=False,
                    )
                )
            )
            == 1,
        )
        jetindx = ak.pad_none(jetindx, 4)
        jetindx = jetindx[:, :4]

        ## other cuts

        MET = ak.zip(
            {
                "pt": events.MET.pt,
                "eta": ak.zeros_like(events.MET.pt),
                "phi": events.MET.phi,
                "mass": ak.zeros_like(events.MET.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        req_MET = MET.pt > 50

        req_metfilter = MET_filters(events, self._campaign)
        event_level = ak.fill_none(
            req_trig & req_jets & req_muon & req_MET & req_lumi & req_metfilter, False
        )
        # Combine masks with the original conditions
        if self.ttaddsel == "c_tt_semilep":
            event_level = ak.fill_none(event_level & req_w_mass & req_c_jets, False)
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
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = (
            event_jet[event_level][:, :4]
            if self.ttaddsel != "c_tt_semilep"
            else c_jets[event_level][:, 0]
        )
        # the selected lepton (SelElectron in the electron mirror: electron SFs, no muon SFs)
        pruned_ev["SelElectron" if elec else "SelMuon"] = event_muon[event_level][:, 0]
        sel_lep = pruned_ev.SelElectron if elec else pruned_ev.SelMuon
        pruned_ev["njet"] = ak.count(event_jet[event_level].pt, axis=1)
        if "PFCands" in events.fields:
            pruned_ev.PFCands = PFCand_link(events, event_level, jetindx)
        if self.ttaddsel != "c_tt_semilep":
            for i in range(4):
                pruned_ev[f"dr_mujet{i}"] = sel_lep.delta_r(
                    pruned_ev.SelJet[:, i]
                )
        else:
            pruned_ev[f"dr_mujet0"] = sel_lep.delta_r(pruned_ev.SelJet)

        ###############################################################
        # Hadronic W mass vs pileup-subtracted track multiplicity     #
        # (semileptonic ttbar; cf. CMS UE-in-ttbar, arXiv:1807.02810) #
        ###############################################################
        if self.ttaddsel != "c_tt_semilep":
            sel4 = pruned_ev.SelJet  # leading-4 selected jets (the ttbar jets)
            # b tag (UParT medium, the 2024 recommendation): the hadronic W is the pair
            # of non-b light jets. (DeepJet has no 2024 working point in the BTV
            # metadata; the earlier btag_wp("DeepFlav") call fell back to > 0.3.)
            try:
                bmask = btag_wp(sel4, self._year, self._campaign, "UParTAK4", "b", "M")
            except Exception:
                # fallback if BTV WP metadata is unavailable for this campaign
                bmask = sel4.btagUParTAK4B > 0.1272
            bmask = ak.fill_none(bmask, False)
            if "hadronFlavour" in sel4.fields:
                # per-jet inputs of the UParT SFs and of the tagging-efficiency histogram (MC)
                pruned_ev["upeff_pt"] = sel4.pt
                pruned_ev["upeff_eta"] = sel4.eta
                pruned_ev["upeff_flav"] = ak.values_astype(sel4.hadronFlavour, np.int64)
                pruned_ev["upeff_tag"] = ak.values_astype(bmask, np.int64)
            # b-jet multiplicity (DeepFlav-M) among the 4 ttbar jets -> lets us
            # slice inclusive / >=1b / >=2b at plot time to purify ttbar
            pruned_ev["nbjet"] = ak.values_astype(ak.sum(bmask, axis=1), np.int64)
            light = sel4[~bmask]
            # all non-b light-jet pairs; pick the one closest to m_W = 80.4 GeV
            pairs = ak.combinations(light, 2, fields=["j1", "j2"])
            pair_mass = (pairs.j1 + pairs.j2).mass
            has_pair = ak.num(pair_mass, axis=1) > 0
            best = ak.argmin(np.abs(pair_mass - 80.4), axis=1, keepdims=True)
            hadW_mass = ak.firsts(pair_mass[best])
            pruned_ev["hadW_valid"] = ak.fill_none(has_pair, False)
            pruned_ev["hadW_mass"] = ak.fill_none(hadW_mass, -1.0)

            # Leptonic side, to separate ttbar from W+jets without touching the
            # hadronic pairing: muon charge (W+jets is charge-asymmetric), mT(mu,
            # MET), MET, and the smaller m(mu, j) of the two jets left after the
            # W pair (the leptonic b in ttbar: endpoint ~153 GeV).
            mu = sel_lep  # the muon, or the electron in the electron mirror
            met = pruned_ev.MET
            pruned_ev["lep_charge"] = ak.values_astype(mu.charge, np.int64)
            pruned_ev["lep_met"] = met.pt
            pruned_ev["lep_mT"] = np.sqrt(
                2.0 * mu.pt * met.pt * (1.0 - np.cos(mu.phi - met.phi))
            )
            jidx = ak.local_index(sel4.pt)
            pidx = ak.combinations(jidx[~bmask], 2, fields=["i1", "i2"])
            i1 = ak.fill_none(ak.firsts(pidx.i1[best]), -1)
            i2 = ak.fill_none(ak.firsts(pidx.i2[best]), -1)
            others = sel4[(jidx != i1) & (jidx != i2)]
            mlb = ak.min((others + mu).mass, axis=1)
            pruned_ev["lep_mlb"] = ak.fill_none(mlb, -1.0)

            # The two jets outside the W pair are the b candidates (with N_b = 2,
            # the two b tags). Hadronic b: the one whose mass with the W pair is
            # closest to m_t = 172.5 GeV; leptonic b: the other. Geometric
            # alternative: the leptonic b is the candidate closest to the muon.
            two = ak.num(others.pt, axis=1) == 2
            wvec = ak.firsts((pairs.j1 + pairs.j2)[best])
            mjjb = (others + wvec).mass
            ihad = ak.argmin(np.abs(mjjb - 172.5), axis=1, keepdims=True)
            ilep_mt = 1 - ihad
            ilep_dr = ak.argmin(others.delta_r(mu), axis=1, keepdims=True)
            blep_mt = ak.firsts(others[ilep_mt])
            blep_dr = ak.firsts(others[ilep_dr])
            pruned_ev["top_valid"] = ak.fill_none(two & has_pair, False)
            pruned_ev["lepb_agree"] = ak.values_astype(
                ak.fill_none(ak.firsts(ilep_mt == ilep_dr), False), np.int64
            )
            pruned_ev["top_mjjb"] = ak.fill_none(ak.firsts(mjjb[ihad]), -1.0)
            pruned_ev["lepb_mlb_mt"] = ak.fill_none((mu + blep_mt).mass, -1.0)
            pruned_ev["lepb_mlb_dr"] = ak.fill_none((mu + blep_dr).mass, -1.0)
            # MC truth of the leptonic-b choice: the leptonic top of a mu+ is a t,
            # its b has partonFlavour +5 (and -5 for a mu-). 1 right b, 0 the
            # other b, 2 not a b (or data).
            if "partonFlavour" in sel4.fields:
                want = 5 * mu.charge

                def truth(bj):
                    pf = ak.fill_none(bj.partonFlavour, 0)
                    return ak.where(pf == want, 1, ak.where(pf == -want, 0, 2))

                pruned_ev["lepb_truth_mt"] = ak.values_astype(truth(blep_mt), np.int64)
                pruned_ev["lepb_truth_dr"] = ak.values_astype(truth(blep_dr), np.int64)
            else:
                pruned_ev["lepb_truth_mt"] = ak.values_astype(ak.full_like(mu.pt, 2), np.int64)
                pruned_ev["lepb_truth_dr"] = ak.values_astype(ak.full_like(mu.pt, 2), np.int64)

            # Track observable: PV tracks (from PV_ndof) minus the charged
            # constituents of the ttbar jets minus the selected muon (1 track).
            # ntrk_PV ~ (PV_ndof + 3)/2; charged constituents per jet ~
            # nConstituents - neMultiplicity (central NanoAODv15 has no PFCands).
            ntrk_pv = (pruned_ev.PV.ndof + 3.0) / 2.0
            charged_const = sel4.nConstituents - sel4.neMultiplicity
            njet_trk = ak.sum(charged_const, axis=1)
            pruned_ev["ntrk_pv"] = ntrk_pv
            pruned_ev["njet_trk"] = njet_trk
            pruned_ev["n_track"] = ntrk_pv - njet_trk - 1.0

            # leptonic-side diagnostics (leptonic b of the m(jjb) choice), against n_track
            pruned_ev["dg_lpt"] = mu.pt
            pruned_ev["dg_dr"] = ak.fill_none(mu.delta_r(blep_mt), -1.0)
            pruned_ev["dg_bpt"] = ak.fill_none(blep_mt.pt, -1.0)
            pruned_ev["dg_ntpv"] = ntrk_pv
            pruned_ev["dg_ntj"] = njet_trk

            if jpsi_mode:
                # b candidates: the two jets outside the W pair (none without a valid pair); the
                # leptonic b of the lepton is the closer one (method 1, as for m(l, b_lep))
                top_ok = ak.fill_none(two & has_pair, False)
                bcands = others[ak.broadcast_arrays(top_ok, others.pt)[0]]
                # b jets for the J/psi matching: the b candidates and every UParT-M tagged selected jet
                jets_sel = event_jet[event_level]
                try:
                    bm_all = btag_wp(jets_sel, self._year, self._campaign, "UParTAK4", "b", "M")
                except Exception:
                    bm_all = jets_sel.btagUParTAK4B > 0.1272
                assoc = JPSI.jet_association(
                    jinfo["p4"][ak.to_numpy(event_level)], jets_sel, bcands, jets_sel[ak.fill_none(bm_all, False)]
                )
                JPSI.store(
                    pruned_ev,
                    jinfo,
                    event_level,
                    assoc,
                    {("e" if elec else "mu"): (mu, ak.fill_none(ak.firsts(ilep_dr), -1))},
                    pruned_ev.nbjet,
                    pruned_ev.n_track,
                    {
                        "mjj": ak.where(pruned_ev.hadW_valid, pruned_ev.hadW_mass, -1.0),
                        "mjjb": ak.where(pruned_ev.top_valid, pruned_ev.top_mjjb, -1.0),
                        ("mlb_e" if elec else "mlb_mu"): ak.where(pruned_ev.top_valid, pruned_ev.lepb_mlb_dr, -1.0),
                    },
                )

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
        # UParT b-tag SFs, method 1a (N_b is counted among the 4 leading jets)
        add_upart_weights(weights, pruned_ev, self._campaign, "semilep", self.isSyst)
        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        # Configure histograms
        if not self.noHist:
            output = histo_writter(
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
