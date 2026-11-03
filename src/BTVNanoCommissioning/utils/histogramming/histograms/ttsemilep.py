import hist as Hist

from BTVNanoCommissioning.utils.upart_sf import ETA_EDGES, PT_EDGES


def get_histograms(axes, **kwargs):
    hists = {}

    c_ttsemilep = kwargs.get("c_ttsemilep", None)
    if c_ttsemilep == None:
        raise ValueError(
            "c_ttsemilep is not specified when running ttsemilep workflow."
        )

    hists["dr_cjet"] = Hist.Hist(
        axes["syst"], axes["flav"], axes["dr"], Hist.storage.Weight()
    )
    if not c_ttsemilep:
        for i in range(4):
            hists[f"dr_mujet{i}"] = Hist.Hist(
                axes["syst"], axes["flav"], axes["dr"], Hist.storage.Weight()
            )
        # Hadronic W mass (non-b light-jet pair closest to 80.4 GeV) and the
        # pileup/UE track observable: PV tracks (from PV_ndof) minus the tracks
        # of the ttbar jets and the selected muon (cf. CMS UE-in-ttbar, 1807.02810).
        hists["hadW_mass"] = Hist.Hist(axes["syst"], axes["mW"], Hist.storage.Weight())
        hists["n_track"] = Hist.Hist(
            axes["syst"], axes["ntrack"], Hist.storage.Weight()
        )
        hists["ntrk_pv"] = Hist.Hist(axes["syst"], axes["ntrk"], Hist.storage.Weight())
        hists["njet_trk"] = Hist.Hist(axes["syst"], axes["ntrk"], Hist.storage.Weight())
        # 2D: hadronic W mass profiled vs the track observable (the deliverable)
        hists["hadW_mass_vs_ntrack"] = Hist.Hist(
            axes["syst"], axes["ntrack"], axes["mW"], Hist.storage.Weight()
        )
        # b-jet multiplicity, and the profile sliced by it (to purify ttbar:
        # slice nbj>=1 or >=2 at plot time)
        hists["nbjet"] = Hist.Hist(axes["syst"], axes["nbj"], Hist.storage.Weight())
        hists["hadW_mass_vs_ntrack_nb"] = Hist.Hist(
            axes["syst"], axes["nbj"], axes["ntrack"], axes["mW"], Hist.storage.Weight()
        )
        # leptonic side: charge split of the full histogram, and each leptonic
        # variable against charge, N_b, coarse n_track and m_jj
        hists["hadW_lep_charge"] = Hist.Hist(
            axes["syst"],
            axes["lepq"],
            axes["nbj"],
            axes["ntrack"],
            axes["mW"],
            Hist.storage.Weight(),
        )
        # leptonic b: m(mu, b_lep) by assignment method and MC truth; hadronic top mass
        hists["hadW_lepb_mlb"] = Hist.Hist(
            axes["syst"],
            axes["lepbmeth"],
            axes["agree"],
            axes["btruth"],
            axes["nbj"],
            axes["ntrack_c"],
            axes["mlbfine"],
            Hist.storage.Weight(),
        )
        hists["hadW_top_mjjb"] = Hist.Hist(
            axes["syst"],
            axes["btruth"],
            axes["nbj"],
            axes["ntrack_c"],
            axes["mjjb"],
            Hist.storage.Weight(),
        )
        # UParT tagging efficiency per jet (MC): hadron flavour x pT x |eta| x tagged, for the
        # method-1a b-tag SFs (utils/upart_sf.py); filled for the nominal weight only
        hists["upart_eff"] = Hist.Hist(
            Hist.axis.IntCategory([0, 4, 5], name="flav"),
            Hist.axis.Variable(PT_EDGES, name="pt"),
            Hist.axis.Variable(ETA_EDGES, name="abseta"),
            Hist.axis.IntCategory([0, 1], name="tag"),
            Hist.storage.Weight(),
        )
        for v in ("mT", "met", "mlb"):
            hists[f"hadW_lep_{v}"] = Hist.Hist(
                axes["syst"],
                axes[f"lep{v}"],
                axes["lepq"],
                axes["nbj"],
                axes["ntrack_c"],
                axes["mW_c"],
                Hist.storage.Weight(),
            )
    for i in ["mu"]:
        hists[f"{i}_pfRelIso04_all"] = Hist.Hist(
            axes["syst"], axes["iso"], Hist.storage.Weight()
        )
        hists[f"{i}_dxy"] = Hist.Hist(axes["syst"], axes["dxy"], Hist.storage.Weight())
        hists[f"{i}_dz"] = Hist.Hist(axes["syst"], axes["dz"], Hist.storage.Weight())

    return hists
