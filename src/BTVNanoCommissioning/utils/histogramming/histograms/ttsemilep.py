import hist as Hist


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
    for i in ["mu"]:
        hists[f"{i}_pfRelIso04_all"] = Hist.Hist(
            axes["syst"], axes["iso"], Hist.storage.Weight()
        )
        hists[f"{i}_dxy"] = Hist.Hist(axes["syst"], axes["dxy"], Hist.storage.Weight())
        hists[f"{i}_dz"] = Hist.Hist(axes["syst"], axes["dz"], Hist.storage.Weight())

    return hists
