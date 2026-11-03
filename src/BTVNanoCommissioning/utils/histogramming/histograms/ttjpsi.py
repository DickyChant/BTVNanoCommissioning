import hist as Hist


def get_histograms(axes, **kwargs):
    """J/psi -> mu mu on top of the ttbar selections (utils/jpsi.py), filled for events with a dimuon in 2.0-4.5 GeV.
    jpsicat: 0 matched to a b jet (control: mostly b decays), 1 in another selected jet, 2 in no jet (signal: 1 and 2); jpsitruth: 0 prompt, 1 from b, 2 no J/psi / data;
    mreg: 0/2 sidebands, 1 signal window; osss: opposite (+1) / same (-1) sign; nmed: muons passing the medium ID (both loose);
    life: lifetime region from l_xy and z0 (utils/jpsi.py life_region): 0 prompt, 1 non-prompt, 2 in between, 3 pileup-like, 4 no vertex."""
    hists = {}
    hists["jpsi_mass"] = Hist.Hist(
        axes["syst"], axes["osss"], axes["jpsinmed"], axes["jpsicat"], axes["jpsitruth"], axes["nbj"], axes["jpsilife"],
        axes["jpsisltrig"], axes["mjpsi"], Hist.storage.Weight(),
    )
    # pseudo-proper decay length, by |z0| (pileup) and N_b
    hists["jpsi_lxy"] = Hist.Hist(
        axes["syst"], axes["mreg"], axes["osss"], axes["jpsicat"], axes["jpsitruth"], axes["nbj"], axes["jpsisltrig"], axes["jpsiz0"],
        axes["jpsilxy"], Hist.storage.Weight(),
    )
    hists["jpsi_pt"] = Hist.Hist(
        axes["syst"], axes["mreg"], axes["osss"], axes["jpsicat"], axes["jpsitruth"], axes["jpsilife"], axes["jpsipt"],
        axes["jpsiabsy"], Hist.storage.Weight(),
    )
    hists["jpsi_sip"] = Hist.Hist(
        axes["syst"], axes["mreg"], axes["osss"], axes["jpsicat"], axes["jpsitruth"], axes["jpsisip"], Hist.storage.Weight()
    )
    hists["jpsi_drj"] = Hist.Hist(
        axes["syst"], axes["mreg"], axes["osss"], axes["jpsitruth"], axes["jpsidr"], Hist.storage.Weight()
    )
    # z = pT(J/psi) / pT(jet), J/psi inside a selected jet
    hists["jpsi_z"] = Hist.Hist(
        axes["syst"], axes["mreg"], axes["osss"], axes["jpsicat"], axes["jpsitruth"], axes["jpsiz"], Hist.storage.Weight()
    )
    # m(l, J/psi) per ttbar lepton, by the reco and MC-truth side of the J/psi b, N_b and coarse n_track
    hists["jpsi_mlj"] = Hist.Hist(
        axes["syst"], axes["lepflav"], axes["mreg"], axes["osss"], axes["jpsirside"], axes["jpsitside"], axes["nbj"],
        axes["ntrack_c"], axes["mlj"], Hist.storage.Weight(),
    )
    # ttbar kinematics of the J/psi events (to show they are ttbar): hadronic W and top (l + jets), m(l, b)
    # (l + jets: b_lep closest to the lepton; e mu: minimax pairing)
    hists["jpsi_tt_mjj"] = Hist.Hist(
        axes["syst"], axes["mreg"], axes["osss"], axes["jpsicat"], axes["nbj"], axes["jpsilife"], axes["mW_c"], Hist.storage.Weight()
    )
    hists["jpsi_tt_mjjb"] = Hist.Hist(
        axes["syst"], axes["mreg"], axes["osss"], axes["jpsicat"], axes["nbj"], axes["jpsilife"], axes["mjjb"], Hist.storage.Weight()
    )
    hists["jpsi_tt_mlb"] = Hist.Hist(
        axes["syst"], axes["lepflav"], axes["mreg"], axes["osss"], axes["jpsicat"], axes["nbj"], axes["jpsilife"], axes["mlbfine"],
        Hist.storage.Weight(),
    )
    return hists
