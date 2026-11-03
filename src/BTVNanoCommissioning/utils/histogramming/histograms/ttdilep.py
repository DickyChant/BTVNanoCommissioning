import hist as Hist

from BTVNanoCommissioning.utils.upart_sf import ETA_EDGES, PT_EDGES


def get_histograms(axes, **kwargs):
    hists = {}

    for i in range(2):
        hists[f"dr_mujet{i}"] = Hist.Hist(
            axes["syst"], axes["flav"], axes["dr"], Hist.storage.Weight()
        )

    for i in ["mu", "ele"]:
        if i == "mu":
            hists[f"{i}_pfRelIso04_all"] = Hist.Hist(
                axes["syst"], axes["iso"], Hist.storage.Weight()
            )
        else:
            hists[f"{i}_pfRelIso03_all"] = Hist.Hist(
                axes["syst"], axes["iso"], Hist.storage.Weight()
            )
        hists[f"{i}_dxy"] = Hist.Hist(axes["syst"], axes["dxy"], Hist.storage.Weight())
        hists[f"{i}_dz"] = Hist.Hist(axes["syst"], axes["dz"], Hist.storage.Weight())

    if kwargs.get("hadw", False):
        # e mu ttbar for the hadronic-W study: m(l, b) per lepton, by pairing
        # (0: geometric, sum of Delta R; 1: minimax mass) and MC truth
        hists["dil_mlb"] = Hist.Hist(
            axes["syst"],
            axes["lepbmeth"],
            axes["btruth"],
            axes["nbj"],
            axes["ntrack_c"],
            axes["mlbfine"],
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
        # the same split by the lepton of the pair (muon-only m(mu, b), and mu vs e)
        hists["dil_mlb_fl"] = Hist.Hist(
            axes["syst"],
            axes["lepflav"],
            axes["lepbmeth"],
            axes["btruth"],
            axes["nbj"],
            axes["ntrack_c"],
            axes["mlbfine"],
            Hist.storage.Weight(),
        )
        hists["dil_ntrack"] = Hist.Hist(
            axes["syst"], axes["nbj"], axes["ntrack"], Hist.storage.Weight()
        )
    return hists
