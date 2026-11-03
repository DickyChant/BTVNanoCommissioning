"""Electron mirror of the hadronic-W / m(l, b) study in one pass over the events (EGamma PD and MC):
  "sl":  e + jets, the mirror of mu + jets (Ele30_WPTight_Gsf, one MVA-iso WP80 electron, pT > 35 GeV)
  "dil": e mu triggered by the single electron (electron pT > 35 GeV, the muon as in the IsoMu24 e mu
         selection), split by whether IsoMu24 also fired
Output per dataset: {"sl": <ttsemilep histograms>, "dil": <ttdilep histograms>}."""
from coffea import processor

from BTVNanoCommissioning.workflows.ttdilep_valid_sf import NanoProcessor as TTdilep
from BTVNanoCommissioning.workflows.ttsemilep_valid_sf import NanoProcessor as TTsemilep


class NanoProcessor(processor.ProcessorABC):
    def __init__(self, year="2022", campaign="Summer22Run3", name="", isSyst=False, isArray=False, noHist=False, chunksize=75000):
        args = (year, campaign, name, isSyst, isArray, noHist, chunksize)
        self.sl = TTsemilep(*args, selectionModifier="tt_semilep_e")
        self.dil = TTdilep(*args, selectionModifier="tt_dilep_hadw_e")
        self._ttrw = "none"

    # the runner sets ttbar_reweights on the processor after construction: pass it on to both
    @property
    def ttbar_reweights(self):
        return self._ttrw

    @ttbar_reweights.setter
    def ttbar_reweights(self, value):
        self._ttrw = value
        self.sl.ttbar_reweights = value
        self.dil.ttbar_reweights = value

    def process(self, events):
        dataset = events.metadata["dataset"]
        # neither processor modifies `events` (the selections work on shifted copies)
        out_sl = self.sl.process(events)[dataset]
        out_dil = self.dil.process(events)[dataset]
        return {dataset: {"sl": out_sl, "dil": out_dil}}

    def postprocess(self, accumulator):
        return accumulator
