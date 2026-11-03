"""ttbar + J/psi -> mu mu in one pass over the events (Muon PD and MC): the mu + jets and e mu (IsoMu24) selections of the
hadronic-W / m(l, b) study unchanged, plus one soft-muon pair near the J/psi mass (utils/jpsi.py)
  "sl":  mu + jets (ttsemilep_valid_sf, selectionModifier tt_semilep_jpsi)
  "dil": e mu      (ttdilep_valid_sf,   selectionModifier tt_dilep_jpsi)
Output per dataset: {"sl": <ttsemilep + ttjpsi histograms>, "dil": <ttdilep + ttjpsi histograms>}."""
from coffea import processor

from BTVNanoCommissioning.workflows.ttdilep_valid_sf import NanoProcessor as TTdilep
from BTVNanoCommissioning.workflows.ttsemilep_valid_sf import NanoProcessor as TTsemilep


class NanoProcessor(processor.ProcessorABC):
    def __init__(self, year="2022", campaign="Summer22Run3", name="", isSyst=False, isArray=False, noHist=False, chunksize=75000):
        args = (year, campaign, name, isSyst, isArray, noHist, chunksize)
        self.sl = TTsemilep(*args, selectionModifier="tt_semilep_jpsi")
        self.dil = TTdilep(*args, selectionModifier="tt_dilep_jpsi")
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
