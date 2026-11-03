"""ttbar + J/psi -> mu mu in e + jets (EGamma PD): the e + jets selection of the electron mirror unchanged
(Ele30_WPTight_Gsf, one MVA-iso WP80 electron, pT > 35 GeV), plus one loose-ID muon pair near the J/psi mass (utils/jpsi.py)
  "sl":  e + jets (ttsemilep_valid_sf, selectionModifier tt_semilep_e_jpsi)
Output per dataset: {"sl": <ttsemilep + ttjpsi histograms>}."""
from coffea import processor

from BTVNanoCommissioning.workflows.ttsemilep_valid_sf import NanoProcessor as TTsemilep


class NanoProcessor(processor.ProcessorABC):
    def __init__(self, year="2022", campaign="Summer22Run3", name="", isSyst=False, isArray=False, noHist=False, chunksize=75000):
        args = (year, campaign, name, isSyst, isArray, noHist, chunksize)
        self.sl = TTsemilep(*args, selectionModifier="tt_semilep_e_jpsi")
        self._ttrw = "none"

    # the runner sets ttbar_reweights on the processor after construction: pass it on
    @property
    def ttbar_reweights(self):
        return self._ttrw

    @ttbar_reweights.setter
    def ttbar_reweights(self, value):
        self._ttrw = value
        self.sl.ttbar_reweights = value

    def process(self, events):
        dataset = events.metadata["dataset"]
        return {dataset: {"sl": self.sl.process(events)[dataset]}}

    def postprocess(self, accumulator):
        return accumulator
