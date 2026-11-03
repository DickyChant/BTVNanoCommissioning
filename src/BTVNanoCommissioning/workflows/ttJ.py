"""ttbar + J/psi -> mu mu, the three channels in one pass over the events. The ttbar selections are those of the
hadronic-W / m(l, b) study, unchanged; on top, one loose-ID muon pair near the J/psi mass (utils/jpsi.py).
  "mujets": mu + jets, IsoMu24          (ttsemilep_valid_sf, selectionModifier tt_semilep_jpsi)
  "ejets":  e + jets, Ele30_WPTight_Gsf (ttsemilep_valid_sf, selectionModifier tt_semilep_e_jpsi)
  "emu":    e mu, IsoMu24               (ttdilep_valid_sf,   selectionModifier tt_dilep_jpsi)
Data runs only the channels of its primary dataset (Muon: mujets and emu; EGamma: ejets), so that no event enters
twice; MC runs all three. Output per dataset: {channel: <ttbar + ttjpsi histograms>}."""
import re

from coffea import processor

from BTVNanoCommissioning.workflows.ttdilep_valid_sf import NanoProcessor as TTdilep
from BTVNanoCommissioning.workflows.ttsemilep_valid_sf import NanoProcessor as TTsemilep

MODES = {
    "mujets": (TTsemilep, "tt_semilep_jpsi"),
    "ejets": (TTsemilep, "tt_semilep_e_jpsi"),
    "emu": (TTdilep, "tt_dilep_jpsi"),
}
PD_CHANNELS = (
    (re.compile(r"(Single)?Muon\d*Run"), ("mujets", "emu")),
    (re.compile(r"(EGamma|SingleElectron)\d*Run"), ("ejets",)),
)


def channels(dataset, is_data):
    if not is_data:
        return tuple(MODES)
    for rx, chans in PD_CHANNELS:
        if rx.match(dataset):
            return chans
    raise ValueError(f"ttJ: no channel for data set {dataset} (expected a Muon or EGamma primary dataset)")


class NanoProcessor(processor.ProcessorABC):
    def __init__(self, year="2022", campaign="Summer22Run3", name="", isSyst=False, isArray=False, noHist=False, chunksize=75000):
        args = (year, campaign, name, isSyst, isArray, noHist, chunksize)
        self.procs = {ch: cls(*args, selectionModifier=mod) for ch, (cls, mod) in MODES.items()}
        self._ttrw = "none"

    # the runner sets ttbar_reweights on the processor after construction: pass it on
    @property
    def ttbar_reweights(self):
        return self._ttrw

    @ttbar_reweights.setter
    def ttbar_reweights(self, value):
        self._ttrw = value
        for p in self.procs.values():
            p.ttbar_reweights = value

    def process(self, events):
        dataset = events.metadata["dataset"]
        is_data = not hasattr(events, "genWeight")
        # no processor modifies `events` (the selections work on shifted copies)
        return {dataset: {ch: self.procs[ch].process(events)[dataset] for ch in channels(dataset, is_data)}}

    def postprocess(self, accumulator):
        return accumulator
