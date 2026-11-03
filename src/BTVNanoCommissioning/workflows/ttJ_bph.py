"""ttbar + J/psi -> mu mu on BPHNano of the low-mass double-muon parking (ParkingDoubleMuonLowMass*): the three ttbar
channels of workflows/ttJ.py with the "_bph" modes. The event is triggered by the J/psi: HLT_DoubleMu4_3_LowMass with
both J/psi muons matched to it, the J/psi taken from the MuMu vertex fit (utils/jpsi.py find_jpsi_bph); the ttbar
leptons are selected offline only, and whether the channel's single-lepton trigger (IsoMu24 / Ele30) also fired is
kept (jpsi_sltrig) for the overlap with the single-lepton-triggered ttJ sample. Every channel runs on every data set
(one parking stream holds all three); MC as in ttJ. Output per dataset: {channel: <ttbar + ttjpsi histograms>}."""
import re

from BTVNanoCommissioning.workflows.ttJ import MODES as TTJ_MODES
from BTVNanoCommissioning.workflows.ttJ import NanoProcessor as TTJProcessor

PARKING = re.compile(r"ParkingDoubleMuonLowMass")


class NanoProcessor(TTJProcessor):
    MODES = {ch: (cls, mod + "_bph") for ch, (cls, mod) in TTJ_MODES.items()}

    def channels(self, dataset, is_data):
        if is_data and not PARKING.search(dataset):
            raise ValueError(f"ttJ_bph: {dataset} is not a ParkingDoubleMuonLowMass data set")
        return tuple(self.MODES)
