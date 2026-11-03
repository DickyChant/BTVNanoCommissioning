"""Workflows of the ttbar + J/psi -> mu mu study (branch ttJ). Each ttbar channel of the hadronic-W / m(l, b) study
with one loose-ID muon pair near the J/psi mass (utils/jpsi.py); "ttJ" runs all three in one pass (workflows/ttJ.py)."""
from functools import partial

from BTVNanoCommissioning.workflows.ttdilep_valid_sf import NanoProcessor as TTdilepValidSFProcessor
from BTVNanoCommissioning.workflows.ttJ import NanoProcessor as TTJProcessor
from BTVNanoCommissioning.workflows.ttsemilep_valid_sf import NanoProcessor as TTsemilepValidSFProcessor

workflows = {}
# mu + jets: IsoMu24, one tight isolated muon pT > 30 GeV, >= 4 jets, MET > 50 GeV (Muon PD)
workflows["ttJ_mujets"] = partial(TTsemilepValidSFProcessor, selectionModifier="tt_semilep_jpsi")
# e + jets: Ele30_WPTight_Gsf, one MVA-iso WP80 electron pT > 35 GeV, >= 4 jets, MET > 50 GeV (EGamma PD)
workflows["ttJ_ejets"] = partial(TTsemilepValidSFProcessor, selectionModifier="tt_semilep_e_jpsi")
# e mu: IsoMu24, opposite sign, >= 2 jets (Muon PD)
workflows["ttJ_emu"] = partial(TTdilepValidSFProcessor, selectionModifier="tt_dilep_jpsi")
# the three channels in one pass (data: the channels of its primary dataset)
workflows["ttJ"] = TTJProcessor

__all__ = ["workflows"]
