"""Check of the two-muon vertex / pseudo-proper decay length (utils/jpsi.py dimuon_vertex) on inclusive J/psi in one
central 2024 Muon NanoAOD file: sideband-subtracted l_xy and z0 of the opposite-sign J/psi (no ttbar selection).
usage: python3 check_lxy_central.py <file> <max events>  (PYTHONPATH=<checkout>/src)"""
import sys
import warnings

import awkward as ak
import numpy as np

warnings.filterwarnings("ignore")
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory  # noqa: E402

from BTVNanoCommissioning.utils import jpsi as JPSI  # noqa: E402

ev = NanoEventsFactory.from_root(sys.argv[1], schemaclass=NanoAODSchema, entry_stop=int(sys.argv[2])).events()
excl = (ev.Muon.pt > 30) & ev.Muon.tightId & (ev.Muon.pfRelIso04_all < 0.12)
info, _ = JPSI.find_jpsi(ev, JPSI.jpsi_muon_mask(ev, excl))
v = ak.to_numpy(info["valid"]) & (ak.to_numpy(info["os"]) == 1)
m = ak.to_numpy(info["mass"])
lxy, lerr, z0, chi2, ok = (np.asarray(info[k]) for k in ("lxy", "lxyerr", "z0", "vchi2", "vok"))
pt = ak.to_numpy(info["pt"])
sig = v & (m > 2.95) & (m < 3.25)
sb = v & (((m > 2.6) & (m < 2.9)) | ((m > 3.3) & (m < 3.6)))
print(f"OS dimuons {v.sum()}, vertex ok {np.mean(ok[v]):.3f}; S-B {sig.sum() - 0.5 * sb.sum():.0f}")
print(f"median l_xy error (window) {np.median(lerr[sig & ok]) * 1e3:.0f} um, median vertex chi2 {np.median(chi2[sig & ok]):.2f}")


def sub(x, sel_extra, bins):
    h1, e = np.histogram(x[sig & ok & sel_extra], bins=bins)
    h2, _ = np.histogram(x[sb & ok & sel_extra], bins=bins)
    return h1 - 0.5 * h2, np.sqrt(h1 + 0.25 * h2), e


z = np.abs(z0)
for lab, extra in (("all", np.ones_like(v)), ("|z0| < 0.1 cm", z < 0.1)):
    h, eh, e = sub(lxy, extra, np.array([-1, -0.3, -0.2, -0.1, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.5, 1, 2, 5]))
    print(f"l_xy [mm], S-B, {lab}:")
    for i in range(len(h)):
        print(f"  [{e[i]:+5.2f},{e[i + 1]:+5.2f})  {h[i]:7.1f} +- {eh[i]:5.1f}  {'#' * int(50 * max(h[i], 0) / max(h.max(), 1))}")
h, eh, e = sub(z, np.ones_like(v), np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 30]))
print("|z0| [cm], S-B:")
for i in range(len(h)):
    print(f"  [{e[i]:5.2f},{e[i + 1]:5.2f})  {h[i]:7.1f} +- {eh[i]:5.1f}")
core = sig & ok & (z < 0.1) & (np.abs(lxy) < 0.1)
for lo, hi in ((5, 10), (10, 20), (20, 100)):
    s = (pt > lo) & (pt < hi)
    q = np.percentile(lxy[core & s], [16, 50, 84]) if (core & s).sum() > 20 else [np.nan] * 3
    print(f"pT {lo}-{hi}: l_xy 16/50/84% in |l|<0.1 mm (window, raw): {q[0] * 1e3:.0f} {q[1] * 1e3:.0f} {q[2] * 1e3:.0f} um; "
          f"non-prompt fraction (l > 0.1 mm, S-B) {(sub(lxy, s & (z < 0.1), np.array([0.1, 50]))[0][0]) / max(sub(lxy, s & (z < 0.1), np.array([-50, 50]))[0][0], 1):.2f}")
