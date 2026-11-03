"""Check of utils/jpsi.py find_jpsi_bph on one 2024 BPHNano parking file (no ttbar selection): trigger-matched J/psi
candidates, the mass peak, l_xy (pseudo-proper, from MuMu_vtx - PV) sideband-subtracted, and z0.
usage: python3 check_lxy_bph.py <BPHNano file>  (PYTHONPATH=<checkout>/src)"""
import sys
import warnings

import awkward as ak
import numpy as np

warnings.filterwarnings("ignore")
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory  # noqa: E402

from BTVNanoCommissioning.utils import jpsi as JPSI  # noqa: E402

ev = NanoEventsFactory.from_root(sys.argv[1], schemaclass=NanoAODSchema).events()
trig = ak.to_numpy(ev.HLT[JPSI.BPH_TRIGGER])
excl = (ev.Muon.pt > 30) & ev.Muon.tightId & (ev.Muon.pfRelIso04_all < 0.12)
info, pair = JPSI.find_jpsi_bph(ev, excl)
v = ak.to_numpy(info["valid"])
m = ak.to_numpy(info["mass"])
print(f"events {len(ev)}, HLT_{JPSI.BPH_TRIGGER} {trig.mean():.3f}, trigger-matched J/psi candidate {v.mean():.3f} (of triggered {v[trig].mean():.3f}),"
      f" central muons matched to the pair per candidate {ak.sum(pair) / max(v.sum(), 1):.2f}")
h, e = np.histogram(m[v], bins=40, range=(2.0, 4.0))
for i in range(0, 40, 2):
    print(f"  {e[i]:.2f}  {h[i] + h[i + 1]:7d}")
lxy, z0 = np.asarray(info["lxy"]), np.abs(np.asarray(info["z0"]))
sig = v & (m > 2.95) & (m < 3.25)
sb = v & (((m > 2.6) & (m < 2.9)) | ((m > 3.3) & (m < 3.6)))
print(f"S-B {sig.sum() - 0.5 * sb.sum():.0f}")
bins = np.array([-1, -0.1, -0.06, -0.02, 0, 0.02, 0.06, 0.1, 0.3, 1, 5])
h1, _ = np.histogram(lxy[sig], bins=bins); h2, _ = np.histogram(lxy[sb], bins=bins)
print("l_xy [mm] S-B:", " ".join(f"[{bins[i]:+.2f},{bins[i + 1]:+.2f}) {h1[i] - 0.5 * h2[i]:.0f}" for i in range(len(h1))))
q = np.percentile(lxy[sig & (np.abs(lxy) < 0.1)], [16, 50, 84])
print(f"prompt core 16/50/84%: {q[0] * 1e3:.0f} {q[1] * 1e3:.0f} {q[2] * 1e3:.0f} um; |z0| > 0.1 cm (S-B) {(sig & (z0 > 0.1)).sum() - 0.5 * (sb & (z0 > 0.1)).sum():.0f}")
