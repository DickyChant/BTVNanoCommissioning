"""Standalone Pythia8 ttbar (l + jets: W+ -> l nu, W- -> q q') at 13.6 TeV, CP5 tune, with J/psi produced in the parton
shower (LETO, CharmoniumShower:all; LDMEs scaled by --ldme), J/psi -> mu mu. Where does a prompt J/psi of a ttbar event come
from, and does the veto of the J/psi in the ttbar decay jets (assoc_veto.py of the J/psi deck) remove the ones from the
top decays? Per event with a J/psi -> mu mu (npz, padded arrays):
  jets      anti-kT R = 0.4 gen jets of the visible particles (the W lepton excluded), pT > 15 GeV, |eta| < 4.5, pT-ordered:
            pt, eta, phi, mass, muf (muon energy fraction), nb / nc (weakly decaying b / c hadrons within 0.4),
            top (1: within 0.4 of a top-decay parton, the b or a W quark after the shower; 0 otherwise)
  lepton    the W lepton (pt, eta, phi, id); met: the neutrino sum (x, y)
  jpsi      pt, eta, phi, the two muons (pt, eta), prompt (no b hadron among the ancestors), origin:
            1 top decay (through the W or the b of a top decay), 2 radiated by a top before its decay, 3 initial-state
            radiation, 4 a second parton scattering (MPI), 5 other (the hard process otherwise)
usage: python3 gen_ttjpsi.py --nev 1000 --seed 1 --ldme 10 --out out.npz"""
import argparse
import math
import time

import numpy as np
import pythia8

from gen_mlb import CP5, dr

NJ, NQ = 12, 4


def bhad(i):
    a = abs(i)
    return (a // 100) % 10 == 5 or (a // 1000) % 10 == 5


def chad(i):
    a = abs(i)
    return ((a // 100) % 10 == 4 or (a // 1000) % 10 == 4) and not bhad(i) and a not in (443, 100443, 10441, 20443, 445, 441)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nev", type=int, default=1000, help="generated ttbar events")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ldme", type=float, default=10.0, help="OniaShower:ldmeFac")
    ap.add_argument("--out", default="gen.npz")
    a = ap.parse_args()
    py = pythia8.Pythia("", False)
    for s in (["Beams:eCM = 13600", "Top:gg2ttbar = on", "Top:qqbar2ttbar = on", "6:m0 = 172.5", "Next:numberCount = 0",
               "Random:setSeed = on", f"Random:seed = {a.seed}"] + CP5
              + ["24:onMode = off", "24:onPosIfAny = 11 13", "24:onNegIfAny = 1 2 3 4 5",
                 "CharmoniumShower:all = on", f"OniaShower:ldmeFac = {a.ldme}", "443:onMode = off", "443:onIfMatch = 13 -13"]):
        if not py.readString(s):
            raise SystemExit(f"bad setting: {s}")
    py.init()
    sj = pythia8.SlowJet(-1, 0.4, 15.0, 4.5, 2, 1)
    E = {k: [] for k in ("lep", "met", "jets", "jpsi", "evt")}
    njpsi = np.zeros(6, int)
    t0 = time.time()
    for ie in range(a.nev):
        if not py.next():
            continue
        ev = py.event
        tops = [i for i in range(ev.size()) if abs(ev[i].id()) == 6 and ev[i].iBotCopy() == i]
        dec, partons, lep = set(), [], None
        for t in tops:
            for d in ev[t].daughterList():
                if abs(ev[d].id()) in (5, 24):
                    dec.add(d)
                    if abs(ev[d].id()) == 5:
                        partons.append(ev[d].iBotCopy())
                    else:
                        w = ev[d].iBotCopy()
                        dec.add(w)
                        for k in ev[w].daughterList():
                            if abs(ev[k].id()) in (11, 13):
                                lep = ev[k].iBotCopy()
                            elif 1 <= abs(ev[k].id()) <= 5:
                                partons.append(ev[k].iBotCopy())
        ps = [i for i in range(ev.size()) if ev[i].id() == 443 and ev[i].daughterList()
              and sorted(abs(ev[d].id()) for d in ev[i].daughterList()) == [13, 13]]
        if lep is None or not ps:
            continue
        rows = []
        for i in ps:
            # ancestry by the first mother
            k, origin, prompt, seen = ev[i].mother1(), 5, True, 0
            while k > 0 and seen < 2000:
                p = ev[k]
                if bhad(p.id()):
                    prompt = False
                if k in dec:
                    origin = 1
                    break
                if abs(p.id()) == 6:
                    origin = 2
                    break
                st = abs(p.status())
                if 31 <= st <= 39:
                    origin = 4
                    break
                if 41 <= st <= 49:
                    origin = 3
                    break
                k, seen = p.mother1(), seen + 1
            mus = sorted(ev[i].daughterList(), key=lambda d: -ev[d].pT())
            m1, m2 = ev[mus[0]], ev[mus[1]]
            rows.append([ev[i].pT(), ev[i].eta(), ev[i].phi(), m1.pT(), m1.eta(), m2.pT(), m2.eta(), float(prompt), float(origin)])
            njpsi[origin] += 1
        ev[lep].statusNeg()
        sj.analyze(ev)
        ev[lep].statusPos()
        hb = [i for i in range(ev.size()) if bhad(ev[i].id()) and ev[i].isHadron() and not any(bhad(ev[d].id()) for d in ev[i].daughterList())]
        hc = [i for i in range(ev.size()) if chad(ev[i].id()) and ev[i].isHadron() and not any(chad(ev[d].id()) or bhad(ev[d].id()) for d in ev[i].daughterList())]
        J = np.zeros((NJ, 8), np.float32)
        for j in range(min(sj.sizeJet(), NJ)):
            p = sj.p(j)
            emu = sum(ev[c].e() for c in sj.constituents(j) if abs(ev[c].id()) == 13)
            near = lambda lst: sum(1 for h in lst if dr(ev[h].eta(), ev[h].phi(), p.eta(), p.phi()) < 0.4)  # noqa: E731
            J[j] = (p.pT(), p.eta(), p.phi(), p.mCalc(), emu / max(p.e(), 1e-9), near(hb), near(hc), float(near(partons) > 0))
        nx = sum(ev[i].px() for i in range(ev.size()) if ev[i].isFinal() and abs(ev[i].id()) in (12, 14, 16))
        ny = sum(ev[i].py() for i in range(ev.size()) if ev[i].isFinal() and abs(ev[i].id()) in (12, 14, 16))
        L = ev[lep]
        E["lep"].append([L.pT(), L.eta(), L.phi(), L.id()])
        E["met"].append([nx, ny])
        E["jets"].append(J)
        jp = np.zeros((NQ, 9), np.float32)
        jp[:, 0] = -1
        for q, r in enumerate(rows[:NQ]):
            jp[q] = r
        E["jpsi"].append(jp)
        E["evt"].append(ie)
    dt = time.time() - t0
    info = py.infoPython()
    print(f"{a.nev} events in {dt:.0f} s ({a.nev / dt:.1f}/s); {len(E['evt'])} with a J/psi -> mu mu; by origin {njpsi[1:].tolist()};"
          f" sigma {info.sigmaGen() * 1e9:.0f} pb")
    np.savez_compressed(a.out, **{k: np.asarray(v, np.float32) for k, v in E.items()}, nev=a.nev, ldme=a.ldme, seed=a.seed,
                        sigma_pb=info.sigmaGen() * 1e9, time_s=dt)


if __name__ == "__main__":
    main()
