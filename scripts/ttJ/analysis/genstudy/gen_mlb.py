"""Standalone Pythia8 ttbar at 13.6 TeV (CMS CP5 tune) for the generator-level dependence of m(l, b) on
the activity around the ttbar system. Runs in the LCG view (Pythia 8.3 python bindings, LHAPDF).

Per event (numpy arrays in the output .npz, one row per leptonic top):
  mlb_jet    m(l, b-jet): anti-kT R = 0.4 gen jet (visible particles, prompt leptons excluded) matched to the
             b quark of the same top (dR < 0.4); -1 if no match
  mlb_q      m(l, b quark after the shower)
  ptl, ptb, ptbq, dr   lepton pT, b-jet pT, b-quark pT, dR(l, b-jet); etal, etab
  lid        PDG id of the lepton (+-11, +-13)
  ntrk       gen analogue of n_track: charged final particles with pT > 0.5 GeV and |eta| < 2.5, minus those
             inside the ttbar jets (the matched b jets, and W jets for the semileptonic channel), minus leptons
  nch        all such charged particles;  nmpi  number of MPI;  evt  event index
usage: python3 gen_mlb.py --config nominal --channel dilep --nev 10000 --seed 1 --out out.npz
"""
import argparse
import math

import numpy as np
import pythia8

CP5 = [
    "Tune:pp = 14", "Tune:ee = 7", "Tune:preferLHAPDF = 2",
    "MultipartonInteractions:ecmPow = 0.03344", "MultipartonInteractions:bProfile = 2",
    "MultipartonInteractions:pT0Ref = 1.41", "MultipartonInteractions:coreRadius = 0.7634",
    "MultipartonInteractions:coreFraction = 0.63", "ColourReconnection:range = 5.176", "SigmaTotal:zeroAXB = off",
    "SpaceShower:alphaSorder = 2", "SpaceShower:alphaSvalue = 0.118", "SigmaProcess:alphaSvalue = 0.118",
    "SigmaProcess:alphaSorder = 2", "MultipartonInteractions:alphaSvalue = 0.118", "MultipartonInteractions:alphaSorder = 2",
    "TimeShower:alphaSorder = 2", "TimeShower:alphaSvalue = 0.118", "SigmaTotal:mode = 0", "SigmaTotal:sigmaEl = 21.89",
    "SigmaTotal:sigmaTot = 100.309", "PDF:pSet = LHAPDF6:NNPDF31_nnlo_as_0118",
    "ParticleDecays:limitTau0 = on", "ParticleDecays:tau0Max = 10",
    "Check:epTolErr = 0.01",
]
CONFIGS = {  # one physics change per configuration, on top of CP5
    "nominal": [],
    "fsr_hi": ["TimeShower:renormMultFac = 0.25"],  # mu_R x 0.5: more FSR
    "fsr_lo": ["TimeShower:renormMultFac = 4.0"],
    "isr_hi": ["SpaceShower:renormMultFac = 0.25"],
    "isr_lo": ["SpaceShower:renormMultFac = 4.0"],
    "mpi_off": ["PartonLevel:MPI = off"],
    "cr_off": ["ColourReconnection:reconnect = off"],
    "cr_qcd": ["ColourReconnection:mode = 1"],  # Pythia defaults of the QCD-based model, not the CMS CR1 tune
    "cr_move": ["ColourReconnection:mode = 2"],  # gluon move, Pythia defaults
    "bfrag_peterson": ["StringZ:usePetersonB = on"],
    "rb_hi": ["StringZ:rFactB = 1.056"],  # CP5 inherits Monash rFactB = 0.855
    "rb_lo": ["StringZ:rFactB = 0.655"],
    # CMS Run-3 colour-reconnection tunes (CMSSW Configuration/Generator/python/MCTunesRun3ECM13p6TeV), and early resonance
    # decays: as in the TuneCP5CR1 / TuneCP5CR2 / Par-ERD-On ttbar samples
    "cms_cr1": ["MultipartonInteractions:pT0Ref = 1.375", "MultipartonInteractions:ecmPow = 0.03283",
                "MultipartonInteractions:coreFraction = 0.4446", "MultipartonInteractions:coreRadius = 0.6046",
                "ColourReconnection:mode = 1", "BeamRemnants:remnantMode = 1", "ColourReconnection:junctionCorrection = 0.238",
                "ColourReconnection:timeDilationPar = 8.58", "ColourReconnection:m0 = 1.721", "StringZ:aLund = 0.38",
                "StringZ:bLund = 0.64", "StringFlav:probQQtoQ = 0.078", "StringFlav:probStoUD = 0.2"],
    "cms_cr2": ["MultipartonInteractions:pT0Ref = 1.454", "MultipartonInteractions:ecmPow = 0.0555",
                "MultipartonInteractions:coreFraction = 0.4392", "MultipartonInteractions:coreRadius = 0.6532",
                "ColourReconnection:mode = 2", "ColourReconnection:m2Lambda = 4.395", "ColourReconnection:fracGluon = 0.9896"],
    "erd_on": ["PartonLevel:earlyResDec = on"],  # top decay products take part in colour reconnection
}
DECAYS = {
    "dilep": ["24:onMode = off", "24:onIfAny = 11 13"],
    "semilep": ["24:onMode = off", "24:onPosIfAny = 11 13", "24:onNegIfAny = 1 2 3 4 5"],  # W+ -> l nu, W- -> qq'
}


def dr(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
    return math.hypot(eta1 - eta2, dphi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nominal", choices=CONFIGS)
    ap.add_argument("--channel", default="dilep", choices=DECAYS)
    ap.add_argument("--nev", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="gen.npz")
    a = ap.parse_args()

    py = pythia8.Pythia("", False)
    for s in (["Beams:eCM = 13600", "Top:gg2ttbar = on", "Top:qqbar2ttbar = on", "6:m0 = 172.5", "Next:numberCount = 0",
               "Random:setSeed = on", f"Random:seed = {a.seed}"] + CP5 + DECAYS[a.channel] + CONFIGS[a.config]):
        if not py.readString(s):
            raise SystemExit(f"bad setting: {s}")
    py.init()
    sj = pythia8.SlowJet(-1, 0.4, 15.0, 4.5, 2, 1)  # anti-kT, R = 0.4, pT > 15 GeV, |eta| < 4.5, visible particles
    rows = {k: [] for k in ("evt", "lid", "mlb_jet", "mlb_q", "ptl", "ptb", "ptbq", "dr", "etal", "etab", "ntrk", "nch", "nmpi")}
    nacc = 0
    while nacc < a.nev:
        if not py.next():
            continue
        ev = py.event
        tops, wq = [], []
        for i in range(ev.size()):
            if abs(ev[i].id()) == 6 and ev[i].iBotCopy() == i:
                tops.append(i)
        leps, bqs = [], []
        for t in tops:
            b = w = None
            for d in ev[t].daughterList():
                if abs(ev[d].id()) == 5:
                    b = ev[d].iBotCopy()
                elif abs(ev[d].id()) == 24:
                    w = ev[d].iBotCopy()
            if b is None or w is None:
                continue
            lep = None
            for d in ev[w].daughterList():
                if abs(ev[d].id()) in (11, 13):
                    lep = ev[d].iBotCopy()
                elif 1 <= abs(ev[d].id()) <= 5:
                    wq.append(ev[d].iBotCopy())
            bqs.append(b)
            leps.append(lep)
        # keep the prompt leptons out of the jets
        prompt = [l for l in leps if l is not None]
        for l in prompt:
            ev[l].statusNeg()
        sj.analyze(ev)
        for l in prompt:
            ev[l].statusPos()
        jets = [(sj.p(j), sj.constituents(j)) for j in range(sj.sizeJet())]

        def match(iq):
            q = ev[iq]
            best, bdr = None, 0.4
            for k, (p, _) in enumerate(jets):
                d = dr(q.eta(), q.phi(), p.eta(), p.phi())
                if d < bdr:
                    best, bdr = k, d
            return best

        bjet = [match(b) for b in bqs]
        ttj = {k for k in bjet if k is not None} | {k for k in (match(q) for q in wq) if k is not None}
        incl = set()
        for k in ttj:
            incl.update(jets[k][1])
        nch = nin = 0
        for i in range(ev.size()):
            p = ev[i]
            if p.isFinal() and p.isCharged() and p.pT() > 0.5 and abs(p.eta()) < 2.5:
                nch += 1
                if i in incl:
                    nin += 1
        nlep = sum(1 for l in prompt if ev[l].pT() > 0.5 and abs(ev[l].eta()) < 2.5)
        ntrk = nch - nin - nlep
        for lep, b, kj in zip(leps, bqs, bjet):
            if lep is None:
                continue
            L, B = ev[lep], ev[b]
            mq = math.sqrt(max((L.e() + B.e()) ** 2 - (L.px() + B.px()) ** 2 - (L.py() + B.py()) ** 2 - (L.pz() + B.pz()) ** 2, 0.0))
            if kj is not None:
                J = jets[kj][0]
                mj = math.sqrt(max((L.e() + J.e()) ** 2 - (L.px() + J.px()) ** 2 - (L.py() + J.py()) ** 2 - (L.pz() + J.pz()) ** 2, 0.0))
                ptb, etab, d = J.pT(), J.eta(), dr(L.eta(), L.phi(), J.eta(), J.phi())
            else:
                mj, ptb, etab, d = -1.0, -1.0, -9.0, -1.0
            for k, v in (("evt", nacc), ("lid", L.id()), ("mlb_jet", mj), ("mlb_q", mq), ("ptl", L.pT()), ("ptb", ptb), ("ptbq", B.pT()), ("dr", d),
                         ("etal", L.eta()), ("etab", etab), ("ntrk", ntrk), ("nch", nch), ("nmpi", py.infoPython().nMPI())):
                rows[k].append(v)
        nacc += 1
    np.savez_compressed(a.out, **{k: np.asarray(v, dtype=np.int32 if k in ("evt", "lid", "ntrk", "nch", "nmpi") else np.float32) for k, v in rows.items()},
                        config=a.config, channel=a.channel, seed=a.seed, nev=a.nev, sigma_pb=py.infoPython().sigmaGen() * 1e9)


if __name__ == "__main__":
    main()
