import hist

axes = {
    "flav": hist.axis.IntCategory([0, 1, 4, 5, 6], name="flav", label="Genflavour"),
    "syst": hist.axis.StrCategory([], name="syst", growth=True),
    "pt": hist.axis.Regular(60, 0, 300, name="pt", label=" $p_{T}$ [GeV]"),
    "jpt": hist.axis.Regular(300, 0, 3000, name="pt", label=" $p_{T}$ [GeV]"),
    "ljpt": hist.axis.Regular(200, 0, 2000, name="pt", label=" $p_{T}$ [GeV]"),
    "softlpt": hist.axis.Regular(25, 0, 25, name="pt", label=" $p_{T}$ [GeV]"),
    "mass": hist.axis.Regular(50, 0, 300, name="mass", label=" mass [GeV]"),
    # hadronic W mass (semileptonic ttbar) and pileup-subtracted track multiplicity
    "mW": hist.axis.Regular(50, 40, 140, name="mass", label="$m_{jj}$ [GeV]"),
    "ntrack": hist.axis.Regular(
        60, -10, 110, name="ntrack", label="$n_{trk}^{PV}-n_{trk}^{t\\bar{t}}$"
    ),
    "ntrk": hist.axis.Regular(60, 0, 120, name="ntrk", label="$n_{trk}$"),
    "lepq": hist.axis.IntCategory([-1, 1], name="charge", label="muon charge"),
    "lepflav": hist.axis.IntCategory([13, 11], name="lepflav", label="lepton |PDG id| of the m(l, b) pair"),
    "lepmT": hist.axis.Regular(16, 0, 160, name="mT", label="$m_T(\\mu, MET)$ [GeV]"),
    "lepmet": hist.axis.Regular(16, 50, 290, name="met", label="MET [GeV]"),
    "lepmlb": hist.axis.Regular(20, 0, 300, name="mlb", label="min $m(\\mu, j)$ [GeV]"),
    "ntrack_c": hist.axis.Variable(
        [-10, 0, 12, 24, 36, 48, 60, 76, 96, 110], name="ntrack", label="$n_{track}$"
    ),
    "mW_c": hist.axis.Regular(25, 40, 140, name="mass", label="$m_{jj}$ [GeV]"),
    "btruth": hist.axis.IntCategory([0, 1, 2], name="btruth", label="leptonic b: 0 wrong, 1 right, 2 not b / data"),
    "agree": hist.axis.IntCategory([0, 1], name="agree", label="the two leptonic-b choices agree"),
    "lepbmeth": hist.axis.IntCategory([0, 1], name="method", label="0: m(jjb) closest to m_t, 1: closest to the muon"),
    "mlbfine": hist.axis.Regular(30, 0, 300, name="mlb", label="$m(\\mu, b_{lep})$ [GeV]"),
    "mjjb": hist.axis.Regular(40, 100, 300, name="mjjb", label="$m(jjb)$ [GeV]"),
    # electron mirror (single-electron trigger): IsoMu24 decision, and the leptonic-side diagnostics vs n_track
    "mutrig": hist.axis.IntCategory([0, 1], name="mutrig", label="HLT_IsoMu24 fired"),
    "dg_lpt": hist.axis.Regular(40, 0, 400, name="lpt", label="lepton $p_{T}$ [GeV]"),
    "dg_dr": hist.axis.Regular(30, 0, 6, name="dr", label="$\\Delta R(\\ell, b)$"),
    "dg_bpt": hist.axis.Regular(40, 0, 400, name="bpt", label="b-jet $p_{T}$ [GeV]"),
    # J/psi -> mu mu on top of the ttbar selections (utils/jpsi.py)
    "mjpsi": hist.axis.Regular(100, 2.0, 4.5, name="mass", label="$m(\\mu\\mu)$ [GeV]"),
    "mreg": hist.axis.IntCategory([0, 1, 2, 3], name="mreg", label="m(mu mu): 0 2.60-2.90, 1 2.95-3.25, 2 3.30-3.60, 3 other"),
    "jpsinmed": hist.axis.IntCategory([0, 1, 2], name="nmed", label="J/psi muons passing the medium ID"),
    "jpsicat": hist.axis.IntCategory([0, 1, 2], name="jpsicat", label="J/psi: 0 matched to a b jet (control), 1 in another jet, 2 in no jet"),
    "jpsitruth": hist.axis.IntCategory([0, 1, 2], name="jpsitruth", label="MC: 0 prompt J/psi, 1 from b, 2 no J/psi / data"),
    "jpsirside": hist.axis.IntCategory([0, 1, 2], name="rside", label="J/psi in the b of: 1 this lepton, 0 the other, 2 none (reco)"),
    "jpsitside": hist.axis.IntCategory([0, 1, 2], name="tside", label="J/psi jet is the b of: 1 this lepton's top, 0 the other, 2 neither (MC)"),
    "jpsipt": hist.axis.Variable([0, 3, 5, 8, 12, 16, 20, 25, 30, 40, 60, 100, 200], name="pt", label="$p_T(J/\\psi)$ [GeV]"),
    "jpsiabsy": hist.axis.Variable([0, 0.9, 1.5, 2.4], name="absy", label="$|y(J/\\psi)|$"),
    "jpsisip": hist.axis.Variable([0, 1, 2, 3, 4, 6, 8, 12, 20, 40, 100], name="sip", label="max SIP3D of the muons"),
    "jpsidr": hist.axis.Variable([0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 2.0, 4.0], name="dr", label="$\\Delta R$(J/$\\psi$, nearest jet)"),
    "jpsiz": hist.axis.Regular(20, 0, 1, name="z", label="$p_T(J/\\psi) / p_T(jet)$"),
    # pseudo-proper decay length from the two-muon vertex (utils/jpsi.py dimuon_vertex) and the J/psi line's z at the PV
    "jpsilxy": hist.axis.Regular(100, -0.5, 1.5, name="lxy", label="$\\ell_{xy}$ [mm]"),
    "jpsiz0": hist.axis.Variable([0, 0.02, 0.05, 0.1, 0.3, 1, 30], name="z0", label="$|z_0|$ [cm]"),
    "jpsilife": hist.axis.IntCategory([0, 1, 2, 3, 4], name="life", label="0 prompt, 1 non-prompt, 2 in between, 3 pileup-like, 4 no vertex"),
    "mlj": hist.axis.Regular(25, 0, 250, name="mlj", label="$m(\\ell, J/\\psi)$ [GeV]"),
    "nbj": hist.axis.IntCategory(
        [0, 1, 2, 3, 4], name="nbj", label="N b-jets (DeepFlav-M)"
    ),
    "bdt": hist.axis.Regular(50, 0, 1, name="bdt", label=" BDT discriminant"),
    "eta": hist.axis.Regular(25, -2.5, 2.5, name="eta", label=" $\eta$"),
    "phi": hist.axis.Regular(30, -3, 3, name="phi", label="$\phi$"),
    "mt": hist.axis.Regular(30, 0, 300, name="mt", label=" $m_{T}$ [GeV]"),
    "iso": hist.axis.Regular(30, 0, 0.05, name="pfRelIso03_all", label="Rel. Iso"),
    "softliso": hist.axis.Regular(
        20, 0.2, 6.2, name="pfRelIso03_all", label="Rel. Iso"
    ),
    "npv": hist.axis.Integer(0, 100, name="npv", label="N PVs"),
    "dr": hist.axis.Regular(20, 0, 8, name="dr", label="$\Delta$R"),
    "dr_s": hist.axis.Regular(20, 0, 0.5, name="dr", label="$\Delta$R"),
    "dr_SV": hist.axis.Regular(20, 0, 1.0, name="dr", label="$\Delta$R"),
    "dxy": hist.axis.Regular(40, -0.05, 0.05, name="dxy", label="$d_{xy}$ [cm]"),
    "dz": hist.axis.Regular(40, -0.01, 0.01, name="dz", label="$d_{z}$ [cm]"),
    "qcddxy": hist.axis.Regular(40, -0.002, 0.002, name="dxy", label="$d_{xy}$ [cm]"),
    "sip3d": hist.axis.Regular(20, 0, 0.2, name="sip3d", label="SIP 3D"),
    "ptratio": hist.axis.Regular(50, 0, 1, name="ratio", label="ratio"),
    "n": hist.axis.Integer(0, 10, name="n", label="N obj"),
    "osss": hist.axis.IntCategory([1, -1], name="osss", label="OS(+)/SS(-)"),
}
