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
