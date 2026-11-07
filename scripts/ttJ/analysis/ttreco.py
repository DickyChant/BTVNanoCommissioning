"""The ttbar system per J/psi candidate, rebuilt from the stored selected jets of the candidate table (ttJ_pixi 79eecfe):
the chi2 assignment of l + jets and the neutrino weighting of e mu, the transverse sum of the ttbar decay products, and the
jet holding the J/psi. Used by ttbar_balance.py (the pT balance) and assoc_veto.py (the J/psi outside the ttbar jets).
See ttbar_balance.py for the method."""
import itertools

import numpy as np

MW, MT = 80.4, 172.5
SW_, ST_H, ST_L = 10.0, 15.0, 20.0
NJ, DR_JET = 8, 0.4
ETA = np.linspace(-3.0, 3.0, 61)
SIG_MET, SIG_ETA = 25.0, 1.4
CH = ("mujets", "ejets", "emu")


def p4(pt, eta, phi, m):
    pt, eta, phi, m = (np.asarray(a, float) for a in (pt, eta, phi, m))
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.stack([px, py, pz, np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2)], axis=-1)


def mass(v):
    return np.sqrt(np.maximum(v[..., 3] ** 2 - (v[..., :3] ** 2).sum(-1), 0))


def rapidity(v):
    return 0.5 * np.log(np.maximum(v[..., 3] + v[..., 2], 1e-9) / np.maximum(v[..., 3] - v[..., 2], 1e-9))


def dphi(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def make_jets(pt, eta, phi, m, btag, isb):
    ok = pt > 0
    z = lambda a: np.where(ok, a, 0)  # noqa: E731
    return dict(pt=pt, eta=eta, phi=phi, mass=m, btag=btag, isb=(isb > 0) & ok, ok=ok, p4=p4(z(pt), z(eta), z(phi), z(m)))


def jets_of(d):
    return make_jets(*(np.stack([d[f"jet{i}_{k}"].to_numpy(float) for i in range(NJ)], axis=1) for k in ("pt", "eta", "phi", "mass", "btag", "isb")))


def add_jets(J, extra):
    """J plus extra jets ({field: (n, k)}), the NJ highest-pT kept: (jets, new index of the first extra jet or -1)"""
    F = ("pt", "eta", "phi", "mass", "btag", "isb")
    A = {k: np.concatenate([np.where(J["ok"], J[k], -99.0), np.where(extra["pt"] > 0, extra[k], -99.0)], axis=1) for k in F}
    o = np.argsort(-A["pt"], axis=1, kind="stable")[:, :NJ]
    first = o == NJ
    inew = np.where(first.any(1), first.argmax(1), -1)
    out = make_jets(*(np.take_along_axis(A[k], o, 1) for k in F))
    return out, np.where((inew >= 0) & out["ok"][np.arange(len(inew)), np.maximum(inew, 0)], inew, -1)


def nu_pz(lep, mx, my):
    """the two pz of the neutrino from m(l nu) = m_W (massless lepton), the real part twice without a solution"""
    ptl2 = lep[:, 0] ** 2 + lep[:, 1] ** 2
    mu = 0.5 * MW ** 2 + lep[:, 0] * mx + lep[:, 1] * my
    a = mu * lep[:, 2] / ptl2
    disc = a ** 2 - (lep[:, 3] ** 2 * (mx ** 2 + my ** 2) - mu ** 2) / ptl2
    r = np.sqrt(np.maximum(disc, 0))
    return np.stack([a - r, a + r], axis=1)


PERM = np.array([(bh, bl, q1, q2) for bh, bl in itertools.permutations(range(NJ), 2)
                 for q1, q2 in itertools.combinations([j for j in range(NJ) if j not in (bh, bl)], 2)])


def reco_ljets(lep, mx, my, J, use):
    """chi2 assignment of four of the usable jets: (t_had, t_lep, index (n, 4) of b_h, b_l, q, q' or -1, chi2)"""
    n = len(lep)
    pz = nu_pz(lep, mx, my)
    nu = np.stack([np.repeat(mx[:, None], 2, 1), np.repeat(my[:, None], 2, 1), pz, np.sqrt(mx[:, None] ** 2 + my[:, None] ** 2 + pz ** 2)], axis=-1)
    jp, tag = J["p4"], J["isb"] & use
    ntag = np.minimum(tag.sum(1), 2)
    best_c, best_p, best_s = np.full(n, np.inf), np.zeros(n, int), np.zeros(n, int)
    for c0 in range(0, len(PERM), 120):
        P = PERM[c0:c0 + 120]
        u = use[:, P].all(-1) & ((tag[:, P[:, 0]].astype(int) + tag[:, P[:, 1]]) == ntag[:, None])
        mqq = mass(jp[:, P[:, 2]] + jp[:, P[:, 3]])
        mth = mass(jp[:, P[:, 2]] + jp[:, P[:, 3]] + jp[:, P[:, 0]])
        lb = lep[:, None, :] + jp[:, P[:, 1]]
        mtl = mass(lb[:, :, None, :] + nu[:, None, :, :])  # (n, perm, 2)
        c2 = ((mqq - MW) / SW_) ** 2 + ((mth - MT) / ST_H) ** 2
        c2 = c2[:, :, None] + ((mtl - MT) / ST_L) ** 2
        c2 = np.where(u[:, :, None], c2, np.inf).reshape(n, -1)
        k = c2.argmin(1)
        cmin = c2[np.arange(n), k]
        better = cmin < best_c
        best_c = np.where(better, cmin, best_c)
        best_p = np.where(better, c0 + k // 2, best_p)
        best_s = np.where(better, k % 2, best_s)
    ok = np.isfinite(best_c)
    idx = np.where(ok[:, None], PERM[best_p], -1)
    r = np.arange(n)
    th = jp[r, PERM[best_p, 0]] + jp[r, PERM[best_p, 2]] + jp[r, PERM[best_p, 3]]
    tl = lep + nu[r, best_s] + jp[r, PERM[best_p, 1]]
    return th, tl, idx, np.where(ok, best_c, np.nan)


def nu_eta_solutions(lep, b):
    """neutrino momenta (n, len(ETA), 2, 4) from m(l nu) = m_W and m(l nu b) = m_t at each neutrino eta, with validity"""
    ch, sh = np.cosh(ETA)[None, :], np.sinh(ETA)[None, :]
    lb = lep + b
    a1 = lep[:, None, 3] * ch - lep[:, None, 2] * sh
    a2 = lb[:, None, 3] * ch - lb[:, None, 2] * sh
    k1, k2 = 0.5 * MW ** 2, 0.5 * (MT ** 2 - mass(lb) ** 2)[:, None]
    # pT (a_i - u_i . c) = k_i, c = (cos phi, sin phi): (k1 u2 - k2 u1) . c = k1 a2 - k2 a1
    wx, wy = k1 * lb[:, None, 0] - k2 * lep[:, None, 0], k1 * lb[:, None, 1] - k2 * lep[:, None, 1]
    R, wn = k1 * a2 - k2 * a1, np.hypot(wx, wy)
    good = np.abs(R) <= wn
    phw, dph = np.arctan2(wy, wx), np.arccos(np.clip(R / np.maximum(wn, 1e-9), -1, 1))
    out, val = [], []
    for s in (-1, 1):
        ph = phw + s * dph
        pt = k1 / (a1 - lep[:, None, 0] * np.cos(ph) - lep[:, None, 1] * np.sin(ph))
        v = good & (pt > 0) & np.isfinite(pt)
        pt = np.where(v, pt, 0)
        out.append(np.stack([pt * np.cos(ph), pt * np.sin(ph), pt * sh, pt * ch], -1))
        val.append(v)
    return np.stack(out, 2), np.stack(val, 2)


def reco_emu(l1, l2, mx, my, J, use):
    """neutrino weighting with the two b candidates of the usable jets: (t1, t2, index (n, 2) of the b candidates, ok)"""
    n = len(l1)
    key = np.where(use, J["isb"] * 10.0 + J["btag"], -np.inf)
    o = np.argsort(-key, axis=1)[:, :2]
    ok = np.isfinite(np.take_along_axis(key, o, 1)).all(1)
    r = np.arange(n)
    b = [J["p4"][r, o[:, 0]], J["p4"][r, o[:, 1]]]
    prior = np.exp(-0.5 * (ETA / SIG_ETA) ** 2)
    best_w = np.zeros(n)
    t1, t2 = np.zeros((n, 4)), np.zeros((n, 4))
    for i1, i2 in ((0, 1), (1, 0)):
        n1, v1 = nu_eta_solutions(l1, b[i1])
        n2, v2 = nu_eta_solutions(l2, b[i2])
        A, B = n1.reshape(n, -1, 4), n2.reshape(n, -1, 4)
        va, vb = v1.reshape(n, -1), v2.reshape(n, -1)
        pa, pb = np.repeat(prior, 2)[None, :] * va, np.repeat(prior, 2)[None, :] * vb
        dx = mx[:, None, None] - A[:, :, None, 0] - B[:, None, :, 0]
        dy = my[:, None, None] - A[:, :, None, 1] - B[:, None, :, 1]
        w = np.exp(-0.5 * (dx ** 2 + dy ** 2) / SIG_MET ** 2) * pa[:, :, None] * pb[:, None, :]
        w = w.reshape(n, -1)
        k = w.argmax(1)
        wm = w[r, k]
        better = wm > best_w
        ia, ib = k // B.shape[1], k % B.shape[1]
        t1 = np.where(better[:, None], l1 + A[r, ia] + b[i1], t1)
        t2 = np.where(better[:, None], l2 + B[r, ib] + b[i2], t2)
        best_w = np.where(better, wm, best_w)
    return t1, t2, np.where(ok[:, None], o, -1), ok & (best_w > 0)


def inputs(d, ch, metcol="puppimet"):
    """jets, the J/psi, the index of the jet holding it (-1: none), leptons and MET of the candidates"""
    n = len(d)
    J = jets_of(d)
    jp = p4(d.pt, d.eta, d.phi, d.mass)
    dr = np.where(J["ok"], np.hypot(J["eta"] - d.eta.to_numpy(float)[:, None], dphi(J["phi"], d.phi.to_numpy(float)[:, None])), 99)
    jn = dr.argmin(1)
    jidx = np.where(dr[np.arange(n), jn] < DR_JET, jn, -1)
    met, mphi = d[metcol].to_numpy(float), d[f"{metcol}_phi"].to_numpy(float)
    leps = ["e", "mu"] if ch == "emu" else ["e" if ch == "ejets" else "mu"]
    L = [p4(d[f"{l}_pt"], d[f"{l}_eta"], d[f"{l}_phi"], np.zeros(n)) for l in leps]
    return J, jp, jidx, L, met * np.cos(mphi), met * np.sin(mphi)


def solve(J, jp, jidx, L, mx, my, ch, exclude):
    """ttbar per candidate (the jet holding the J/psi left out if `exclude`): transverse sum and full reconstruction, the role
    of the jet holding the J/psi, njets, validity"""
    n = len(mx)
    use = J["ok"] & ~((np.arange(NJ)[None, :] == jidx[:, None]) & exclude)
    # in chunks of candidates (the e mu weights are (n, 122, 122) per pairing)
    size = 300 if ch == "emu" else 2000
    parts = []
    for c0 in range(0, n, size):
        sl = slice(c0, c0 + size)
        Jc = {k: a[sl] for k, a in J.items()}
        if ch == "emu":
            parts.append(reco_emu(L[0][sl], L[1][sl], mx[sl], my[sl], Jc, use[sl]))
        else:
            th, tl, ix, c2 = reco_ljets(L[0][sl], mx[sl], my[sl], Jc, use[sl])
            parts.append((th, tl, ix, np.isfinite(c2)))
    t1, t2, idx, ok = (np.concatenate([p_[k] for p_ in parts]) for k in range(4))
    roles = {0: "b", 1: "b"} if ch == "emu" else {0: "b_had", 1: "b_lep", 2: "W jet", 3: "W jet"}
    # transverse sum of the ttbar decay products (the assignment only picks the jets)
    r = np.arange(n)
    tt_t = sum(Lk[:, :2] for Lk in L) + np.stack([mx, my], 1)
    for k in range(idx.shape[1]):
        tt_t = tt_t + np.where(ok[:, None], J["p4"][r, np.maximum(idx[:, k], 0), :2], 0)
    role = np.where(jidx < 0, "none", "extra").astype(object)
    for k, nm in roles.items():
        role = np.where(ok & (jidx >= 0) & (idx[:, k] == jidx), nm, role)
    holder = np.where((jidx >= 0)[:, None], J["p4"][r, np.maximum(jidx, 0)], jp)
    return dict(ok=ok, idx=idx, ttx=tt_t[:, 0], tty=tt_t[:, 1], tt=t1 + t2, jp=jp, jidx=jidx, role=role, njets=J["ok"].sum(1),
                holder=holder, holder_pt=np.hypot(holder[:, 0], holder[:, 1]))


def reconstruct(d, ch, exclude, metcol="puppimet", nojpsi=False):
    J, jp, jidx, L, mx, my = inputs(d, ch, metcol)
    if nojpsi:
        mx, my = mx + jp[:, 0], my + jp[:, 1]
    return solve(J, jp, jidx, L, mx, my, ch, exclude)


def observables(R_, jpt, jphi, jy):
    ttpt, ttphi = np.hypot(R_["ttx"], R_["tty"]), np.arctan2(R_["tty"], R_["ttx"])
    ux, uy = np.cos(jphi), np.sin(jphi)
    return dict(dphi=dphi(jphi, ttphi), ppar=R_["ttx"] * ux + R_["tty"] * uy, pperp=-R_["ttx"] * uy + R_["tty"] * ux, ptt=ttpt,
                dy=np.abs(jy - rapidity(R_["tt"])))
