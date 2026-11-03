"""UParT (AK4) b-tagging scale factors with the fixed-working-point method 1a.

Event weight = prod_tagged SF * prod_untagged (1 - SF * eff) / (1 - eff), with the central BTV SFs
(UParTAK4_comb for b and c jets, UParTAK4_light for light jets) and MC tagging efficiencies `eff`
(hadron flavour x pT x |eta|) measured in the analysis selection, per process group, and stored in
data/BTV/upart_eff_<channel>.json (built by `eff_json` from the `upart_eff` histograms of an MC
pass without SFs). The b/c and the light SFs are varied separately.
"""

import json
import os

import awkward as ak
import correctionlib
import numpy as np

BTV_JSON = {
    "Summer24": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/btagging.json.gz",
}
PT_EDGES = [20, 30, 50, 70, 100, 140, 200, 300, 1000]
ETA_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
GROUPS = (
    ("tt", ("TT",)),
    ("wjets", ("WtoMuNu", "WtoENu")),
    ("dy", ("DYto",)),
    ("st", ("TbarW", "TW")),
    ("vv", ("WW", "WZ", "ZZ")),
)
_CACHE = {}


def group_of(dataset):
    """Process group of a dataset name (efficiency maps are per group); "all" if unknown."""
    for g, prefixes in GROUPS:
        if dataset.startswith(prefixes):
            return g
    return "all"


def _load(campaign, channel):
    key = (campaign, channel)
    if key not in _CACHE:
        cset = (
            correctionlib.CorrectionSet.from_file(BTV_JSON[campaign])
            if campaign in BTV_JSON
            else None
        )
        path = os.path.join(
            os.path.dirname(__file__), "..", "data", "BTV", f"upart_eff_{channel}.json"
        )
        eff = json.load(open(path)) if os.path.exists(path) else None
        _CACHE[key] = (cset, eff)
    return _CACHE[key]


def upart_weights(pt, eta, flav, tagged, campaign, channel, group="all", wp="M"):
    """Per-event method-1a weights from jagged per-jet arrays.

    Returns {"nom", "bc_up", "bc_dn", "l_up", "l_dn"} (numpy, one value per event), or None when
    the campaign has no UParT SFs or the efficiency map for the channel is missing.
    """
    cset, eff = _load(campaign, channel)
    if cset is None or eff is None:
        return None
    emap = eff["maps"].get(group, eff["maps"]["all"])
    counts = ak.to_numpy(ak.num(pt, axis=1))
    pt = np.clip(ak.to_numpy(ak.flatten(pt)), 20.0, 999.0)
    aeta = np.clip(np.abs(ak.to_numpy(ak.flatten(eta))), 0.0, 2.499)
    flav = ak.to_numpy(ak.flatten(flav)).astype(int)
    tag = ak.to_numpy(ak.flatten(tagged)).astype(bool)

    pe, ee = eff["pt_edges"], eff["eta_edges"]
    ipt = np.clip(np.searchsorted(pe, pt, side="right") - 1, 0, len(pe) - 2)
    ieta = np.clip(np.searchsorted(ee, aeta, side="right") - 1, 0, len(ee) - 2)
    e = np.empty(len(pt))
    for f in (0, 4, 5):
        m = flav == f
        e[m] = np.asarray(emap[str(f)])[ipt[m], ieta[m]]
    e = np.clip(e, 1e-4, 1.0 - 1e-4)
    heavy = flav > 0

    def event_weight(syst_bc, syst_light):
        sf = np.ones(len(pt))
        if heavy.any():
            sf[heavy] = cset["UParTAK4_comb"].evaluate(
                syst_bc, wp, flav[heavy], aeta[heavy], pt[heavy]
            )
        if (~heavy).any():
            sf[~heavy] = cset["UParTAK4_light"].evaluate(
                syst_light, wp, flav[~heavy], aeta[~heavy], pt[~heavy]
            )
        w = np.where(tag, sf, np.clip(1.0 - sf * e, 0.0, None) / (1.0 - e))
        return ak.to_numpy(ak.prod(ak.unflatten(w, counts), axis=1))

    return {
        "nom": event_weight("central", "central"),
        "bc_up": event_weight("up", "central"),
        "bc_dn": event_weight("down", "central"),
        "l_up": event_weight("central", "up"),
        "l_dn": event_weight("central", "down"),
    }


def add_upart_weights(weights, pruned_ev, campaign, channel, isSyst):
    """Add the UParT SF weight ("btagSF") and, with systematics, its b/c and light variations,
    from the jets stored as pruned_ev.upeff_{pt,eta,flav,tag}. No-op without MC flavour or maps.
    """
    if "upeff_flav" not in pruned_ev.fields:
        return
    w = upart_weights(
        pruned_ev.upeff_pt,
        pruned_ev.upeff_eta,
        pruned_ev.upeff_flav,
        pruned_ev.upeff_tag,
        campaign,
        channel,
        group=group_of(pruned_ev.metadata.get("dataset", "")),
    )
    if w is None:
        return
    nom = np.where(w["nom"] > 0, w["nom"], 1.0)
    weights.add("btagSF", nom)
    if isSyst != False:
        one = np.ones(len(nom))
        weights.add("btagSF_bc", one, w["bc_up"] / nom, w["bc_dn"] / nom)
        weights.add("btagSF_light", one, w["l_up"] / nom, w["l_dn"] / nom)


def eff_json(hists_by_group, path, min_count=20.0):
    """Write the efficiency maps eff[group][flav][ipt][ieta] = tagged / all from summed `upart_eff`
    histograms (dict group -> hist). A group keeps its own value in a bin only when it expects at
    least `min_count` tagged and `min_count` untagged jets there (effective entries times the
    inclusive efficiency); otherwise the bin takes the inclusive map ("all", the sum over groups).
    """

    def ratio(h):
        v, w2 = h.values(), h.variances()  # (flav, pt, abseta, tag)
        tot, tot2 = v.sum(axis=-1), w2.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            e = np.where(tot > 0, v[..., 1] / tot, 0.0)
            neff = np.where(tot2 > 0, tot**2 / tot2, 0.0)
        return e, neff

    flavs = list(next(iter(hists_by_group.values())).axes["flav"])
    # inclusive map: each group weighted by its effective statistics (sum w / sum w^2 rescales its
    # generator weights to effective entries), so the map is not set by the generator-weight scale
    e_all, n_all = ratio(
        sum(
            h * (h.values().sum() / h.variances().sum())
            for h in hists_by_group.values()
            if h.variances().sum() > 0
        )
    )
    maps = {"all": e_all}
    for g, h in hists_by_group.items():
        e, n = ratio(h)
        enough = (n * e_all >= min_count) & (n * (1.0 - e_all) >= min_count)
        maps[g] = np.where(enough, e, e_all)
    h0 = next(iter(hists_by_group.values()))
    out = {
        "pt_edges": list(h0.axes["pt"].edges),
        "eta_edges": list(h0.axes["abseta"].edges),
        "maps": {
            g: {str(f): m[i].tolist() for i, f in enumerate(flavs)}
            for g, m in maps.items()
        },
        "neff_all": {str(f): n_all[i].tolist() for i, f in enumerate(flavs)},
    }
    json.dump(out, open(path, "w"), indent=1)
    return out
