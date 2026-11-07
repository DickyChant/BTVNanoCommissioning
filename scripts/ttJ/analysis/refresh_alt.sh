#!/bin/bash
# The ttbar generator / tune variants (pass hadw/out_alt, cluster 897658, the code of the data pass run5): candidate tables
# (dump_cands.py, sumw.json) -> assoc_veto.py -> gen_variants.py (prompt and b-decay J/psi by MC truth per generator, against
# the data) -> figures -> slides.
# usage: refresh_alt.sh [step to start from: dump|veto|cmp|figs]
set -e
D=$(cd "$(dirname "$0")" && pwd)
# the pass area (run5m/, mc5/, out_mc5/, ...) and the checkout whose pixi.toml defines the analysis environment
H=${TTJ_STORE:?"set TTJ_STORE to the directory of the passes"}
REPO=${TTJ_REPO:-$(cd "$D/../../.." 2>/dev/null && pwd)}
[ -f "$REPO/pixi.toml" ] || { echo "set TTJ_REPO to the BTVNanoCommissioning checkout (its pixi.toml)"; exit 1; }
source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/${TTJ_PIXI:-v0.84.0}/setup.sh > /dev/null
# every step in the analysis environment of pixi.toml (the grid environment plus the fit and plotting packages)
py() { (cd "$D" && pixi run --manifest-path "$REPO/pixi.toml" -e analysis python "$@"); }
cd "$D"
OUT=$H/out_alt
MA=$H/mcalt
START=${1:-dump}
mkdir -p $MA
step() { case "$START" in dump) return 0;; veto) [ "$1" != dump ];; cmp) [[ "$1" =~ ^(cmp|figs)$ ]];; figs) [ "$1" = figs ];; esac; }
if step dump; then py dump_cands.py $OUT $MA/cands > $MA/dump.log 2>&1; tail -3 $MA/dump.log; fi
if step veto; then py assoc_veto.py $MA/cands $MA/cands_assoc; fi
if step cmp; then py gen_variants.py $H/mc5/cands_assoc $H/mc5/cands/sumw.json $MA/cands_assoc $MA/cands/sumw.json xsec_map.json $H/run5m/sweights_assoc.parquet gen_variants.json | tee gen_variants.log; fi
if step figs; then
  py make_figs_mc.py && py build_deck_assoc.py
fi
echo "refreshed $(date)"
