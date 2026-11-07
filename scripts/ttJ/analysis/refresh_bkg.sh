#!/bin/bash
# The non-ttbar backgrounds (pass hadw/out_bkg, cluster 897661: single top tW / t / s, W + jets MLM 1-4J, ttW, ttZ, WW / WZ / ZZ,
# QCD mu-enriched; Summer24 NanoAODv15, the code of the data pass run5): build_ttJ.py summary (selected events by cross section),
# candidate tables -> assoc_veto.py -> mc_compare.py with the backgrounds next to the ttbar simulation -> figures -> slides.
# usage: refresh_bkg.sh [step to start from: build|veto|cmp|figs]
set -e
D=$(cd "$(dirname "$0")" && pwd)
# the pass area (run5m/, mc5/, out_mc5/, ...) and the checkout whose pixi.toml defines the analysis environment
H=${TTJ_STORE:?"set TTJ_STORE to the directory of the passes"}
REPO=${TTJ_REPO:-$(cd "$D/../../.." 2>/dev/null && pwd)}
[ -f "$REPO/pixi.toml" ] || { echo "set TTJ_REPO to the BTVNanoCommissioning checkout (its pixi.toml)"; exit 1; }
source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/${TTJ_PIXI:-v0.84.0}/setup.sh > /dev/null
# every step in the analysis environment of pixi.toml (the grid environment plus the fit and plotting packages)
py() { (cd "$D" && env -u PYTHONHOME -u PYTHONPATH pixi run --manifest-path "$REPO/pixi.toml" -e analysis python "$@"); }
cd "$D"
OUT=$H/out_bkg
MB=$H/mcbkg
START=${1:-build}
mkdir -p $MB
step() { case "$START" in build) return 0;; veto) [ "$1" != build ];; cmp) [[ "$1" =~ ^(cmp|figs)$ ]];; figs) [ "$1" = figs ];; esac; }
if step build; then
  py build_ttJ.py $OUT $MB/results_bkg.json > $MB/build_bkg.log 2>&1 &
  py dump_cands.py $OUT $MB/cands > $MB/dump.log 2>&1
  wait
fi
if step veto; then py assoc_veto.py $MB/cands $MB/cands_assoc; fi
if step cmp; then
  TTJ_BKG=$MB/cands_assoc:$MB/cands/sumw.json:$MB/results_bkg.json py mc_compare.py $H/mc5/cands_assoc $H/mc5/cands/sumw.json xsec_map.json $H/mc5/results_mc.json \
    $H/run5m/sweights_assoc.parquet results_run3.json mc/splot.json mc_compare.json | tee mc_compare.log
fi
if step figs; then
  py make_figs_mc.py && py build_deck_assoc.py
fi
echo "refreshed $(date)"
