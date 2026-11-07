#!/bin/bash
# Data against the CMS ttbar simulation (Summer24 TTtoLNu2Q + TTto2L2Nu, POWHEG + Pythia8 CP5; pass hadw/out_mc5, cluster 897589,
# the code of the data pass run5): build_ttJ.py summaries (MC by cross section; and MC as pseudo-data, TTJ_AS_DATA=1), candidate
# tables (dump_cands.py, sumw.json) -> assoc_veto.py -> the data chain on the MC as pseudo-data (life_from_cands.py, fit_lxy.py,
# fit_jpsi.py, splot_prompt.py; outputs in mc/) for the closure -> mc_compare.py -> figures -> slides.
# usage: refresh_mc.sh [step to start from: build|veto|fit|cmp|figs]
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
OUT=$H/out_mc5
MC=$H/mc5
MD=$D/mc
M=$H/run5m
START=${1:-build}
mkdir -p $MC $MD
step() { case "$START" in build) return 0;; veto) [ "$1" != build ];; fit) [[ "$1" =~ ^(fit|cmp|figs)$ ]];; cmp) [[ "$1" =~ ^(cmp|figs)$ ]];; figs) [ "$1" = figs ];; esac; }
if step build; then
  py build_ttJ.py $OUT $MC/results_mc.json > $MC/build_mc.log 2>&1 &
  TTJ_AS_DATA=1 py build_ttJ.py $OUT $MC/results_asdata.json > $MC/build_asdata.log 2>&1 &
  py dump_cands.py $OUT $MC/cands && py sum_npv.py $OUT $MD/npv.json
  wait
fi
if step veto; then py assoc_veto.py $MC/cands $MC/cands_assoc; fi
if step fit; then
  py life_from_cands.py $MC/results_asdata.json $MC/cands_assoc $MD/results_run3.json --signal assoc
  py fit_lxy.py $MD/results_run3.json $MD/fit_lxy.json > $MD/fit_lxy.log 2>&1 || tail -5 $MD/fit_lxy.log
  py fit_jpsi.py $MD/results_run3.json $MD/fit_jpsi.json > $MD/fit_jpsi.log 2>&1 || tail -5 $MD/fit_jpsi.log
  SIGNAL=assoc py splot_prompt.py $MC/cands_assoc $MD/fit_jpsi.json $MD/results_run3.json $MD/splot.json $MC/sweights_assoc.parquet > $MD/splot.log 2>&1 || tail -5 $MD/splot.log
  grep -E "^(notb|ttj|inb)" $MD/splot.log || true
fi
# the non-ttbar backgrounds next to the ttbar simulation once refresh_bkg.sh has made their tables
BKG=""; [ -f $H/mcbkg/results_bkg.json ] && BKG=$H/mcbkg/cands_assoc:$H/mcbkg/cands/sumw.json:$H/mcbkg/results_bkg.json
if step cmp; then TTJ_BKG=$BKG py mc_compare.py $MC/cands_assoc $MC/cands/sumw.json xsec_map.json $MC/results_mc.json $M/sweights_assoc.parquet results_run3.json $MD/splot.json mc_compare.json; fi
if step figs; then
  py make_figs_mc.py && py build_deck_assoc.py
fi
echo "refreshed $(date)"
