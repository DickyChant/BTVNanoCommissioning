#!/bin/bash
# The tt + J/psi deck with the signal of associated production: the J/psi outside every ttbar decay jet. From the merged
# run5 + run5b pass (hadw/run5m: candidate tables and the build_ttJ.py summaries of the linked outputs, build_summaries.sh):
#   assoc_veto.py (veto columns) -> life_from_cands.py (summaries with the new categories, all years and per year) -> fit_lxy.py,
#   fit_jpsi.py -> splot_prompt.py (SIGNAL=assoc) -> ttbar_balance.py -> wpair_check.py; gen_assoc.py on the Pythia8 + LETO
#   samples (genstudy/ttjpsi), gen_leak_z.py (reweighted to the data z), gen_cutscan.py (tighter selections), gen_compare.py (data
#   against Pythia, normalised with the b-decay J/psi) -> figures -> slides.     usage: refresh_assoc.sh [step to start from: veto|life|fit|splot|bal|figs]
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
M=$H/run5m
G=$H/genstudy/ttjpsi
YEARS="2022 2023 2024 2025 2026"
START=${1:-veto}
step() { case "$START" in veto) return 0;; life) [ "$1" != veto ];; fit) [[ "$1" =~ ^(fit|splot|bal|figs)$ ]];; splot) [[ "$1" =~ ^(splot|bal|figs)$ ]];; bal) [[ "$1" =~ ^(bal|figs)$ ]];; figs) [ "$1" = figs ];; esac; }
if step veto; then py assoc_veto.py $M/cands $M/cands_assoc; fi
if step life; then
  py life_from_cands.py $M/results_run5m.json $M/cands_assoc results_run3.json --signal assoc
  for Y in $YEARS; do py life_from_cands.py $M/years/results_$Y.json $M/cands_assoc years/results_$Y.json --signal assoc --year $Y; done
fi
if step fit; then
  py fit_lxy.py results_run3.json fit_lxy.json | grep -E "^(notb|prompt sig)" || true
  py fit_jpsi.py results_run3.json fit_jpsi.json | grep -E "^2D" || true
  for Y in $YEARS; do py fit_lxy.py years/results_$Y.json years/fit_lxy_$Y.json > years/fit_lxy_$Y.log 2>&1
    py fit_jpsi.py years/results_$Y.json years/fit_jpsi_$Y.json > years/fit_jpsi_$Y.log 2>&1; done
fi
cp -p $M/npv.json npv.json
if step splot; then SIGNAL=assoc py splot_prompt.py $M/cands_assoc fit_jpsi.json results_run3.json splot.json $M/sweights_assoc.parquet | grep -E "^(notb|ttj|DPS)"; fi
if step bal; then py -u ttbar_balance.py $M/sweights_assoc.parquet ttbar_balance.json > ttbar_balance.log 2>&1; grep -E "f\(same|prompt |emb" ttbar_balance.log | head -20; fi
if step figs; then
  py wpair_check.py $M/sweights_assoc.parquet wpair_check.json
  GEN_TABLE=$M/gen_table.parquet py gen_assoc.py gen_assoc.json $G/ttjpsi_*.npz
  py gen_leak_z.py $M/gen_table.parquet $M/sweights_assoc.parquet gen_leak_z.json
  py gen_cutscan.py $M/gen_table.parquet $M/sweights_assoc.parquet gen_cutscan.json
  py gen_compare.py $M/gen_table.parquet $M/sweights_assoc.parquet splot.json gen_leak_z.json gen_compare.json > gen_compare.log 2>&1
  py make_figs_run3.py && py make_figs_balance.py && py make_figs_assoc.py && py make_figs_gencmp.py && py make_figs_an.py && py build_deck_assoc.py
fi
echo "refreshed $(date)"
