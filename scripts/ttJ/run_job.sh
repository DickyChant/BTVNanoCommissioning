#!/bin/bash
# One batch job of the ttJ workflow: runner.py over one sub-json (a few files of one data set).
# usage: run_job.sh <job name> [isSyst] [campaign] [year]      (the sub-json is $TTJ_SUBJSON/<job name>.json;
#        campaign / year default to $TTJ_CAMPAIGN / $TTJ_YEAR, then Summer24 / 2024)
# Environment (set in the submit file, see submit.jdl.template):
#   TTJ_CODE     checkout of the ttJ branch readable from the batch nodes (e.g. on EOS); jobs read the code at run time,
#                so do not edit it while jobs run
#   TTJ_SUBJSON  directory of the sub-jsons (scripts/ttJ/make_jobs.py)
#   TTJ_OUT      output directory: <TTJ_OUT>/<job name>/<job name>.coffea
#   TTJ_CONDA    optional: a conda environment with coffea 0.7 (activated with its etc/profile.d/conda.sh) instead of
#                the pixi environment of $TTJ_CODE/pixi.lock, which the job installs on its local disk (pixi from cvmfs)
#   TTJ_PIXI_CACHE  directory (e.g. on EOS) of the package caches of the lock files (scripts/ttJ/pixi_cache.sh); unset or
#                no cache of this lock file: the job downloads the packages (about 300 MB) from conda-forge / PyPI
#   TTJ_PIXI     optional: pixi version on /cvmfs/cms-griddata.cern.ch/cat/sw/pixi (default v0.84.0)
#   X509_USER_PROXY  grid proxy readable from the batch node (for xrootd)
#   TTJ_WF       workflow (default ttJ; ttJ_bph for BPHNano of the low-mass double-muon parking)
NAME=$1; SYST=${2:-False}; WF=${TTJ_WF:-ttJ}; CAMPAIGN=${3:-${TTJ_CAMPAIGN:-Summer24}}; YEAR=${4:-${TTJ_YEAR:-2024}}
SUB=$TTJ_SUBJSON/${NAME}.json
echo "[job] host=$(hostname) wf=$WF name=$NAME isSyst=$SYST campaign=$CAMPAIGN year=$YEAR start=$(date)"
export HOME=$PWD
if [ -n "$TTJ_CONDA" ]; then
  # spread the start-up reads of a shared conda environment (hundreds of jobs start at once)
  sleep $((RANDOM % 600))
  source "$(dirname "$(dirname "$TTJ_CONDA")")/etc/profile.d/conda.sh" && conda activate "$TTJ_CONDA"
else
  # the pixi environment of the lock file, installed in the job from a local package cache: the cache of this lock file
  # from $TTJ_PIXI_CACHE (scripts/ttJ/pixi_cache.sh, an offline install), else the packages from conda-forge / PyPI
  sleep $((RANDOM % 60))
  export PIXI_CACHE_DIR=$PWD/rattler
  source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/${TTJ_PIXI:-v0.84.0}/setup.sh >/dev/null
  mkdir -p "$PWD/env" && cp "$TTJ_CODE/pixi.toml" "$TTJ_CODE/pixi.lock" "$PWD/env/"
  if [ -n "$TTJ_PIXI_CACHE" ]; then
    C=$TTJ_PIXI_CACHE/$(sha256sum "$TTJ_CODE/pixi.lock" | cut -c1-16).tar.zst
    case $C in /eos/cms/*) U=root://eoscms.cern.ch/$C ;; /eos/user/*) U=root://eosuser.cern.ch/$C ;; *) U=$C ;; esac
    t0=$(date +%s)
    { xrdcp -s -f "$U" "$PWD/pixi_cache.tar.zst" || cp "$C" "$PWD/pixi_cache.tar.zst"; } 2>/dev/null \
      && tar -I zstd -xf "$PWD/pixi_cache.tar.zst" -C "$PWD" && echo "[job] package cache $C in $(($(date +%s) - t0)) s" \
      || echo "[job] no package cache $C: packages from the network"
    rm -f "$PWD/pixi_cache.tar.zst"
  fi
  ok=0
  for try in 1 2 3; do
    t0=$(date +%s); pixi install --manifest-path "$PWD/env/pixi.toml" --locked && ok=1 && break
    echo "[job] pixi install failed (try $try), retrying"; sleep $((30 + RANDOM % 90))
  done
  [ $ok = 1 ] || { echo "[job] no pixi environment"; exit 1; }
  echo "[job] pixi environment installed in $(($(date +%s) - t0)) s ($(du -sh "$PIXI_CACHE_DIR" | cut -f1) of packages)"
  eval "$(pixi shell-hook --manifest-path "$PWD/env/pixi.toml" --frozen)"
fi
export PYTHONPATH=$TTJ_CODE/src:$PYTHONPATH
# an environment on EOS sometimes fails to import under batch load (half-imported modules such as numba.typed):
# retry until the whole workflow package imports
for try in $(seq 1 10); do
  (cd "$TTJ_CODE" && python -c 'import coffea.lumi_tools, numba.typed, BTVNanoCommissioning.workflows' 2>/dev/null) && break
  echo "[job] env import failed (try $try), retrying"; sleep $((60 + RANDOM % 180))
  [ -n "$TTJ_CONDA" ] && conda activate "$TTJ_CONDA"
done
export HADW_STUDY_HISTS=1  # fill only the study histograms
TMPJ=$PWD/${NAME}_global.json
sed 's#root://xrootd-cms.infn.it/#root://cms-xrd-global.cern.ch/#g' "$SUB" > "$TMPJ"
FNALJ=$PWD/${NAME}_fnal.json
sed 's#root://xrootd-cms.infn.it/#root://cmsxrootd.fnal.gov/#g' "$SUB" > "$FNALJ"
cd "$TTJ_CODE"
# up to four attempts: the redirector of the sub-json first, then the global one, then the FNAL one (the European and
# global redirectors sometimes send a file to a server that fails with "Redirect limit has been reached")
for J in "$SUB" "$TMPJ" "$FNALJ" "$SUB"; do
  python runner.py --wf $WF --json "$J" --campaign $CAMPAIGN --year $YEAR --executor iterative \
    --chunk 40000 --skipbadfiles --isSyst $SYST --ttbar-reweights full -o ${NAME}.coffea --outputdir "$TTJ_OUT" 2> >(grep -v -i 'warn' >&2)
  rc=$?; echo "[job] runner exit=$rc json=$J $(date)"
  [ $rc -eq 0 ] && [ -f "$TTJ_OUT/${NAME}/${NAME}.coffea" ] && break
  sleep $((30 + RANDOM % 90))
done
ls -l "$TTJ_OUT/${NAME}/${NAME}.coffea" 2>&1
exit $rc
