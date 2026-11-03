#!/bin/bash
# One batch job of the ttJ workflow: runner.py over one sub-json (a few files of one data set).
# usage: run_job.sh <job name> [isSyst]      (the sub-json is $TTJ_SUBJSON/<job name>.json)
# Environment (set in the submit file, see submit.jdl.template):
#   TTJ_CODE     checkout of the ttJ branch readable from the batch nodes (e.g. on EOS); jobs read the code at run time,
#                so do not edit it while jobs run
#   TTJ_SUBJSON  directory of the sub-jsons (scripts/ttJ/make_jobs.py)
#   TTJ_OUT      output directory: <TTJ_OUT>/<job name>/<job name>.coffea
#   TTJ_CONDA    conda environment with coffea 0.7 (activated with its etc/profile.d/conda.sh)
#   X509_USER_PROXY  grid proxy readable from the batch node (for xrootd)
NAME=$1; SYST=${2:-False}
SUB=$TTJ_SUBJSON/${NAME}.json
echo "[job] host=$(hostname) name=$NAME isSyst=$SYST start=$(date)"
# spread the start-up reads of a shared conda environment (hundreds of jobs start at once)
sleep $((RANDOM % 600))
source "$(dirname "$(dirname "$TTJ_CONDA")")/etc/profile.d/conda.sh" && conda activate "$TTJ_CONDA"
export PYTHONPATH=$TTJ_CODE/src:$PYTHONPATH
# a conda environment on EOS sometimes fails to import under batch load (half-imported modules such as numba.typed):
# retry until the whole workflow package imports
for try in $(seq 1 10); do
  (cd "$TTJ_CODE" && python -c 'import coffea.lumi_tools, numba.typed, BTVNanoCommissioning.workflows' 2>/dev/null) && break
  echo "[job] env import failed (try $try), retrying"; sleep $((60 + RANDOM % 180))
  conda activate "$TTJ_CONDA"
done
export HOME=$PWD
export HADW_STUDY_HISTS=1  # fill only the study histograms
TMPJ=$PWD/${NAME}_global.json
sed 's#root://xrootd-cms.infn.it/#root://cms-xrd-global.cern.ch/#g' "$SUB" > "$TMPJ"
cd "$TTJ_CODE"
# up to four attempts: the redirector of the sub-json first, then the global one
for J in "$SUB" "$TMPJ" "$SUB" "$TMPJ"; do
  python runner.py --wf ttJ --json "$J" --campaign Summer24 --year 2024 --executor iterative \
    --chunk 40000 --skipbadfiles --isSyst $SYST --ttbar-reweights full -o ${NAME}.coffea --outputdir "$TTJ_OUT" 2> >(grep -v -i 'warn' >&2)
  rc=$?; echo "[job] runner exit=$rc json=$J $(date)"
  [ $rc -eq 0 ] && [ -f "$TTJ_OUT/${NAME}/${NAME}.coffea" ] && break
  sleep $((30 + RANDOM % 90))
done
ls -l "$TTJ_OUT/${NAME}/${NAME}.coffea" 2>&1
exit $rc
