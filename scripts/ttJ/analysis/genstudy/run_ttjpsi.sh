#!/bin/bash
# One batch job of the gen-level Pythia 8 + LETO ttbar sample (gen_ttjpsi.py; J/psi in the parton shower):
#   run_ttjpsi.sh <nev> <seed> <ldme factor> <output dir>   ->   <output dir>/ttjpsi_<seed>.npz
# Pythia 8.317 with its python bindings from the LCG_110 view (not in pixi.toml: conda-forge has no LETO-enabled Pythia
# build with the python module pinned to this version).
set -e
D=$(cd "$(dirname "$0")" && pwd)
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
OUT=$4/ttjpsi_$2.npz
PYTHONPATH=$D:$PYTHONPATH python3 $D/gen_ttjpsi.py --nev $1 --seed $2 --ldme $3 --out gen.npz > gen.log 2>&1 || { tail -30 gen.log; exit 1; }
grep "events in" gen.log
cp gen.npz $OUT.tmp && mv $OUT.tmp $OUT && ls -l $OUT
