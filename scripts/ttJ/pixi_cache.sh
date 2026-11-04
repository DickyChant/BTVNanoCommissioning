#!/bin/bash
# Package cache of the pixi environment for the batch jobs: installs pixi.lock of this checkout into an empty cache and
# writes it to <dir>/<first 16 hex digits of sha256(pixi.lock)>.tar.zst (about 300 MB). With TTJ_PIXI_CACHE=<dir>,
# run_job.sh copies it and installs offline instead of downloading the packages from conda-forge / PyPI in every job.
# Run it again after every `pixi lock` (a new lock file has a new name; jobs without a matching cache use the network).
# usage: scripts/ttJ/pixi_cache.sh <dir, e.g. on EOS> [pixi version on cvmfs, default v0.84.0]
set -e
CODE=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$1; V=${2:-v0.84.0}
[ -n "$OUT" ] || { sed -n '2,6p' "$0"; exit 1; }
KEY=$(sha256sum "$CODE/pixi.lock" | cut -c1-16)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/env" && cp "$CODE/pixi.toml" "$CODE/pixi.lock" "$W/env/"
export PIXI_CACHE_DIR=$W/rattler
source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/$V/setup.sh >/dev/null
pixi install --manifest-path "$W/env/pixi.toml" --locked
mkdir -p "$OUT"
tar -I "zstd -3 -T0" -cf "$OUT/$KEY.tar.zst.tmp" -C "$W" rattler && mv "$OUT/$KEY.tar.zst.tmp" "$OUT/$KEY.tar.zst"
ls -l "$OUT/$KEY.tar.zst"
