---
name: ttj-lxbatch
description: Run the ttJ (ttbar + J/psi -> mu mu) workflow of this branch on CERN lxbatch with HTCondor + eossubmit - smoke tests, job splitting, submission from lxplus, monitoring, resubmission and the summary builder. Use whenever running, monitoring or resubmitting ttJ jobs, or building results from their outputs.
---

# ttJ on lxbatch

The workflow `ttJ` (`src/BTVNanoCommissioning/workflows/ttJ.py`) runs the three ttbar channels with a J/psi -> mu mu
candidate in one pass: data runs only the channels of its primary dataset (Muon: `mujets`, `emu`; EGamma: `ejets`),
MC runs all three. Single channels: `ttJ_mujets`, `ttJ_ejets`, `ttJ_emu`. Batch tools are in `scripts/ttJ/`.

Placeholders below: `<eos-out>` an EOS area with quota for outputs, `<code>` a checkout of this branch on EOS,
`<submit>` a submit directory on EOS, `<proxy>` a grid proxy file on EOS, `<schedd>` the schedd that `condor_submit`
reports. Never write the user's account name, e-mail, proxy path or private paths into files that get committed.

The software is the pixi environment of `pixi.toml` / `pixi.lock` at the top of the checkout. Pixi itself comes from
cvmfs: `source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/v0.84.0/setup.sh`. It puts pixi on PATH and keeps the package
cache in `$TMPDIR`. Every batch job installs the environment on its local disk, which takes seconds and about 1 GB
(`run_job.sh`).
- Offline install: the job copies the package cache of the lock file (300 MB) from `<pixi-cache>` on EOS
  (`TTJ_PIXI_CACHE` in the submit file). Without a cache, every job downloads the packages from conda-forge / PyPI.
- Write the cache with `scripts/ttJ/pixi_cache.sh <pixi-cache>` once per lock file, before submitting.
- Setting `TTJ_CONDA=<conda-env>` uses a conda env instead.

## Before anything

- **Credentials are the user's.** Kerberos (`kinit`) and the grid proxy (`voms-proxy-init --voms cms --valid 192:00
  --out <proxy>`) need the user; ask them to run these (`! <command>` in Claude Code) and wait. When ssh to lxplus or
  `condor_q` starts failing with authentication errors, the ticket has expired: ask again, do not retry in a loop.
- **Quota:** `eos quota <eos-out>` (home areas fill up; a group area is safer). Outputs and condor logs go to EOS,
  never to the AFS home.
- **Code is read at run time.** Jobs import `<code>` while they run: never edit it while jobs are queued or running.
  For a new campaign copy a fresh checkout (e.g. `<code>_v2`).

## 1. Smoke test (always first)

One MC file and one file of each primary dataset, two chunks each:

```bash
source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/v0.84.0/setup.sh
cd <code> && pixi install --locked && eval "$(pixi shell-hook --frozen)"   # .pixi/ in the checkout (gitignored)
export PYTHONPATH=<code>/src:$PYTHONPATH X509_USER_PROXY=<proxy> HADW_STUDY_HISTS=1
python runner.py --wf ttJ --json one_file.json --campaign Summer24 --year 2024 --executor iterative \
  --chunk 50000 --max 2 --skipbadfiles --isSyst False --ttbar-reweights full -o smoke.coffea --outputdir <scratch>
```

Check with `coffea.util.load`: per dataset the expected channel keys, non-zero `nbjet` (l + jets) or `dil_ntrack` (e mu)
entries, and in MC filled `jpsi_mass` / `jpsi_lxy`. In data a few chunks rarely hold a J/psi (about 1e-3 per ttbar event).

## 2. Jobs

```bash
python3 scripts/ttJ/make_jobs.py <submit>/subjson <submit>/jobs.txt 10 Muon2024.json EGamma2024.json   # 10 files per data job
```

Fileset JSONs come from `scripts/fetch.py` (DAS). Copy `scripts/ttJ/submit.jdl.template` to `<submit>/submit.jdl`, fill
the `@...@` placeholders, and create `<submit>/logs` and `<submit>/out`. Keep `max_materialize` at about 400 (all jobs
read the code on EOS) and `transfer_output_files = ""`: otherwise condor copies the job's pixi environment back.
Right after submitting, run `condor_q -name <schedd> <cluster>.0 -better-analyze`: in November 2026 late
materialization (`max_materialize`) on the eossubmit schedd wrapped the job requirements into a loop that matched no
machine ("never PreVerRequirements"), and only a submission without `max_materialize` started.
Before a pass, point the files with a replica on CERN EOS at `root://eoscms.cern.ch//eos/cms/store/...` in the
sub-jsons. Through the redirectors some files fail with "Redirect limit has been reached", and FUSE `/eos/...` paths
can stall a batch job for hours.
Use the "tomorrow" flavour; 40-file jobs have hit the 8 h "workday" limit.

## 3. Submit (from the agent's machine, non-interactively)

```bash
ssh -o BatchMode=yes lxplus "cd <submit> && module load lxbatch/eossubmit >/dev/null 2>&1 && condor_submit submit.jdl"
```

Write down the cluster id and the schedd name it prints; every later query names both. The user may have other
clusters on the same schedd: only ever act on the cluster ids of this campaign.

## 4. Monitor

```bash
ssh -o BatchMode=yes lxplus "condor_q -name <schedd> <cluster> -totals"        # idle / running / held
ssh -o BatchMode=yes lxplus "condor_q -name <schedd> <cluster> -hold -af HoldReason | sort | uniq -c"
find <eos-out> -mindepth 2 -name '*.coffea' | wc -l                               # finished jobs
```

- Count outputs with `-mindepth 2`: failed attempts can leave ~20-byte stub files at the top level of `<eos-out>`.
- A job's stdout/stderr: `<submit>/out/<name>.out|.err`. The meaningful lines:
  - "package cache ... in N s", then "pixi environment installed in N s": the software set-up. Two pixi WARN lines
    about repodata in the .err are expected when the node cannot reach conda-forge.
  - "env import failed": the import retry.
  - "runner exit=1" with an xrootd error: a file access problem. The job retries through the global redirector,
    then the FNAL one.
- A job with little CPU time after hours of wall time (`condor_q <cluster> -af RemoteUserCpu RemoteWallClockTime`)
  is stalled on file reads: remove it and resubmit it with XRootD URLs.
- Poll every 10-20 min in a background loop, not in a tight loop; report progress as done / total with a time estimate.

## 5. Resubmit what is missing

When the cluster has left the queue: list the names in `jobs.txt` without `<eos-out>/<name>/<name>.coffea`, write them
to `jobs_resub.txt`, and submit a copy of submit.jdl that queues from it. Look at a few `.err` files first: a
systematic error needs a fix, not a resubmission.

## 6. Results

```bash
python3 scripts/ttJ/build_ttJ.py <eos-out> [<more out dirs>] results.json
```

It sums the nominal histograms per channel and writes the J/psi spectra, the sideband-subtracted yields per ttbar event
(window 2.95-3.25 GeV minus half of 2.60-2.90 and 3.30-3.60 GeV), the b-matched / not-b-matched categories, the
lifetime regions (prompt |l_xy| < 0.06 mm, non-prompt l_xy > 0.1 mm, pileup-like |z0| >= 0.1 cm) and the ttbar
kinematics of the J/psi events. It also reads the older hadw_jpsi / hadw_e_jpsi outputs ("sl"/"dil" channels).

The analysis after the summaries (candidate tables, the ttbar-jet veto, the lifetime and m x l_xy fits, the simulation,
generator and background comparisons, figures, deck, analysis-note figures) is in `scripts/ttJ/analysis/` (README.md
there); it runs in the `analysis` environment of `pixi.toml` (the grid environment plus the fit and plotting packages):
`pixi run -e analysis python <script>`.

## Changing the environment

Edit `pixi.toml`, run `pixi lock`, and commit `pixi.lock` with it: the jobs install with `--locked`, so a stale lock
fails at the start of the job. Then run `scripts/ttJ/pixi_cache.sh <pixi-cache>`. The cache name is the lock file's
hash, so jobs with a new lock file and no cache fall back to the network ("no package cache" in the job output).
Validate a new environment before a pass. Run one chunk of a few data sets in the old and the new environment on
the same machine, then compare the outputs histogram by histogram and column by column. They should be identical
unless a physics package changed on purpose. Across batch nodes of different CPU types the candidate columns differ
in the float32 rounding (relative 1e-7, more for small differences such as sv_dr); the histograms stay identical.

## Do not

- `condor_rm` anything without listing it first and checking the cluster id is this campaign's.
- Delete or overwrite outputs without the user's go-ahead; move superseded outputs aside instead.
- Change credentials, permissions or another user's environment; the conda env may belong to someone else - read only.
