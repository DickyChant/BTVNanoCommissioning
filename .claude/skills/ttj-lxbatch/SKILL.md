---
name: ttj-lxbatch
description: Run the ttJ / ttJ_bph (ttbar + J/psi -> mu mu, single-lepton-triggered or on BPHNano of the low-mass double-muon parking) workflows of this branch on CERN lxbatch with HTCondor + eossubmit - smoke tests, job splitting, submission from lxplus, monitoring, resubmission and the summary builder. Use whenever running, monitoring or resubmitting ttJ jobs, or building results from their outputs.
---

# ttJ on lxbatch

The workflow `ttJ` (`src/BTVNanoCommissioning/workflows/ttJ.py`) runs the three ttbar channels with a J/psi -> mu mu
candidate in one pass: data runs only the channels of its primary dataset (Muon: `mujets`, `emu`; EGamma: `ejets`),
MC runs all three. Single channels: `ttJ_mujets`, `ttJ_ejets`, `ttJ_emu`. Batch tools are in `scripts/ttJ/`.

Placeholders below: `<eos-out>` an EOS area with quota for outputs, `<code>` a checkout of this branch on EOS,
`<submit>` a submit directory on EOS, `<conda-env>` a conda env with coffea 0.7 (the BTVNanoCommissioning
environment), `<proxy>` a grid proxy file on EOS, `<schedd>` the schedd that `condor_submit` reports. Never write
the user's account name, e-mail, proxy path or private paths into files that get committed.

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
source <conda-env>/../../etc/profile.d/conda.sh && conda activate <conda-env>
export PYTHONPATH=<code>/src:$PYTHONPATH X509_USER_PROXY=<proxy> HADW_STUDY_HISTS=1
cd <code> && python runner.py --wf ttJ --json one_file.json --campaign Summer24 --year 2024 --executor iterative \
  --chunk 50000 --max 2 --skipbadfiles --isSyst False --ttbar-reweights full -o smoke.coffea --outputdir <scratch>
```

Check with `coffea.util.load`: per dataset the expected channel keys, non-zero `nbjet` (l + jets) or `dil_ntrack` (e mu)
entries, and in MC filled `jpsi_mass` / `jpsi_lxy`. In data a few chunks rarely hold a J/psi (about 1e-3 per ttbar event).

## 2. Jobs

```bash
python3 scripts/ttJ/make_jobs.py <submit>/subjson <submit>/jobs.txt 10 Muon2024.json EGamma2024.json   # 10 files per data job
```

Fileset JSONs come from `scripts/fetch.py` (DAS). Copy `scripts/ttJ/submit.jdl.template` to `<submit>/submit.jdl`, fill
the `@...@` placeholders, and create `<submit>/logs` and `<submit>/out`. Keep `max_materialize` at about 400: more
concurrent jobs make a shared EOS conda environment fail to import (run_job.sh retries, but jobs then start slowly).
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
- A job's stdout/stderr: `<submit>/out/<name>.out|.err`; "env import failed" lines are the import retry, "runner exit=1"
  with an xrootd error is a file access problem (the job retries via the global redirector).
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

## BPHNano: low-mass double-muon parking (workflow ttJ_bph)

- **Inputs:** BPHNano of ParkingDoubleMuonLowMass* = NanoAODv15 plus the BPH collections (`MuMu_*` vertex fit,
  `BPHMuon_*` with per-muon trigger flags). No central 2024 BPHNano exists (private 2024C/D/E productions, two
  streams); central 2025 PromptReco parking NanoAOD is BPHNano (eight streams). The streams split the events, so a
  subset of streams is a subset of the luminosity: say which streams were used.
- **Trigger, carefully:** the event must fire `HLT_DoubleMu4_3_LowMass` and both J/psi muons must be matched to it
  (`BPHMuon_fired_HLT_DoubleMu4_3_LowMass`): the J/psi is what triggered the event. The ttbar leptons are selected
  offline only. Whether the channel's single-lepton trigger (IsoMu24 / Ele30) also fired is kept as the `sltrig` axis:
  those events are also in the single-lepton-triggered ttJ sample, so remove them (sltrig = 0) before combining the two.
  Do not mix other parking paths in without asking: their prescales and thresholds differ by run.
- **J/psi:** from `MuMu` (opposite sign only, so no same-sign control): the highest `svprob` candidate in 2.0-4.5 GeV
  (fit mass); l_xy = m (L_xy . p_T) / p_T^2 from `MuMu_vtx - PV`. `MuMu_l_xy` itself is the plain transverse decay length.
- **Normalisation:** every selected event has a triggered dimuon, so "per ttbar event" is not the single-lepton
  sample's quantity; compare yields per luminosity.
- **Jobs:** `TTJ_WF=ttJ_bph` in the submit file; 20 files per job (the files are small).

## Do not

- `condor_rm` anything without listing it first and checking the cluster id is this campaign's.
- Delete or overwrite outputs without the user's go-ahead; move superseded outputs aside instead.
- Change credentials, permissions or another user's environment; the conda env may belong to someone else - read only.
