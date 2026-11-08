# ttbar + J/psi analysis chain (after the batch pass)

The steps after the ttJ batch pass (`scripts/ttJ/run_job.sh`): candidate tables, the veto of the ttbar decay jets, the
lifetime fits, the unbinned m x l_xy fit (sPlot), the CMS ttbar simulation (closure, data / simulation), the generator /
tune variants and the non-ttbar backgrounds, our private generator-level Pythia 8 + LETO study, the figures (cms-plotting
rules, plotlyhep), the HTML deck and the figures of the analysis note.

## Two kinds of simulation

Kept apart in every figure, slide, caption and the animation:

| | CMS simulation | private gen-level study (`genstudy/`) |
|---|---|---|
| what | official CMS samples (NanoAOD) through the ttJ workflow of the data pass | our own Pythia 8.317 sample, generated for this study: not a CMS sample |
| J/psi | b-hadron decays, prompt J/psi from hadronisation only (no onia in the shower) | LETO: onia in the parton shower (`CharmoniumShower:all`), LDMEs scaled up and divided out |
| detector | full simulation and reconstruction | none: gen jets, b tag by efficiency, smeared MET |
| figures | CMS label ('CMS Simulation' alone; 'CMS' with data), legends 'CMS ttbar sim.' | label 'Pythia 8 + LETO, private, gen level' (`cmsdeck.label(gen=True)`), never a CMS label alone; legends 'gen. Pythia + LETO' |
| deck | tag CMS SIMULATION · FULL DETECTOR, kicker 'CMS simulation · ...' | tag PRIVATE GEN LEVEL · PYTHIA 8 + LETO, kicker 'Gen study · ...' |

## Environment

Everything runs in the `analysis` environment of the repository's `pixi.toml`: the grid environment of the batch jobs
(coffea 0.7.22, Python 3.10, numpy, pandas, scipy, pyarrow pinned as in the jobs) plus iminuit, plotly, kaleido, pillow and
plotlyhep. pixi comes from CVMFS:

```bash
source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/v0.84.0/setup.sh
pixi install -e analysis --manifest-path <checkout>/pixi.toml      # once
pixi run -e analysis --manifest-path <checkout>/pixi.toml python <script> ...
```

kaleido (PNG / PDF export) uses a Chrome found on the machine; where there is none:
`pixi run -e analysis python -c "import kaleido; kaleido.get_chrome_sync()"`.
Other environments: `anim` (Manim, `anim/ttjpsi.py`); `anim/tts/pixi.toml` for the narration (Kokoro text-to-speech, run locally, and faster-whisper); the private gen-level Pythia 8.317 + LETO sample (`genstudy/`) uses
the LCG_110 view (its Pythia python bindings).

## Configuration

| variable | meaning |
|---|---|
| `TTJ_STORE` | the pass area: `run5m/` (data candidate tables, summaries), `mc5/`, `out_mc5/`, `mcalt/`, `out_alt/`, `mcbkg/`, `out_bkg/`, `genstudy/ttjpsi/` |
| `TTJ_REPO` | the checkout with `pixi.toml` (default: this checkout, when the scripts run in place) |
| `TTJ_AUTHOR` | name on the deck cover and footer (default: none) |
| `TTJ_PREV_SPLOT` | the unbinned fit of the previous deck (the 6 Nov signal definition), quoted on the slides |
| `FIG_PREVIEW` | directory for PNG previews of every figure (outside the web area); not set: JSON only |
| `FIG_ONLY` | comma-separated name prefixes: PNG previews only for those figures |
| `TTJ_BTEMPLATE` | `btemplate.json`: the fits use the per-category b-decay shape (default: the b-jet shape) |
| `TTJ_PIXI` | pixi version on CVMFS (default v0.84.0) |

## Running

The scripts read and write next to themselves (the deck directory). Copy them into a deck directory and run the refresh
scripts there:

```bash
cp -rL scripts/ttJ/analysis/. <www>/<date>_jpsi_assoc/files/
cd <www>/<date>_jpsi_assoc/files
export TTJ_STORE=<pass area> TTJ_REPO=<checkout> TTJ_AUTHOR="<name>"
bash refresh_assoc.sh        # data: veto -> life -> fit -> splot -> bal -> figs (start from any step: refresh_assoc.sh fit)
bash refresh_mc.sh           # ttbar simulation: build -> veto -> fit (closure on the MC as pseudo-data) -> cmp -> figs
bash refresh_alt.sh          # generator / tune variants: dump -> veto -> cmp -> figs
bash refresh_bkg.sh          # non-ttbar backgrounds: build -> veto -> cmp -> figs
pixi run -e analysis --manifest-path $TTJ_REPO/pixi.toml python stage_an.py <an_figures dir>   # PDFs + MANIFEST.txt for the note
```

## The scripts

| step | script | output |
|---|---|---|
| summaries of a pass | `build_ttJ.py` (link to `../build_ttJ.py`; `TTJ_AS_DATA=1`: MC as pseudo-data), `sum_npv.py` | `results*.json`, `npv.json` |
| candidate tables | `dump_cands.py` | `cands/cands_<channel>.parquet`, `sumw.json` |
| ttbar-jet veto (signal: J/psi outside every ttbar decay jet) | `assoc_veto.py` (with `ttreco.py`, the chi2 ttbar reconstruction) | `cands_assoc/` |
| lifetime categories | `life_from_cands.py` | `results_run3.json`, `years/` |
| l_xy fit, mass and m x l_xy fits | `fit_lxy.py`, `fit_jpsi.py` | `fit_lxy.json`, `fit_jpsi.json` |
| unbinned m x l_xy fit, sWeights, DPS rate | `splot_prompt.py` (`SIGNAL=assoc`) | `splot.json`, `sweights_assoc.parquet` |
| pT balance, W-pair check | `ttbar_balance.py`, `wpair_check.py` | `ttbar_balance.json`, `wpair_check.json` |
| private gen-level Pythia 8 + LETO (not a CMS sample) | `genstudy/gen_ttjpsi.py` (batch: `genstudy/run_ttjpsi.sh`), `gen_assoc.py`, `gen_leak_z.py`, `gen_cutscan.py`, `gen_compare.py` | `gen_*.json` |
| data / ttbar simulation, closure, backgrounds | `mc_compare.py` (`TTJ_BKG`), `mc_template_fit.py` | `mc_compare.json`, `mc_template.json` |
| generators and tunes | `gen_variants.py` | `gen_variants.json` |
| b-decay l_xy shape per category (MC truth), the fits with it (`TTJ_BTEMPLATE=btemplate.json`) | `btemplate_mc.py` | `btemplate.json`, `btmpl/` |
| closure of the prompt fits with toys (the data's candidate count per channel, category, sign and mass region drawn from the simulation, injected prompt J/psi, the whole chain per toy; hours, run by hand) | `toy_closure.py` (`--retry-failed`; with `TTJ_BTEMPLATE` for the per-category template), `toy_summary.py` | `toys.jsonl`, `toy_closure.json`, `toy_closure_btmpl.json` |
| figures | `make_figs_run3.py`, `make_figs_balance.py`, `make_figs_assoc.py`, `make_figs_gencmp.py`, `make_figs_mc.py`, `make_figs_an.py` (with `cmsdeck.py` on the cms-plotting skill's `cmsplot_plotly.py`) | `figs/*.json` |
| deck | `build_deck_assoc.py` (with `deck_lib.py`) | `slides.html`, `../index.html` |
| analysis-note figures | `stage_an.py` | `<an dir>/*.pdf`, `MANIFEST.txt`, `MC_SAMPLES.txt` |
| animation (the project's stand-alone video, not part of the slides) | `anim/ttjpsi.py` (`pixi run -e anim manim render ...`) | one video at a fixed web location, overwritten on every update |
| its narration: the text per segment (`narration()` in the scene, from the same numbers as the frames), speech, mixing, a transcription check | `ttjpsi.py --narration voice/narration.json`, `anim/voice.py` (Kokoro; the scene holds each segment until it is spoken), `anim/mix.py` (voice + subtitles on the joined video), `anim/check_voice.py` (faster-whisper) | `anim/voice/` (not in git), the narrated mp4 and its `.srt` |

Cross sections of the simulated samples: `xsec_map.json`. Plotting rules of the project: `.claude/cms-plotting/` at the
repository root, with the cms-plotting skill (github.com/DickyChant/cms-plotting).
