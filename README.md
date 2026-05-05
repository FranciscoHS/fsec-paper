# Feature-Selectivity Evidence: code & data

Anonymised companion repository for our ICML 2026 submission. This repo
contains the code and cached intermediate data needed to reproduce
every figure in the paper, plus the upstream scripts that generate
those caches from open-weight LLMs.

The headline analysis fits a superellipse exponent `p` to the 2D
iso-response contour of two contrastive feature directions, and finds
that contrastive (and other candidate-feature) directions sit
robustly above the isotropic reference `p = 2`, while non-feature
baselines (PCA, random) cluster at `p ≈ 2`. See the paper for the full
discussion.

## Quickstart — reproduce every paper figure (no GPU)

```bash
pip install -r requirements.txt
bash render_figures.sh
```

`render_figures.sh` runs each plotting script in turn and copies the
output to a paper-side filename under
`results/exp_map/figures/paper/`. The seven files it produces are
exactly the ones included from `paper.tex`:

| Figure | Output |
| ------ | ------ |
| Fig 1a (combo sweep)              | `combo_sweep_gemma_L2_Gender_Refusal_to25deg.pdf` |
| Fig 1b (fitted boundary)          | `fig3_boundary_Gender_Refusal_gemma_L2.pdf` |
| Fig 2  (composition table)        | `composition_table_gender_tense.pdf` |
| Fig 3  (direction-family beeswarm)| `fig_beeswarm_directions.pdf` |
| Fig 4  (robustness beeswarm)      | `fig4_n_robust_beeswarm.pdf` |
| Appx   (direction overlap heatmap)| `appendix_directions_overlap.pdf` |
| Appx   (fit residuals)            | `appendix_residuals_beeswarm.pdf` |

End-to-end runtime on a laptop: about a minute (the residuals beeswarm
is the slowest single step at ~30s; everything else is sub-second).
The threshold and metric ablation columns are recomputed from the
bundled fits on every run; no hidden caches.

## Repo layout

```
src/                 Model loaders + data utilities (load_model,
                     forward_from_layer*, FineWeb-anchor helper).
data/                Contrastive prompt sets used to build the 33
                     direction-of-mean (DoM) directions: 14 binary
                     semantic concepts, 10 natural languages, 9
                     programming languages.
scripts/exp_map/
  lib/               Shared library: direction registry, parametric
                     2-direction grid, superellipse fitter, anchor
                     activation pipeline.
  sweep_2d.py        Per-pair 2D iso-response sweep (the slow step:
                     ~40 s/pair on H100).
  fit_pairs.py       Fit a superellipse to every sweep_2d output.
  refit_thrfixed_all.py
                     Drives fit_pairs.py over every ablation
                     condition with a per-condition fixed-L^2
                     threshold (the canonical paper protocol).
  recommend_fixed_threshold.py
                     Helper used by refit_thrfixed_all.py to pick the
                     per-condition threshold from the axis-max
                     distribution.
  {melbo,pca,sae_directions_top,random}_directions.py
                     Build the alternative direction families used in
                     the direction-type beeswarm.
  plotting/          One file per paper figure. Read PKLs from
                     results/, write PDF + PNG to
                     results/exp_map/figures/.
results/exp_map/data/
  directions/        DoM and baseline-family direction caches
                     (`dirs_<target>_L<layer>[_variant].pkl`).
  fits/              Per-condition fitted exponents
                     (`fits_<target>_L<layer>[_variant].pkl`).
  sweeps_2d/         Per-pair iso-response surfaces; inputs to
                     fit_pairs.py. Filename pattern:
                     `sweep2d_<target>_L<layer>_<a>__<b>_fineweb_60deg[_variant].pkl`.
                     ~13k files, ~1.6 GB — required for the slow path
                     below.
render_figures.sh    Top-level "rebuild every figure" entry point.
LICENSE              MIT.
requirements.txt     Python deps (numpy, scipy, matplotlib,
                     transformers, torch, datasets).
```

## Re-running upstream stages

Two layers above the figure-level reproduction, the repo also lets
you regenerate the cached PKLs:

### Refit (no GPU)

Regenerate `results/exp_map/data/fits/*` from the bundled
`sweeps_2d/*` cache. Useful for double-checking the fitting protocol.

```bash
python scripts/exp_map/refit_thrfixed_all.py --exact
```

Runtime: a few minutes for all conditions.

### Re-sweep from model weights (GPU)

The slowest stage. Loads each LLM and runs the 2D iso-response sweep
end-to-end. Models are fetched from HuggingFace; gated repos
(Gemma, Llama) require you to set `HUGGINGFACE_HUB_TOKEN` from your
own account before running.

```bash
# canonical Gemma run
python scripts/exp_map/sweep_2d.py --target gemma --layer 2

# repeat for each cross-model column
for tgt in qwen llama mistral aya yi; do
    python scripts/exp_map/sweep_2d.py --target $tgt --layer 2
done
```

Wallclock guidance (single 80 GB H100, ~40 s/pair):
- Single 528-pair canonical run (one model, one layer): ~6 h.
- All 6 cross-model + 3 cross-layer Gemma + 7 ablation variants: ~36 h
  total.

After re-sweeping, regenerate the fits and figures with
`refit_thrfixed_all.py --exact` followed by `render_figures.sh`.

### Models used

| Target  | Hugging Face repo            |
| ------- | ---------------------------- |
| gemma   | `google/gemma-2-9b`          |
| qwen    | `Qwen/Qwen3-1.7B`            |
| llama   | `meta-llama/Llama-3.1-8B`    |
| mistral | `mistralai/Mistral-7B-v0.3`  |
| aya     | `CohereForAI/aya-expanse-8b` |
| yi      | `01-ai/Yi-1.5-9B`            |

## Citation

(Anonymous submission; citation will be filled in upon acceptance.)

## License

MIT (see `LICENSE`).
