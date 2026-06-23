# Feature-specific error correction (FSEC): code & data

Companion repository for our paper on feature-specific error
correction in LLMs. This repo contains the code and cached intermediate
data needed to reproduce every figure in the paper, plus the upstream
scripts that generate those caches from open-weight LLMs.

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

`render_figures.sh` runs every plotting script in turn. LLM figure
outputs land in `results/figures/`; toy-model figure outputs in
`toy_model/runs/misalignment_sweep_H1024/`.

LLM figures (one per `\includegraphics` in `paper.tex`):

| Figure | Output |
| ------ | ------ |
| Fig 1 (combo sweep)               | `combo_sweep_gemma_L2_Wealth_Gender_to50deg_thrpair.pdf` |
| Fig 2 (fitted boundary)           | `fig3_boundary_Wealth_Gender_gemma_L2_exact.pdf` |
| Fig 3 (composition table)         | `composition_table_wealth_gender.pdf` |
| Fig 4 (direction-family beeswarm) | `robustness_beeswarm_gemma_L2_..._dirfamilies_morebaselines.pdf` |
| Fig 5 (misalignment sweep)        | `misalignment_llm_gemma_L2_thrpair_exact.pdf` |
| Fig 6 (robustness beeswarm)       | `robustness_beeswarm_gemma_L2_..._thrpair_exact.pdf` |
| Appx  (direction overlap heatmap) | `appendix_directions_overlap.pdf` |
| Appx  (fit residuals)             | `residuals_beeswarm_gemma_L2_ov0p1_thrpair_exact.pdf` |

Toy-model figures:

| Figure | Output |
| ------ | ------ |
| Toy boundary (LLM Fig 1b analog) | `toy_boundary_d8192_H1024_pair2-3_taufrac0.5.pdf` |
| Toy misalignment sweep           | `lp_vs_angle_no_feature_orth_clean.png` |

End-to-end runtime on a laptop: about a minute for the LLM figures
(residuals beeswarm is the slowest single step at ~30s). The toy
boundary plot needs ~30 s on CPU (or seconds on a GPU); the
misalignment sweep plot is sub-second since it only renders a cached
NPZ. The threshold and metric ablation columns are recomputed from
the bundled fits on every run; no hidden caches.

## Repo layout

```
src/                 Model loaders + data utilities (load_model,
                     forward_from_layer*, FineWeb-anchor helper).
data/                Contrastive prompt sets used to build the 33
                     direction-of-mean (DoM) directions: 14 binary
                     semantic concepts, 10 natural languages, 9
                     programming languages.
scripts/
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
                     results/figures/.
results/
  directions/        DoM and baseline-family direction caches
                     (`dirs_<target>_L<layer>[_variant].pkl`).
  fits/              Per-condition fitted exponents
                     (`fits_<target>_L<layer>[_variant].pkl`).
  sweeps_2d/         Per-pair iso-response surfaces; inputs to
                     fit_pairs.py. Filename pattern:
                     `sweep2d_<target>_L<layer>_<a>__<b>_fineweb_60deg[_variant].pkl`.
                     ~13k files, ~1.6 GB — required for the slow path
                     below.
toy_model/           Random-sparse autoencoder toy model + cached runs
                     for the toy-model figures. See `toy_model/scripts/`
                     and `toy_model/runs/` below.
  scripts/           Toy-model code (BaselineConfig, build_baked_model,
                     boundary_radii, the misalignment sweep, plotting).
  mft_denoising/     Helper package: dataset stream + TwoLayerNet.
  runs/misalignment_sweep_H1024/
                     Phase 2 baked-model NPZs for d in {4096, 8192,
                     16384}, plus the cached misalignment-sweep result
                     used by the toy figures.
render_figures.sh    Top-level "rebuild every figure" entry point
                     (LLM + toy).
LICENSE              MIT.
requirements.txt     Python deps (numpy, scipy, matplotlib,
                     transformers, torch, datasets).
```

## Re-running upstream stages

Two layers above the figure-level reproduction, the repo also lets
you regenerate the cached PKLs:

### Refit (no GPU)

Regenerate `results/fits/*` from the bundled
`sweeps_2d/*` cache. Useful for double-checking the fitting protocol.

```bash
python scripts/refit_thrfixed_all.py --exact
```

Runtime: a few minutes for all conditions.

### Re-sweep from model weights (GPU)

The slowest stage. Loads each LLM and runs the 2D iso-response sweep
end-to-end. Models are fetched from HuggingFace; gated repos
(Gemma, Llama) require you to set `HUGGINGFACE_HUB_TOKEN` from your
own account before running.

```bash
# canonical Gemma run
python scripts/sweep_2d.py --target gemma --layer 2

# repeat for each cross-model column
for tgt in qwen llama mistral aya yi; do
    python scripts/sweep_2d.py --target $tgt --layer 2
done
```

Wallclock guidance (single 80 GB H100, ~40 s/pair):
- Single 528-pair canonical run (one model, one layer): ~6 h.
- All 6 cross-model + 3 cross-layer Gemma + 7 ablation variants: ~36 h
  total.

After re-sweeping, regenerate the fits and figures with
`refit_thrfixed_all.py --exact` followed by `render_figures.sh`.

### Toy-model slow path (GPU helpful)

The two toy-model figures read from cached NPZs under
`toy_model/runs/misalignment_sweep_H1024/`. To regenerate those caches
from scratch:

```bash
cd toy_model
# Phase 2: pick (p*, c_in*, c_out*, lambda_on) per d (one Phase 2 NPZ
# per d). ~minutes per d on a single RTX 4090.
python -m scripts.random_sparse_phase2_pselect --d 4096  --H 1024 --lambda_on 500  --out_dir runs/misalignment_sweep_H1024
python -m scripts.random_sparse_phase2_pselect --d 8192  --H 1024 --lambda_on 1000 --out_dir runs/misalignment_sweep_H1024
python -m scripts.random_sparse_phase2_pselect --d 16384 --H 1024 --lambda_on 2000 --out_dir runs/misalignment_sweep_H1024

# Misalignment sweep (the figure-feeding step). Default uses a feature-
# axis-orthogonal random direction; the no_feature_orth flag samples an
# isotropic random unit vector instead (this is the variant plotted in
# the paper). ~4 min on RTX 4090 for the full d-sweep.
python -m scripts.random_sparse_misalignment_sweep \
    --ds 4096 8192 16384 --H 1024 \
    --phase2_dir runs/misalignment_sweep_H1024 \
    --out_dir   runs/misalignment_sweep_H1024 \
    --no_feature_orth
```

After re-sweeping, run `bash render_figures.sh` from the repo root to
regenerate every paper figure (LLM + toy).

### Models used

| Target  | Hugging Face repo            |
| ------- | ---------------------------- |
| gemma   | `google/gemma-2-9b`          |
| qwen    | `Qwen/Qwen3-1.7B`            |
| llama   | `meta-llama/Llama-3.1-8B`    |
| mistral | `mistralai/Mistral-7B-v0.3`  |
| aya     | `CohereForAI/aya-expanse-8b` |
| yi      | `01-ai/Yi-1.5-9B`            |

## License

MIT (see `LICENSE`).
