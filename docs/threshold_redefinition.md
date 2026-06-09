# Threshold redefinition: random-direction reference scale

## Motivation

Reviewer feedback: the current threshold `T` is defined per feature-pair as
`min(max(L2 along d1), max(L2 along d2))` — "the smaller of the two single-axis
maxima" — then `T = f * median` over pairs. This is methodologically convoluted:
it involves feature *pairs* unnecessarily (in the feature-feature case the `min()`
collapses to a single direction), and the "population" / "smaller of two maxima"
language is unclear in the paper.

## New definition

A single scalar per model, anchored on **random directions** (no feature pairs):

```
T = 0.5 * median_i [ max_alpha  mean_anchors L2(alpha; random_i) ]
```

- Sample ~33 isotropic random unit directions at the model's residual width.
- Sweep activations 1D along each, alpha in [0, 60] deg (matches existing sweeps).
- Per direction: mean L2 over anchors, then max over angle = that direction's
  plateau height.
- Median over directions, halve it. The 0.5 fraction is ablated downstream, so
  it is only a starting point.

Decisions (confirmed with user):
- **Angle range: 0-60 deg** — matches all existing feature sweeps and the
  contour-fit ranges; responses plateau before 60 deg.
- **Aggregation: mean-over-anchors then median-over-directions** — mirrors the
  current code's mean-then-reduce structure.

## Why this is a small change

The expensive 2D feature-pair response surfaces are already collected and do NOT
change. The threshold is only the *level* at which we slice those surfaces to
extract the iso-response contour. We recompute one scalar `T` per model and
re-slice; the 2D grids -> contour fit -> figures pipeline is otherwise untouched.

The threshold was always a 1D quantity: even the old code only read the grid
*edges* (`g[:,0]`, `g[0,:]`), each a single-direction sweep. So the threshold
never needed the 2D interior; the new method just makes the anchor set explicit
(random directions) and 1D.

## Work plan

1. **[done] Random directions for all models.** `scripts/random_directions.py`
   generalized to per-target residual width (WIDTHS); generated
   `dirs_<model>_L2_random.pkl` for llama/qwen/mistral/aya/yi (no GPU).
2. **[done] Threshold rewrite.** `scripts/recommend_fixed_threshold.py` now
   computes `T = f * median(random plateau height)` from the EDGES of random x
   random sweeps. `scripts/refit_thrfixed_all.py` builds the matching random-ref
   suffix per condition (`ref_random_suffix`) and defaults `f = 0.5`.
   `scripts/sweep_2d.py` now honours `--pairs`/`--pairs_file` in the
   `--directions_pkl` branch so we sweep only a 17-pair reference set, not 528.
3. **[done, no GPU] Gemma cells from existing data.** The gemma main cell + all
   6 direction-family cells share the existing `_dirrandom` sweeps; regenerated
   their `_thrfixed_exact` fits with the new T (= 123, was ~150).
4. **[GPU] Random-reference sweeps for the remaining 17 conditions.** Driver:
   `scripts/run_random_ref_sweeps.sh` (17 pairs each, resumable). Conditions:
   5 other models at L2; gemma L5/L10/L20, M-7/M-12, additive, src
   wiki_en/wiki_zh/code, cos+kl+l2, pos-2/pos-3.
5. **[after GPU] Recompute + re-render.** `refit_thrfixed_all.py --exact` (the
   figures consume `_thrfixed_exact` fits, NOT plain `_thrfixed`), then
   `render_figures.sh`.
6. **Manuscript prose** (separate repo, `error-correction-paper`): rewrite the
   threshold paragraph; clarify or remove the "plateau" framing and the
   "population" / "smaller of two maxima" wording.

## Rollback

Checkpoint commit precedes any change. The 2D feature sweeps under
`results/sweeps_2d/` are inputs and are not modified.
