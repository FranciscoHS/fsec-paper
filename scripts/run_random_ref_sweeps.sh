#!/usr/bin/env bash
# RETIRED (per-pair threshold switch). No longer part of the pipeline: the
# threshold is now computed per-pair from each sweep's own grid, so the random-
# direction reference sweeps below are no longer needed. Kept for provenance.
#
# Generate the random-direction REFERENCE sweeps used to set the per-condition
# threshold T = 0.5 * median_i(max_alpha mean_anchors metric(alpha; random_i)).
#
# We only need each random direction's 1D plateau height, which lives on the
# EDGE of a random x random 2D sweep (g[:,0], g[0,:]). REF_PAIRS is 17 pairs
# chaining all 33 random directions, so each condition is ~17 sweeps, not the
# full 528. Filenames land exactly where recommend_fixed_threshold.py globs
# (see ref_random_suffix() in refit_thrfixed_all.py).
#
# Resumable: --skip_existing. Run from repo root on a GPU pod.
set -euo pipefail

REF_PAIRS=results/directions/ref_pairs_random.txt
COMMON="--pairs_file $REF_PAIRS --skip_existing"
run() { echo "### $*"; python scripts/sweep_2d.py $COMMON "$@"; }

GEMMA_RAND=results/directions/dirs_gemma_L2_random.pkl

# --- Model column: 5 other models at L2 (main condition) ---
for m in llama qwen mistral aya yi; do
  run --target "$m" --layer 2 \
      --directions_pkl "results/directions/dirs_${m}_L2_random.pkl"
done

# --- Gemma ablation columns (all reuse the L2 random-direction cache) ---
# Perturb-layer
for L in 5 10 20; do
  run --target gemma --layer "$L" --directions_pkl "$GEMMA_RAND"
done
# Measure-layer
for off in -7 -12; do
  run --target gemma --layer 2 --measure_offset "$off" --directions_pkl "$GEMMA_RAND"
done
# Method (additive)
run --target gemma --layer 2 --mode additive --directions_pkl "$GEMMA_RAND"
# Anchor source
for src in wiki_en wiki_zh code; do
  run --target gemma --layer 2 --anchor_source "$src" --directions_pkl "$GEMMA_RAND"
done
# Metric (one sweep records cos+kl+l2; serves both cos and kl threshold cells)
run --target gemma --layer 2 --metrics cos,kl,l2 --directions_pkl "$GEMMA_RAND"
# Token position
for p in -2 -3; do
  run --target gemma --layer 2 --perturb_pos "$p" --directions_pkl "$GEMMA_RAND"
done

echo "ALL RANDOM-REF SWEEPS DONE"
