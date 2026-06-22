#!/usr/bin/env bash
# Reproduce every figure in the paper from the bundled caches.
set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------
# LLM figures (read PKLs from results/, write PDF + PNG to results/figures/).
# ----------------------------------------------------------------------

# Fig 1 — combo sweep (Wealth x Gender, 0..50 deg). Threshold is the
# per-pair scale T_pair = 0.5 * min(axis maxima) on this pair's own median
# grid (= 139 for Wealth x Gender at gemma L2). Omitting --threshold lets
# the script compute it, so the line is the same per-pair contour level the
# exponent is fit on.
python scripts/plotting/plot_combo_sweep.py \
    --target gemma --layer 2 \
    --d1 Wealth --d2 Gender \
    --max_angle 50 --out_tag thrpair

# Fig 2 — fitted boundary (same per-pair threshold T_pair=139, auto-computed)
python scripts/plotting/plot_fig3_boundary.py \
    --target gemma --layer 2 --d1 Wealth --d2 Gender --exact

# Fig 3 — composition table (static, no PKL inputs)
python scripts/plotting/plot_composition_table_wealth_gender.py

# Fig 4 — direction-family beeswarm (includes the random-difference baseline
# column; reads fits_gemma_L2_dirrandomdiffavg_fineweb_thrpair_exact.pkl)
python scripts/plotting/plot_beeswarm_direction_types.py \
    --target gemma --layer 2 --max_overlap 0.10 \
    --exclude_dirs HonestyShort,TensePresent,Formal \
    --use_thrpair --exact

# Fig 5 — LLM misalignment sweep: fitted p vs rotation off the contrastive
# directions toward random (reads fits_gemma_L2_dirrotcontrastive_*_thrpair_exact.pkl)
python -m scripts.plotting.plot_misalignment_llm \
    --target gemma --layer 2 --exact

# Fig 6 — robustness beeswarm across ablation axes
python scripts/plotting/plot_robustness_beeswarm.py \
    --target gemma --layer 2 --max_overlap 0.10 \
    --exclude_dirs Formal,HonestyShort,TensePresent \
    --use_thrpair --exact

# Appendix — direction-overlap heatmap + LaTeX longtable
python scripts/plotting/build_directions_appendix.py

# Appendix — fit residual beeswarm
python scripts/plotting/plot_residuals_beeswarm.py

# ----------------------------------------------------------------------
# Toy-model figures (run from toy_model/; outputs in
# toy_model/runs/misalignment_sweep_H1024/).
# ----------------------------------------------------------------------

(
  cd toy_model
  # Toy boundary plot — single feature pair (j=2, k=3) at perfect
  # alignment. Builds a TwoLayerNet from the bundled phase 2 NPZ and
  # forward-passes a phi grid; needs a CUDA GPU if available, falls back
  # to CPU otherwise.
  python -m scripts.plot_toy_boundary \
      --phase2_npz runs/misalignment_sweep_H1024/phase2_d8192_H1024.npz \
      --out_dir   runs/misalignment_sweep_H1024

  # Toy sweep plot (appendix) — Fig-1a-equivalent: L^2 response vs
  # perturbation magnitude along feature axes, their combination, and a
  # random-direction baseline, with the plateau-breaking threshold.
  python -m scripts.plot_toy_sweep \
      --phase2_npz runs/misalignment_sweep_H1024/phase2_d8192_H1024.npz \
      --out_dir   runs/misalignment_sweep_H1024

  # Toy misalignment sweep — fitted superellipse exponent vs cosine
  # similarity to the feature axes (median over 80 pairs, IQR shaded).
  # Reads the bundled lp_vs_angle_no_feature_orth.npz; pure CPU plot.
  python -m scripts.plot_misalignment_lp \
      --npz runs/misalignment_sweep_H1024/lp_vs_angle_no_feature_orth.npz \
      --out runs/misalignment_sweep_H1024/lp_vs_angle_no_feature_orth_clean.png
)

echo
echo "All figures regenerated."
echo "  LLM figures:        results/figures/"
echo "  Toy-model figures:  toy_model/runs/misalignment_sweep_H1024/"
