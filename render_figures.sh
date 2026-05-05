#!/usr/bin/env bash
# Reproduce every figure included in the paper from the bundled PKLs.
# Output canonical names (= the names referenced by paper.tex) under
# results/figures/paper/.
set -euo pipefail
cd "$(dirname "$0")"

OUT=results/figures
PAPER_OUT=$OUT/paper
mkdir -p "$PAPER_OUT"

# Fig 1a — combo sweep (Gender x Refusal, 0..50 deg, threshold L^2=150)
python scripts/plotting/plot_combo_sweep.py \
    --target gemma --layer 2 \
    --d1 Gender --d2 Refusal \
    --max_angle 50 --threshold 150 --out_tag _to50deg_thr150
cp "$OUT/combo_sweep_gemma_L2_Gender_Refusal_to50deg__to50deg_thr150.pdf" \
   "$PAPER_OUT/combo_sweep_gemma_L2_Gender_Refusal_to25deg.pdf"

# Fig 1b — fitted boundary
python scripts/plotting/plot_fig3_boundary.py \
    --target gemma --layer 2 --d1 Gender --d2 Refusal
cp "$OUT/fig3_boundary_Gender_Refusal_gemma_L2.pdf" \
   "$PAPER_OUT/fig3_boundary_Gender_Refusal_gemma_L2.pdf"

# Fig 2 — composition table (static, no PKL inputs)
python scripts/plotting/plot_composition_table.py
cp "$OUT/composition_table_gender_tense.pdf" \
   "$PAPER_OUT/composition_table_gender_tense.pdf"

# Fig 3 — direction-family beeswarm
python scripts/plotting/plot_beeswarm_direction_types.py \
    --target gemma --layer 2 --max_overlap 0.10 \
    --exclude_dirs HonestyShort,TensePresent,Formal \
    --use_thrfixed --exact
cp "$OUT/robustness_beeswarm_gemma_L2_ov0p1_excl-Formal-HonestyShort-TensePresent_thrfixed_exact_dirfamilies_morebaselines.pdf" \
   "$PAPER_OUT/fig_beeswarm_directions.pdf"

# Fig 4 — robustness beeswarm across ablation axes
python scripts/plotting/plot_robustness_beeswarm.py \
    --target gemma --layer 2 --max_overlap 0.10 \
    --exclude_dirs Formal,HonestyShort,TensePresent \
    --use_thrfixed --exact
cp "$OUT/robustness_beeswarm_gemma_L2_ov0p1_excl-Formal-HonestyShort-TensePresent_thrfixed_exact.pdf" \
   "$PAPER_OUT/fig4_n_robust_beeswarm.pdf"

# Appendix — direction-overlap heatmap + LaTeX longtable
python scripts/plotting/build_directions_appendix.py
cp "$OUT/appendix_directions_overlap.pdf" \
   "$PAPER_OUT/appendix_directions_overlap.pdf"

# Appendix — fit residual beeswarm
python scripts/plotting/plot_residuals_beeswarm.py
cp "$OUT/residuals_beeswarm_gemma_L2_ov0p1_thrfixed_exact.pdf" \
   "$PAPER_OUT/appendix_residuals_beeswarm.pdf"

echo
echo "All figures regenerated. Paper-named copies are in $PAPER_OUT/"
ls -la "$PAPER_OUT"
