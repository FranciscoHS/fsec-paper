# J-lens directions on Qwen3.6-27B

Does the superellipse exponent `p` exceed 2 for **J-lens** (Jacobian Lens)
token directions, as it does for contrastive feature directions?

Short answer: no. On Qwen3.6-27B, contrastive directions sit at `p ≈ 2.40`,
while J-lens directions sit at `p ≈ 1.91` (Chinese, n=100) and `p ≈ 2.15`
(English, n=10) — i.e. at the isotropic reference `p = 2`.

## Reproduce the figure (no GPU, ~5 s)

```bash
pip install numpy matplotlib
python jlens/plot_beeswarm.py
```

Writes `jlens/qwen36_p_beeswarm.{png,pdf}` from the cached fits in
`jlens/data/`. The script is self-contained (numpy + matplotlib only), so
this directory can be lifted out of the repo as a standalone artifact.

## Setup

- **Model** Qwen3.6-27B (64 layers). Perturb at **L36**, measure L2 response
  at L62 (penultimate). L36 sits inside the workspace band (~30–90% depth,
  per [Nanda's review](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper)),
  and the method works best on directions far from the output.
- **J-lens direction** for token `t` at layer `l` is `J_l^T W_U[t]`, using the
  pre-fitted lens from HF `neuronpedia/jacobian-lens` (`qwen3.6-27b`, n=1000).
- **Pairs** Analysis runs on *pairs* of directions, orthogonalised before the
  2D sweep — so pairs need low `|cos|`, else orthogonalisation distorts them.
  Threshold `|cos| < 0.1`. Chinese: 100 pairs (200 distinct tokens) from a pool
  of 1200 single-token Chinese words; all pairs come out at `|cos| ≤ 0.0005`.
  English: it was hard to find low-`|cos|` J-lens directions, so the threshold
  was relaxed to **0.25** to reach n=10 (at 0.1 only 1 pair survives). We do
  not know why this is hard for English but not Chinese.
- **Sweeps** 30 FineWeb 5-token anchors, 41×41 angle grid to 20°, per-pair
  checkpointing.

## Fitting protocol

Both families go through the paper's protocol, via `jlens/refit_protocol.py`
(which calls `scripts/lib/superellipse.py`, i.e. the same code as
`fit_pairs.py --per_pair_threshold --exact`):

    T_pair = 0.50 * min(axis-1 max, axis-2 max) on the pair's own median grid
    factors (0.5, 1.0, 2.0) x T_pair, exact-geodesic normalization, 200 bootstrap
    reported cell = ("1.0xT", window)

**One residual difference:** the angle window. Contrastive sweeps run to 60°,
the J-lens sweeps only to 20° (finer grid over a narrower cone, inherited from
the qwen3-8b J-lens config), so J-lens is fit at its own 20° limit. Closing
this would need re-running the sweeps to 60° on a GPU, not a re-fit. Note the
narrower cone is not obviously worse here — J-lens response rises steeply
(median L2 ≈ 268 by 60°), so 20° may sit better inside the plateau-breaking
regime.

To regenerate the fits from the bundled sweep grids:

```bash
python jlens/refit_protocol.py    # ~20 min, CPU only
```

## Regenerating the caches (GPU)

`upstream/` holds the scripts that produced `data/`, for auditability. They
are **not turnkey** — they were written against the research repo
(`steering-plateaus`) and depend on its `src/model.py` (Qwen3.5/3.6 hybrid
support: `model.model.language_model.layers`, mrope, `transformers>=5.0`) and
`scripts/exp_map/lib/` sweep primitives. Order: `download_inspect_lens` →
`build_jlens_zh_pairs` / `concrete_en_pairs` → `sweep_isoplateau_jlens`;
`select_qwen36_pairs` for the contrastive leg. Needs an 80 GB card; the
100-pair Chinese sweep took ~2.8 h on a B200.
