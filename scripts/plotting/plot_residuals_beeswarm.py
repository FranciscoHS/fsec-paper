"""Beeswarm of per-pair median fit residual across all ablation
conditions, mirroring plot_robustness_beeswarm.py.

Y-axis is the median ``mean_radial_frac`` across the per-pair
robustness cells (one number per pair). Columns: the seven ablations
shared with the robustness beeswarm (Model, Perturb layer, Measure
layer, Metric, Threshold, Method, Anchor source), then Token position
(pos -1 / -2 / -3), then Direction family (Contrastive / MELBO / SAE
/ PCA / Random) at the right edge.

We reuse the existing ``render`` and column loaders by monkey-patching
``plot_robustness_beeswarm.median_p_for_pair`` to return the median
residual instead of the median exponent. All filter / overlap /
exclusion logic and the per-sub-group bootstrap CI then run unchanged.

Usage:
  python scripts/plotting/plot_residuals_beeswarm.py
"""
from __future__ import annotations
import os, sys, pickle, argparse
sys.path.insert(0, ".")
import numpy as np

import scripts.plotting.plot_robustness_beeswarm as P
from scripts.plotting.plot_robustness_beeswarm import (
    col_model, col_perturb_layer, col_measure_layer, col_metric,
    col_method, col_anchor_source, col_token_position,
    _apply_filters, _filter_exclude,
    _filter_by_overlap, render, FITS_DIR, OUT_DIR, DATA_DIR,
)
from scripts.plotting.plot_beeswarm_direction_types import (
    _filter_family_overlap, FAMILIES,
)


def median_residual_for_pair(robust_fit_dict) -> float:
    """One number per pair: the residual of the single threshold fit
    (1.0xT, full 60-degree window), matching median_p_for_pair."""
    cell = robust_fit_dict.get(("1.0xT", 60.0))
    if cell is None:
        return float("nan")
    r = cell.get("mean_radial_frac", np.nan)
    return float(r) if isinstance(r, float) and np.isfinite(r) else float("nan")


def col_threshold_residual(target: str, layer: int,
                            max_overlap: float | None = None
                            ) -> dict:
    """Residual analogue of col_threshold's thrfixed branch: one
    sub-group per threshold factor (0.5xT, 1.0xT, 2.0xT), per-pair value
    is the median residual across max_alpha cells at that factor.
    Reads ``mean_radial_frac`` directly from the cells, bypassing the
    p_median field that the main col_threshold expects.
    """
    fp = os.path.join(FITS_DIR,
        f"fits_{target}_L{layer}_thrpair_exact.pkl")
    if not os.path.exists(fp): return {}
    with open(fp, "rb") as f: full = pickle.load(f)
    out: dict[str, dict] = {}
    for pair_key, fit in full.items():
        for cell_key, cell in fit.items():
            if cell_key == "__levels__": continue
            tl, _ma = cell_key
            r = cell.get("mean_radial_frac", np.nan)
            if not (isinstance(r, float) and np.isfinite(r)): continue
            out.setdefault(tl, {}).setdefault(
                frozenset(pair_key), []).append(r)
    out = {tl: {pair: float(np.median(rs))
                 for pair, rs in d.items()}
           for tl, d in out.items()}
    out = {k: _filter_by_overlap(v, target, layer, max_overlap)
           for k, v in out.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--max_overlap", type=float, default=0.10)
    ap.add_argument("--exclude_dirs", default="Formal,HonestyShort,TensePresent")
    ap.add_argument("--ref_pct", type=float, default=None,
                    help="horizontal reference line, in percent. None "
                         "(default) draws no reference line.")
    ap.add_argument("--y_max", type=float, default=0.125,
                    help="upper y-limit for the residual axis. Default "
                         "0.125 (= 12.5%%) clips ~0.02%% of dots and "
                         "keeps the bulk of the distribution legible.")
    args = ap.parse_args()
    excl = {n.strip() for n in args.exclude_dirs.split(",") if n.strip()}

    # All fits we read are the exact-geodesic, per-pair-threshold variants —
    # same as the canonical robustness beeswarm. The col_* loaders pick up
    # USE_THRPAIR / USE_EXACT from the module.
    P.USE_THRPAIR = True
    P.USE_EXACT = True
    # Swap the per-pair aggregator: residual instead of p_median.
    P.median_p_for_pair = median_residual_for_pair

    target, layer, mo = args.target, args.layer, args.max_overlap
    cols = [
        ("Model",               col_model(layer, max_overlap=mo)),
        ("Perturbation layer",  col_perturb_layer(target, max_overlap=mo)),
        ("Measurement layer",   col_measure_layer(target, layer, max_overlap=mo)),
        ("Fit metric",          col_metric(target, layer, max_overlap=mo)),
        ("Response threshold",  col_threshold_residual(target, layer, max_overlap=mo)),
        ("Perturbation method", col_method(target, layer, max_overlap=mo)),
        ("Anchor source",       col_anchor_source(target, layer, max_overlap=mo)),
        ("Token position",      col_token_position(target, layer, max_overlap=mo)),
    ]
    cols = [(label, _apply_filters(d, target, layer, mo, excl))
            for label, d in cols]

    # Direction family column: same logic as plot_beeswarm_direction_types
    # but rolled into one column with multiple sub-groups (one per family),
    # so it slots into the same render() call as the other ablations.
    fam_subs = {}
    for label, fits_suffix, dir_suffix, do_filter in FAMILIES:
        fp = os.path.join(FITS_DIR,
            f"fits_{target}_L{layer}{fits_suffix}_thrpair_exact.pkl")
        d = P._load_fit_pair_dict(fp)
        if not d:
            print(f"  Direction family / {label}: no fits at {fp}")
            continue
        if do_filter:
            d = _filter_family_overlap(d, target, layer, dir_suffix, mo)
            if dir_suffix is None:        # Contrastive: also drop excl
                d = _filter_exclude(d, excl)
        fam_subs[label] = d
    cols.append(("Direction family", fam_subs))

    # Counts.
    print("Columns with data:")
    for label, subs in cols:
        for sub, ps in subs.items():
            print(f"  {label:18s} {sub:>18s}  n={len(ps)}")

    out_png = os.path.join(
        OUT_DIR,
        f"residuals_beeswarm_{target}_L{layer}_ov0p1_thrpair_exact.png")
    out_pdf = out_png.replace(".png", ".pdf")
    ref_y = (args.ref_pct / 100.0) if args.ref_pct is not None else None
    ref_label = (rf"${args.ref_pct:g}\%$ reference"
                 if args.ref_pct is not None else "")
    render(cols, out_png, out_pdf,
           show_sub_legends=True, show_stats_text=False,
           ylabel="median superellipse fit residual",
           ref_y=ref_y, ref_label=ref_label,
           ylim=(0.0, args.y_max))


if __name__ == "__main__":
    main()
