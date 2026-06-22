"""Phase-3 numbers audit for the per-pair threshold switch.

ONE master table: every column x condition (sub-group), with the per-pair
  median p, pooled mean p, direction-bootstrap 95% CI on the mean,
  median fit residual, and n
all read from the ``_thrpair_exact`` fits via the SAME column loaders and
filters that the figures use (plot_robustness_beeswarm + the direction-family
and misalignment loaders), so this is the literal source of truth for every
number quoted in the paper.

Filters match render_figures.sh: max_overlap=0.10, exclude
HonestyShort/TensePresent/Formal on the contrastive/DoM groups.

Usage:
  python scripts/audit_thrpair_numbers.py
  python scripts/audit_thrpair_numbers.py --md results/thrpair_audit.md
"""
from __future__ import annotations
import os, sys, glob, pickle, re, argparse
sys.path.insert(0, ".")
import numpy as np

import scripts.plotting.plot_robustness_beeswarm as P
from scripts.plotting.plot_robustness_beeswarm import (
    col_model, col_perturb_layer, col_measure_layer, col_metric,
    col_threshold, col_method, col_anchor_source, col_token_position,
    _apply_filters, _filter_exclude, _filter_by_overlap,
    _bootstrap_pooled_mean_ci, FITS_DIR,
)
from scripts.plotting.plot_beeswarm_direction_types import (
    _filter_family_overlap, FAMILIES,
)

MAX_OVERLAP = 0.10
EXCLUDE = {"HonestyShort", "TensePresent", "Formal"}


# ---- per-pair residual loader (mirror of median_p_for_pair) ----

def _residual_for_pair(fit) -> float:
    cell = fit.get(("1.0xT", 60.0))
    if cell is None:
        return np.nan
    r = cell.get("mean_radial_frac", np.nan)
    return float(r) if isinstance(r, float) and np.isfinite(r) else np.nan


def _residuals_for_fits_file(fp: str) -> dict[frozenset, float]:
    if not os.path.exists(fp):
        return {}
    with open(fp, "rb") as f:
        full = pickle.load(f)
    out = {}
    for pair_key, fit in full.items():
        r = _residual_for_pair(fit)
        if np.isfinite(r):
            out[frozenset(pair_key)] = r
    return out


def _row(label, sub, pairs_p, resid_lookup):
    """One audit row from a {pair: p} sub-group dict."""
    ps = np.array([v for v in pairs_p.values() if np.isfinite(v)])
    if ps.size == 0:
        return None
    mean, lo, hi = _bootstrap_pooled_mean_ci({sub: pairs_p})
    # median residual over the same pairs (where residual is available)
    rs = np.array([resid_lookup[k] for k in pairs_p
                   if k in resid_lookup and np.isfinite(resid_lookup[k])])
    med_r = float(np.median(rs)) if rs.size else np.nan
    return {
        "column": label, "sub": sub, "n": int(ps.size),
        "median": float(np.median(ps)), "mean": mean, "lo": lo, "hi": hi,
        "median_resid": med_r,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--md", default="results/thrpair_audit.md",
                    help="write a Markdown table here too")
    args = ap.parse_args()
    target, layer, mo = args.target, args.layer, MAX_OVERLAP

    P.USE_THRPAIR = True
    P.USE_EXACT = True

    rows = []

    # --- Fig-4 robustness columns (p-valued; same loaders as the beeswarm) ---
    p_cols = [
        ("Model",               col_model(layer, max_overlap=mo)),
        ("Perturbation layer",  col_perturb_layer(target, max_overlap=mo)),
        ("Measurement layer",   col_measure_layer(target, layer, max_overlap=mo)),
        ("Fit metric",          col_metric(target, layer, max_overlap=mo)),
        ("Response threshold",  col_threshold(target, layer, max_overlap=mo)),
        ("Perturbation method", col_method(target, layer, max_overlap=mo)),
        ("Anchor source",       col_anchor_source(target, layer, max_overlap=mo)),
        ("Token position",      col_token_position(target, layer, max_overlap=mo)),
    ]
    p_cols = [(label, _apply_filters(d, target, layer, mo, EXCLUDE))
              for label, d in p_cols]

    # residual lookups, keyed the same way the loaders key fits files
    def resid(fname):
        return _residuals_for_fits_file(os.path.join(FITS_DIR, fname))

    sfx = "_thrpair_exact"
    # Build a residual lookup by reading every fits file referenced; for the
    # columns we read residuals from the matching fits file per sub-group.
    # Map each (column, sub) to its source fits file where determinable;
    # otherwise fall back to the canonical contrastive file.
    canon_resid = resid(f"fits_{target}_L{layer}{sfx}.pkl")

    SUB_RESID_FILE = {
        # metric
        "KL": f"fits_{target}_L{layer}_cos+kl+l2_metric_kl{sfx}.pkl",
        r"$1-\cos$": f"fits_{target}_L{layer}_cos+kl+l2_metric_cos{sfx}.pkl",
        # measure
        "penult-5": f"fits_{target}_L{layer}_M-7{sfx}.pkl",
        "penult-10": f"fits_{target}_L{layer}_M-12{sfx}.pkl",
        # method
        "additive": f"fits_{target}_L{layer}_additive{sfx}.pkl",
        # anchor source
        "Wikipedia (en)": f"fits_{target}_L{layer}_srcwiki_en{sfx}.pkl",
        "Wikipedia (zh)": f"fits_{target}_L{layer}_srcwiki_zh{sfx}.pkl",
        "Code": f"fits_{target}_L{layer}_srccode{sfx}.pkl",
        # token position
        r"pos $-2$": f"fits_{target}_L{layer}_pos-2{sfx}.pkl",
        r"pos $-3$": f"fits_{target}_L{layer}_pos-3{sfx}.pkl",
    }
    MODEL_RESID = {
        "gemma-2-9b": canon_resid,
    }
    for short, full in {"qwen":"qwen3-1.7b","llama":"llama-3.1-8b",
                        "mistral":"mistral-7b-v0.3","aya":"aya-expanse-8b",
                        "yi":"yi-1.5-9b"}.items():
        MODEL_RESID[full] = resid(f"fits_{short}_L{layer}{sfx}.pkl")
    PERTURB_RESID = {f"L={L}": resid(f"fits_{target}_L{L}{sfx}.pkl")
                     for L in (5,10,20)}

    for label, subs in p_cols:
        for sub, pairs_p in subs.items():
            if sub in MODEL_RESID:
                rl = MODEL_RESID[sub]
            elif sub in PERTURB_RESID:
                rl = PERTURB_RESID[sub]
            elif sub in SUB_RESID_FILE:
                rl = resid(SUB_RESID_FILE[sub])
            else:
                rl = canon_resid
            r = _row(label, sub, pairs_p, rl)
            if r:
                rows.append(r)

    # --- Fig-3 direction-family column ---
    for fam_label, fits_suffix, dir_suffix, do_filter in FAMILIES:
        fp = os.path.join(FITS_DIR,
            f"fits_{target}_L{layer}{fits_suffix}{sfx}.pkl")
        d = P._load_fit_pair_dict(fp)
        if not d:
            print(f"  [skip] Direction family / {fam_label}: no fits at {fp}")
            continue
        if do_filter:
            d = _filter_family_overlap(d, target, layer, dir_suffix, mo)
            if dir_suffix is None:
                d = _filter_exclude(d, EXCLUDE)
        rl = _residuals_for_fits_file(fp)
        if do_filter and dir_suffix is None:
            rl = _filter_exclude(_filter_by_overlap(rl, target, layer, mo), EXCLUDE)
        r = _row("Direction family", fam_label, d, rl)
        if r:
            rows.append(r)

    # --- Misalignment theta sweep (rotated contrastive) ---
    rx = re.compile(rf"_dirrotcontrastive_th(\d+)_s(\d+){re.escape(sfx)}\.pkl$")
    by_theta: dict[int, dict] = {}
    by_theta_r: dict[int, list] = {}
    for fp in sorted(glob.glob(os.path.join(FITS_DIR,
            f"fits_{target}_L{layer}_dirrotcontrastive_th*_s*{sfx}.pkl"))):
        m = rx.search(os.path.basename(fp))
        if not m: continue
        theta = int(m.group(1))
        with open(fp, "rb") as f: full = pickle.load(f)
        for pk, fit in full.items():
            p = P.median_p_for_pair(fit)
            if np.isfinite(p):
                # key by (theta, seed, pair) to keep them distinct
                by_theta.setdefault(theta, {})[(m.group(2),) + tuple(pk)] = p
            rr = _residual_for_pair(fit)
            if np.isfinite(rr):
                by_theta_r.setdefault(theta, []).append(rr)
    for theta in sorted(by_theta):
        pairs_p = by_theta[theta]
        ps = np.array(list(pairs_p.values()))
        rs = np.array(by_theta_r.get(theta, []))
        rows.append({
            "column": "Misalignment", "sub": f"theta={theta}",
            "n": int(ps.size), "median": float(np.median(ps)),
            "mean": float(ps.mean()), "lo": np.nan, "hi": np.nan,
            "median_resid": float(np.median(rs)) if rs.size else np.nan,
        })

    # --- print + write ---
    hdr = f"{'column':<20s} {'sub-group':<18s} {'n':>4s} {'median':>7s} {'mean':>7s} {'95% CI':>16s} {'med_resid':>9s}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        ci = (f"[{r['lo']:.3f},{r['hi']:.3f}]"
              if np.isfinite(r["lo"]) else "       --       ")
        lines.append(
            f"{r['column']:<20s} {r['sub']:<18s} {r['n']:>4d} "
            f"{r['median']:>7.3f} {r['mean']:>7.3f} {ci:>16s} "
            f"{r['median_resid']*100:>8.2f}%")
    table = "\n".join(lines)
    print(table)

    if args.md:
        md = ["# Per-pair threshold audit (source of truth)\n",
              f"target={target} L={layer}, max_overlap={MAX_OVERLAP}, "
              f"exclude={sorted(EXCLUDE)}\n",
              "| column | sub-group | n | median p | mean p | 95% CI | median residual |",
              "|---|---|---:|---:|---:|---|---:|"]
        for r in rows:
            ci = (f"[{r['lo']:.3f}, {r['hi']:.3f}]"
                  if np.isfinite(r["lo"]) else "--")
            md.append(f"| {r['column']} | {r['sub']} | {r['n']} | "
                      f"{r['median']:.3f} | {r['mean']:.3f} | {ci} | "
                      f"{r['median_resid']*100:.2f}% |")
        os.makedirs(os.path.dirname(args.md), exist_ok=True)
        with open(args.md, "w") as f:
            f.write("\n".join(md) + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
