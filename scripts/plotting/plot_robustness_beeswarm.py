"""Robustness beeswarm of fitted superellipse exponent p across many
axes of variation in the exp-map workspace.

Each dot = one fit on one pair. Default = gemma, L=2 perturb,
measure = penult, L^2 metric, geodesic perturbation, threshold = 50%.
Columns vary one axis at a time:

  1. Model            — gemma, qwen, llama, mistral, aya, yi
  2. Perturb layer    — gemma at L in {2, 5, 10, 20}
  3. Measure layer    — gemma at L=2, measure in {penult, -5, -10}
  4. Metric           — gemma default, fit on {L2, cos, KL}
  5. Threshold        — gemma default, threshold frac in {0.25, 0.5, 0.75}
  6. Perturb method   — gemma default, geodesic vs additive
  7. Direction type   — (deferred)

Reads fits from results/fits/ and (for the threshold
sweep) refits the raw sweep2d pkls in
results/sweeps_2d/. Missing groups are silently skipped
so the script can be run as data lands.

Usage:
  python scripts/plotting/plot_robustness_beeswarm.py
"""
from __future__ import annotations
import os, sys, glob, pickle, argparse, re
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.lib.superellipse import (
    extract_contour, axis_intercept, fit_superellipse, threshold_levels,
)

OUT_DIR = "results/figures"
DATA_DIR = "results"
FITS_DIR = os.path.join(DATA_DIR, "fits")
SWEEPS_DIR = os.path.join(DATA_DIR, "sweeps_2d")


# ----- helpers -----

_OVERLAP_CACHE: dict[tuple[str, int], dict[frozenset, float]] = {}


def _load_dom_overlaps(target: str, layer: int) -> dict[frozenset, float]:
    """Return {frozenset(pair): |<d1, d2>|} for the DoM directions at
    (target, layer). Used by --max_overlap to drop pairs whose raw DoMs
    are too non-orthogonal — for those pairs, the perpendicular component
    after Gram-Schmidt is a tiny residual blown up to unit norm and the
    fitted n is dominated by an orthonormalization artifact rather than
    real iso-perturbation geometry."""
    key = (target, layer)
    if key in _OVERLAP_CACHE: return _OVERLAP_CACHE[key]
    fp = os.path.join(DATA_DIR, "directions",
                       f"dirs_{target}_L{layer}.pkl")
    if not os.path.exists(fp):
        _OVERLAP_CACHE[key] = {}
        return {}
    with open(fp, "rb") as f: blob = pickle.load(f)
    dirs = {k: v.cpu().numpy().astype(np.float64)
            for k, v in blob["directions"].items()}
    names = sorted(dirs)
    out = {}
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            out[frozenset({a, b})] = float(abs(np.dot(dirs[a], dirs[b])))
    _OVERLAP_CACHE[key] = out
    return out


def _filter_by_overlap(d: dict[frozenset, float],
                        target: str, layer: int,
                        max_overlap: float | None
                        ) -> dict[frozenset, float]:
    """Keep only pairs whose raw DoM overlap <= max_overlap. No-op if
    max_overlap is None or the overlap table is missing."""
    if max_overlap is None: return d
    overlaps = _load_dom_overlaps(target, layer)
    if not overlaps: return d
    return {k: v for k, v in d.items()
            if overlaps.get(k, 0.0) <= max_overlap}


def _filter_exclude(d: dict[frozenset, float],
                     exclude: set[str] | None
                     ) -> dict[frozenset, float]:
    """Drop any pair that contains a direction name in ``exclude``. Used
    to prune redundant directions (e.g. HonestyShort/TensePresent/Formal)
    at plot time without re-running fits."""
    if not exclude: return d
    return {k: v for k, v in d.items() if not (k & exclude)}


def _apply_filters(subs: dict[str, dict[frozenset, float]],
                    target: str, layer: int,
                    max_overlap: float | None,
                    exclude: set[str] | None,
                    ) -> dict[str, dict[frozenset, float]]:
    out = {}
    for sub_label, pairs in subs.items():
        kept = _filter_by_overlap(pairs, target, layer, max_overlap)
        kept = _filter_exclude(kept, exclude)
        out[sub_label] = kept
    return out


def median_p_for_pair(robust_fit_dict) -> float:
    """One number per pair: median p over the 6 robustness cells."""
    ps = []
    for k, cell in robust_fit_dict.items():
        if k == "__levels__": continue
        if np.isfinite(cell.get("p_median", np.nan)):
            ps.append(cell["p_median"])
    if not ps: return np.nan
    return float(np.median(ps))


USE_THRFIXED = False  # toggled via main() / module-level setter
USE_EXACT = False     # toggled via main() / module-level setter


def _fits_suffix() -> str:
    """Trailing suffix on every fits filename driven by global config:
    '_thrfixed' if USE_THRFIXED, with '_exact' appended if USE_EXACT."""
    s = ""
    if USE_THRFIXED: s += "_thrfixed"
    if USE_EXACT: s += "_exact"
    return s


def load_default_fits(target: str, layer: int = 2) -> dict[frozenset, float]:
    """Returns {frozenset(pair): median_p}. Dedupes by unordered pair so
    historical Tier-1-order duplicates (e.g. (Tense, Era) and (Era, Tense)
    coexisting in the qwen/llama/mistral/aya fits) collapse to one entry."""
    sfx = _fits_suffix()
    fp = os.path.join(FITS_DIR, f"fits_{target}_L{layer}{sfx}.pkl")
    if not os.path.exists(fp):
        return {}
    with open(fp, "rb") as f:
        full = pickle.load(f)
    out = {}
    for pair_key, fit in full.items():
        out[frozenset(pair_key)] = median_p_for_pair(fit)
    return out


def fit_pair_at_threshold(sweep_pkl_path: str,
                           threshold_frac: float,
                           max_alpha_deg: float = 60.0) -> float:
    """Re-fit a single sweep pkl at a chosen fraction of the per-pair
    base threshold (used by the threshold-sweep column)."""
    with open(sweep_pkl_path, "rb") as f:
        d = pickle.load(f)
    angles = d["angles_deg"]
    grid = np.median(d["l2"], axis=0)
    base = min(grid[:, 0].max(), grid[0, :].max())
    thresh = threshold_frac * base
    contour = extract_contour(angles, grid, thresh)
    if contour.size == 0: return np.nan
    edge_eps = 1.0
    grid_max = float(angles.max())
    contour = contour[(contour[:, 0] < grid_max - edge_eps)
                       & (contour[:, 1] < grid_max - edge_eps)]
    contour = contour[(contour[:, 0] <= max_alpha_deg)
                       & (contour[:, 1] <= max_alpha_deg)]
    if len(contour) < 3: return np.nan
    t1 = axis_intercept(angles, grid[:, 0], thresh)
    t2 = axis_intercept(angles, grid[0, :], thresh)
    if not (np.isfinite(t1) and np.isfinite(t2) and t1 > 0 and t2 > 0):
        return np.nan
    fit = fit_superellipse(contour[:, 0] / t1, contour[:, 1] / t2)
    return fit["p"]


def sweeps_for(target: str, layer: int) -> list[str]:
    pat = os.path.join(SWEEPS_DIR,
        f"sweep2d_{target}_L{layer}_*_fineweb_*deg.pkl")
    return sorted(glob.glob(pat))


def _pair_from_sweep_filename(path: str) -> frozenset | None:
    """Parse pair frozenset from sweep pkl filename. Filenames have form
    sweep2d_{target}_L{layer}_{A}__{B}_fineweb_{deg}deg.pkl — directions
    are separated by a literal double underscore."""
    name = os.path.basename(path).replace(".pkl", "")
    m = re.match(r"sweep2d_[^_]+_L\d+_(.+)_fineweb_\d+deg", name)
    if not m: return None
    middle = m.group(1)
    if "__" not in middle: return None
    a, b = middle.split("__", 1)
    return frozenset({a, b})


def _load_fit_pair_dict(filepath: str) -> dict[frozenset, float]:
    """Load a fits pkl and return {frozenset(pair): median_p}, dropping
    non-finite entries."""
    if not os.path.exists(filepath): return {}
    with open(filepath, "rb") as f: full = pickle.load(f)
    out = {}
    for pair_key, fit in full.items():
        p = median_p_for_pair(fit)
        if np.isfinite(p):
            out[frozenset(pair_key)] = p
    return out


def _bootstrap_pooled_mean_ci(subs_pairs: dict, n_iter: int = 5000,
                               seed: int = 0) -> tuple[float, float, float]:
    """Direction-level cluster bootstrap of the pooled mean across
    sub-groups. For each sub-group: resample its directions with
    replacement, rebuild all unordered pairs, look up fit values, and
    accumulate. Pool across sub-groups and take the mean. CI = central
    95% (2.5–97.5 percentile of the bootstrap distribution).

    subs_pairs: dict[sub_label, dict[frozenset(pair), p]].
    Returns (point_mean, lo, hi). Self-pairs (same direction sampled
    twice) are dropped from each iteration.
    """
    rng = np.random.RandomState(seed)
    sub_data = []
    for pairs in subs_pairs.values():
        if not pairs: continue
        dirs = sorted({d for pair in pairs for d in pair})
        n = len(dirs)
        if n < 2: continue
        idx = {d: i for i, d in enumerate(dirs)}
        P = np.full((n, n), np.nan)
        for pair, p in pairs.items():
            a, b = list(pair)
            ia, ib = idx[a], idx[b]
            P[ia, ib] = P[ib, ia] = p
        ii, jj = np.triu_indices(n, k=1)
        sub_data.append((P, ii, jj, n))

    pooled = np.array([p for pairs in subs_pairs.values()
                        for p in pairs.values()])
    pooled = pooled[np.isfinite(pooled)]
    if not sub_data or pooled.size == 0:
        return float("nan"), float("nan"), float("nan")
    point_mean = float(pooled.mean())

    means = np.empty(n_iter)
    for b in range(n_iter):
        total_sum = 0.0
        total_count = 0
        for P, ii, jj, n in sub_data:
            r = rng.randint(0, n, size=n)
            ra = r[ii]; rb = r[jj]
            vals = P[ra, rb]
            mask = np.isfinite(vals) & (ra != rb)
            total_sum += vals[mask].sum()
            total_count += int(mask.sum())
        means[b] = total_sum / total_count if total_count > 0 else np.nan
    means = means[np.isfinite(means)]
    if means.size == 0:
        return point_mean, float("nan"), float("nan")
    return (point_mean,
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


# ----- column loaders -----

MODEL_FULL = {
    "gemma": "gemma-2-9b", "qwen": "qwen3-1.7b", "llama": "llama-3.1-8b",
    "mistral": "mistral-7b-v0.3", "aya": "aya-expanse-8b", "yi": "yi-1.5-9b",
}


def col_model(layer: int = 2,
               max_overlap: float | None = None
               ) -> dict[str, dict[frozenset, float]]:
    out = {}
    for tgt in ["gemma", "qwen", "llama", "mistral", "aya", "yi"]:
        d = load_default_fits(tgt, layer)
        if not d: continue
        d = {k: v for k, v in d.items() if np.isfinite(v)}
        d = _filter_by_overlap(d, tgt, layer, max_overlap)
        out[MODEL_FULL[tgt]] = d
    return out


def col_perturb_layer(target: str = "gemma",
                       max_overlap: float | None = None
                       ) -> dict[str, dict[frozenset, float]]:
    out = {}
    for L in [2, 5, 10, 20]:
        d = load_default_fits(target, L)
        if not d: continue
        d = {k: v for k, v in d.items() if np.isfinite(v)}
        d = _filter_by_overlap(d, target, L, max_overlap)
        out[f"L={L}"] = d
    return out


def col_measure_layer(target: str = "gemma",
                       layer: int = 2,
                       max_overlap: float | None = None
                       ) -> dict[str, dict[frozenset, float]]:
    """Default = penultimate (n_layers-2). Variants shift the
    measurement layer earlier in the network."""
    out = {}
    default = load_default_fits(target, layer)
    if default:
        out["penult"] = {k: v for k, v in default.items() if np.isfinite(v)}
    extra = _fits_suffix()
    for label, suffix in [("penult-5", "_M-7"), ("penult-10", "_M-12")]:
        fp = os.path.join(FITS_DIR,
                           f"fits_{target}_L{layer}{suffix}{extra}.pkl")
        d = _load_fit_pair_dict(fp)
        if d: out[label] = d
    out = {k: _filter_by_overlap(v, target, layer, max_overlap)
           for k, v in out.items()}
    return out


def col_metric(target: str = "gemma",
                layer: int = 2,
                max_overlap: float | None = None
                ) -> dict[str, dict[frozenset, float]]:
    """Same sweep grid, different distance fitted to the iso-contour."""
    out = {}
    default = load_default_fits(target, layer)
    if default:
        out[r"$L^2$"] = {k: v for k, v in default.items() if np.isfinite(v)}
    pretty = {"cos": r"$1-\cos$", "kl": "KL"}
    extra = _fits_suffix()
    for m in ["cos", "kl"]:
        fp = os.path.join(
            FITS_DIR,
            f"fits_{target}_L{layer}_cos+kl+l2_metric_{m}{extra}.pkl")
        d = _load_fit_pair_dict(fp)
        if d: out[pretty[m]] = d
    out = {k: _filter_by_overlap(v, target, layer, max_overlap)
           for k, v in out.items()}
    return out


def col_threshold(target: str = "gemma",
                   layer: int = 2,
                   fracs=(0.25, 0.50, 0.75),
                   max_overlap: float | None = None
                   ) -> dict[str, dict[frozenset, float]]:
    """Iso-contour threshold = fraction of per-pair max axis L^2.
    In fixed-T mode, loads the _thrfixed fits and splits per
    threshold_factor sub-group (0.5×T / 1.0×T / 2.0×T)."""
    if USE_THRFIXED:
        out = {}
        fp = os.path.join(FITS_DIR,
                           f"fits_{target}_L{layer}{_fits_suffix()}.pkl")
        if not os.path.exists(fp): return out
        with open(fp, "rb") as f: full = pickle.load(f)
        for pair_key, fit in full.items():
            for cell_key, cell in fit.items():
                if cell_key == "__levels__": continue
                tl, ma = cell_key
                p = cell.get("p_median", np.nan)
                if not np.isfinite(p): continue
                out.setdefault(tl, {}).setdefault(
                    frozenset(pair_key), []).append(p)
        # collapse the per-pair list to its median (across max_alphas)
        out = {tl: {pair: float(np.median(ps))
                     for pair, ps in d.items()}
               for tl, d in out.items()}
        out = {k: _filter_by_overlap(v, target, layer, max_overlap)
               for k, v in out.items()}
        return out
    out = {}
    files = sweeps_for(target, layer)
    if not files: return out
    for f in fracs:
        sub = {}
        for fp in files:
            pair = _pair_from_sweep_filename(fp)
            if pair is None: continue
            p = fit_pair_at_threshold(fp, threshold_frac=f)
            if np.isfinite(p):
                sub[pair] = p
        if sub:
            sub = _filter_by_overlap(sub, target, layer, max_overlap)
            out[f"{int(f*100)}%"] = sub
    return out


def col_method(target: str = "gemma",
                layer: int = 2,
                max_overlap: float | None = None
                ) -> dict[str, dict[frozenset, float]]:
    """Geodesic exp-map (norm-matched on sphere) vs additive
    (a + sin(α) R d_perp)."""
    out = {}
    default = load_default_fits(target, layer)
    if default:
        out["norm-matched"] = {k: v for k, v in default.items()
                                if np.isfinite(v)}
    extra = _fits_suffix()
    fp = os.path.join(FITS_DIR,
                       f"fits_{target}_L{layer}_additive{extra}.pkl")
    d = _load_fit_pair_dict(fp)
    if d: out["additive"] = d
    out = {k: _filter_by_overlap(v, target, layer, max_overlap)
           for k, v in out.items()}
    return out


def col_anchor_source(target: str = "gemma",
                       layer: int = 2,
                       max_overlap: float | None = None
                       ) -> dict[str, dict[frozenset, float]]:
    """Anchor distribution: FineWeb-edu (existing baseline) vs Wikipedia
    English vs Wikipedia Mandarin vs Python code. Same DoM directions
    across all sub-groups; only the evaluation-anchor distribution
    varies."""
    out = {}
    default = load_default_fits(target, layer)
    if default:
        out["FineWeb-edu"] = {k: v for k, v in default.items()
                               if np.isfinite(v)}
    sources = [("Wikipedia (en)", "_srcwiki_en"),
               ("Wikipedia (zh)", "_srcwiki_zh"),
               ("Code",           "_srccode")]
    extra = _fits_suffix()
    for label, suffix in sources:
        fp = os.path.join(FITS_DIR,
                           f"fits_{target}_L{layer}{suffix}{extra}.pkl")
        d = _load_fit_pair_dict(fp)
        if d: out[label] = d
    out = {k: _filter_by_overlap(v, target, layer, max_overlap)
           for k, v in out.items()}
    return out


_LANG_NAMES = {"Arabic", "Chinese", "Dutch", "French", "German", "Italian",
                "Japanese", "Korean", "Portuguese", "Russian", "Spanish"}
_CODE_NAMES = {"Cpp", "Go", "Haskell", "Java", "JavaScript", "Lisp",
                "Python", "Rust", "TypeScript"}


def _pair_kind(pair: frozenset) -> str:
    """Classify an unordered direction pair: 'lang' (natural language),
    'code' (programming language), 'sem' (everything else, e.g.
    Gender/Tense/Era/Sentiment/...). Returned label is sorted
    alphabetically (e.g. 'codexlang', 'langxsem')."""
    a, b = list(pair)
    def kind(n):
        if n in _LANG_NAMES: return "lang"
        if n in _CODE_NAMES: return "code"
        return "sem"
    ka, kb = sorted([kind(a), kind(b)])
    return f"{ka}×{kb}"


def col_pair_kind(target: str = "gemma",
                   layer: int = 2,
                   max_overlap: float | None = None
                   ) -> dict[str, dict[frozenset, float]]:
    """Split the canonical 528-pair fits into 6 sub-groups by pair-type:
    lang×lang, code×code, sem×sem, code×lang, lang×sem, code×sem.

    Sub-groups are ordered by descending mean overlap (the most-aligned
    groups first, so the cluster structure reads visually). Pairs in
    each sub-group are then optionally overlap-filtered as usual."""
    full = load_default_fits(target, layer)
    if not full: return {}
    full = {k: v for k, v in full.items() if np.isfinite(v)}
    full = _filter_by_overlap(full, target, layer, max_overlap)
    grouped: dict[str, dict[frozenset, float]] = {}
    for pair, p in full.items():
        grouped.setdefault(_pair_kind(pair), {})[pair] = p
    order = ["lang×lang", "code×code", "code×lang",
             "code×sem", "lang×sem", "sem×sem"]
    out = {}
    for k in order:
        if k in grouped: out[k] = grouped[k]
    # Append any unrecognised label so we don't silently drop data.
    for k in grouped:
        if k not in out: out[k] = grouped[k]
    return out


def col_direction_type(target: str = "gemma",
                        layer: int = 2,
                        max_overlap: float | None = None
                        ) -> dict[str, dict[frozenset, float]]:
    """DoM (contrastive prompts), SAE-decoder rows (random latent
    indices), MELBO unsupervised steering vectors, isotropic random
    unit vectors. 33 directions per family, paired C(33, 2) = 528.

    The overlap filter only applies to the DoM sub-group — the other
    direction families don't go through the contrastive-DoM pipeline,
    so the orthogonality concern is DoM-specific."""
    out = {}
    default = load_default_fits(target, layer)
    if default:
        dom = {k: v for k, v in default.items() if np.isfinite(v)}
        dom = _filter_by_overlap(dom, target, layer, max_overlap)
        out["DoM contrastive"] = dom
    families = [("SAE decoder",    "_dirsae_random"),
                ("MELBO",          "_dirmelbo"),
                ("random",         "_dirrandom")]
    for label, suffix in families:
        fp = os.path.join(FITS_DIR, f"fits_{target}_L{layer}{suffix}.pkl")
        d = _load_fit_pair_dict(fp)
        if d: out[label] = d
    return out


# ----- rendering -----

def render(columns_data, png_path, pdf_path,
            show_sub_legends: bool = True,
            show_stats_text: bool = False,
            ylabel: str = r"fitted superellipse exponent $p$",
            ref_y: float | None = 2.0,
            ref_label: str = r"$p=2$ (Euclidean)",
            ylim: tuple[float, float] | None = None,
            fontscale: float = 1.0):
    """columns_data is a list of (column_label, sub_label_to_values_dict).

    Within each column, all sub-groups share the same x position; they're
    distinguished only by colour. The mean +/- 2 sigma band and the median
    line are computed over all sub-groups pooled (i.e., across the levels
    of that column's axis of variation) — this matches the "vary one
    thing" reading without privileging any single level.

    ``show_sub_legends``: when False, suppresses the per-column color
    legend (use this when each column only has a single sub-group, so
    the column label alone identifies the data).
    ``show_stats_text``: when True, prints "μ=X.XX  Md=Y.YY" above each
    column near the top of the panel.
    """
    # palette for sub-groups (max 6); cycles if needed.
    SUB_COLORS = ["#3a6ea5", "#D9822B", "#3aa54a", "#a53a3a",
                  "#7B3294", "#1f9d9d"]

    n_cols = len(columns_data)
    # Wider columns when stats are printed underneath so the CI line
    # doesn't overflow into the next column. Also widened for plain
    # mode at the new larger fontsize so multi-word column labels (e.g.
    # "Perturbation method") don't collide with their neighbours.
    width_per_col = 4.6 if show_stats_text else 3.4
    fig, ax = plt.subplots(figsize=(width_per_col * n_cols + 2.5, 8.0))
    rng = np.random.RandomState(0)

    x_pos = 1.0
    label_y_main = -0.03
    label_y_legend = -0.10

    for col_label, subs in columns_data:
        sub_labels = list(subs)
        # Pooled stats kept around for the per-column stats text line
        # (when show_stats_text=True). The diamonds and median bars
        # below are split per sub-group so ablation effects are
        # visible.
        pooled = np.concatenate([np.array(list(subs[s].values()))
                                  for s in sub_labels if len(subs[s])])
        if len(pooled):
            mean, lo, hi = _bootstrap_pooled_mean_ci(subs)

        # Per-sub-group mean +/- 95% CI diamonds and median bars,
        # offset horizontally within the column. CI is the
        # direction-level cluster bootstrap on that single sub-group;
        # diamond + median bar are colored to match the sub-group's
        # dots so the reader can pair them up by colour.
        non_empty_idx = [i for i, s in enumerate(sub_labels) if len(subs[s])]
        n_sub_plot = len(non_empty_idx)
        if n_sub_plot == 1:
            offsets = {non_empty_idx[0]: 0.0}
        elif n_sub_plot > 1:
            xs = np.linspace(-0.27, 0.27, n_sub_plot)
            offsets = {idx: float(xs[k]) for k, idx in enumerate(non_empty_idx)}
        else:
            offsets = {}
        for s_idx, sub in enumerate(sub_labels):
            data = np.array(list(subs[sub].values()))
            if not len(data): continue
            color = SUB_COLORS[s_idx % len(SUB_COLORS)]
            m_s, lo_s, hi_s = _bootstrap_pooled_mean_ci({sub: subs[sub]})
            x_s = x_pos + offsets[s_idx]
            if np.isfinite(m_s):
                ax.errorbar(x_s, m_s,
                            yerr=[[m_s - lo_s], [hi_s - m_s]],
                            fmt="D", color=color, ecolor=color,
                            markersize=7, markerfacecolor="white",
                            markeredgewidth=1.8, elinewidth=1.5,
                            capsize=4, capthick=1.5, zorder=5)
            med_s = float(np.median(data))
            half_bar = 0.30 / max(n_sub_plot, 1)
            ax.hlines(med_s, x_s - half_bar, x_s + half_bar,
                      colors=color, lw=2.5, zorder=6)
        # Plot each sub-group's dots at the SAME x_pos, colour-coded.
        for s_idx, sub in enumerate(sub_labels):
            data = np.array(list(subs[sub].values()))
            if not len(data): continue
            xs = x_pos + rng.uniform(-0.30, 0.30, size=len(data))
            color = SUB_COLORS[s_idx % len(SUB_COLORS)]
            ax.scatter(xs, data, color=color, s=12, alpha=0.28,
                       edgecolor="none", zorder=3, label=sub)
        # Per-column legend (small, just below the column label).
        # Anchor uses get_xaxis_transform so x is in DATA coords (matches
        # the column's x_pos exactly) and y is in axes-fraction.
        leg_handles = [plt.Line2D([0], [0], marker="o", color=color,
                                  markeredgecolor="none", linestyle="",
                                  markersize=6,
                                  label=sub)
                       for s_idx, sub in enumerate(sub_labels)
                       for color in [SUB_COLORS[s_idx % len(SUB_COLORS)]]
                       if len(subs[sub])]
        if show_stats_text and len(pooled):
            # Two-line treatment: family name (larger, semibold) on top,
            # mean + 95% CI underneath in smaller font. Median stays as
            # the black bar inside the panel.
            ax.text(x_pos, label_y_main, col_label,
                    ha="center", va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=22 * fontscale, fontweight="semibold",
                    color="#222")
            ax.text(x_pos, label_y_main - 0.07,
                    f"mean = {mean:.2f}\n[{lo:.2f}, {hi:.2f}]",
                    ha="center", va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=20 * fontscale, color="#444",
                    linespacing=1.05)
        else:
            ax.text(x_pos, label_y_main, col_label, ha="center", va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=18 * fontscale,
                    fontweight="bold", color="#222")
        if leg_handles and show_sub_legends:
            leg = ax.legend(handles=leg_handles,
                            loc="upper center",
                            bbox_to_anchor=(x_pos, label_y_legend),
                            bbox_transform=ax.get_xaxis_transform(),
                            ncol=1, fontsize=18 * fontscale, frameon=False,
                            handlelength=1.4, handletextpad=0.6,
                            borderaxespad=0.0,
                            labelcolor="linecolor")
            for txt in leg.get_texts():
                txt.set_fontweight("medium")
            ax.add_artist(leg)
        x_pos += 1.0

    # Reference horizontal line + legend for it. Defaults to the p=2
    # (Euclidean) line; callers can pass a different ref_y / ref_label
    # to reuse this renderer for other quantities (e.g. fit residual).
    legend_handles = []
    if ref_y is not None:
        ref_line = ax.axhline(ref_y, ls="--", color="#D9822B", lw=1.2,
                               alpha=0.85, label=ref_label)
        legend_handles.append(ref_line)
    mean_handle = plt.Line2D([0], [0], marker="D", color="#222222",
                              linestyle="", markerfacecolor="white",
                              markeredgewidth=1.5, markersize=7,
                              label="mean (bar = 95% CI, dir bootstrap)")
    median_handle = plt.Line2D([0], [0], color="black", lw=2.5,
                               label="median")
    legend_handles += [mean_handle, median_handle]
    ax.set_ylabel(ylabel, fontsize=24 * fontscale)
    ax.tick_params(axis="y", labelsize=20 * fontscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(0.2, x_pos - 1.0 + 0.8)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=legend_handles,
              loc="upper right", fontsize=18 * fontscale,
              frameon=True, framealpha=0.9)
    # Manual layout — tight_layout clobbers the legend anchor placement.
    # Reserve bottom margin for per-column color legends; if those are
    # off, the column labels alone need only a small margin.
    if show_sub_legends:
        bottom = 0.34
    elif show_stats_text:
        # room for family name + two-line CI block, scaled with fontsize
        bottom = 0.18 + 0.06 * fontscale
    else:
        bottom = 0.08
    fig.subplots_adjust(left=0.04, right=0.995, top=0.97, bottom=bottom)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    # Don't use bbox_inches="tight" — it crops the per-column legends
    # (which live in the bottom margin via xaxis-transform anchors) out
    # of the saved figure.
    fig.savefig(png_path, dpi=220, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma",
                    help="default target for the 'vary one axis' columns")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--columns", default="model,perturb,measure,metric,threshold,method,direction,anchor_source",
                    help="comma-separated subset of columns to plot")
    ap.add_argument("--max_overlap", type=float, default=None,
                    help="filter pairs by raw DoM overlap |<d1,d2>|. "
                         "Pairs with overlap exceeding this threshold are "
                         "dropped (their fitted n is dominated by the "
                         "Gram-Schmidt orthonormalization artifact, not "
                         "real geometry). Suggested value 0.15.")
    ap.add_argument("--exclude_dirs", default="",
                    help="comma-separated list of direction names to "
                         "drop from every column (any pair touching one "
                         "of these names is excluded). Use to prune "
                         "redundant directions at plot time without "
                         "re-running fits.")
    ap.add_argument("--use_thrfixed", action="store_true",
                    help="load fits_<...>_thrfixed.pkl files instead "
                         "of canonical fits. Each condition uses its "
                         "own per-condition fixed L^2 (or fixed metric) "
                         "threshold from refit_thrfixed_all.py.")
    ap.add_argument("--exact", action="store_true",
                    help="load exact-geodesic fits (fits_<...>_exact.pkl "
                         "or fits_<...>_thrfixed_exact.pkl) instead of "
                         "the small-angle-approximation fits. Adds "
                         "'_exact' to the output filename.")
    ap.add_argument("--show_stats_text", action="store_true",
                    help="print pooled mean (with 95% CI) under each "
                         "column label. Also enables a wider per-column "
                         "spacing so the CI text doesn't overflow into "
                         "neighbouring columns.")
    args = ap.parse_args()
    global USE_THRFIXED, USE_EXACT
    USE_THRFIXED = args.use_thrfixed
    USE_EXACT = args.exact

    mo = args.max_overlap
    excl = {n.strip() for n in args.exclude_dirs.split(",") if n.strip()}
    columns = []
    def _add(label, d):
        if d:
            columns.append((label, _apply_filters(d, args.target,
                                                    args.layer, mo, excl)))
    if "model" in args.columns:
        _add("Model", col_model(args.layer, max_overlap=mo))
    if "perturb" in args.columns:
        _add("Perturbation layer", col_perturb_layer(args.target, max_overlap=mo))
    if "measure" in args.columns:
        _add("Measurement layer", col_measure_layer(args.target, args.layer, max_overlap=mo))
    if "metric" in args.columns:
        _add("Fit metric", col_metric(args.target, args.layer, max_overlap=mo))
    if "threshold" in args.columns:
        _add("Response threshold", col_threshold(args.target, args.layer, max_overlap=mo))
    if "method" in args.columns:
        _add("Perturbation method", col_method(args.target, args.layer, max_overlap=mo))
    if "direction" in args.columns:
        _add("Direction family", col_direction_type(args.target, args.layer, max_overlap=mo))
    if "anchor_source" in args.columns:
        _add("Anchor source", col_anchor_source(args.target, args.layer, max_overlap=mo))
    if "pair_kind" in args.columns:
        # Each pair-kind gets its own column so the cluster structure
        # reads horizontally (low-n high-overlap clusters on the left,
        # near-orthogonal sem×sem on the right).
        grouped = col_pair_kind(args.target, args.layer, max_overlap=mo)
        for kind_label, pairs in grouped.items():
            sub = _apply_filters({kind_label: pairs}, args.target,
                                  args.layer, mo, excl)
            if any(sub.values()):
                columns.append((kind_label, sub))

    if not columns:
        print("no columns have data yet — run fit_pairs.py first")
        return

    # report counts
    print("Columns with data:")
    for label, subs in columns:
        for sub, ps in subs.items():
            print(f"  {label:18s} {sub:>10s}  n={len(ps)}")
    # report bootstrap CIs alongside the figure
    print("\nDirection-bootstrap 95% CIs (pooled per column):")
    for label, subs in columns:
        mean, lo, hi = _bootstrap_pooled_mean_ci(subs)
        print(f"  {label:18s}  mean={mean:.4f}  CI=[{lo:.4f}, {hi:.4f}]"
              f"  half-width={0.5*(hi-lo):.4f}")

    overlap_tag = (f"_ov{args.max_overlap:g}".replace(".", "p")
                   if args.max_overlap is not None else "")
    excl_tag = f"_excl-{'-'.join(sorted(excl))}" if excl else ""
    if args.use_thrfixed: excl_tag += "_thrfixed"
    if args.exact: excl_tag += "_exact"
    # Distinguish a non-default --columns choice in the filename so we
    # don't overwrite the canonical full-beeswarm pdf.
    cols_tag = ("" if args.columns ==
                "model,perturb,measure,metric,threshold,method,direction,anchor_source"
                else "_" + args.columns.replace(",", "+"))
    out_png = os.path.join(
        OUT_DIR,
        f"robustness_beeswarm_{args.target}_L{args.layer}"
        f"{overlap_tag}{excl_tag}{cols_tag}.png")
    out_pdf = out_png.replace(".png", ".pdf")
    render(columns, out_png, out_pdf,
           show_stats_text=args.show_stats_text)


if __name__ == "__main__":
    main()
