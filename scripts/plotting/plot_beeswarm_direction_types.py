"""Beeswarm with one column per direction family.

Default family set:

  1. Contrastive (DoM)
  2. MELBO
  3. SAE-eval       — top-33 SAE features by mean |activation| over
                       the 30 evaluation anchor prompts
  4. SAE-FineWeb    — top-33 SAE features by mean |activation| over
                       a 10k-prompt FineWeb sample
  5. PCA-FineWeb    — top-33 right singular vectors of the same 10k
                       FineWeb activations (PCA, no SAE basis)
  6. Random         — isotropic unit vectors

Every column is overlap-filtered at ``|cos| < max_overlap`` against its
own direction set (not just the Contrastive set as the original beeswarm
did). For SAE-eval / SAE-FineWeb this filter is load-bearing — the
top-activating SAE features tend to be highly correlated with each
other; for PCA-FineWeb / Random it is essentially a no-op (the
directions are orthogonal by construction).

Reuses ``plot_robustness_beeswarm.render`` so the visual style stays
identical to the canonical beeswarm.
"""
from __future__ import annotations
import os, sys, pickle, argparse
sys.path.insert(0, ".")
import numpy as np

from scripts.plotting.plot_robustness_beeswarm import (
    render, _load_fit_pair_dict, _filter_by_overlap,
    _filter_exclude,
    DATA_DIR, FITS_DIR, OUT_DIR,
)


# label, fits suffix, direction-pkl suffix (or None to use the
# Contrastive cache via _load_dom_overlaps), apply overlap filter?
FAMILIES = [
    ("Contrastive",  "",                None,             True),
    ("MELBO",        "_dirmelbo",       "_melbo",         True),
    ("SAE",          "_dirsae_fineweb", "_sae_fineweb",   True),
    ("PCA",          "_dirpca_fineweb", "_pca_fineweb",   True),
    ("Random",       "_dirrandom",      "_random",        True),
]


_PATH_OVERLAP_CACHE: dict[str, dict[frozenset, float]] = {}


def _load_overlaps_from_pkl(path: str) -> dict[frozenset, float]:
    """Pairwise ``|<d_i, d_j>|`` over all directions in a single
    direction cache pkl. Cached per path so per-column filtering doesn't
    re-compute the table each call."""
    if path in _PATH_OVERLAP_CACHE:
        return _PATH_OVERLAP_CACHE[path]
    if not os.path.exists(path):
        _PATH_OVERLAP_CACHE[path] = {}
        return {}
    with open(path, "rb") as f:
        blob = pickle.load(f)
    dirs = {k: v.cpu().numpy().astype(np.float64)
            for k, v in blob["directions"].items()}
    names = sorted(dirs)
    out = {}
    for i, a in enumerate(names):
        da = dirs[a]
        na = float(np.linalg.norm(da))
        if na == 0: continue
        da_u = da / na
        for j in range(i + 1, len(names)):
            b = names[j]
            db = dirs[b]
            nb = float(np.linalg.norm(db))
            if nb == 0: continue
            out[frozenset({a, b})] = float(abs(np.dot(da_u, db / nb)))
    _PATH_OVERLAP_CACHE[path] = out
    return out


def _filter_family_overlap(d: dict[frozenset, float],
                            target: str, layer: int,
                            dir_suffix: str | None,
                            max_overlap: float
                            ) -> dict[frozenset, float]:
    """Family-internal overlap filter. ``dir_suffix=None`` routes to the
    Contrastive overlap table (matching the original DoM-only behaviour);
    any other suffix loads ``dirs_<target>_L<layer><dir_suffix>.pkl``."""
    if max_overlap is None: return d
    if dir_suffix is None:
        # Defer to the existing DoM overlap loader (cached upstream).
        return _filter_by_overlap(d, target, layer, max_overlap)
    fp = os.path.join(DATA_DIR, "directions",
                       f"dirs_{target}_L{layer}{dir_suffix}.pkl")
    overlaps = _load_overlaps_from_pkl(fp)
    if not overlaps:
        # No direction pkl on disk → can't filter; fall back to no-op so
        # we don't silently drop the whole column.
        return d
    return {k: v for k, v in d.items()
            if overlaps.get(k, 0.0) <= max_overlap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--max_overlap", type=float, default=0.10)
    ap.add_argument("--exclude_dirs", default="",
                    help="DoM-only: comma-sep names to drop (e.g. "
                         "HonestyShort,TensePresent,Formal). Other "
                         "families don't share these names.")
    ap.add_argument("--use_thrfixed", action="store_true",
                    help="load fits_<...>_thrfixed.pkl files instead "
                         "of canonical fits.")
    ap.add_argument("--exact", action="store_true",
                    help="load the exact-geodesic fits (fits_<...>_exact"
                         ".pkl or fits_<...>_thrfixed_exact.pkl) and "
                         "append '_exact' to the output filename.")
    ap.add_argument("--include_sae_random", action="store_true",
                    help="also include the original SAE-random column "
                         "(top-33 random latent indices). Off by default "
                         "now that SAE-eval and SAE-FineWeb subsume it.")
    args = ap.parse_args()
    excl = {n.strip() for n in args.exclude_dirs.split(",") if n.strip()}
    thr = "_thrfixed" if args.use_thrfixed else ""
    if args.exact: thr += "_exact"

    families = list(FAMILIES)
    if args.include_sae_random:
        # Insert just before the Random column for parity with the
        # original 4-column figure.
        families.insert(-1, ("SAE-random", "_dirsae_random",
                              "_sae", True))

    columns = []
    for entry in families:
        label, fits_suffix, dir_suffix, do_filter = entry
        fits_fp = os.path.join(
            FITS_DIR,
            f"fits_{args.target}_L{args.layer}{fits_suffix}{thr}.pkl")
        fits = _load_fit_pair_dict(fits_fp)
        if not fits:
            print(f"  {label}: no fits at {fits_fp}, skipping")
            continue
        n_before = len(fits)
        if do_filter:
            fits = _filter_family_overlap(fits, args.target, args.layer,
                                           dir_suffix, args.max_overlap)
            if dir_suffix is None:
                # Contrastive: also apply name-based exclusions.
                fits = _filter_exclude(fits, excl)
            note = f"|cos| <= {args.max_overlap}"
            if dir_suffix is None:
                if excl: note += f", excl {sorted(excl)}"
            kept = len(fits)
            print(f"  {label:<12s} n={kept}/{n_before}  ({note})")
        else:
            print(f"  {label:<12s} n={len(fits)}  (no overlap filter)")
        columns.append((label, {label: fits}))

    if not columns:
        print("no data"); return

    overlap_tag = (f"_ov{args.max_overlap:g}".replace(".", "p")
                   if args.max_overlap is not None else "")
    excl_tag = f"_excl-{'-'.join(sorted(excl))}" if excl else ""
    if args.use_thrfixed: excl_tag += "_thrfixed"
    if args.exact: excl_tag += "_exact"
    if args.include_sae_random: excl_tag += "_withsaerand"
    out_png = os.path.join(
        OUT_DIR,
        f"robustness_beeswarm_{args.target}_L{args.layer}"
        f"{overlap_tag}{excl_tag}_dirfamilies_morebaselines.png")
    out_pdf = out_png.replace(".png", ".pdf")
    render(columns, out_png, out_pdf, show_sub_legends=False,
           show_stats_text=True, fontscale=1.4)


if __name__ == "__main__":
    main()
