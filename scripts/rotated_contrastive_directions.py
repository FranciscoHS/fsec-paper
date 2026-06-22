"""Build angle-resolved rotated contrastive direction caches for the LLM
misalignment sweep (item 1 of the baseline-strengthening plan).

For each tilt ``theta`` and ensemble seed ``s``, every canonical contrastive
direction ``d_i`` is rotated off its true axis toward a fixed, seeded random
direction ``w_i``:

    d_i(theta) = cos(theta) * d_i + sin(theta) * w_i_perp

where ``w_i_perp`` is the component of the random draw orthogonal to ``d_i``
(so the rotation is a clean geodesic tilt and ``<d_i, d_i(theta)> = cos theta``
exactly). ``w_i`` depends only on ``(seed, direction name)`` — NOT on theta —
so the tilts for a fixed seed trace one geodesic per direction, exactly as
the toy ``random_sparse_misalignment_sweep`` does. theta = 0 reproduces the
real pair; theta = 90deg sends each direction to a random orthogonal axis.

The pair members are re-orthonormalized at sweep time (``sweep_2d.py`` calls
``parametrize.orthonormalize`` against each anchor), so this builder only
needs to emit per-direction rotated unit vectors.

Outputs, one cache per (theta, seed):
  results/directions/dirs_<target>_L<L>_rotcontrastive_th<deg>_s<seed>.pkl
  schema matches the DoM cache (same direction NAMES as the contrastive
  cache, so the overlap-filtered pairs file applies unchanged);
  ``family='rotcontrastive_th<deg>_s<seed>'`` so the downstream
  ``sweep_2d.py`` / ``fit_pairs.py`` filenames don't collide across tilts.

Also writes the overlap-filtered pair list (the pairs the main beeswarm
reports, ``|cos| <= max_overlap`` on the *true* contrastive geometry) for
``sweep_2d.py --pairs_file``.

Usage:
  python -u scripts/rotated_contrastive_directions.py --target gemma --layer 2
"""
from __future__ import annotations
import os, sys, pickle, hashlib, math, argparse
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F

OUT_DIR = "results/directions"
os.makedirs(OUT_DIR, exist_ok=True)

# Mirror the toy 7-point angle grid: [0, 15, 30, 45, 60, 75, 90] deg.
DEFAULT_THETAS_DEG = [0, 15, 30, 45, 60, 75, 90]
DEFAULT_SEEDS = [0, 1, 2, 3]
MAX_OVERLAP = 0.10


def _theta_tag(theta_deg: float) -> str:
    """Stable integer-degree tag for filenames/family strings."""
    return f"{int(round(theta_deg)):02d}"


def _seeded_random_dir(d_i: torch.Tensor, name: str, seed: int) -> torch.Tensor:
    """Unit random vector orthogonal to ``d_i``, deterministic in
    ``(seed, name)`` (independent of theta). Per-direction sub-seed via a
    hash of the name so each direction gets an independent geodesic."""
    sub = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(sub)
    w = torch.randn(d_i.shape[0], generator=g, dtype=torch.float64)
    d = d_i.double()
    w = w - (w @ d) * d                      # project off the true axis
    w = w / w.norm().clamp_min(1e-12)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--thetas_deg", type=float, nargs="+",
                    default=DEFAULT_THETAS_DEG,
                    help="tilt grid in degrees (default mirrors the toy "
                         "7-point sweep)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="ensemble seeds; each is one set of random tilt "
                         "targets w_i (default 0 1 2 3)")
    ap.add_argument("--max_overlap", type=float, default=MAX_OVERLAP,
                    help="overlap threshold for the emitted pairs file "
                         "(|cos| on the true contrastive directions)")
    ap.add_argument("--exclude_dirs", default="",
                    help="comma-sep direction names to drop from the pairs "
                         "file (match the main beeswarm's exclusions)")
    ap.add_argument("--names", default="",
                    help="comma-sep direction names to restrict to. Default "
                         "(empty) = the canonical registry set "
                         "(semantic+lang+code). Load-bearing: the on-disk "
                         "dirs_<target>_L<L>.pkl may be a SUPERSET extended "
                         "with non-canonical directions (e.g. persona/aair "
                         "probes) by other experiments; without this filter "
                         "the misalignment sweep would run on the wrong "
                         "population.")
    ap.add_argument("--max_pairs", type=int, default=40,
                    help="cap the overlap-filtered pairs file to this many "
                         "pairs via a seeded random subsample (the full "
                         "|cos|<=max_overlap set is ~400 pairs; the "
                         "misalignment sweep re-sweeps every pair at each "
                         "theta x seed, so the full set is multi-day GPU). "
                         "0 or negative = keep all.")
    ap.add_argument("--pairs_seed", type=int, default=0,
                    help="seed for the --max_pairs subsample")
    ap.add_argument("--pairs_file", default=None,
                    help="path for the overlap-filtered pairs file. Default "
                         "results/directions/rotcontrastive_pairs_"
                         "<target>_L<L>_ov<max_overlap>.txt")
    args = ap.parse_args()

    base_path = os.path.join(
        OUT_DIR, f"dirs_{args.target}_L{args.layer}.pkl")
    with open(base_path, "rb") as f:
        base = pickle.load(f)
    dirs = {k: v.float() for k, v in base["directions"].items()}
    if args.names.strip():
        keep = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        from scripts.lib import registry
        keep = registry.all_direction_names()
    missing = [n for n in keep if n not in dirs]
    if missing:
        print(f"WARNING: {len(missing)} requested names absent from cache: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    dirs = {n: dirs[n] for n in keep if n in dirs}
    names = sorted(dirs)
    print(f"loaded {len(base['directions'])} directions from {base_path}; "
          f"restricted to {len(names)} canonical contrastive directions")

    # --- overlap-filtered pairs file (true-geometry |cos| <= max_overlap) ---
    excl = {n.strip() for n in args.exclude_dirs.split(",") if n.strip()}
    D = {k: v.double() for k, v in dirs.items()}
    kept_pairs = []
    for i, a in enumerate(names):
        if a in excl: continue
        for b in names[i + 1:]:
            if b in excl: continue
            cos = float(abs(D[a] @ D[b]))
            if cos <= args.max_overlap:
                kept_pairs.append((a, b))
    n_filtered = len(kept_pairs)
    if args.max_pairs and args.max_pairs > 0 and n_filtered > args.max_pairs:
        rng = np.random.default_rng(args.pairs_seed)
        sel = sorted(rng.choice(n_filtered, size=args.max_pairs,
                                replace=False))
        kept_pairs = [kept_pairs[i] for i in sel]
        print(f"subsampled {args.max_pairs} of {n_filtered} filtered pairs "
              f"(seed={args.pairs_seed})")
    pairs_file = args.pairs_file or os.path.join(
        OUT_DIR,
        f"rotcontrastive_pairs_{args.target}_L{args.layer}_"
        f"ov{args.max_overlap:g}".replace(".", "p") + ".txt")
    with open(pairs_file, "w") as f:
        f.write("a,b\n")
        for a, b in kept_pairs:
            f.write(f"{a},{b}\n")
    print(f"wrote {len(kept_pairs)} overlap-filtered pairs "
          f"(|cos| <= {args.max_overlap}) -> {pairs_file}")

    # --- precompute per-(seed, name) random tilt targets w_i ---
    w_dirs = {s: {n: _seeded_random_dir(dirs[n], n, s) for n in names}
              for s in args.seeds}

    n_written = 0
    for theta_deg in args.thetas_deg:
        ca = math.cos(math.radians(theta_deg))
        sa = math.sin(math.radians(theta_deg))
        for s in args.seeds:
            fam = f"rotcontrastive_th{_theta_tag(theta_deg)}_s{s}"
            rot = {}
            realized_cos = []
            for n in names:
                d = dirs[n].double()
                rd = ca * d + sa * w_dirs[s][n]
                rd = rd / rd.norm().clamp_min(1e-12)
                realized_cos.append(float(abs(rd @ d)))
                rot[n] = rd.float()
            out = {
                "directions": rot,
                "family": fam,
                "signs": {n: base.get("signs", {}).get(n, "rot")
                          for n in names},
                "prompt_set_hashes": {
                    n: {"family": fam,
                        "base_name": n,
                        "theta_deg": float(theta_deg),
                        "ensemble_seed": s}
                    for n in names},
                "theta_deg": float(theta_deg),
                "cos_theta": ca,
                "ensemble_seed": s,
                "base_cache": base_path,
                "layer": args.layer,
                "model": args.target,
                "n_dirs": len(names),
            }
            out_path = os.path.join(
                OUT_DIR,
                f"dirs_{args.target}_L{args.layer}_{fam}.pkl")
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(out, f)
            os.replace(tmp, out_path)
            n_written += 1
            print(f"  theta={theta_deg:>4.0f} s={s}  "
                  f"realized cos median={np.median(realized_cos):.4f} "
                  f"(target {ca:.4f})  -> {os.path.basename(out_path)}")

    print(f"done: {n_written} rotated caches over "
          f"{len(args.thetas_deg)} thetas x {len(args.seeds)} seeds")


if __name__ == "__main__":
    main()
