"""Build a cache of 33 random in-distribution *difference* directions at
L=2: draw random anchor pairs (i, j) from a 10k-prompt FineWeb activation
sample and set ``d = (h_i - h_j) / ||h_i - h_j||``.

This is the headline non-feature null (item 2 of the baseline-strengthening
plan). It is identical in construction (a difference of two activations)
and geometry (lives on the real activation-difference manifold, inheriting
its anisotropy and covariance) to a real contrastive direction, and differs
only in carrying no coherent semantic axis. Expectation: ``p ~ 2``.

Output:
  results/directions/dirs_<target>_L<L>_randomdiff_fineweb.pkl
  schema: matches the existing direction-cache schema (33 unit vectors,
  ``family='randomdiff_fineweb'``); per-direction ``prompt_set_hashes``
  records the sampled (i, j) anchor indices + seed.

Activations are reused from
  results/activations/acts_<target>_L<L>_fineweb_<n>.pkl
via ``actlib.fineweb_acts_n``. If that cache already exists, no model load
is needed (pure index-and-subtract job).

Usage:
  python -u scripts/random_diff_directions.py --target gemma --layer 2
"""
from __future__ import annotations
import os, sys, pickle, argparse, time
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from scripts.lib import activations as actlib

OUT_DIR = "results/directions"
os.makedirs(OUT_DIR, exist_ok=True)

N_DIRS = 33
N_FINEWEB = 10000
SEED = 42


def ts(): return time.strftime("%H:%M:%S")


def _load_acts(args):
    """Return the (N, D) FineWeb last-token activations, loading the cached
    pkl directly when present (no model) and only falling back to a model
    forward pass when it is missing."""
    cache_pkl = os.path.join(
        "results/activations",
        f"acts_{args.target}_L{args.layer}_fineweb_{args.n_fineweb}.pkl")
    if os.path.exists(cache_pkl):
        print(f"[{ts()}] loading cached activations {cache_pkl}", flush=True)
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)["activations"].float()

    # Cache miss: load the model and extract (this also writes the cache).
    print(f"[{ts()}] activation cache missing; loading {args.model}",
          flush=True)
    from src.model import load_model
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    fw = actlib.fineweb_acts_n(
        model, tokenizer, device, args.target, args.layer,
        n=args.n_fineweb, seed=args.fineweb_seed, batch_size=args.fwd_batch)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return fw["activations"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DIRS,
                    help="number of random difference directions to build")
    ap.add_argument("--n_fineweb", type=int, default=N_FINEWEB,
                    help="size of the FineWeb activation pool to sample from")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="RNG seed for the anchor-pair draw")
    ap.add_argument("--k_avg", type=int, default=1,
                    help="number of activation-difference pairs averaged per "
                         "direction. 1 = single random difference (headline "
                         "null). Set to the contrastive pair-count (K=30) for "
                         "the construction-matched non-semantic twin of a "
                         "contrastive direction; writes family "
                         "'randomdiffavg_fineweb' (distinct file/suffix so it "
                         "doesn't collide with the k_avg=1 sweeps).")
    ap.add_argument("--fineweb_seed", type=int, default=SEED,
                    help="seed of the FineWeb activation cache (only used "
                         "when the cache has to be (re)built)")
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--model", default="gemma-2-9b",
                    help="HF model id used by src.model.load_model (only "
                         "loaded if the activation cache is missing)")
    ap.add_argument("--fwd_batch", type=int, default=32)
    args = ap.parse_args()

    acts = _load_acts(args)            # (N, D) float32 CPU
    N, D = acts.shape
    print(f"  acts: {tuple(acts.shape)}", flush=True)
    need = 2 * args.n * args.k_avg
    if N < need:
        raise ValueError(f"need >= {need} activations for {args.n} "
                         f"directions x {args.k_avg} disjoint pairs; have {N}")

    # Each direction averages `k_avg` disjoint activation-difference pairs,
    # then normalizes. k_avg=1 -> a single random difference (the headline
    # difference null). k_avg=K matching the contrastive pair-count (K=30)
    # -> the construction-identical *non-semantic* twin of a contrastive
    # direction: same estimator (mean of K activation differences), the only
    # change being that the K pairs share no concept, so coherent feature
    # structure averages out and p collapses toward 2. All n*k_avg pairs are
    # globally disjoint (one big permutation), mirroring the random-partition
    # framing.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)[:need]
    idx_i = perm[0::2]                           # (n*k_avg,)
    idx_j = perm[1::2]
    diffs = (acts[idx_i] - acts[idx_j]).reshape(args.n, args.k_avg, D)
    raw = diffs.mean(dim=1)                       # (n, D)
    norms = raw.norm(dim=-1)
    dirs = F.normalize(raw, dim=-1)

    avg = args.k_avg > 1
    family = "randomdiffavg_fineweb" if avg else "randomdiff_fineweb"
    prefix = "randomdiffavg" if avg else "randomdiff"
    idx_i_g = idx_i.reshape(args.n, args.k_avg)
    idx_j_g = idx_j.reshape(args.n, args.k_avg)
    names = [f"{prefix}_{i:03d}" for i in range(args.n)]
    out = {
        "directions": {n: dirs[i].clone() for i, n in enumerate(names)},
        "family": family,
        "signs": {n: family.split("_")[0] for n in names},
        "prompt_set_hashes": {
            n: {"family": family,
                "k_avg": args.k_avg,
                "anchor_i": idx_i_g[i].tolist(),
                "anchor_j": idx_j_g[i].tolist(),
                "n_fineweb": args.n_fineweb,
                "fineweb_seed": args.fineweb_seed,
                "pair_seed": args.seed,
                "mean_diff_norm": float(norms[i].item())}
            for i, n in enumerate(names)},
        "k_avg": args.k_avg,
        "n_fineweb": args.n_fineweb,
        "fineweb_seed": args.fineweb_seed,
        "seed": args.seed,
        "layer": args.layer,
        "model": args.target,
        "model_full": args.model,
        "d_model": D,
        "n_dirs": args.n,
    }
    out_path = os.path.join(
        OUT_DIR, f"dirs_{args.target}_L{args.layer}_{family}.pkl")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, out_path)
    print(f"[{ts()}] saved {args.n} {family} unit vectors "
          f"(k_avg={args.k_avg}) -> {out_path}")
    print(f"  ||d|| range: "
          f"[{dirs.norm(dim=-1).min():.6f}, {dirs.norm(dim=-1).max():.6f}]")
    print(f"  diff-norm range: [{norms.min():.2f}, {norms.max():.2f}]")
    print(f"  pairwise |cos| max (off-diag): "
          f"{(dirs @ dirs.T).fill_diagonal_(0).abs().max():.4e}")


if __name__ == "__main__":
    main()
