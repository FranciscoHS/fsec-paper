"""Build a cache of 33 MELBO unit-direction steering vectors at L=2 by
running uncapped MELBO once per anchor (one direction per anchor) on 33
FineWeb activations.

Setup:
  inject_layer  = 2 (default; matches exp_map sweeps)
  measure_layer = n_layers - 2 (penultimate)
  reward        = L^2 of (h_perturbed - h_ref) at measure_layer (uncapped)
  opt_angle     = 10 deg (the angle at which the response is measured
                  during optimization)
  n_steps       = 100, n_restarts = 3, no refinement
  anchors       = 33 distinct FineWeb anchors (seed=42), one direction each

Output:
  results/exp_map/data/directions/dirs_gemma_L2_melbo.pkl
  {
    'directions':       {f'melbo_{i:03d}': Tensor[D]},
    'family':           'melbo',
    'signs':            {name: 'melbo_l2'},
    'prompt_set_hashes': {name: {anchor_idx, opt_angle, ...}},
    'anchor_indices':   [0, 1, ..., 32],
    'rewards':          [r_0, ..., r_32],
    'opt_angle_deg':    10.0,
    'n_steps':          100, 'n_restarts': 3,
    'seed':             42, 'layer': 2, 'model': 'gemma',
  }

Usage:
  python -u scripts/exp_map/melbo_directions.py
"""
from __future__ import annotations
import os, sys, pickle, argparse, time
sys.path.insert(0, ".")
import numpy as np
import torch

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.melbo.run_iterative_melbo_hf import (
    extract_context_and_activation,
    forward_to_L,
    capped_melbo_one_direction,
)

OUT_DIR = "results/exp_map/data/directions"
os.makedirs(OUT_DIR, exist_ok=True)

N_DIRS = 33
SEED = 42
SEQ_LEN = 5
OPT_ANGLE_DEG = 10.0
N_STEPS = 100
N_RESTARTS = 3


def ts(): return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DIRS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--inject_layer", type=int, default=2)
    ap.add_argument("--measure_offset", type=int, default=-2,
                    help="measure_layer = n_layers + offset; default penult")
    ap.add_argument("--opt_angle", type=float, default=OPT_ANGLE_DEG)
    ap.add_argument("--n_steps", type=int, default=N_STEPS)
    ap.add_argument("--n_restarts", type=int, default=N_RESTARTS)
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--model", default="gemma-2-9b")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[{ts()}] loading {args.model}", flush=True)
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    d_model = model._plateau_info['d_model']
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers + args.measure_offset
    print(f"  d_model={d_model}  n_layers={n_layers}  "
          f"inject={args.inject_layer}  measure={measure_layer}", flush=True)

    print(f"[{ts()}] loading {args.n} FineWeb sequences (seed={args.seed})",
          flush=True)
    cache_dir = f"/workspace/fineweb_cache_{args.target}"
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        args.n, tokenizer, seq_len=args.seq_len, seed=args.seed,
        cache_dir=cache_dir)
    contexts, activations = extract_context_and_activation(
        model, tokenizer, token_lists, args.inject_layer, device)
    print(f"  contexts={tuple(contexts.shape)}  "
          f"activations={tuple(activations.shape)}", flush=True)

    angle_rad = np.deg2rad(args.opt_angle)

    directions = []
    rewards = []
    cos_dists = []
    failures = []

    for i in range(args.n):
        ctx = contexts[i].unsqueeze(0).to(device)
        act = activations[i].to(device)
        a_norm = act.norm()
        a_unit = act / a_norm

        with torch.no_grad():
            ref_L = forward_to_L(model, ctx, act.unsqueeze(0),
                                 args.inject_layer, measure_layer)

        opt_seed = args.seed * 1000 + i * 7
        torch.manual_seed(opt_seed)
        np.random.seed(opt_seed)

        t0 = time.time()
        print(f"[{ts()}] anchor {i+1}/{args.n}  ||a||={a_norm:.2f}  "
              f"opt_seed={opt_seed}", flush=True)
        d, reward, hist = capped_melbo_one_direction(
            model, ctx, act, ref_L, a_unit, a_norm,
            angle_rad, target_cos=0.1, prev_directions=[],
            inject_layer=args.inject_layer, measure_layer=measure_layer,
            device=device, d_model=d_model,
            n_steps=args.n_steps, lr=args.lr, n_restarts=args.n_restarts,
            uncapped=True, reward_type='l2',
        )
        if d is None:
            print(f"  -> no direction found", flush=True)
            failures.append(i)
            continue
        directions.append(d.detach().cpu())
        rewards.append(float(reward))
        cos_dists.append(float(hist[-1][1]) if hist else float('nan'))
        print(f"  -> reward={reward:.4f}  cos_dist={cos_dists[-1]:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    n_found = len(directions)
    if n_found == 0:
        raise RuntimeError("MELBO produced 0 directions")

    names = [f"melbo_{i:03d}" for i in range(n_found)]
    out = {
        "directions": {n: d.float() for n, d in zip(names, directions)},
        "family": "melbo",
        "signs": {n: "melbo_l2" for n in names},
        "prompt_set_hashes": {n: {"family": "melbo",
                                   "anchor_idx": i,
                                   "opt_angle_deg": args.opt_angle,
                                   "inject_layer": args.inject_layer,
                                   "measure_layer": measure_layer,
                                   "n_steps": args.n_steps,
                                   "n_restarts": args.n_restarts,
                                   "reward": rewards[i]}
                              for i, n in enumerate(names)},
        "anchor_indices": list(range(n_found)),
        "rewards": rewards,
        "cos_dists": cos_dists,
        "failures": failures,
        "opt_angle_deg": args.opt_angle,
        "n_steps": args.n_steps,
        "n_restarts": args.n_restarts,
        "inject_layer": args.inject_layer,
        "measure_layer": measure_layer,
        "measure_offset": args.measure_offset,
        "reward_type": "l2",
        "uncapped": True,
        "seed": args.seed,
        "layer": args.inject_layer,
        "model": args.target,
        "model_full": args.model,
        "d_model": d_model,
        "n_dirs": n_found,
    }
    out_path = os.path.join(
        OUT_DIR, f"dirs_{args.target}_L{args.inject_layer}_melbo.pkl")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, out_path)

    D = torch.stack(directions)
    cos = (D @ D.T).fill_diagonal_(0)
    print(f"[{ts()}] saved {n_found} MELBO directions -> {out_path}",
          flush=True)
    print(f"  ||d|| range: "
          f"[{D.norm(dim=-1).min():.6f}, {D.norm(dim=-1).max():.6f}]")
    print(f"  pairwise |cos| max (off-diag): {cos.abs().max():.4f}")
    print(f"  reward stats: min={min(rewards):.3f} "
          f"med={np.median(rewards):.3f} max={max(rewards):.3f}")
    if failures:
        print(f"  failed anchors: {failures}")


if __name__ == "__main__":
    main()
