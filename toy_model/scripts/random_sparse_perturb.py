"""Perturbation sweep for the random-sparse tied tanh3 baseline.

We don't train. We:
  1) sample tied random sparse E (shape H x d), with iid {0, +1, -1} entries;
  2) scan c_in (input scale), fitting c_out by closed-form on-only LS per c_in;
  3) wrap (c_in * E, c_out * E.T) into a TwoLayerNet with tanh3 activation;
  4) run the existing 1D + 2D frontier sweeps from mft_denoising.probe.

Default config: d=512, H=128, sparsity=2, p=0.05, tanh3.

Outputs go to runs/random_sparse_d{d}_H{H}_p{p}_tied_tanh3/frontier/.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch

from mft_denoising.data import DataConfig, TwoHotStream
from mft_denoising.nn import TwoLayerNet
from mft_denoising.probe import run_robustness_sweep
from scripts.random_sparse_baseline import (
    BaselineConfig, build_weights, scan_one_param, fit_c_out, activation,
)


def build_baked_model(cfg: BaselineConfig, c_in: float, c_out: float, E, D,
                      device=None):
    """Return a TwoLayerNet whose fc1.weight = c_in*E, fc2.weight = c_out*D, tanh3 act."""
    model = TwoLayerNet(
        input_size=cfg.d, hidden_size=cfg.H, activation="tanh3",
    )
    with torch.no_grad():
        model.fc1.weight.copy_(c_in * E)
        model.fc1.bias.zero_()
        model.fc2.weight.copy_(c_out * D)
        model.fc2.bias.zero_()
    model.eval()
    if device is None and torch.cuda.is_available():
        device = "cuda"
    if device is not None:
        model = model.to(device)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--sparsity", type=int, default=2)
    p.add_argument("--noise_variance", type=float, default=0.03)
    p.add_argument("--p", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--scan_min", type=float, default=0.05)
    p.add_argument("--scan_max", type=float, default=8.0)
    p.add_argument("--scan_n", type=int, default=121)
    p.add_argument("--alpha_max", type=float, default=2.0)
    p.add_argument("--n_grid", type=int, default=80)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or (
        f"runs/random_sparse_d{args.d}_H{args.H}_p{args.p}_tied_tanh3/frontier"
    )
    os.makedirs(out_dir, exist_ok=True)

    cfg = BaselineConfig(
        d=args.d, H=args.H, sparsity=args.sparsity,
        noise_variance=args.noise_variance, p=args.p, tied=True,
        activation="tanh3", n_eval=args.n_eval, seed=args.seed,
        out_dir=out_dir,
    )

    # 1) build weights
    E, D = build_weights(cfg.d, cfg.H, cfg.p, cfg.tied, cfg.seed)

    # 2) sample val data and scan c_in (closed-form c_out per c_in)
    torch.manual_seed(cfg.seed)
    data_cfg = DataConfig(d=cfg.d, sparsity=cfg.sparsity,
                          noise_variance=cfg.noise_variance, seed=cfg.seed,
                          device="cpu")
    stream = TwoHotStream(data_cfg)
    x_noisy, x_clean = stream.sample_batch(cfg.n_eval)
    scan = np.linspace(args.scan_min, args.scan_max, args.scan_n)
    best, _ = scan_one_param(E, D, x_clean, x_noisy, "tanh3", scan)
    best_c_in, best_c_on, best_snr, _, _ = best
    print(f"best c_in={best_c_in:.4f}  c_out={best_c_on:.4f}  SNR={best_snr:.3f}")

    # 3) bake the model and run sweeps
    model = build_baked_model(cfg, c_in=best_c_in, c_out=best_c_on, E=E, D=D)

    pairs = [(2, 3), (4, 5), (10, 11)]
    active_sets = [(0, 1), (5, 6), (20, 21)]
    run_robustness_sweep(
        model,
        pairs=pairs,
        active_sets=active_sets,
        out_prefix=os.path.join(out_dir, "frontier"),
        alpha_max=args.alpha_max,
        n_grid=args.n_grid,
        thresholds=(0.3, 0.5, 0.7),
    )

    with open(os.path.join(out_dir, "frontier_meta.json"), "w") as f:
        json.dump({
            "config": asdict(cfg),
            "fit": {"c_in": best_c_in, "c_out": best_c_on, "snr": best_snr},
            "pairs": pairs, "active_sets": active_sets,
            "alpha_max": args.alpha_max, "n_grid": args.n_grid,
        }, f, indent=2)
    print(f"saved frontier_meta.json to {out_dir}")


if __name__ == "__main__":
    main()
