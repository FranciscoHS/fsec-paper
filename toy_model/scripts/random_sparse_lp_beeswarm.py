"""Lp-exponent beeswarm for the random-sparse tied tanh3 baseline.

For each pair of directions (u, v), sweep angle φ ∈ [0, π/2] over a 2-D direction
    direction(φ) = cos(φ)·u + sin(φ)·v
in input space (centered at x_clean). Find the radius α at which the L² output
deviation ||f(x+α·direction) − f(x)||₂ first crosses τ. Fit
    (x/r₀)^n + (y/r_{π/2})^n = 1
to those (α(φ)·cos φ, α(φ)·sin φ) points to recover n.

We do this for:
  - feature pairs: u = e_j, v = e_k for two inactive features j, k ∉ S
  - random pairs: random unit vectors orthogonal to each other and to S

Default: d=512, H=128, sparsity=2, p=0.05, tied, tanh3.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize_scalar

from mft_denoising.data import DataConfig, TwoHotStream
from scripts.random_sparse_baseline import (
    BaselineConfig, build_weights, scan_one_param,
)
from scripts.random_sparse_perturb import build_baked_model


@torch.no_grad()
def boundary_radii(model, d, S, u, v, phis, taus, alpha_max, n_alpha):
    """Vectorized over phis: builds (n_phi * n_alpha, d) inputs in one go and
    forwards in memory-bounded chunks. Equivalent results to the old loop."""
    device = next(model.parameters()).device
    x_clean = torch.zeros(d, device=device)
    for s in S:
        x_clean[s] = 1.0
    base = model(x_clean.unsqueeze(0))[0]
    alphas = torch.linspace(0.0, alpha_max, n_alpha, device=device)

    if not torch.is_tensor(phis):
        phis_t = torch.tensor(np.asarray(phis), dtype=torch.float32, device=device)
    else:
        phis_t = phis.to(device).to(torch.float32)
    n_phi = phis_t.numel()

    # directions: (n_phi, d)
    directions = (torch.cos(phis_t).unsqueeze(1) * u.unsqueeze(0)
                  + torch.sin(phis_t).unsqueeze(1) * v.unsqueeze(0))
    # x: (n_phi, n_alpha, d) — broadcast over alpha and direction
    x = (x_clean.view(1, 1, d)
         + alphas.view(1, n_alpha, 1) * directions.view(n_phi, 1, d))
    x_flat = x.view(n_phi * n_alpha, d)

    # Forward in chunks of ~100 M floats (~400 MB at fp32) to bound GPU memory.
    chunk = max(1, 100_000_000 // max(d, 1))
    diff_norms = []
    for s_idx in range(0, x_flat.shape[0], chunk):
        out_chunk = model(x_flat[s_idx:s_idx + chunk])
        diff_norms.append((out_chunk - base.unsqueeze(0)).norm(p=2, dim=1))
    l2 = torch.cat(diff_norms, dim=0).view(n_phi, n_alpha)

    alphas_cpu = alphas.detach().cpu().numpy()
    out = {tau: np.full(n_phi, np.nan) for tau in taus}
    for tau in taus:
        # earliest α index along each phi where l2 > tau
        above = (l2 > tau).int()
        # If any True, the first True is at argmax(above) (since values are 0/1
        # and argmax breaks ties by first occurrence).
        any_above = above.any(dim=1)
        first_idx = above.argmax(dim=1)
        first_idx_cpu = first_idx.detach().cpu().numpy()
        any_above_cpu = any_above.detach().cpu().numpy()
        out[tau][any_above_cpu] = alphas_cpu[first_idx_cpu[any_above_cpu]]
    return out


def fit_n(xn, yn):
    mask = (xn > 1e-3) & (yn > 1e-3) & np.isfinite(xn) & np.isfinite(yn)
    x = xn[mask]; y = yn[mask]
    if len(x) < 10:
        return np.nan
    def loss(n): return float(np.sum((x ** n + y ** n - 1.0) ** 2))
    res = minimize_scalar(loss, bounds=(0.5, 60.0), method="bounded",
                          options={"xatol": 1e-3})
    return res.x


def fit_from_radii(rs, phis):
    if not (np.isfinite(rs[0]) and np.isfinite(rs[-1])):
        return np.nan
    xs = rs * np.cos(phis) / rs[0]
    ys = rs * np.sin(phis) / rs[-1]
    return fit_n(xs, ys)


def random_orthonormal_pair(rng, d, S, device):
    u_np = rng.standard_normal(d); u_np[list(S)] = 0; u_np /= np.linalg.norm(u_np)
    v_np = rng.standard_normal(d); v_np[list(S)] = 0
    v_np -= (v_np @ u_np) * u_np; v_np /= np.linalg.norm(v_np)
    u = torch.from_numpy(u_np.astype(np.float32)).to(device)
    v = torch.from_numpy(v_np.astype(np.float32)).to(device)
    return u, v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--sparsity", type=int, default=2)
    p.add_argument("--noise_variance", type=float, default=0.03)
    p.add_argument("--p", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--n_pairs", type=int, default=80)
    p.add_argument("--n_phis", type=int, default=60)
    p.add_argument("--alpha_max", type=float, default=8.0)
    p.add_argument("--n_alpha", type=int, default=1200)
    p.add_argument("--scan_min", type=float, default=0.05)
    p.add_argument("--scan_max", type=float, default=8.0)
    p.add_argument("--scan_n", type=int, default=121)
    p.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    p.add_argument("--rand_tau", type=float, default=None,
                   help="threshold to use for random-direction fits (defaults to taus[0])")
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or (
        f"runs/random_sparse_d{args.d}_H{args.H}_p{args.p}_tied_tanh3"
    )
    os.makedirs(out_dir, exist_ok=True)

    cfg = BaselineConfig(
        d=args.d, H=args.H, sparsity=args.sparsity,
        noise_variance=args.noise_variance, p=args.p, tied=True,
        activation="tanh3", n_eval=args.n_eval, seed=args.seed, out_dir=out_dir,
    )

    # build & fit
    E, D = build_weights(cfg.d, cfg.H, cfg.p, cfg.tied, cfg.seed)
    torch.manual_seed(cfg.seed)
    data_cfg = DataConfig(d=cfg.d, sparsity=cfg.sparsity,
                          noise_variance=cfg.noise_variance, seed=cfg.seed,
                          device="cpu")
    stream = TwoHotStream(data_cfg)
    x_noisy, x_clean_b = stream.sample_batch(cfg.n_eval)
    scan = np.linspace(args.scan_min, args.scan_max, args.scan_n)
    best, _ = scan_one_param(E, D, x_clean_b, x_noisy, "tanh3", scan)
    c_in, c_out, snr, _, _ = best
    print(f"c_in={c_in:.4f}  c_out={c_out:.4f}  SNR={snr:.3f}")
    model = build_baked_model(cfg, c_in=c_in, c_out=c_out, E=E, D=D)
    device = next(model.parameters()).device

    # sample pairs
    S = tuple(range(cfg.sparsity))
    rng = np.random.default_rng(args.seed)
    inactive = [i for i in range(cfg.d) if i not in S]

    feat_pairs = set()
    while len(feat_pairs) < args.n_pairs:
        a, b = rng.choice(inactive, size=2, replace=False)
        feat_pairs.add((int(min(a, b)), int(max(a, b))))
    feat_pairs = sorted(feat_pairs)

    phis = np.linspace(0.0, math.pi / 2, args.n_phis)

    ns_by_tau = {t: [] for t in args.taus}
    for ix, (j, k) in enumerate(feat_pairs):
        u = torch.zeros(cfg.d, device=device); u[j] = 1.0
        v = torch.zeros(cfg.d, device=device); v[k] = 1.0
        rs_dict = boundary_radii(model, cfg.d, S, u, v, phis, args.taus,
                                 args.alpha_max, args.n_alpha)
        for tau in args.taus:
            ns_by_tau[tau].append(fit_from_radii(rs_dict[tau], phis))
        if (ix + 1) % 20 == 0:
            print(f"  feat-pair {ix + 1}/{args.n_pairs}", flush=True)

    rand_tau = args.rand_tau if args.rand_tau is not None else args.taus[0]
    random_ns = []
    for ix in range(args.n_pairs):
        u, v = random_orthonormal_pair(rng, cfg.d, S, device)
        rs_dict = boundary_radii(model, cfg.d, S, u, v, phis, [rand_tau],
                                 args.alpha_max, args.n_alpha)
        random_ns.append(fit_from_radii(rs_dict[rand_tau], phis))
        if (ix + 1) % 20 == 0:
            print(f"  random-pair {ix + 1}/{args.n_pairs}", flush=True)

    npz_path = os.path.join(out_dir, "lp_distributions.npz")
    np.savez(npz_path, taus=np.array(args.taus),
             random_ns=np.array(random_ns),
             **{f"feat_tau_{t}": np.array(ns_by_tau[t]) for t in args.taus})
    print(f"saved {npz_path}")

    # beeswarm
    columns = [(f"τ={t}", np.array([v for v in ns_by_tau[t] if np.isfinite(v)]),
                "tab:blue") for t in args.taus]
    columns.append((f"random pairs\n(τ={rand_tau})",
                    np.array([v for v in random_ns if np.isfinite(v)]),
                    "tab:orange"))

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    xpos = np.arange(len(columns))
    xtick_labels = []
    for ix, (name, vals, color) in enumerate(columns):
        if len(vals) == 0:
            xtick_labels.append(f"{name}\n(no fits)")
            continue
        jx = xpos[ix] + (rng.random(len(vals)) - 0.5) * 0.35
        ax.scatter(jx, vals, s=20, alpha=0.6, edgecolor="black", linewidth=0.3,
                   color=color)
        ax.hlines(np.median(vals), xpos[ix] - 0.3, xpos[ix] + 0.3,
                  color="black", linewidth=2.0)
        xtick_labels.append(f"{name}\nmed={np.median(vals):.2f}\n"
                            f"mean={np.mean(vals):.2f}\nN={len(vals)}")

    ax.axhline(2, color="tab:purple", ls="--", alpha=0.6, label="L²")
    ax.set_xticks(xpos); ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_ylabel("fitted Lp exponent n")
    ax.set_xlabel("output L² threshold (blue: feature pairs;  orange: random orthogonal pairs)")
    ax.set_title(f"random-sparse tied tanh³ — d={cfg.d}, H={cfg.H}, p={cfg.p} "
                 f"— Lp exponent, N={args.n_pairs} pairs/col")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "lp_exponent_beeswarm.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
