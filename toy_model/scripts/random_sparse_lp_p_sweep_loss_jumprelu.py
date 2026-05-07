"""Same analysis as random_sparse_lp_p_sweep_loss.py, but with JumpReLU
activation instead of tanh^3.

JumpReLU is positively 1-homogeneous in (c_in, theta): scaling c_in by k and
theta by k gives the same output (modulo c_out absorbing k). So we fix c_in=1
and scan theta only. c_out is closed-form per theta via the weighted-MSE
optimum, same as the tanh3 script. τ is set per-p as a fixed fraction of the
peak L² output deviation along a feature axis.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from mft_denoising.data import DataConfig, TwoHotStream
from mft_denoising.nn import TwoLayerNet
from scripts.random_sparse_baseline import BaselineConfig, build_weights
from scripts.random_sparse_lp_beeswarm import (
    boundary_radii, fit_from_radii, random_orthonormal_pair,
)


def fit_loss_optimal_jumprelu(E, D, x_clean, x_noisy, lambda_on, theta_scan):
    """Pick (theta, c_out) minimizing lambda_on-weighted MSE.
    c_in is fixed at 1 (absorbed into theta and c_out). Returns (theta, c_out, loss)."""
    mask_on = (x_clean > 0.5).to(x_clean.dtype)
    mask_off = 1.0 - mask_on
    z = x_noisy @ E.t()              # pre-activation, shared across theta
    best = None
    for v in theta_scan:
        theta = float(v)
        h = z * (z > theta).to(z.dtype)
        f = h @ D.t()
        num = lambda_on * float((f * x_clean * mask_on).sum().item())
        den = (lambda_on * float((f * f * mask_on).sum().item())
               + float((f * f * mask_off).sum().item()))
        c_out = num / max(den, 1e-12)
        err = c_out * f - x_clean
        loss = (lambda_on * float((err ** 2 * mask_on).sum().item())
                + float((err ** 2 * mask_off).sum().item())) / x_clean.numel()
        if best is None or loss < best[2]:
            best = (theta, c_out, loss)
    return best


def build_baked_jumprelu(cfg, theta, c_out, E, D, device=None):
    """Build a TwoLayerNet with JumpReLU(theta) activation, weights (E, c_out·D)."""
    model = TwoLayerNet(
        input_size=cfg.d, hidden_size=cfg.H, activation="jumprelu",
        activation_kwargs=dict(theta_init=theta, learn_threshold=False, eps=0.1),
    )
    with torch.no_grad():
        model.fc1.weight.copy_(E)
        model.fc1.bias.zero_()
        model.fc2.weight.copy_(c_out * D)
        model.fc2.bias.zero_()
        # set theta buffer to scalar
        model.act.theta.fill_(theta)
    model.eval()
    if device is None and torch.cuda.is_available():
        device = "cuda"
    if device is not None:
        model = model.to(device)
    return model


def peak_l2_along_feature(model, d, S, j, alpha_max, n_alpha):
    device = next(model.parameters()).device
    x_clean = torch.zeros(d, device=device)
    for s in S:
        x_clean[s] = 1.0
    base = model(x_clean.unsqueeze(0))[0]
    alphas = torch.linspace(0.0, alpha_max, n_alpha, device=device)
    u = torch.zeros(d, device=device); u[j] = 1.0
    x = x_clean.unsqueeze(0) + alphas.unsqueeze(1) * u.unsqueeze(0)
    out = model(x)
    return float((out - base.unsqueeze(0)).norm(p=2, dim=1).max().item())


def fit_one_p(d, H, sparsity, noise_variance, p, seed, n_eval, n_pairs, n_phis,
              alpha_max, n_alpha, theta_scan, lambda_on, tau_frac):
    cfg = BaselineConfig(d=d, H=H, sparsity=sparsity,
                         noise_variance=noise_variance, p=p, tied=True,
                         activation="jumprelu", n_eval=n_eval, seed=seed, out_dir="")
    E, D = build_weights(cfg.d, cfg.H, cfg.p, cfg.tied, cfg.seed)
    torch.manual_seed(seed)
    stream = TwoHotStream(DataConfig(d=cfg.d, sparsity=cfg.sparsity,
                                     noise_variance=cfg.noise_variance,
                                     seed=seed, device="cpu"))
    x_noisy, x_clean = stream.sample_batch(n_eval)
    fit_device = "cuda" if torch.cuda.is_available() else "cpu"
    E_d, D_d = E.to(fit_device), D.to(fit_device)
    x_n_d, x_c_d = x_noisy.to(fit_device), x_clean.to(fit_device)
    theta, c_out, loss = fit_loss_optimal_jumprelu(
        E_d, D_d, x_c_d, x_n_d, lambda_on, theta_scan)
    print(f"  p={p}: theta={theta:.3f}  c_out={c_out:.4f}  "
          f"train_loss(λ={lambda_on:g})={loss:.4f}", flush=True)

    model = build_baked_jumprelu(cfg, theta=theta, c_out=c_out, E=E, D=D)
    device = next(model.parameters()).device

    S = tuple(range(cfg.sparsity))
    inactive_first = next(i for i in range(cfg.d) if i not in S)
    peak = peak_l2_along_feature(model, cfg.d, S, inactive_first, alpha_max, n_alpha)
    tau = tau_frac * peak
    print(f"     peak_L2(feat) = {peak:.4f}  ->  tau = {tau_frac:.2f} * peak = {tau:.4f}",
          flush=True)

    rng = np.random.default_rng(seed)
    inactive = [i for i in range(cfg.d) if i not in S]
    feat_pairs = set()
    while len(feat_pairs) < n_pairs:
        a, b = rng.choice(inactive, size=2, replace=False)
        feat_pairs.add((int(min(a, b)), int(max(a, b))))
    feat_pairs = sorted(feat_pairs)
    phis = np.linspace(0.0, math.pi / 2, n_phis)

    feat_ns = []
    for j, k in feat_pairs:
        u = torch.zeros(cfg.d, device=device); u[j] = 1.0
        v = torch.zeros(cfg.d, device=device); v[k] = 1.0
        rs_dict = boundary_radii(model, cfg.d, S, u, v, phis, [tau], alpha_max, n_alpha)
        feat_ns.append(fit_from_radii(rs_dict[tau], phis))

    random_ns = []
    for _ in range(n_pairs):
        u, v = random_orthonormal_pair(rng, cfg.d, S, device)
        rs_dict = boundary_radii(model, cfg.d, S, u, v, phis, [tau], alpha_max, n_alpha)
        random_ns.append(fit_from_radii(rs_dict[tau], phis))

    return {
        "theta": float(theta), "c_out": float(c_out), "train_loss": float(loss),
        "peak_l2_feat": float(peak), "tau": float(tau),
        "feat": np.array(feat_ns), "rand": np.array(random_ns),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--ps", type=float, nargs="+", required=True)
    p.add_argument("--lambda_on", type=float, required=True)
    p.add_argument("--tau_frac", type=float, default=0.5)
    p.add_argument("--sparsity", type=int, default=2)
    p.add_argument("--noise_variance", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--n_pairs", type=int, default=80)
    p.add_argument("--n_phis", type=int, default=60)
    p.add_argument("--alpha_max", type=float, default=25.0)
    p.add_argument("--n_alpha", type=int, default=2500)
    p.add_argument("--theta_min", type=float, default=0.0)
    p.add_argument("--theta_max", type=float, default=2.5)
    p.add_argument("--theta_n", type=int, default=51)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"d={args.d} H={args.H} d/H={args.d/args.H:.1f} λ={args.lambda_on} "
          f"τ_frac={args.tau_frac}  ps={args.ps}", flush=True)

    theta_scan = np.linspace(args.theta_min, args.theta_max, args.theta_n)

    results = {}
    for pv in args.ps:
        print(f"=== p={pv} ===", flush=True)
        results[pv] = fit_one_p(
            d=args.d, H=args.H, sparsity=args.sparsity,
            noise_variance=args.noise_variance, p=pv, seed=args.seed,
            n_eval=args.n_eval, n_pairs=args.n_pairs, n_phis=args.n_phis,
            alpha_max=args.alpha_max, n_alpha=args.n_alpha,
            theta_scan=theta_scan, lambda_on=args.lambda_on,
            tau_frac=args.tau_frac,
        )

    save = {
        "ps": np.array(args.ps),
        "lambda_on": np.array(args.lambda_on),
        "tau_frac": np.array(args.tau_frac),
        "theta": np.array([results[pv]["theta"] for pv in args.ps]),
        "c_out": np.array([results[pv]["c_out"] for pv in args.ps]),
        "train_loss": np.array([results[pv]["train_loss"] for pv in args.ps]),
        "peak_l2_feat": np.array([results[pv]["peak_l2_feat"] for pv in args.ps]),
        "tau": np.array([results[pv]["tau"] for pv in args.ps]),
    }
    save.update({f"feat_p{pv}": results[pv]["feat"] for pv in args.ps})
    save.update({f"rand_p{pv}": results[pv]["rand"] for pv in args.ps})
    npz_path = os.path.join(args.out_dir,
        f"lp_p_loss_jumprelu_d{args.d}_H{args.H}_lam{args.lambda_on:g}.npz")
    np.savez(npz_path, **save)
    print(f"saved {npz_path}", flush=True)

    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(args.ps), 5.6))
    rng = np.random.default_rng(0)
    xpos, xtick_labels = [], []
    cols = []
    for ix, pv in enumerate(args.ps):
        feat = results[pv]["feat"]; feat = feat[np.isfinite(feat)]
        rand = results[pv]["rand"]; rand = rand[np.isfinite(rand)]
        cols.append((2 * ix, feat, "tab:blue",
                     f"feat\np={pv:g}\nmed={np.median(feat):.2f}\nN={len(feat)}"))
        cols.append((2 * ix + 1, rand, "tab:orange",
                     f"rand\np={pv:g}\nmed={np.median(rand):.2f}\nN={len(rand)}"))
    for x, vals, color, label in cols:
        if len(vals) == 0:
            xpos.append(x); xtick_labels.append(label + "\n(empty)"); continue
        jx = x + (rng.random(len(vals)) - 0.5) * 0.35
        ax.scatter(jx, vals, s=18, alpha=0.6, edgecolor="black",
                   linewidth=0.3, color=color)
        ax.hlines(np.median(vals), x - 0.3, x + 0.3, color="black", linewidth=2.0)
        xpos.append(x); xtick_labels.append(label)
    ax.axhline(2, color="tab:purple", ls="--", alpha=0.6, label="L²")
    ax.set_xticks(xpos); ax.set_xticklabels(xtick_labels, fontsize=7.0)
    ax.set_ylabel("fitted Lp exponent n")
    ax.set_title(
        f"random sparse tied JumpReLU — d={args.d}, H={args.H} "
        f"(d/H={args.d/args.H:.1f}), loss-optimal fit (λ={args.lambda_on:g}), "
        f"τ = {args.tau_frac:g}·peak_L2(feat)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir,
        f"lp_p_loss_jumprelu_d{args.d}_H{args.H}_lam{args.lambda_on:g}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
