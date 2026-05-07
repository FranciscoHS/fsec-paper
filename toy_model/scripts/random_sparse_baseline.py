"""Dmitry's random-sparse baseline.

Build a denoiser y = c_out * D * sigma(c_in * E x) where E (encoder, shape H x d) and
D (decoder, shape d x H) are *frozen random sparse matrices* with iid entries

    P(0)   = 1 - p
    P(+1)  = p / 2
    P(-1)  = p / 2

We do NOT learn E or D. We learn at most two scalars (c_in, c_out) — c_out has a
closed form given c_in, so it's a 1-D scan over c_in with c_out fitted analytically
each step. For JumpReLU we instead scan the threshold theta (c_in fixed at 1).

We support two coupling modes:

    --tied           D = E^T  (the MFT-aligned construction).
    --untied         D and E sampled independently from the same distribution.

The untied case is the literal reading of "sample Encoder[i,k] and Decoder[i,k]
from the same distribution"; we expect it to fail because the diagonal of the
effective linear map D * J(sigma) * E is mean-zero. Tied is the construction
that should work.

Outputs (under --out_dir):
    config.json
    summary.json                   reconstruction metrics
    enc_dec_weight_scatter.png     pair plot of (E[i,k], D[k,i])
    fit_curve.png                  reconstruction loss vs c_in / theta scan

Default config: d=512, H=128, sparsity=2 (4x more features than neurons).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import matplotlib.pyplot as plt
import torch

from mft_denoising.data import DataConfig, TwoHotStream


# ---------------------------- weight construction ----------------------------

def sample_sparse_pm1(shape, p, generator):
    """iid entries in {0, +1, -1} with P(+1)=P(-1)=p/2, P(0)=1-p."""
    u = torch.rand(*shape, generator=generator)
    s = torch.zeros(*shape)
    s[u < p / 2] = 1.0
    s[(u >= p / 2) & (u < p)] = -1.0
    return s


def build_weights(d, H, p, tied, seed):
    g = torch.Generator().manual_seed(seed)
    E = sample_sparse_pm1((H, d), p, g)              # encoder rows
    if tied:
        D = E.t().clone()                             # (d, H)
    else:
        D = sample_sparse_pm1((d, H), p, g)
    return E, D


# ---------------------------- forward model ----------------------------------

def activation(z, kind, theta):
    if kind == "tanh3":
        return torch.tanh(z) ** 3
    if kind == "relu":
        return torch.clamp(z, min=0.0)
    if kind == "jumprelu":
        return z * (z > theta).to(z.dtype)
    raise ValueError(kind)


def forward(E, D, x, c_in, c_out, kind, theta):
    return c_out * (activation(c_in * (x @ E.t()), kind, theta) @ D.t())


# ---------------------------- scalar fitting ---------------------------------

@torch.no_grad()
def fit_c_out(f, x_clean, lambda_on=1.0):
    """Two scalar fits, both closed-form:
        c_on   = <f, x*>_on / ||f||^2_on        (fits ON-coords only)
        c_w    = lambda_on * <f, x*>_on / (lambda_on * ||f||^2_on + ||f||^2_off)

    With c_on, on-coords match in LS sense and off-coords reveal the irreducible
    noise floor of the random baseline. We pick the scan param by SNR, defined as
        snr = mean(c_on * f[on]) / std(c_on * f[off]) = 1 / std_after_scaling
    so high SNR == low off-noise after fitting on-coords to 1.

    Returns (c_on, snr, c_w, weighted_err)."""
    mask_on = (x_clean > 0.5)
    f_on = f[mask_on]
    f_off = f[~mask_on]
    on_norm2 = float((f_on * f_on).sum().item())
    if on_norm2 == 0.0:
        return 0.0, 0.0, 0.0, float("inf")
    c_on = float((f_on * x_clean[mask_on]).sum().item()) / on_norm2
    scaled_on = c_on * f_on
    scaled_off = c_on * f_off
    snr = float(scaled_on.mean().item()) / max(float(scaled_off.std().item()), 1e-12)

    num_w = lambda_on * float((f_on * x_clean[mask_on]).sum().item())
    den_w = lambda_on * on_norm2 + float((f_off * f_off).sum().item())
    c_w = num_w / den_w
    resid = c_w * f - x_clean
    err_w = (lambda_on * (resid[mask_on] ** 2).sum()
             + (resid[~mask_on] ** 2).sum()).item() / x_clean.numel()
    return c_on, snr, c_w, err_w


@torch.no_grad()
def scan_one_param(E, D, x_clean, x_noisy, kind, scan, lambda_on=1.0):
    """Scan c_in (for tanh3/relu) or theta (for jumprelu). Pick best by SNR."""
    rows = []
    for v in scan:
        if kind == "jumprelu":
            c_in, theta = 1.0, float(v)
        else:
            c_in, theta = float(v), 0.0
        f = (activation(c_in * (x_noisy @ E.t()), kind, theta) @ D.t())
        c_on, snr, c_w, err_w = fit_c_out(f, x_clean, lambda_on=lambda_on)
        rows.append((float(v), c_on, snr, c_w, err_w))
    # best by largest SNR
    rows_sorted = sorted(rows, key=lambda r: -r[2])
    best = rows_sorted[0]
    return best, rows


# ---------------------------- evaluation -------------------------------------

@torch.no_grad()
def reconstruction_quality(E, D, x_clean, x_noisy, c_in, c_out, kind, theta,
                           on_thr=0.5, off_tol=0.1):
    out = forward(E, D, x_noisy, c_in, c_out, kind, theta)
    mask_on = (x_clean > 0.5)
    mask_off = ~mask_on
    on_vals = out[mask_on]
    off_vals = out[mask_off]
    return {
        "n_samples": int(x_clean.shape[0]),
        "mse": float(((out - x_clean) ** 2).mean().item()),
        "mse_on": float(((out - x_clean)[mask_on] ** 2).mean().item()),
        "mse_off": float((out[mask_off] ** 2).mean().item()),
        "on_correct_at_0p5": float((on_vals > on_thr).float().mean().item()),
        "off_correct_at_0p1": float((off_vals.abs() < off_tol).float().mean().item()),
        "mean_on_output": float(on_vals.mean().item()),
        "mean_off_output": float(off_vals.mean().item()),
    }


# ---------------------------- plots ------------------------------------------

def plot_enc_dec_scatter(E, D, save_path, figsize=(10, 8)):
    """Scatter of (E[i, k], D[k, i]) over all (i, k) hidden-feature pairs.

    E has shape (H, d); D has shape (d, H). For each (i, k), pair E[i, k] with
    D[k, i]. Vectorized: flatten E row-major and flatten D.T row-major.
    """
    encoder_weights_full = E.numpy()                    # (H, d)
    decoder_weights_full = D.numpy()                    # (d, H)
    encoder_pairs = encoder_weights_full.flatten()      # E[0,0], E[0,1], ...
    decoder_pairs = decoder_weights_full.T.flatten()    # D[0,0], D[1,0], ... matches E layout

    xlim = (encoder_pairs.min() - 0.5, encoder_pairs.max() + 0.5)
    ylim = (decoder_pairs.min() - 0.5, decoder_pairs.max() + 0.5)

    # Jitter so discrete {-1, 0, +1} clusters are visible (no-op for continuous weights).
    n_unique = len(np.unique(encoder_pairs)) + len(np.unique(decoder_pairs))
    rng = np.random.default_rng(0)
    if n_unique <= 12:
        jx = encoder_pairs + (rng.random(encoder_pairs.size) - 0.5) * 0.3
        jy = decoder_pairs + (rng.random(decoder_pairs.size) - 0.5) * 0.3
    else:
        jx, jy = encoder_pairs, decoder_pairs

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(jx, jy, s=2, alpha=0.25, color="blue", rasterized=True, zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Encoder Weight", fontsize=12)
    ax.set_ylabel("Decoder Weight", fontsize=12)
    ax.set_title("Encoder-Decoder Weight Pairs\n(encoder[i,j] vs decoder[j,i])",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    stats_text = (
        "Statistics:\n"
        f"Total pairs: {len(encoder_pairs):,}\n"
        f"Encoder range: [{encoder_pairs.min():.4f}, {encoder_pairs.max():.4f}]\n"
        f"Decoder range: [{decoder_pairs.min():.4f}, {decoder_pairs.max():.4f}]\n"
        f"Encoder mean: {encoder_pairs.mean():.4f}\n"
        f"Decoder mean: {decoder_pairs.mean():.4f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


def plot_scan(rows, kind, save_path):
    rows = np.array(rows)              # (n, 5): param, c_on, snr, c_w, err_w
    rows = rows[np.argsort(rows[:, 0])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(rows[:, 0], rows[:, 2], color="tab:blue", marker=".")
    ax.set_xlabel("theta" if kind == "jumprelu" else "c_in")
    ax.set_ylabel("SNR  =  mean(y[on]) / std(y[off])")
    ax.set_title(f"SNR vs scan param ({kind})")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(rows[:, 0], rows[:, 4], color="tab:orange", marker=".")
    ax.set_xlabel("theta" if kind == "jumprelu" else "c_in")
    ax.set_ylabel("weighted err (lambda_on)")
    ax.set_title(f"weighted-LS err vs scan param ({kind})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


# ---------------------------- main -------------------------------------------

@dataclass
class BaselineConfig:
    d: int = 512
    H: int = 128
    sparsity: int = 2
    noise_variance: float = 0.03
    p: float = 0.05
    tied: bool = True
    activation: str = "jumprelu"   # "jumprelu" | "tanh3" | "relu"
    n_eval: int = 4096
    seed: int = 42
    out_dir: str = "runs/random_sparse_d512_H128"
    lambda_on: float = 10.0
    # scan config:
    scan_min: float = 0.0
    scan_max: float = 2.0
    scan_n: int = 41


def run(cfg: BaselineConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    # 1) build data
    data_cfg = DataConfig(d=cfg.d, sparsity=cfg.sparsity,
                          noise_variance=cfg.noise_variance, seed=cfg.seed,
                          device="cpu")
    stream = TwoHotStream(data_cfg)
    x_noisy, x_clean = stream.sample_batch(cfg.n_eval)

    # 2) build random weights
    E, D = build_weights(cfg.d, cfg.H, cfg.p, cfg.tied, cfg.seed)

    # 3) scan one param + closed-form c_out
    if cfg.activation == "jumprelu":
        scan = np.linspace(cfg.scan_min, cfg.scan_max, cfg.scan_n)
    else:
        # for tanh3/relu, scan c_in
        scan = np.linspace(max(cfg.scan_min, 1e-3), cfg.scan_max, cfg.scan_n)

    best, rows = scan_one_param(E, D, x_clean, x_noisy, cfg.activation, scan,
                                lambda_on=cfg.lambda_on)
    best_param, best_c_on, best_snr, best_c_w, best_err_w = best
    if cfg.activation == "jumprelu":
        c_in, theta = 1.0, best_param
    else:
        c_in, theta = best_param, 0.0
    print(f"best: param={best_param:.4f}  c_on={best_c_on:.4f}  snr={best_snr:.3f}  "
          f"c_w={best_c_w:.4f}  err_w={best_err_w:.4f}")

    # 4) eval at the best — both fitting modes
    metrics_on = reconstruction_quality(E, D, x_clean, x_noisy,
                                        c_in, best_c_on, cfg.activation, theta)
    metrics_w = reconstruction_quality(E, D, x_clean, x_noisy,
                                       c_in, best_c_w, cfg.activation, theta)
    print("metrics (on-only fit):", json.dumps(metrics_on, indent=2))
    print("metrics (weighted fit):", json.dumps(metrics_w, indent=2))

    # 5) baselines for context
    zero_mse = float((x_clean ** 2).mean().item())          # predict 0
    identity_mse = float(((x_noisy - x_clean) ** 2).mean().item())  # passthrough
    baselines = {"zero_mse": zero_mse, "identity_mse": identity_mse}

    # 6) plots
    plot_enc_dec_scatter(E, D, os.path.join(cfg.out_dir, "enc_dec_weight_scatter.png"))
    plot_scan(rows, cfg.activation, os.path.join(cfg.out_dir, "fit_curve.png"))

    # 7) persist
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump({
            "config": asdict(cfg),
            "fit": {"c_in": c_in, "theta": theta,
                    "c_on": best_c_on, "snr": best_snr,
                    "c_w": best_c_w, "err_w": best_err_w},
            "metrics_on": metrics_on,
            "metrics_w": metrics_w,
            "baselines": baselines,
            "scan": [[float(r[0]), float(r[1]), float(r[2]),
                      float(r[3]), float(r[4])] for r in rows],
        }, f, indent=2)
    print(f"saved {cfg.out_dir}/summary.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--sparsity", type=int, default=2)
    p.add_argument("--noise_variance", type=float, default=0.03)
    p.add_argument("--p", type=float, default=0.05,
                   help="density of nonzeros in E (and D if untied)")
    p.add_argument("--untied", action="store_true",
                   help="sample D independently of E (default tied: D = E^T)")
    p.add_argument("--activation", choices=["tanh3", "relu", "jumprelu"],
                   default="jumprelu")
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--scan_min", type=float, default=0.0)
    p.add_argument("--scan_max", type=float, default=2.0)
    p.add_argument("--scan_n", type=int, default=41)
    p.add_argument("--lambda_on", type=float, default=10.0,
                   help="match the trained model's loss weighting")
    args = p.parse_args()

    out_dir = args.out_dir or (
        f"runs/random_sparse_d{args.d}_H{args.H}_p{args.p}"
        f"_{'untied' if args.untied else 'tied'}_{args.activation}"
    )
    cfg = BaselineConfig(
        d=args.d, H=args.H, sparsity=args.sparsity,
        noise_variance=args.noise_variance, p=args.p,
        tied=not args.untied, activation=args.activation,
        n_eval=args.n_eval, seed=args.seed,
        out_dir=out_dir,
        lambda_on=args.lambda_on,
        scan_min=args.scan_min, scan_max=args.scan_max, scan_n=args.scan_n,
    )
    run(cfg)


if __name__ == "__main__":
    main()
