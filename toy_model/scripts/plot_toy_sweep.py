"""LLM-style response-vs-perturbation sweep plot for the toy random-sparse
model. Toy-model analog of scripts/plotting/plot_combo_sweep.py from the LLM
repo (Figure 1a): the 1a-equivalent companion to plot_toy_boundary.py (1b).

For the baked random-sparse autoencoder we perturb the clean active-feature
input additively along a direction and measure the downstream L^2 response as
a function of the perturbation magnitude alpha (the toy analog of the LLM's
perturbation angle). We plot:

  - two inactive feature axes e_j, e_k,
  - their equal-weight combination (e_j + e_k)/sqrt(2),
  - a random-direction baseline (median over --n_random isotropic directions),

with a horizontal threshold tau = tau_frac * peak L^2 along e_j and dashed
verticals at each curve's plateau-breaking magnitude. Feature axes break the
plateau (cross tau) at smaller magnitudes than random directions -- the toy
analog of the privileging seen in the LLM.

Medians only (no bands), matching the LLM Fig 1a convention.

Usage:
    python -m scripts.plot_toy_sweep \\
        --phase2_npz runs/misalignment_sweep_H1024/phase2_d8192_H1024.npz \\
        --out_dir   runs/misalignment_sweep_H1024
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.random_sparse_baseline import BaselineConfig, build_weights
from scripts.random_sparse_perturb import build_baked_model
from scripts.random_sparse_lp_p_sweep_loss_jumprelu import build_baked_jumprelu
from scripts.random_sparse_misalignment_sweep import (
    peak_l2_along_feature, sample_pair_random_directions)


def response_along(model, x_clean, base, direction, alphas):
    """L^2(out(x_clean + alpha * direction) - base) over the alpha grid.
    direction: (d,) unit tensor; alphas: (n_alpha,) tensor. Returns (n_alpha,)."""
    x = x_clean.unsqueeze(0) + alphas.unsqueeze(1) * direction.unsqueeze(0)
    out = model(x)
    return (out - base.unsqueeze(0)).norm(p=2, dim=1)


def crossing(alphas, curve, tau):
    """Linear-interpolated alpha where curve first crosses tau; nan if never."""
    a = np.asarray(alphas); c = np.asarray(curve)
    if c.max() < tau:
        return float("nan")
    for i in range(1, len(c)):
        if (c[i - 1] - tau) * (c[i] - tau) <= 0:
            y0, y1 = float(c[i - 1]), float(c[i])
            if y1 == y0:
                return float(a[i])
            t = (tau - y0) / (y1 - y0)
            return float(a[i - 1] + t * (a[i] - a[i - 1]))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--pair", default="2,3",
                    help="comma-separated (j, k) inactive-feature indices.")
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument("--noise_variance", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_eval", type=int, default=4096)
    ap.add_argument("--alpha_max", type=float, default=25.0)
    ap.add_argument("--n_alpha", type=int, default=2500)
    ap.add_argument("--tau_frac", type=float, default=0.5,
                    help="threshold tau as a fraction of peak L^2 along e_j.")
    ap.add_argument("--n_random", type=int, default=10,
                    help="random directions for the baseline (median curve).")
    ap.add_argument("--x_max", type=float, default=None,
                    help="x-axis (alpha) cap for the plot; default alpha_max.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    j, k = (int(s) for s in args.pair.split(","))

    # ---- load phase 2 NPZ + build model (mirrors plot_toy_boundary.py) -----
    z = np.load(args.phase2_npz, allow_pickle=False)
    p_val = float(z["p_star"])
    c_out = float(z["c_out_star"])
    theta = float(z["theta_star"]) if "theta_star" in z.files else 0.0
    c_in = float(z["c_in_star"]) if "c_in_star" in z.files else float("nan")
    activation = (str(z["activation"]) if "activation" in z.files else "tanh3")
    H = int(z["H"]) if "H" in z.files else None
    d = int(z["d"]) if "d" in z.files else None
    if H is None or d is None:
        stem = os.path.basename(args.phase2_npz).replace(".npz", "")
        d = int(stem.split("_d")[-1].split("_H")[0])
        H = int(stem.split("_H")[-1])
    print(f"# d={d} H={H} act={activation} p*={p_val} c_out*={c_out:.4f}")

    cfg = BaselineConfig(d=d, H=H, sparsity=args.sparsity,
                         noise_variance=args.noise_variance, p=p_val, tied=True,
                         activation=activation, n_eval=args.n_eval,
                         seed=args.seed, out_dir="")
    E, D = build_weights(cfg.d, cfg.H, cfg.p, cfg.tied, cfg.seed)
    if activation == "tanh3":
        model = build_baked_model(cfg, c_in=c_in, c_out=c_out, E=E, D=D)
    elif activation == "jumprelu":
        model = build_baked_jumprelu(cfg, theta=theta, c_out=c_out, E=E, D=D)
    else:
        raise ValueError(f"unknown activation: {activation}")
    model = model.to(args.device)
    device = next(model.parameters()).device

    if j in range(cfg.sparsity) or k in range(cfg.sparsity):
        raise SystemExit(f"pair ({j},{k}) intersects active set")

    # ---- clean input, base, threshold --------------------------------------
    S = tuple(range(cfg.sparsity))
    x_clean = torch.zeros(cfg.d, device=device)
    for s in S:
        x_clean[s] = 1.0
    base = model(x_clean.unsqueeze(0))[0]
    peak = peak_l2_along_feature(model, cfg.d, S, j, args.alpha_max, args.n_alpha)
    tau = float(args.tau_frac) * peak
    print(f"peak L^2(e_{j}) = {peak:.4f}   tau = {tau:.4f} "
          f"({args.tau_frac*100:.0f}% of peak)")

    alphas = torch.linspace(0.0, args.alpha_max, args.n_alpha, device=device)
    a_np = alphas.detach().cpu().numpy()

    def axis(i):
        e = torch.zeros(cfg.d, device=device); e[i] = 1.0
        return e

    with torch.no_grad():
        e_j, e_k = axis(j), axis(k)
        combo = (e_j + e_k); combo = combo / combo.norm()
        curves = {
            f"e_{j}":  response_along(model, x_clean, base, e_j, alphas).cpu().numpy(),
            f"e_{k}":  response_along(model, x_clean, base, e_k, alphas).cpu().numpy(),
            "combo":   response_along(model, x_clean, base, combo, alphas).cpu().numpy(),
        }
        # random baseline: median over n_random isotropic directions
        rng = np.random.default_rng(args.seed)
        rand_stack = []
        for _ in range(args.n_random):
            w_u, _ = sample_pair_random_directions(rng, cfg.d, S, j, k,
                                                   no_feature_orth=True)
            w = torch.from_numpy(w_u).to(device)
            rand_stack.append(
                response_along(model, x_clean, base, w, alphas).cpu().numpy())
        rand_curve = np.median(np.stack(rand_stack, 0), axis=0)

    # ---- plateau-breaking magnitudes ---------------------------------------
    breaks = {name: crossing(a_np, c, tau) for name, c in curves.items()}
    breaks["random"] = crossing(a_np, rand_curve, tau)
    print("plateau-breaking alpha: " +
          "  ".join(f"{n}={v:.3f}" for n, v in breaks.items()))

    # ---- plot (medians only, matching LLM Fig 1a) --------------------------
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    C = {f"e_{j}": "#ff7f00", f"e_{k}": "#377eb8", "combo": "#3aa54a"}
    for name, c in curves.items():
        lab = (rf"$e_{{{j}}}$" if name == f"e_{j}"
               else rf"$e_{{{k}}}$" if name == f"e_{k}"
               else rf"$(e_{{{j}}}+e_{{{k}}})/\sqrt{{2}}$")
        ax.plot(a_np, c, "-", color=C[name], lw=2.4, label=lab)
    ax.plot(a_np, rand_curve, "--", color="#888888", lw=2.0, label="Random")
    ax.axhline(tau, color="#444444", ls=":", lw=1.3,
               label=rf"Threshold ($\tau = {tau:.2f}$)")
    for name, color in [(f"e_{j}", C[f"e_{j}"]), (f"e_{k}", C[f"e_{k}"]),
                        ("combo", C["combo"]), ("random", "#888888")]:
        b = breaks[name]
        if np.isfinite(b):
            ax.axvline(b, color=color, ls="--", lw=1.4, alpha=0.85)

    x_max = args.x_max if args.x_max is not None else args.alpha_max
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, tau * 2.0)
    ax.set_xlabel(r"Perturbation magnitude $\alpha$", fontsize=13)
    ax.set_ylabel(r"$L^2$ response", fontsize=13)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    stem = f"toy_sweep_d{d}_H{H}_pair{j}-{k}_taufrac{args.tau_frac:g}{tag}"
    for ext in ("png", "pdf"):
        path = os.path.join(args.out_dir, f"{stem}.{ext}")
        plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
