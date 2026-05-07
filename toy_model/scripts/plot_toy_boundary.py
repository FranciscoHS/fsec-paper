"""LLM-style iso-response boundary plot for the toy random-sparse model.

Analog of scripts/exp_map/plotting/plot_fig3_boundary.py from the
LLM repo, but for the random-sparse autoencoder. Picks one inactive
feature pair (j, k), no misalignment (alpha=0 -> u=e_j, v=e_k),
sweeps phi in [0, pi/2], extracts the iso-tau boundary radii, fits
the superellipse exponent n, and overlays:

  - p=2 ellipse with semi-axes (r0, r1) = (r(0), r(pi/2))
  - p=infinity rectangle with the same semi-axes
  - fitted super-ellipse curve scaled to (r0, r1)
  - data points (r(phi_i)*cos phi_i, r(phi_i)*sin phi_i)

All in raw (alpha_along_e_j, alpha_along_e_k) units.

Usage:
    python -m scripts.plot_toy_boundary \\
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
from scripts.random_sparse_lp_beeswarm import boundary_radii, fit_n
from scripts.random_sparse_misalignment_sweep import peak_l2_along_feature


def superellipse_first_quadrant(n: float, n_pts: int = 200):
    """First-quadrant unit superellipse |x|^n + |y|^n = 1, parameterised
    by t in [0, pi/2]:  x = cos(t)^(2/n), y = sin(t)^(2/n).

    For n=2 this is a quarter-circle; n=infinity becomes the corner
    (1,0)->(1,1)->(0,1) which we clip to a large finite n."""
    t = np.linspace(0.0, np.pi / 2, n_pts)
    n = max(float(n), 1e-6)
    return np.cos(t) ** (2.0 / n), np.sin(t) ** (2.0 / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2_npz", required=True,
                    help="path to phase2[_act]_d{d}_H{H}.npz with the baked "
                         "model parameters (p_star, c_in_star, c_out_star or "
                         "theta_star, lambda_on_final, activation).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--pair", default="2,3",
                    help="comma-separated (j, k) inactive-feature indices. "
                         "Defaults to the first two indices outside the "
                         "active set S=(0..sparsity-1).")
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument("--noise_variance", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_eval", type=int, default=4096)
    ap.add_argument("--n_phis", type=int, default=120,
                    help="phi grid points in [0, pi/2].")
    ap.add_argument("--alpha_max", type=float, default=25.0,
                    help="upper bound on alpha along (u, v); must exceed the "
                         "expected boundary radius at every phi.")
    ap.add_argument("--n_alpha", type=int, default=2500,
                    help="alpha grid resolution per phi.")
    ap.add_argument("--tau_frac", type=float, default=0.5,
                    help="threshold tau as a fraction of peak L^2 along the "
                         "first inactive feature axis (matches the toy-model "
                         "convention in random_sparse_misalignment_sweep.py).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    j_str, k_str = args.pair.split(",")
    j, k = int(j_str), int(k_str)

    # ---- load phase 2 NPZ --------------------------------------------------
    z = np.load(args.phase2_npz, allow_pickle=False)
    p_val = float(z["p_star"])
    c_out = float(z["c_out_star"])
    theta = float(z["theta_star"]) if "theta_star" in z.files else 0.0
    c_in = float(z["c_in_star"]) if "c_in_star" in z.files else float("nan")
    activation = (str(z["activation"]) if "activation" in z.files else "tanh3")
    H = int(z["H"]) if "H" in z.files else None
    d = int(z["d"]) if "d" in z.files else None
    if H is None or d is None:
        # Recover (d, H) from the filename if not stored in NPZ.
        base = os.path.basename(args.phase2_npz)
        # phase2[_act]_d{d}_H{H}.npz
        try:
            stem = base.replace(".npz", "")
            d = int(stem.split("_d")[-1].split("_H")[0])
            H = int(stem.split("_H")[-1])
        except Exception as e:
            raise SystemExit(f"could not infer (d, H) from {base}: {e}")

    print(f"# d={d} H={H} act={activation} p*={p_val} "
          f"c_in*={c_in:.4f} c_out*={c_out:.4f} theta*={theta:.4f}")

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

    # ---- threshold tau -----------------------------------------------------
    S = tuple(range(cfg.sparsity))
    inactive_first = next(i for i in range(cfg.d) if i not in S)
    peak = peak_l2_along_feature(model, cfg.d, S, inactive_first,
                                  args.alpha_max, args.n_alpha)
    tau = float(args.tau_frac) * peak
    print(f"peak L^2(feat axis e_{inactive_first}) = {peak:.4f}")
    print(f"tau = {tau:.4f}  ({args.tau_frac*100:.1f}% of peak)")

    # ---- u = e_j, v = e_k (alpha = 0, perfectly aligned) -------------------
    if j in S or k in S:
        raise SystemExit(f"pair ({j},{k}) intersects active set S={S}")
    if j == k:
        raise SystemExit(f"pair must have j != k; got {j},{k}")
    device = next(model.parameters()).device
    u = torch.zeros(cfg.d, device=device); u[j] = 1.0
    v = torch.zeros(cfg.d, device=device); v[k] = 1.0

    # ---- boundary radii r(phi) at this single tau --------------------------
    phis = np.linspace(0.0, math.pi / 2, args.n_phis)
    rs_dict = boundary_radii(model, cfg.d, S, u, v, phis, [tau],
                              args.alpha_max, args.n_alpha)
    rs = np.asarray(rs_dict[tau], dtype=np.float64)
    if not (np.isfinite(rs[0]) and np.isfinite(rs[-1])):
        raise SystemExit("boundary did not reach tau on at least one of the "
                         "two axes — increase --alpha_max or pick a smaller "
                         "--tau_frac")
    r0 = float(rs[0])         # along e_j
    r1 = float(rs[-1])        # along e_k
    print(f"axis intercepts: r0={r0:.4f} (along e_{j})  "
          f"r1={r1:.4f} (along e_{k})")

    # ---- fit n in normalised coords ---------------------------------------
    xs_norm = rs * np.cos(phis) / r0
    ys_norm = rs * np.sin(phis) / r1
    n_fit = float(fit_n(xs_norm, ys_norm))
    if np.isfinite(n_fit):
        # Per-point radial fit residual in normalised coords.
        mask = (xs_norm > 1e-3) & (ys_norm > 1e-3) \
            & np.isfinite(xs_norm) & np.isfinite(ys_norm)
        x = xs_norm[mask]; y = ys_norm[mask]
        rho = np.abs((np.abs(x) ** n_fit + np.abs(y) ** n_fit) ** (1.0 / n_fit) - 1.0)
        mean_rho = float(np.mean(rho))
        n_pts = int(mask.sum())
    else:
        mean_rho = float("nan"); n_pts = 0
    print(f"fitted n = {n_fit:.3f}   mean radial fraction = {mean_rho:.4f}   "
          f"n_pts = {n_pts}")

    # ---- plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    # p=infinity rectangle (semi-axes r0, r1)
    xi, yi = superellipse_first_quadrant(20)
    ax.plot(r0 * xi, r1 * yi, ls=(0, (1.8, 2.5)),
            color="#D9822B", lw=1.6, alpha=0.95,
            label=r"Per-feature threshold ($p\to\infty$)")
    # p=2 ellipse
    xc, yc = superellipse_first_quadrant(2)
    ax.plot(r0 * xc, r1 * yc, "-",
            color="#1F77B4", lw=2.6, alpha=0.95,
            label=r"Euclidean combination ($p=2$)")
    # fitted super-ellipse
    if np.isfinite(n_fit):
        xf, yf = superellipse_first_quadrant(n_fit)
        ax.plot(r0 * xf, r1 * yf, "-",
                color="#3a7d3a", lw=2.0, alpha=0.95,
                label=rf"Super-ellipse fit ($p_{{\mathrm{{fit}}}} = {n_fit:.2f}$)",
                zorder=4)
    # data points
    xs_raw = rs * np.cos(phis)
    ys_raw = rs * np.sin(phis)
    ax.plot(xs_raw, ys_raw, "o",
            color="#3a3a3a", markersize=4.0, alpha=0.85,
            markeredgecolor="none",
            label="Data", zorder=5)

    # ticks roughly 5–7 across each axis
    def _spacing(span):
        for s in [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]:
            if span / s <= 8: return s
        return 100.0
    sp1 = _spacing(r0 * 1.18)
    sp2 = _spacing(r1 * 1.18)
    ax.set_xticks(np.arange(0, r0 * 1.18 + 1e-6, sp1))
    ax.set_yticks(np.arange(0, r1 * 1.18 + 1e-6, sp2))
    ax.grid(True, which="major", color="#bbbbbb", lw=0.5, alpha=0.45)
    ax.set_xlim(-0.03 * r0, 1.15 * r0)
    ax.set_ylim(-0.03 * r1, 1.15 * r1)
    ax.set_aspect("equal")
    ax.set_xlabel(rf"$\alpha(\varphi_i)\cos\varphi_i$  (along $e_{{{j}}}$)",
                  fontsize=12)
    ax.set_ylabel(rf"$\alpha(\varphi_i)\sin\varphi_i$  (along $e_{{{k}}}$)",
                  fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="lower left", frameon=True, framealpha=0.95)
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    stem = f"toy_boundary_d{d}_H{H}_pair{j}-{k}_taufrac{args.tau_frac:g}{tag}"
    out_png = os.path.join(args.out_dir, f"{stem}.png")
    out_pdf = os.path.join(args.out_dir, f"{stem}.pdf")
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
