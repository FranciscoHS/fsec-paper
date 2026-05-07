"""Phase 2 — pick (p*, scale*) for random-sparse tied at fixed (d, H), with
auto λ-bumping. Supports two activations:

  * tanh³ (default): scan c_in (with closed-form c_out per c_in); θ is unused.
  * jumprelu: scan θ (with closed-form c_out per θ); c_in is fixed at 1
    (jumprelu is positively 1-homogeneous in (c_in, θ), so the scan over θ
    alone is sufficient — c_in is absorbed).

For each p in --ps:
  1. Build tied random sparse weights E (D = Eᵀ).
  2. Fit the activation-specific scalar pair on a val batch of n_eval = 4096
     by minimizing the λ-weighted training MSE.
  3. Compute val metrics: train_loss, mean(y[on]), mean(|y[off]|),
     MSE_on, MSE_off, MSE_total. Compute predict-zero baseline λ·s/d.

Pick p* with lowest train_loss. If max_p mean(y[on]) < 0.15, double λ and
retry. Bail if no λ ≤ 8× starting passes (Phase 2 plan §2 step 4).

Saves phase2[_jumprelu]_d{d}_H{H}.npz with all-p arrays + selected (p*, ...).

Usage:
    # tanh3
    python -m scripts.random_sparse_phase2_pselect \
        --d 4096 --H 1024 --lambda_on 500 \
        --ps 0.025 0.05 0.075 0.1 0.125 0.15 \
        --out_dir runs/misalignment_sweep_H1024

    # jumprelu
    python -m scripts.random_sparse_phase2_pselect \
        --d 4096 --H 1024 --lambda_on 500 \
        --activation jumprelu \
        --ps 0.025 0.05 0.075 0.1 0.125 0.15 \
        --out_dir runs/misalignment_sweep_H1024_jumprelu
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from mft_denoising.data import DataConfig, TwoHotStream
from scripts.random_sparse_baseline import build_weights


@torch.no_grad()
def fit_loss_optimal_tanh3(E, D, x_clean, x_noisy, lambda_on, scan):
    """tanh³: scan c_in (θ unused), closed-form c_out.

    Returns (c_in*, c_out*, weighted_train_loss).
    Optimization: precompute z_base = x_noisy @ Eᵀ once.
    """
    mask_on = (x_clean > 0.5).to(x_clean.dtype)
    mask_off = 1.0 - mask_on
    n_total = x_clean.numel()
    z_base = x_noisy @ E.t()  # (B, H), encoder pass shared across scan
    best = None
    for v in scan:
        c_in = float(v)
        h = torch.tanh(c_in * z_base) ** 3
        f = h @ D.t()
        f_x_on = (f * x_clean * mask_on).sum().item()
        f_sq_on = (f * f * mask_on).sum().item()
        f_sq_off = (f * f * mask_off).sum().item()
        num = lambda_on * f_x_on
        den = lambda_on * f_sq_on + f_sq_off
        c_out = num / max(den, 1e-12)
        err = c_out * f - x_clean
        loss = (lambda_on * (err ** 2 * mask_on).sum().item()
                + (err ** 2 * mask_off).sum().item()) / n_total
        if best is None or loss < best[2]:
            best = (c_in, c_out, loss)
    return best


@torch.no_grad()
def fit_loss_optimal_jumprelu(E, D, x_clean, x_noisy, lambda_on, theta_scan):
    """jumprelu: scan θ (c_in fixed at 1), closed-form c_out per θ.

    JumpReLU is positively 1-homogeneous in (c_in, θ): scaling c_in by k and
    θ by k leaves the output unchanged once c_out absorbs the scale, so we
    fix c_in=1 and only scan θ. Returns (theta*, c_out*, weighted_train_loss).
    """
    mask_on = (x_clean > 0.5).to(x_clean.dtype)
    mask_off = 1.0 - mask_on
    n_total = x_clean.numel()
    z_base = x_noisy @ E.t()  # (B, H), pre-activation, shared across theta
    best = None
    for v in theta_scan:
        theta = float(v)
        h = z_base * (z_base > theta).to(z_base.dtype)
        f = h @ D.t()
        f_x_on = (f * x_clean * mask_on).sum().item()
        f_sq_on = (f * f * mask_on).sum().item()
        f_sq_off = (f * f * mask_off).sum().item()
        num = lambda_on * f_x_on
        den = lambda_on * f_sq_on + f_sq_off
        c_out = num / max(den, 1e-12)
        err = c_out * f - x_clean
        loss = (lambda_on * (err ** 2 * mask_on).sum().item()
                + (err ** 2 * mask_off).sum().item()) / n_total
        if best is None or loss < best[2]:
            best = (theta, c_out, loss)
    return best


@torch.no_grad()
def val_metrics(E, D, x_clean, x_noisy, activation, c_in, c_out, theta,
                lambda_on):
    z = c_in * (x_noisy @ E.t())
    if activation == "tanh3":
        h = torch.tanh(z) ** 3
    elif activation == "jumprelu":
        h = z * (z > theta).to(z.dtype)
    else:
        raise ValueError(f"unknown activation: {activation}")
    f = c_out * (h @ D.t())
    mask_on = (x_clean > 0.5)
    mask_off = ~mask_on
    f_on = f[mask_on]
    f_off = f[mask_off]
    err = f - x_clean
    n_total = x_clean.numel()
    train_loss = (lambda_on * (err ** 2)[mask_on].sum().item()
                  + (err ** 2)[mask_off].sum().item()) / n_total
    return {
        "train_loss": float(train_loss),
        "mean_on": float(f_on.mean().item()),
        "mean_abs_off": float(f_off.abs().mean().item()),
        "mse_on": float(((f_on - 1.0) ** 2).mean().item()),
        "mse_off": float((f_off ** 2).mean().item()),
        "mse_total": float((err ** 2).mean().item()),
    }


def run_for_lambda(args, lambda_on, x_clean, x_noisy, fit_device):
    """Run the full p-sweep at the given λ. Returns dict per p plus selection."""
    if args.activation == "tanh3":
        scan = np.linspace(args.scan_min, args.scan_max, args.scan_n)
    else:
        scan = np.linspace(args.theta_min, args.theta_max, args.theta_n)
    s = args.sparsity
    predict_zero_loss = lambda_on * s / args.d

    per_p = {}
    for pv in args.ps:
        E, D = build_weights(args.d, args.H, pv, True, args.seed)
        E_d = E.to(fit_device)
        D_d = D.to(fit_device)
        if args.activation == "tanh3":
            c_in, c_out, fit_loss = fit_loss_optimal_tanh3(
                E_d, D_d, x_clean, x_noisy, lambda_on, scan)
            theta = 0.0
        else:
            theta, c_out, fit_loss = fit_loss_optimal_jumprelu(
                E_d, D_d, x_clean, x_noisy, lambda_on, scan)
            c_in = 1.0
        m = val_metrics(E_d, D_d, x_clean, x_noisy, args.activation,
                        c_in, c_out, theta, lambda_on)
        per_p[pv] = {
            "c_in": c_in, "c_out": c_out, "theta": theta, **m,
            "predict_zero_loss": predict_zero_loss,
        }
        if args.activation == "tanh3":
            scalar_str = f"c_in={c_in:.3f}"
        else:
            scalar_str = f"theta={theta:.3f}"
        print(f"  p={pv:<6g}  {scalar_str}  c_out={c_out:.4f}  "
              f"train_loss={m['train_loss']:.4f}  pred0={predict_zero_loss:.4f}  "
              f"mean(y[on])={m['mean_on']:.3f}  mean|y[off]|={m['mean_abs_off']:.4f}",
              flush=True)
        del E, D, E_d, D_d
        if fit_device == "cuda":
            torch.cuda.empty_cache()

    # Soft gate: max over p of mean_on must be ≥ 0.15
    max_mean_on = max(per_p[pv]["mean_on"] for pv in args.ps)
    passed_gate = max_mean_on >= args.gate_min_mean_on
    # Pick p* by lowest train_loss
    p_star = min(args.ps, key=lambda pv: per_p[pv]["train_loss"])
    return per_p, passed_gate, p_star, max_mean_on


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--lambda_on", type=float, required=True,
                   help="initial λ — bumped 2× if soft gate fails (cap 8× initial)")
    p.add_argument("--ps", type=float, nargs="+",
                   default=[0.025, 0.05, 0.075, 0.1, 0.125, 0.15])
    p.add_argument("--sparsity", type=int, default=2)
    p.add_argument("--noise_variance", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--activation", choices=["tanh3", "jumprelu"], default="tanh3")
    p.add_argument("--scan_min", type=float, default=0.05,
                   help="(tanh3) c_in scan lower bound")
    p.add_argument("--scan_max", type=float, default=8.0,
                   help="(tanh3) c_in scan upper bound")
    p.add_argument("--scan_n", type=int, default=121,
                   help="(tanh3) c_in scan resolution")
    p.add_argument("--theta_min", type=float, default=0.0,
                   help="(jumprelu) θ scan lower bound")
    p.add_argument("--theta_max", type=float, default=4.0,
                   help="(jumprelu) θ scan upper bound")
    p.add_argument("--theta_n", type=int, default=41,
                   help="(jumprelu) θ scan resolution")
    p.add_argument("--gate_min_mean_on", type=float, default=0.15,
                   help="soft gate: max_p mean(y[on]) must be ≥ this")
    p.add_argument("--lambda_max_mult", type=float, default=8.0,
                   help="cap λ at this multiple of the starting value")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fit_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"d={args.d} H={args.H} d/H={args.d/args.H:.1f}  ps={args.ps}  "
          f"lambda_on(start)={args.lambda_on}  gate≥{args.gate_min_mean_on}  "
          f"device={fit_device}", flush=True)

    # Sample val batch once. Same data is reused across λ retries.
    torch.manual_seed(args.seed)
    stream = TwoHotStream(DataConfig(
        d=args.d, sparsity=args.sparsity, noise_variance=args.noise_variance,
        seed=args.seed, device="cpu"))
    x_noisy_cpu, x_clean_cpu = stream.sample_batch(args.n_eval)
    x_noisy = x_noisy_cpu.to(fit_device)
    x_clean = x_clean_cpu.to(fit_device)

    lambda_start = float(args.lambda_on)
    lambda_cap = lambda_start * args.lambda_max_mult
    lambda_now = lambda_start
    history = []
    final = None

    while True:
        print(f"=== λ = {lambda_now:g} ===", flush=True)
        per_p, passed, p_star, max_mean_on = run_for_lambda(
            args, lambda_now, x_clean, x_noisy, fit_device)
        history.append({"lambda_on": lambda_now,
                        "max_mean_on": max_mean_on,
                        "passed_gate": passed,
                        "p_star": p_star,
                        "train_loss_star": per_p[p_star]["train_loss"]})
        if passed:
            print(f"  PASSED gate (max mean_on={max_mean_on:.3f} ≥ "
                  f"{args.gate_min_mean_on}). p*={p_star}.", flush=True)
            final = (lambda_now, per_p, p_star)
            break
        next_lambda = lambda_now * 2.0
        if next_lambda > lambda_cap + 1e-9:
            print(f"  FAILED gate at λ={lambda_now:g} "
                  f"(max mean_on={max_mean_on:.3f}). Cap at {lambda_cap:g} "
                  f"reached — abandoning.", flush=True)
            final = (lambda_now, per_p, p_star)  # save best-effort selection
            break
        print(f"  failed gate (max mean_on={max_mean_on:.3f}). "
              f"Bumping λ → {next_lambda:g}", flush=True)
        lambda_now = next_lambda

    lambda_final, per_p_final, p_star = final
    last_passed = history[-1]["passed_gate"]

    save_dict = {
        "ps": np.array(args.ps),
        "activation": np.array(args.activation),
        "lambda_on_start": np.array(lambda_start),
        "lambda_on_final": np.array(lambda_final),
        "passed_gate": np.array(last_passed),
        "p_star": np.array(p_star),
        "c_in_star": np.array(per_p_final[p_star]["c_in"]),
        "c_out_star": np.array(per_p_final[p_star]["c_out"]),
        "theta_star": np.array(per_p_final[p_star]["theta"]),
        "train_loss_star": np.array(per_p_final[p_star]["train_loss"]),
        "mean_on_star": np.array(per_p_final[p_star]["mean_on"]),
    }
    for k in ("c_in", "c_out", "theta", "train_loss", "mean_on", "mean_abs_off",
              "mse_on", "mse_off", "mse_total", "predict_zero_loss"):
        save_dict[k] = np.array([per_p_final[pv][k] for pv in args.ps])

    suffix = "" if args.activation == "tanh3" else f"_{args.activation}"
    out_npz = os.path.join(args.out_dir,
                           f"phase2{suffix}_d{args.d}_H{args.H}.npz")
    np.savez(out_npz, **save_dict)
    print(f"saved {out_npz}", flush=True)

    # Also drop a small JSON report for human eyes.
    report = {
        "d": args.d, "H": args.H, "ratio": args.d / args.H,
        "activation": args.activation,
        "lambda_on_start": lambda_start, "lambda_on_final": lambda_final,
        "passed_gate": bool(last_passed),
        "gate_min_mean_on": args.gate_min_mean_on,
        "p_star": float(p_star),
        "selected": per_p_final[p_star],
        "per_p": {f"{pv:g}": per_p_final[pv] for pv in args.ps},
        "lambda_history": history,
    }
    out_json = os.path.join(args.out_dir,
                            f"phase2{suffix}_d{args.d}_H{args.H}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
