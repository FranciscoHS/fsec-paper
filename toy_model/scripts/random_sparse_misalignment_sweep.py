"""Phase 3 + Phase 4 — misalignment angle sweep & plot.

For each d in --ds, read phase2_d{d}_H{H}.npz (or accept CLI overrides) for the
selected (p*, c_in*, c_out*, λ_final). Then:

  1. Bake a TwoLayerNet at (c_in*·E, c_out*·D), tanh³.
  2. Compute peak_L²(feat) = max ‖f(x_clean + α·e_j) − f(x_clean)‖₂ along the
     first inactive-feature axis. Set τ = tau_frac · peak_L²(feat).
  3. Sample N inactive-feature pairs (j, k). Per pair, sample raw vectors
     (w_u_raw, w_v_raw); zero entries in S ∪ {j, k}; Gram–Schmidt to get a
     unit, orthogonal pair (w_u, w_v) in the subspace orthogonal to all of
     S, e_j, e_k.
  4. For each angle α in --angles_deg:
        u(α) = cos α · e_j + sin α · w_u
        v(α) = cos α · e_k + sin α · w_v
        v(α) ← v(α) − ⟨v(α), u(α)⟩ u(α);  v(α) ← v(α) / ‖v(α)‖
     Run boundary_radii on the orthonormal (u, v) at threshold τ over
     n_phis values of φ. Fit (x/r₀)^n + (y/r_{π/2})^n = 1.

Saves runs/<out_dir>/lp_vs_angle.npz and runs/<out_dir>/lp_vs_angle.png.
The plot has one line per d (median ± IQR shading), with a horizontal
reference at n = 2. Legend annotates each line with (λ, p*, mean(y[on])).

Usage:
    python -m scripts.random_sparse_misalignment_sweep \
        --ds 4096 8192 16384 --H 1024 \
        --phase2_dir runs/misalignment_sweep_H1024 \
        --out_dir   runs/misalignment_sweep_H1024
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from scripts.random_sparse_baseline import BaselineConfig, build_weights
from scripts.random_sparse_perturb import build_baked_model
from scripts.random_sparse_lp_beeswarm import fit_from_radii
from scripts.random_sparse_lp_p_sweep_loss_jumprelu import build_baked_jumprelu


# ---------- chunked, memory-safe boundary_radii (handles d=16384) -----------

@torch.no_grad()
def boundary_radii_chunked(model, d, S, u, v, phis, taus, alpha_max, n_alpha,
                           input_chunk_floats=200_000_000):
    """Same semantics as random_sparse_lp_beeswarm.boundary_radii, but chunks
    over phi BEFORE materializing the (n_phi, n_alpha, d) input tensor.

    The original boundary_radii materializes a (60, 2500, 16384) ≈ 9.8 GB
    fp32 input at d=16384, which is uncomfortable. We split it into phi
    chunks sized so that input_chunk_floats covers (chunk_phi · n_alpha · d).
    """
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

    directions = (torch.cos(phis_t).unsqueeze(1) * u.unsqueeze(0)
                  + torch.sin(phis_t).unsqueeze(1) * v.unsqueeze(0))  # (n_phi, d)

    phi_chunk = max(1, int(input_chunk_floats // max(n_alpha * d, 1)))
    l2 = torch.empty(n_phi, n_alpha, device=device)
    for s_phi in range(0, n_phi, phi_chunk):
        e_phi = min(s_phi + phi_chunk, n_phi)
        cp = e_phi - s_phi
        x = (x_clean.view(1, 1, d)
             + alphas.view(1, n_alpha, 1) * directions[s_phi:e_phi].view(cp, 1, d))
        out = model(x.view(cp * n_alpha, d)).view(cp, n_alpha, d)
        l2[s_phi:e_phi] = (out - base.view(1, 1, d)).norm(p=2, dim=2)

    alphas_cpu = alphas.detach().cpu().numpy()
    out_dict = {tau: np.full(n_phi, np.nan) for tau in taus}
    for tau in taus:
        above = (l2 > tau).int()
        any_above = above.any(dim=1)
        first_idx = above.argmax(dim=1)
        first_idx_cpu = first_idx.detach().cpu().numpy()
        any_above_cpu = any_above.detach().cpu().numpy()
        out_dict[tau][any_above_cpu] = alphas_cpu[first_idx_cpu[any_above_cpu]]
    return out_dict


# ---------- helpers ---------------------------------------------------------

@torch.no_grad()
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


def sample_pair_random_directions(rng, d, S, j, k, no_feature_orth: bool = False):
    """Return unit vectors (w_u, w_v) in R^d, mutually orthogonal.

    Default: also orthogonal to S, e_j, e_k (sample in the
    (d − |S| − 2)-dim subspace). With ``no_feature_orth=True``, drop
    that constraint and sample fully isotropic Gaussians in R^d (still
    Gram-Schmidted between u and v); the random direction can then have
    nonzero components along feature axes."""
    if no_feature_orth:
        forbidden_idx = []
    else:
        forbidden_idx = sorted(set(S) | {int(j), int(k)})
    while True:
        w_u = rng.standard_normal(d).astype(np.float32)
        if forbidden_idx: w_u[forbidden_idx] = 0.0
        n_u = float(np.linalg.norm(w_u))
        if n_u > 1e-12:
            break
    w_u /= n_u
    while True:
        w_v = rng.standard_normal(d).astype(np.float32)
        if forbidden_idx: w_v[forbidden_idx] = 0.0
        w_v -= float(np.dot(w_v, w_u)) * w_u
        n_v = float(np.linalg.norm(w_v))
        if n_v > 1e-12:
            break
    w_v /= n_v
    return w_u, w_v


def build_uv_at_angle(d, j, k, w_u, w_v, alpha, device):
    """Return (u(α), v(α)) as torch.float32 tensors on device, orthonormal."""
    e_j = torch.zeros(d, device=device); e_j[int(j)] = 1.0
    e_k = torch.zeros(d, device=device); e_k[int(k)] = 1.0
    w_u_t = torch.from_numpy(w_u).to(device)
    w_v_t = torch.from_numpy(w_v).to(device)
    ca = float(math.cos(alpha))
    sa = float(math.sin(alpha))
    u = ca * e_j + sa * w_u_t                      # already unit (orthogonal terms)
    v = ca * e_k + sa * w_v_t                      # already unit
    # Numerical-safety renormalize and orthogonalize v against u
    u = u / u.norm().clamp_min(1e-12)
    v = v - (v @ u) * u
    v = v / v.norm().clamp_min(1e-12)
    return u, v


# ---------- per-d experiment ------------------------------------------------

def run_one_d(d, H, phase2_npz, args):
    """Returns dict with 'angles', 'ns' (n_angles, n_pairs), plus metadata."""
    if phase2_npz is None:
        # CLI override path
        p_val = args.cli_p
        c_in = args.cli_c_in
        c_out = args.cli_c_out
        theta = args.cli_theta if args.cli_theta is not None else 0.0
        lambda_on = args.cli_lambda_on
        activation = args.activation
        mean_on = float("nan")
        train_loss = float("nan")
        passed_gate = None
    else:
        z = np.load(phase2_npz, allow_pickle=False)
        p_val = float(z["p_star"])
        c_in = float(z["c_in_star"])
        c_out = float(z["c_out_star"])
        theta = float(z["theta_star"]) if "theta_star" in z.files else 0.0
        lambda_on = float(z["lambda_on_final"])
        mean_on = float(z["mean_on_star"])
        train_loss = float(z["train_loss_star"])
        passed_gate = bool(z["passed_gate"])
        activation = (str(z["activation"]) if "activation" in z.files
                      else "tanh3")

    if args.activation is not None and args.activation != activation:
        raise SystemExit(f"--activation={args.activation} but Phase 2 NPZ was "
                         f"saved with activation={activation}")

    scalar_str = (f"c_in*={c_in:.4f}" if activation == "tanh3"
                  else f"theta*={theta:.4f}")
    print(f"\n### d={d} H={H} (d/H={d/H:.1f})  act={activation}  "
          f"p*={p_val}  {scalar_str}  c_out*={c_out:.4f}  "
          f"λ={lambda_on}  mean(y[on])*={mean_on:.3f}  "
          f"passed={passed_gate}", flush=True)

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
    device = next(model.parameters()).device

    S = tuple(range(cfg.sparsity))
    inactive_first = next(i for i in range(cfg.d) if i not in S)
    peak = peak_l2_along_feature(model, cfg.d, S, inactive_first,
                                 args.alpha_max, args.n_alpha)
    if args.taus_abs is not None:
        taus = [float(t) for t in args.taus_abs]
        tau_fracs = [t / peak for t in taus]    # for legend only
        print(f"  peak_L2(feat axis e_{inactive_first}) = {peak:.4f}  "
              f"(τ values are absolute, not fractions of peak)", flush=True)
        for tau, frac in zip(taus, tau_fracs):
            print(f"    τ = {tau:.4f}  ({frac*100:.2f}% of peak)", flush=True)
    else:
        tau_fracs = list(args.tau_fracs)
        taus = [float(frac) * peak for frac in tau_fracs]
        print(f"  peak_L2(feat axis e_{inactive_first}) = {peak:.4f}", flush=True)
        for frac, tau in zip(tau_fracs, taus):
            print(f"    τ_frac={frac:g}  →  τ = {tau:.4f}", flush=True)

    # Sample N feature pairs + per-pair raw (w_u, w_v).
    rng = np.random.default_rng(args.seed)
    inactive = [i for i in range(cfg.d) if i not in S]
    feat_pairs = set()
    while len(feat_pairs) < args.n_pairs:
        a, b = rng.choice(inactive, size=2, replace=False)
        feat_pairs.add((int(min(a, b)), int(max(a, b))))
    feat_pairs = sorted(feat_pairs)

    pair_dirs = []
    for (j, k) in feat_pairs:
        w_u, w_v = sample_pair_random_directions(
            rng, cfg.d, S, j, k, no_feature_orth=args.no_feature_orth)
        pair_dirs.append((j, k, w_u, w_v))

    angles = np.array(args.angles, dtype=np.float64)
    n_angles = angles.size
    n_taus = len(tau_fracs)
    phis = np.linspace(0.0, math.pi / 2, args.n_phis)

    # Shape (n_taus, n_angles, n_pairs). All τ are computed in one call per
    # (pair, angle) — boundary_radii_chunked supports multi-τ for free.
    ns = np.full((n_taus, n_angles, args.n_pairs), np.nan, dtype=np.float64)
    # Realized cosine of u against e_j and v against e_k after build_uv_at_angle
    # (post-renormalization). With the default feature-axis orthogonalization
    # these recover cos(α) exactly; with --no_feature_orth they fluctuate by
    # ~1/sqrt(d) per pair around cos(α), so we record them per-pair so the
    # plot can use the empirical cosine as its x-axis.
    cos_uj = np.full((n_angles, args.n_pairs), np.nan, dtype=np.float64)
    cos_vk = np.full((n_angles, args.n_pairs), np.nan, dtype=np.float64)
    for ai, alpha in enumerate(angles):
        for pi, (j, k, w_u, w_v) in enumerate(pair_dirs):
            u, v = build_uv_at_angle(cfg.d, j, k, w_u, w_v, float(alpha), device)
            cos_uj[ai, pi] = float(u[int(j)].item())
            cos_vk[ai, pi] = float(v[int(k)].item())
            rs_dict = boundary_radii_chunked(
                model, cfg.d, S, u, v, phis, taus,
                args.alpha_max, args.n_alpha)
            for ti, tau in enumerate(taus):
                ns[ti, ai, pi] = fit_from_radii(rs_dict[tau], phis)
        meds = " ".join(f"τ={taus[ti]:.3g}:n={np.nanmedian(ns[ti, ai]):.3f}"
                        for ti in range(n_taus))
        print(f"  α = {math.degrees(alpha):5.1f}°  median n: {meds}  "
              f"median |cos(u,e_j)| = "
              f"{float(np.nanmedian(np.abs(cos_uj[ai]))):.4f}",
              flush=True)

    # Free GPU memory before the next d.
    del model, E, D
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "d": d, "H": H, "activation": activation,
        "p_star": p_val, "c_in": c_in, "c_out": c_out, "theta": theta,
        "lambda_on": lambda_on, "mean_on": mean_on, "train_loss": train_loss,
        "passed_gate": passed_gate,
        "peak_l2_feat": peak,
        "tau_fracs": np.array(tau_fracs), "taus": np.array(taus),
        "angles": angles, "ns": ns,
        "cos_uj": cos_uj, "cos_vk": cos_vk,
    }


# ---------- plotting --------------------------------------------------------

def plot_lp_vs_angle(results, out_png, H, title_suffix=""):
    """One subplot per τ_frac, 3 lines per panel (one per d/H)."""
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    activations = {r["activation"] for r in results}
    act_label = (next(iter(activations)) if len(activations) == 1
                 else "+".join(sorted(activations)))

    # Assume all results share the same τ list (set at CLI time).
    tau_fracs = list(results[0]["tau_fracs"])
    n_taus = len(tau_fracs)
    cols = min(n_taus, 3)
    rows = (n_taus + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.6 * rows),
                             squeeze=False, sharex=True, sharey=False)

    xt = [0, math.pi / 12, math.pi / 6, math.pi / 4, math.pi / 3,
          5 * math.pi / 12, math.pi / 2]
    xt_labels = ["0", "π/12", "π/6", "π/4", "π/3", "5π/12", "π/2"]

    for ti, frac in enumerate(tau_fracs):
        ax = axes[ti // cols][ti % cols]
        for ix, r in enumerate(results):
            color = palette[ix % len(palette)]
            ratio = r["d"] / H
            ns_t = r["ns"][ti]  # (n_angles, n_pairs)
            med = np.array([np.nanmedian(ns_t[ai]) for ai in range(ns_t.shape[0])])
            q25 = np.array([np.nanpercentile(ns_t[ai], 25)
                            for ai in range(ns_t.shape[0])])
            q75 = np.array([np.nanpercentile(ns_t[ai], 75)
                            for ai in range(ns_t.shape[0])])
            if r["activation"] == "tanh3":
                scalar = f"c_in={r['c_in']:.2f}"
            else:
                scalar = f"θ={r['theta']:.2f}"
            # Build legend entry only on the first subplot to keep things tidy.
            label = (f"{ratio:.0f}× (d={r['d']}, λ={r['lambda_on']:g}, "
                     f"p={r['p_star']:g}, {scalar}, "
                     f"mean(y[on])={r['mean_on']:.2f})") if ti == 0 else None
            ax.fill_between(r["angles"], q25, q75, color=color, alpha=0.18,
                            linewidth=0)
            ax.plot(r["angles"], med, color=color, marker="o", linewidth=1.6,
                    label=label)
        ax.axhline(2, color="k", ls="--", alpha=0.5,
                   label=("L²" if ti == 0 else None))
        ax.set_xticks(xt); ax.set_xticklabels(xt_labels, fontsize=8)
        ax.grid(alpha=0.3)
        # Show the absolute τ for each d (likely shared across d, but list it
        # explicitly when not). frac is "fraction of d=first_d's peak L²".
        taus_at_this_panel = [float(r["taus"][ti]) for r in results]
        if all(np.isclose(t, taus_at_this_panel[0], rtol=0.005)
               for t in taus_at_this_panel):
            tau_str = f"τ = {taus_at_this_panel[0]:.3g}"
        else:
            tau_str = "τ = " + ",".join(f"{t:.2g}" for t in taus_at_this_panel)
        ax.set_title(f"{tau_str}  (≈{frac*100:.1f}% peak)", fontsize=9)
        if ti % cols == 0:
            ax.set_ylabel("Lp exponent n  (median, IQR shaded)")
        if ti // cols == rows - 1:
            ax.set_xlabel("misalignment angle  α")

    # Hide any unused panels in the last row.
    for ti in range(n_taus, rows * cols):
        axes[ti // cols][ti % cols].set_visible(False)

    fig.suptitle(f"Lp exponent vs feature-axis misalignment, τ-sweep  "
                 f"(random sparse tied {act_label}, H={H}){title_suffix}",
                 fontsize=11)
    if any(ax.get_legend_handles_labels()[1] for row in axes for ax in row):
        axes[0][0].legend(loc="best", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}", flush=True)


# ---------- main ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", type=int, nargs="+", required=True)
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--phase2_dir", type=str, default=None,
                   help="dir containing phase2_d{d}_H{H}.npz; required unless "
                        "all CLI overrides are set (single d only).")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--activation", choices=["tanh3", "jumprelu"], default=None,
                   help="if set, must match the Phase 2 NPZ; selects which "
                        "phase2[_act]_d{d}_H{H}.npz file to load. Default: "
                        "tanh3.")
    p.add_argument("--sparsity", type=int, default=2)
    p.add_argument("--noise_variance", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--n_pairs", type=int, default=80)
    p.add_argument("--n_phis", type=int, default=60)
    p.add_argument("--alpha_max", type=float, default=25.0)
    p.add_argument("--n_alpha", type=int, default=2500)
    p.add_argument("--tau_fracs", type=float, nargs="+", default=[0.5],
                   help="output L² thresholds, as fractions of peak_L²(feat axis). "
                        "Pass several to sweep — boundary radii for all τ are "
                        "computed in one forward pass per pair.")
    p.add_argument("--taus_abs", type=float, nargs="+", default=None,
                   help="absolute output L² thresholds; overrides --tau_fracs "
                        "when set. Useful for probing small τ where the "
                        "boundary lives near the activation gate.")
    p.add_argument("--angles", type=float, nargs="+",
                   default=[0.0,
                            math.pi / 12, math.pi / 6, math.pi / 4,
                            math.pi / 3, 5 * math.pi / 12, math.pi / 2],
                   help="misalignment angles α in radians (default: 7-pt sweep)")
    # CLI overrides for a single d, when phase2 is unavailable
    p.add_argument("--cli_p", type=float, default=None)
    p.add_argument("--cli_c_in", type=float, default=None)
    p.add_argument("--cli_c_out", type=float, default=None)
    p.add_argument("--cli_theta", type=float, default=None)
    p.add_argument("--cli_lambda_on", type=float, default=None)
    p.add_argument("--no_feature_orth", action="store_true",
                   help="don't orthogonalize the random direction against "
                        "the feature axes — sample (w_u, w_v) as fully "
                        "isotropic Gaussians in R^d (still Gram-Schmidted "
                        "between u and v). Default keeps the original "
                        "behaviour (orthogonal to S and (j,k)).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Activation suffix appears on BOTH the phase2 input filenames and the
    # output filenames. The no_feature_orth flag only changes the random-
    # direction sampling and does not require its own phase2 cache, so it
    # is appended to the output suffix only.
    suffix = "" if (args.activation is None or args.activation == "tanh3") \
        else f"_{args.activation}"
    out_suffix = suffix + ("_no_feature_orth" if args.no_feature_orth else "")

    using_overrides = all(v is not None for v in
                          (args.cli_p, args.cli_c_in, args.cli_c_out,
                           args.cli_lambda_on))
    if not using_overrides and args.phase2_dir is None:
        raise SystemExit("Either --phase2_dir or all CLI overrides "
                         "(--cli_p, --cli_c_in, --cli_c_out, --cli_lambda_on) "
                         "must be provided.")
    if using_overrides and len(args.ds) != 1:
        raise SystemExit("CLI overrides only support a single --ds value.")
    if using_overrides and args.activation is None:
        args.activation = "tanh3"

    results = []
    for d in args.ds:
        if using_overrides:
            phase2_npz = None
        else:
            phase2_npz = os.path.join(args.phase2_dir,
                                      f"phase2{suffix}_d{d}_H{args.H}.npz")
            if not os.path.exists(phase2_npz):
                raise SystemExit(f"missing {phase2_npz} — run Phase 2 first")
        results.append(run_one_d(d, args.H, phase2_npz, args))

    save = {"angles": np.array(args.angles),
            "H": np.array(args.H),
            "ds": np.array(args.ds),
            "tau_fracs": np.array(args.tau_fracs),
            "n_pairs": np.array(args.n_pairs),
            "n_phis": np.array(args.n_phis)}
    for r in results:
        d = r["d"]
        save[f"ns_d{d}"] = r["ns"]                # (n_taus, n_angles, n_pairs)
        save[f"taus_d{d}"] = r["taus"]            # (n_taus,)
        save[f"cos_uj_d{d}"] = r["cos_uj"]        # (n_angles, n_pairs)
        save[f"cos_vk_d{d}"] = r["cos_vk"]        # (n_angles, n_pairs)
        save[f"meta_d{d}"] = np.array(json.dumps({
            "d": r["d"], "H": r["H"], "activation": r["activation"],
            "p_star": r["p_star"],
            "c_in": r["c_in"], "c_out": r["c_out"], "theta": r["theta"],
            "lambda_on": r["lambda_on"], "mean_on": r["mean_on"],
            "train_loss": r["train_loss"],
            "passed_gate": r["passed_gate"],
            "peak_l2_feat": r["peak_l2_feat"],
            "tau_fracs": list(r["tau_fracs"]),
            "taus": list(r["taus"]),
        }))
    out_npz = os.path.join(args.out_dir, f"lp_vs_angle{out_suffix}.npz")
    np.savez(out_npz, **save)
    print(f"saved {out_npz}", flush=True)

    out_png = os.path.join(args.out_dir, f"lp_vs_angle{out_suffix}.png")
    plot_lp_vs_angle(results, out_png, args.H)


if __name__ == "__main__":
    main()
