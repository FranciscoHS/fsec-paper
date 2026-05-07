"""
Phase 3 frontier probing.

Given a trained denoiser ``model`` whose dictionary axes are the d coordinates,
we ask: what region of input space (around a clean signal x_clean) causes a
spurious off-feature j to "turn on" in the output?

Active set S indexes the coords of x_clean that are 1 (the two-hot signal).
We perturb x_clean along the standard basis vectors e_j, e_k for off-features
j, k ∉ S and inspect the output coordinates out[j], out[k].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch


# ---------- core helpers ------------------------------------------------------------

@torch.no_grad()
def response(model, x_clean: torch.Tensor, perturbation: torch.Tensor) -> torch.Tensor:
    """Return ``model(x_clean + perturbation)`` (no grad). Auto-batches."""
    model.eval()
    device = next(model.parameters()).device
    x = (x_clean + perturbation).to(device)
    return model(x).detach().cpu()


def is_nontrivial(out: torch.Tensor, j: int, threshold: float = 0.5) -> torch.Tensor:
    """Boolean: did out[j] (per-sample) cross ``threshold``?"""
    if out.dim() == 1:
        return out[j] > threshold
    return out[..., j] > threshold


def make_clean(d: int, S: Sequence[int], device=None) -> torch.Tensor:
    """Two-hot clean signal on coords S."""
    x = torch.zeros(d, device=device)
    for i in S:
        x[i] = 1.0
    return x


def basis_perturb(d: int, j: int, alpha: float, device=None) -> torch.Tensor:
    e = torch.zeros(d, device=device)
    e[j] = alpha
    return e


# ---------- 1D sweep ----------------------------------------------------------------

@dataclass
class Sweep1DResult:
    alphas: np.ndarray
    out_j: np.ndarray
    j: int
    S: Tuple[int, ...]
    threshold_crossings: dict  # τ → α* (first α ≥ 0 where out_j > τ); None if never

    def write_summary(self) -> str:
        lines = [f"1D sweep S={self.S} j={self.j}"]
        for τ, α in self.threshold_crossings.items():
            lines.append(f"  τ={τ}: α* = {α if α is None else f'{α:.3f}'}")
        return "\n".join(lines)


@torch.no_grad()
def sweep_1d(
    model,
    S: Sequence[int],
    j: int,
    alpha_min: float = -3.0,
    alpha_max: float = 3.0,
    n_points: int = 200,
    thresholds: Sequence[float] = (0.1, 0.3, 0.5, 0.7),
    d: Optional[int] = None,
) -> Sweep1DResult:
    model.eval()
    device = next(model.parameters()).device
    if d is None:
        d = model.fc1.in_features

    alphas = torch.linspace(alpha_min, alpha_max, n_points, device=device)
    x_clean = make_clean(d, S, device=device).expand(n_points, d).clone()
    e_j = torch.zeros_like(x_clean)
    e_j[:, j] = alphas
    x = x_clean + e_j
    out = model(x).cpu()
    out_j = out[:, j].numpy()
    alphas_np = alphas.cpu().numpy()

    crossings: dict = {}
    for τ in thresholds:
        # smallest α ≥ 0 with out_j > τ; otherwise None
        pos_mask = (alphas_np >= 0) & (out_j > τ)
        crossings[float(τ)] = float(alphas_np[pos_mask][0]) if pos_mask.any() else None

    return Sweep1DResult(
        alphas=alphas_np,
        out_j=out_j,
        j=int(j),
        S=tuple(int(s) for s in S),
        threshold_crossings=crossings,
    )


def plot_sweep_1d(res: Sweep1DResult, save_path: Optional[str] = None, figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(res.alphas, res.out_j, color="black", linewidth=1.5)
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(res.threshold_crossings)))
    for (τ, α), c in zip(res.threshold_crossings.items(), colors):
        ax.axhline(τ, color=c, linestyle="--", alpha=0.7, label=f"τ={τ}")
        if α is not None:
            ax.axvline(α, color=c, linestyle=":", alpha=0.5)
    ax.set_xlabel(f"α (perturbation along e_{res.j})")
    ax.set_ylabel(f"out[{res.j}]")
    ax.set_title(f"1D sweep S={res.S} j={res.j}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig, ax


# ---------- 2D frontier -------------------------------------------------------------

@dataclass
class Sweep2DResult:
    alphas: np.ndarray         # (G,)
    betas: np.ndarray          # (G,)
    out_j: np.ndarray          # (G, G)
    out_k: np.ndarray          # (G, G)
    j: int
    k: int
    S: Tuple[int, ...]


@torch.no_grad()
def sweep_2d(
    model,
    S: Sequence[int],
    j: int,
    k: int,
    alpha_max: float = 2.0,
    n_grid: int = 100,
    d: Optional[int] = None,
    batch_size: int = 4096,
) -> Sweep2DResult:
    model.eval()
    device = next(model.parameters()).device
    if d is None:
        d = model.fc1.in_features

    alphas = torch.linspace(-alpha_max, alpha_max, n_grid, device=device)
    betas = torch.linspace(-alpha_max, alpha_max, n_grid, device=device)
    A, B = torch.meshgrid(alphas, betas, indexing="ij")
    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    N = A_flat.numel()

    x_clean = make_clean(d, S, device=device)
    out_j_all = torch.empty(N)
    out_k_all = torch.empty(N)

    for s in range(0, N, batch_size):
        e = batch_size
        sub_a = A_flat[s:s + e]
        sub_b = B_flat[s:s + e]
        nb = sub_a.numel()
        x = x_clean.unsqueeze(0).expand(nb, d).clone()
        x[:, j] = x[:, j] + sub_a
        x[:, k] = x[:, k] + sub_b
        out = model(x).cpu()
        out_j_all[s:s + nb] = out[:, j]
        out_k_all[s:s + nb] = out[:, k]

    return Sweep2DResult(
        alphas=alphas.cpu().numpy(),
        betas=betas.cpu().numpy(),
        out_j=out_j_all.numpy().reshape(n_grid, n_grid),
        out_k=out_k_all.numpy().reshape(n_grid, n_grid),
        j=int(j),
        k=int(k),
        S=tuple(int(s) for s in S),
    )


def _overlay_lp_curves(ax, alpha_star: float, alpha_max: float, colors=None):
    """Overlay L1 (diamond), L2 (circle), L∞ (square) at radius α*."""
    if alpha_star is None or alpha_star <= 0:
        return
    if colors is None:
        colors = {"L1": "tab:red", "L2": "tab:blue", "Linf": "tab:green"}
    th = np.linspace(0, 2 * np.pi, 400)
    # L2 circle
    ax.plot(alpha_star * np.cos(th), alpha_star * np.sin(th),
            color=colors["L2"], linestyle="-", linewidth=1.2, label="L2")
    # L1 diamond
    s = alpha_star
    ax.plot([s, 0, -s, 0, s], [0, s, 0, -s, 0],
            color=colors["L1"], linestyle="-", linewidth=1.2, label="L1")
    # L∞ square
    ax.plot([s, s, -s, -s, s], [s, -s, -s, s, s],
            color=colors["Linf"], linestyle="-", linewidth=1.2, label="L∞")


def plot_sweep_2d(
    res: Sweep2DResult,
    thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    alpha_star: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize=(15, 5),
):
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    extent = [res.betas.min(), res.betas.max(), res.alphas.min(), res.alphas.max()]

    # Heatmap out[j]
    ax = axes[0]
    im = ax.imshow(res.out_j, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r",
                   vmin=-1.5, vmax=1.5)
    ax.set_xlabel(f"β (along e_{res.k})")
    ax.set_ylabel(f"α (along e_{res.j})")
    ax.set_title(f"out[{res.j}]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Heatmap max(out[j], out[k])
    ax = axes[1]
    mx = np.maximum(res.out_j, res.out_k)
    im = ax.imshow(mx, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax.set_xlabel(f"β (along e_{res.k})")
    ax.set_ylabel(f"α (along e_{res.j})")
    ax.set_title(f"max(out[{res.j}], out[{res.k}])")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Frontier mask: union of {out[j] > τ} and {out[k] > τ} for each τ
    ax = axes[2]
    A, B = np.meshgrid(res.alphas, res.betas, indexing="ij")
    base = np.zeros_like(res.out_j)
    ax.imshow(base, origin="lower", extent=extent, aspect="auto",
              cmap="Greys", vmin=0, vmax=1)
    cmap_τ = plt.cm.plasma(np.linspace(0.15, 0.85, len(thresholds)))
    for τ, c in zip(thresholds, cmap_τ):
        mask = (res.out_j > τ) | (res.out_k > τ)
        ax.contour(B, A, mask.astype(float), levels=[0.5], colors=[c], linewidths=1.5)
        ax.plot([], [], color=c, label=f"τ={τ}")  # legend proxy
    if alpha_star is not None:
        _overlay_lp_curves(ax, alpha_star=alpha_star, alpha_max=res.alphas.max())
    ax.set_xlabel(f"β (along e_{res.k})")
    ax.set_ylabel(f"α (along e_{res.j})")
    ax.set_title(f"frontier {{out[{res.j}]>τ}} ∪ {{out[{res.k}]>τ}}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    fig.suptitle(f"S={res.S}  (j,k)=({res.j},{res.k})")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig, axes


# ---------- batch driver ------------------------------------------------------------

def run_robustness_sweep(
    model,
    pairs: Iterable[Tuple[int, int]],
    active_sets: Iterable[Sequence[int]],
    out_prefix: str,
    alpha_max: float = 2.0,
    n_grid: int = 80,
    thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    alpha_star_thresholds_for_1d: Sequence[float] = (0.1, 0.3, 0.5, 0.7),
) -> List[Tuple[Sweep1DResult, Sweep2DResult]]:
    """For each (S, (j, k)) run a 1D and 2D sweep, save plots."""
    results = []
    for S in active_sets:
        for (j, k) in pairs:
            tag = f"S{'-'.join(map(str, S))}_j{j}_k{k}"
            r1 = sweep_1d(model, S, j, thresholds=alpha_star_thresholds_for_1d)
            plot_sweep_1d(r1, save_path=f"{out_prefix}_{tag}_1d.png")
            print(r1.write_summary())
            α_star = r1.threshold_crossings.get(0.5)
            r2 = sweep_2d(model, S, j, k, alpha_max=alpha_max, n_grid=n_grid)
            plot_sweep_2d(r2, thresholds=thresholds, alpha_star=α_star,
                          save_path=f"{out_prefix}_{tag}_2d.png")
            results.append((r1, r2))
    return results
