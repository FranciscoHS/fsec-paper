"""Exponential-map perturbation primitives.

Chart label: exp_map_v1.

For an anchor activation a in R^D at perturbation layer L, with R = ||a||
and a_hat = a/R, and one or two contrastive directions d1, (d2):

  1. Project to the tangent plane at a_hat and orthonormalize.
  2. Parametrize a perturbation by tangent coords (alpha_1, alpha_2) in
     radians. The perturbed activation is

         v   = alpha_1 d1_perp + alpha_2 d2_perp
         r   = ||v||
         a'  = R (cos r * a_hat + sin r * v / r)     if r > eps_r
         a'  = R * a_hat                              if r <= eps_r

Properties (in the r > 0 branch, when d1_perp and d2_perp are orthonormal):
  - ||a'|| = R exactly.
  - arccos(a_hat . a' / R) = r exactly.
  - direction in chart (alpha_1, alpha_2)/r equals tangent direction v/r.

Sign convention: directions have no canonical sign. The caller fixes the
sign at extraction time and must record direction_signs in the output PKL
header. The chart is invariant under simultaneous (alpha_i -> -alpha_i,
d_i -> -d_i), so superellipse-p fits are sign-agnostic.
"""
from __future__ import annotations

import torch

CHART = "exp_map_v1"
EPS = 1e-6
EPS_R = 1e-12


class DegenerateDirection(ValueError):
    pass


def orthonormalize(
    a_hat: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor | None = None,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Project d1 (and optionally d2) onto the tangent plane at a_hat and
    orthonormalize via Gram-Schmidt. Returns (d1_perp, d2_perp).

    Raises DegenerateDirection if any direction is collinear with a_hat
    or with the previous direction (norm < eps * original norm).
    """
    n1 = d1.norm()
    d1p = d1 - (d1 @ a_hat) * a_hat
    if d1p.norm() <= eps * n1:
        raise DegenerateDirection("d1 is collinear with a_hat")
    d1p = d1p / d1p.norm()

    if d2 is None:
        return d1p, None

    n2 = d2.norm()
    d2p = d2 - (d2 @ a_hat) * a_hat
    d2p = d2p - (d2p @ d1p) * d1p
    if d2p.norm() <= eps * n2:
        raise DegenerateDirection("d2 is collinear with a_hat or d1")
    d2p = d2p / d2p.norm()
    return d1p, d2p


def exp_map_2d(
    a: torch.Tensor,
    d1_perp: torch.Tensor,
    d2_perp: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    eps_r: float = EPS_R,
) -> torch.Tensor:
    """Apply exp-map perturbation. d1_perp, d2_perp must already be
    orthonormal tangent vectors at a_hat = a/||a|| (use orthonormalize).

    alpha1, alpha2 may be scalars or any broadcastable shape S. Returns
    new_a of shape S + (D,).
    """
    R = a.norm()
    a_hat = a / R
    alpha1 = torch.as_tensor(alpha1, dtype=a.dtype, device=a.device)
    alpha2 = torch.as_tensor(alpha2, dtype=a.dtype, device=a.device)
    r = torch.sqrt(alpha1 * alpha1 + alpha2 * alpha2)
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    # safe sin_r / r without dividing by 0: where r is tiny, return R*a_hat.
    # We compute (sin_r / r) * (alpha1 d1 + alpha2 d2) and clamp r away
    # from zero, then mask afterwards.
    r_safe = torch.where(r > eps_r, r, torch.ones_like(r))
    coeff = sin_r / r_safe  # shape S
    v_dir = (
        coeff.unsqueeze(-1) * (alpha1.unsqueeze(-1) * d1_perp
                               + alpha2.unsqueeze(-1) * d2_perp)
    )
    out = R * (cos_r.unsqueeze(-1) * a_hat + v_dir)
    # zero-perturbation branch: r <= eps_r -> exactly R * a_hat
    mask = (r <= eps_r).unsqueeze(-1)
    out = torch.where(mask, (R * a_hat).expand_as(out), out)
    return out


def exp_map_1d(
    a: torch.Tensor,
    d_perp: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Single-direction rotation. d_perp must already be a unit tangent
    vector at a_hat. alpha is a scalar or any shape S; returns S + (D,).

    new_a = R * (cos alpha * a_hat + sin alpha * d_perp).
    """
    R = a.norm()
    a_hat = a / R
    alpha = torch.as_tensor(alpha, dtype=a.dtype, device=a.device)
    cos_a = torch.cos(alpha).unsqueeze(-1)
    sin_a = torch.sin(alpha).unsqueeze(-1)
    return R * (cos_a * a_hat + sin_a * d_perp)


def composite_direction(
    a_hat: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    eps: float = EPS,
) -> dict:
    """Build the composite direction for compositional steering.

    Returns a dict with both the orthogonalized sum (canonical, decision D6)
    and the raw sum (recorded for revisit-ability). Both are unit vectors.

    The orthogonalized sum is normalized in the tangent plane.
    """
    d1p, d2p = orthonormalize(a_hat, d1, d2, eps=eps)
    raw = d1 + d2
    raw_norm = raw.norm()
    if raw_norm <= eps:
        raise DegenerateDirection("raw composite has zero norm")
    raw_unit = raw / raw_norm
    ortho = d1p + d2p
    ortho_norm = ortho.norm()
    if ortho_norm <= eps:
        raise DegenerateDirection("orthogonalized composite has zero norm")
    ortho_unit = ortho / ortho_norm
    return {"ortho": ortho_unit, "raw": raw_unit, "d1_perp": d1p, "d2_perp": d2p}


def chart_distance(a_hat: torch.Tensor, new_a: torch.Tensor) -> torch.Tensor:
    """Great-circle angle between a_hat and new_a/||new_a||, in radians."""
    R = new_a.norm(dim=-1, keepdim=True)
    n_hat = new_a / R
    cos_theta = (n_hat * a_hat).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.arccos(cos_theta)


def project_to_chart(
    a: torch.Tensor,
    d1_perp: torch.Tensor,
    d2_perp: torch.Tensor,
    new_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover (alpha1, alpha2) tangent coords from a perturbed activation,
    assuming new_a was produced by exp_map_2d with the same (a, d1_perp,
    d2_perp). Used by the round-trip validation test.

    Returns (alpha1, alpha2) of shape new_a.shape[:-1].
    """
    R = a.norm()
    a_hat = a / R
    n_hat = new_a / new_a.norm(dim=-1, keepdim=True)
    cos_r = (n_hat * a_hat).sum(dim=-1).clamp(-1.0, 1.0)
    r = torch.arccos(cos_r)
    # tangential component of n_hat
    tangential = n_hat - cos_r.unsqueeze(-1) * a_hat
    t_norm = tangential.norm(dim=-1, keepdim=True)
    # avoid 0/0 where r == 0
    safe = t_norm.clamp(min=EPS_R)
    v_dir = tangential / safe
    # alpha_i = r * (v_dir . d_i_perp)
    a1 = r * (v_dir * d1_perp).sum(dim=-1)
    a2 = r * (v_dir * d2_perp).sum(dim=-1)
    # zero-r branch: alphas are 0
    zero = (r <= EPS_R)
    a1 = torch.where(zero, torch.zeros_like(a1), a1)
    a2 = torch.where(zero, torch.zeros_like(a2), a2)
    return a1, a2


def eps_thresholds() -> dict:
    """Return the eps thresholds used by this module, for PKL header."""
    return {"eps": EPS, "eps_r": EPS_R, "chart": CHART}
