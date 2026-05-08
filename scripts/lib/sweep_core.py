"""Shared sweep primitives in the exp-map chart.

Both sweep_1d and sweep_2d build perturbed activations with parametrize.exp_map_*
(or, in additive mode, with `a + sin(alpha) R d_perp`) and push them through
the model from perturb_layer to measure_layer. Returns L^2 (and optionally
1 - cos sim and KL) between perturbed and unperturbed measurement-layer
activations / final-layer logits.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
from src.model import (forward_from_layer_to_layer, forward_from_layer,
                       forward_from_layer_to_layer_at_pos)

from scripts.lib.parametrize import (
    orthonormalize, exp_map_2d, exp_map_1d,
)


def build_perturbed_pts_2d(a32, d1p, d2p, alphas_rad, mode="geodesic"):
    """Return (n*n, D) tensor of perturbed activations on device, fp32.

    geodesic: new_a = R*(cos r * a_hat + sin r * v / r)
    additive: new_a = a + sin(alpha_1) * R * d1_perp + sin(alpha_2) * R * d2_perp
    """
    R = a32.norm()
    a_hat = a32 / R
    n = len(alphas_rad)
    A1, A2 = torch.meshgrid(alphas_rad, alphas_rad, indexing="ij")
    if mode == "geodesic":
        return exp_map_2d(a32, d1p, d2p, A1.flatten(), A2.flatten())
    elif mode == "additive":
        s1 = torch.sin(A1).flatten().unsqueeze(-1)
        s2 = torch.sin(A2).flatten().unsqueeze(-1)
        return a32.unsqueeze(0) + R * (s1 * d1p + s2 * d2p)
    else:
        raise ValueError(f"unknown mode {mode}")


@torch.no_grad()
def sweep_1d_one_anchor(model, ctx, anchor, d, perturb_layer: int,
                        measure_layer: int, device, alphas_rad: np.ndarray,
                        batch_size: int = 64) -> np.ndarray:
    """L^2 distance at measure_layer for each alpha along direction d.

    ctx: (1, T-1, D) on device, model_dtype.
    anchor: (D,) float32 on device.
    d: (D,) float32 on device, not necessarily normalized or tangent.
    Returns (n_steps,) float32.
    """
    model_dtype = next(model.parameters()).dtype
    a32 = anchor.to(torch.float32)
    a_hat = a32 / a32.norm()
    d_perp, _ = orthonormalize(a_hat, d.to(torch.float32), None)
    alphas_t = torch.from_numpy(alphas_rad.astype(np.float32)).to(device)
    pts = exp_map_1d(a32, d_perp, alphas_t).to(model_dtype)
    n = pts.shape[0]
    ref = forward_from_layer_to_layer(
        model, ctx, a32[None, None].to(model_dtype),
        perturb_layer, measure_layer).float()
    out = np.zeros(n, dtype=np.float32)
    for st in range(0, n, batch_size):
        ed = min(st + batch_size, n)
        batch = pts[st:ed].unsqueeze(1)  # (B, 1, D)
        pf = forward_from_layer_to_layer(
            model, ctx, batch, perturb_layer, measure_layer).float()
        out[st:ed] = (pf - ref).norm(dim=-1).cpu().numpy()
    return out


@torch.no_grad()
def sweep_2d_one_anchor(model, ctx, anchor, d1, d2, perturb_layer: int,
                        measure_layer: int, device, alphas_rad: np.ndarray,
                        batch_size: int = 64,
                        mode: str = "geodesic",
                        record_cos: bool = False,
                        record_kl: bool = False,
                        single_batch_ref: bool = False,
                        pos: int = -1,
                        full_hidden=None) -> dict:
    """2D grid sweep in (alpha_1, alpha_2).

    Returns dict with at least 'l2': (n, n); optionally 'cos': 1 - cos sim
    of perturbed vs unperturbed measurement-layer activation, and 'kl':
    KL(p_ref || p_perturbed) of final-layer logits softmax (summed over
    vocab). grid[i, j] = metric at alpha_1 = alphas_rad[i], alpha_2 =
    alphas_rad[j].

    If single_batch_ref=True: run all n*n points in a single forward
    pass and use pf[0] (= alpha=(0,0), exactly the unperturbed anchor)
    as the reference within that batch. This eliminates the bf16
    batched-vs-unbatched matmul rounding bias that otherwise produces
    a ~few-units L^2 floor at alpha=(0,0). Requires enough VRAM to hold
    n*n inputs and intermediate activations; not currently supported
    with record_kl (which would need a separate full-model forward).

    Token-position ablation: pos != -1 routes through
    forward_from_layer_to_layer_at_pos with `full_hidden: (1, T, D)`
    (residual stream at perturb_layer, all positions) and `anchor` =
    activation at position `pos`. d1, d2 are still applied as
    perturbation directions in the tangent plane at anchor/||anchor||,
    and L^2 is read at position `pos`. record_kl and single_batch_ref
    are not supported when pos != -1.
    """
    model_dtype = next(model.parameters()).dtype
    a32 = anchor.to(torch.float32)
    a_hat = a32 / a32.norm()
    d1p, d2p = orthonormalize(a_hat, d1.to(torch.float32), d2.to(torch.float32))
    alphas_t = torch.from_numpy(alphas_rad.astype(np.float32)).to(device)
    pts32 = build_perturbed_pts_2d(a32, d1p, d2p, alphas_t, mode=mode)
    pts = pts32.to(model_dtype)
    n_total = pts.shape[0]
    n = len(alphas_rad)

    if pos != -1:
        if full_hidden is None:
            raise ValueError("full_hidden required when pos != -1")
        if record_kl:
            raise NotImplementedError(
                "record_kl not supported when pos != -1")
        if single_batch_ref:
            raise NotImplementedError(
                "single_batch_ref not supported when pos != -1")
        fh = full_hidden.to(device=device, dtype=model_dtype)
        # Reference: splice the anchor itself back at `pos`. Round-trip
        # equivalent to running the unperturbed full_hidden, and matches
        # the bf16 batched-matmul shape the perturbed forward sees.
        ref_act = forward_from_layer_to_layer_at_pos(
            model, fh, a32.to(model_dtype).unsqueeze(0),
            pos, perturb_layer, measure_layer).float()
        l2 = np.zeros(n_total, dtype=np.float32)
        cos = np.zeros(n_total, dtype=np.float32) if record_cos else None
        for st in range(0, n_total, batch_size):
            ed = min(st + batch_size, n_total)
            pf = forward_from_layer_to_layer_at_pos(
                model, fh, pts[st:ed],
                pos, perturb_layer, measure_layer).float()
            l2[st:ed] = (pf - ref_act).norm(dim=-1).cpu().numpy()
            if record_cos:
                cos[st:ed] = (1.0 - F.cosine_similarity(ref_act, pf, dim=-1)
                              ).cpu().numpy()
        out = {"l2": l2.reshape(n, n)}
        if record_cos:
            out["cos"] = cos.reshape(n, n)
        return out

    if single_batch_ref:
        if record_kl:
            raise NotImplementedError(
                "single_batch_ref + record_kl not supported; KL needs a "
                "full-model forward and we don't currently fit that into "
                "one batch.")
        # row-major: pts[0] corresponds to (alphas[0], alphas[0]) = (0,0),
        # which exp_map_2d returns as exactly the anchor.
        pf = forward_from_layer_to_layer(
            model, ctx, pts.unsqueeze(1),
            perturb_layer, measure_layer).float()
        ref = pf[0:1]
        l2 = (pf - ref).norm(dim=-1).cpu().numpy()
        out = {"l2": l2.reshape(n, n)}
        if record_cos:
            cos = (1.0 - F.cosine_similarity(ref, pf, dim=-1)
                   ).cpu().numpy()
            out["cos"] = cos.reshape(n, n)
        return out

    ref_act = forward_from_layer_to_layer(
        model, ctx, a32[None, None].to(model_dtype),
        perturb_layer, measure_layer).float()
    if record_kl:
        ref_logits = forward_from_layer(
            model, ctx, a32[None, None].to(model_dtype), perturb_layer).float()
        ref_logp = F.log_softmax(ref_logits, dim=-1)
    l2 = np.zeros(n_total, dtype=np.float32)
    cos = np.zeros(n_total, dtype=np.float32) if record_cos else None
    kl = np.zeros(n_total, dtype=np.float32) if record_kl else None
    for st in range(0, n_total, batch_size):
        ed = min(st + batch_size, n_total)
        batch = pts[st:ed].unsqueeze(1)
        pf = forward_from_layer_to_layer(
            model, ctx, batch, perturb_layer, measure_layer).float()
        l2[st:ed] = (pf - ref_act).norm(dim=-1).cpu().numpy()
        if record_cos:
            cos[st:ed] = (1.0 - F.cosine_similarity(ref_act, pf, dim=-1)
                          ).cpu().numpy()
        if record_kl:
            logits = forward_from_layer(
                model, ctx, batch, perturb_layer).float()
            logp = F.log_softmax(logits, dim=-1)
            kld = (ref_logp.exp() * (ref_logp - logp)).sum(dim=-1).squeeze()
            kl[st:ed] = kld.cpu().numpy()
    out = {"l2": l2.reshape(n, n)}
    if record_cos: out["cos"] = cos.reshape(n, n)
    if record_kl: out["kl"] = kl.reshape(n, n)
    return out


def run_1d_per_anchors(model, contexts, activations, d, perturb_layer,
                        measure_layer, device, alphas_rad) -> np.ndarray:
    """L2 grid (n_anchors, n_steps)."""
    model_dtype = next(model.parameters()).dtype
    out = np.zeros((len(activations), len(alphas_rad)), dtype=np.float32)
    for i in range(len(activations)):
        ctx = contexts[i].to(device=device, dtype=model_dtype)
        a = activations[i].to(device=device)
        d_dev = d.to(device=device, dtype=torch.float32)
        out[i] = sweep_1d_one_anchor(model, ctx, a, d_dev, perturb_layer,
                                     measure_layer, device, alphas_rad)
    return out


def run_2d_per_anchors(model, contexts, activations, d1, d2,
                        perturb_layer, measure_layer, device,
                        alphas_rad, mode: str = "geodesic",
                        record_cos: bool = False,
                        record_kl: bool = False,
                        single_batch_ref: bool = False,
                        pos: int = -1,
                        full_hidden_list=None) -> dict:
    """Per-metric grids of shape (n_anchors, n, n). Always includes 'l2';
    optionally 'cos' and 'kl' if requested.

    Token-position ablation: pos != -1 requires
    `full_hidden_list[i]: (1, T, D)` (residual stream at perturb_layer,
    all positions) and `activations[i]` should be the activation at
    position `pos`. `contexts` is unused in that case.
    """
    model_dtype = next(model.parameters()).dtype
    n = len(alphas_rad)
    out = {"l2": np.zeros((len(activations), n, n), dtype=np.float32)}
    if record_cos:
        out["cos"] = np.zeros_like(out["l2"])
    if record_kl:
        out["kl"] = np.zeros_like(out["l2"])
    if pos != -1 and full_hidden_list is None:
        raise ValueError("full_hidden_list required when pos != -1")
    for i in range(len(activations)):
        if pos != -1:
            ctx = None
            fh = full_hidden_list[i]
        else:
            ctx = contexts[i].to(device=device, dtype=model_dtype)
            fh = None
        a = activations[i].to(device=device)
        d1_dev = d1.to(device=device, dtype=torch.float32)
        d2_dev = d2.to(device=device, dtype=torch.float32)
        r = sweep_2d_one_anchor(model, ctx, a, d1_dev, d2_dev,
                                perturb_layer, measure_layer, device,
                                alphas_rad, mode=mode,
                                record_cos=record_cos, record_kl=record_kl,
                                single_batch_ref=single_batch_ref,
                                pos=pos, full_hidden=fh)
        for k in out:
            out[k][i] = r[k]
    return out
