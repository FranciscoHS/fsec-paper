"""Build a cache of 33 SAE-decoder directions at L=2 by ranking SAE
features by mean |activation| over a chosen prompt set.

Two ranking modes (``--rank_set``):

  - ``anchors``: rank features by mean ``|f(act_i)|`` across the 30
    evaluation anchor prompts (the same ones used in sweep_2d.py with
    ``--anchor_source fineweb`` at ``N_ANCHORS = 30``). Output:
    ``dirs_<target>_L<L>_sae_eval.pkl``  (family ``sae_eval``).

  - ``fineweb``: rank features by mean ``|f(act_i)|`` across an
    n-prompt FineWeb sample (default ``--n_fineweb 10000``). Output:
    ``dirs_<target>_L<L>_sae_fineweb.pkl``  (family ``sae_fineweb``).

In both cases each direction is the L2-normalised SAE decoder row
``W_dec[k]`` for the chosen latent index ``k``. The activation pass uses
the SAE encoder + JumpReLU threshold so the ranking matches what the SAE
itself considers "active" on the prompt distribution.

Loads the SAE directly from ``google/gemma-scope-9b-pt-res :: layer_<L>/
width_<W>/average_l0_<L0>/params.npz`` (same path as
``sae_directions.py``) so we don't depend on sae_lens / transformer_lens.

Usage:
  python -u scripts/sae_directions_top.py --rank_set anchors
  python -u scripts/sae_directions_top.py --rank_set fineweb \\
      --n_fineweb 10000
"""
from __future__ import annotations
import os, sys, pickle, argparse, time
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model
from scripts.lib import activations as actlib

OUT_DIR = "results/directions"
os.makedirs(OUT_DIR, exist_ok=True)

N_DIRS = 33
SEED = 42

# Canonical L0 mapping for gemma-scope-9b-pt-res. Kept in sync with
# sae_directions.py.
CANONICAL_L0 = {
    (2, "16k"): 67,
}


def ts(): return time.strftime("%H:%M:%S")


def _encode_chunked(activations: torch.Tensor,
                     W_enc: torch.Tensor, b_enc: torch.Tensor,
                     b_dec: torch.Tensor,
                     threshold: torch.Tensor | None,
                     device: torch.device,
                     batch_size: int = 256) -> torch.Tensor:
    """Encode (N, d_in) activations through the SAE in chunks. Returns
    the per-feature mean ``|activation|`` as a (d_sae,) tensor on CPU.

    Uses JumpReLU (``feature = pre * (pre > threshold)``) when
    ``threshold`` is provided; falls back to ReLU otherwise.
    """
    n = activations.shape[0]
    d_sae = W_enc.shape[1]
    sum_abs = torch.zeros(d_sae, dtype=torch.float64, device=device)
    for start in range(0, n, batch_size):
        chunk = activations[start:start + batch_size].to(device).float()
        pre = (chunk - b_dec.unsqueeze(0)) @ W_enc + b_enc.unsqueeze(0)
        if threshold is not None:
            feats = pre * (pre > threshold.unsqueeze(0)).float()
        else:
            feats = F.relu(pre)
        sum_abs += feats.abs().double().sum(dim=0)
    return (sum_abs / n).cpu().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank_set", required=True,
                    choices=["anchors", "fineweb"],
                    help="anchors: rank over the 30 evaluation anchors. "
                         "fineweb: rank over a 10k-prompt FineWeb sample.")
    ap.add_argument("--n", type=int, default=N_DIRS,
                    help="number of top features to keep")
    ap.add_argument("--n_fineweb", type=int, default=10000,
                    help="size of the FineWeb sample used for "
                         "rank_set=fineweb. Ignored for rank_set=anchors.")
    ap.add_argument("--n_anchors", type=int, default=actlib.N_ANCHORS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--repo",
                    default="google/gemma-scope-9b-pt-res")
    ap.add_argument("--l0", type=int, default=None,
                    help="average_l0 of the SAE variant; default = "
                         "canonical L0 from CANONICAL_L0[(layer, width)]")
    ap.add_argument("--model", default="gemma-2-9b",
                    help="HF model id used by src.model.load_model")
    ap.add_argument("--encode_batch", type=int, default=256)
    ap.add_argument("--fwd_batch", type=int, default=32,
                    help="forward batch size when extracting fineweb acts")
    args = ap.parse_args()

    l0 = args.l0
    if l0 is None:
        key = (args.layer, args.width)
        if key not in CANONICAL_L0:
            raise SystemExit(f"no canonical L0 mapped for {key}; pass --l0")
        l0 = CANONICAL_L0[key]

    rel_path = f"layer_{args.layer}/width_{args.width}/average_l0_{l0}/params.npz"
    print(f"[{ts()}] downloading {args.repo} :: {rel_path}", flush=True)
    from huggingface_hub import hf_hub_download
    fpath = hf_hub_download(repo_id=args.repo, filename=rel_path)
    print(f"  -> {fpath}", flush=True)

    npz = np.load(fpath)
    print(f"[{ts()}] keys: {list(npz.keys())}", flush=True)

    def _to_t(x): return torch.from_numpy(np.asarray(x).copy()).float()

    W_enc_np = npz["W_enc"]
    W_dec_np = npz["W_dec"]
    b_enc_np = npz["b_enc"]
    b_dec_np = npz["b_dec"]
    threshold_np = npz.get("threshold")
    if threshold_np is None:
        # Some Gemma Scope variants spell it differently; fall back to
        # any obvious matches before giving up to ReLU.
        for k in ("thresholds", "log_threshold", "jumprelu_threshold"):
            if k in npz.files:
                threshold_np = npz[k]
                print(f"  using threshold key '{k}'")
                break

    print(f"  W_enc: {W_enc_np.shape}  W_dec: {W_dec_np.shape}  "
          f"b_enc: {b_enc_np.shape}  b_dec: {b_dec_np.shape}  "
          f"threshold: "
          f"{None if threshold_np is None else threshold_np.shape}",
          flush=True)
    n_latents, d_model = W_dec_np.shape
    if W_enc_np.shape != (d_model, n_latents):
        raise SystemExit(f"unexpected W_enc shape {W_enc_np.shape}; "
                         f"expected {(d_model, n_latents)}")

    print(f"[{ts()}] loading {args.model}", flush=True)
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)

    if args.rank_set == "anchors":
        print(f"[{ts()}] loading {args.n_anchors} fineweb anchors "
              f"(seed={args.seed})", flush=True)
        fw = actlib.fineweb_acts(model, tokenizer, device, args.target,
                                  args.layer, n=args.n_anchors)
        acts = fw["activations"]
        rank_label = "sae_eval"
        rank_meta = {
            "rank_set": "anchors",
            "n_rank": int(acts.shape[0]),
            "anchor_seed": args.seed,
        }
    else:
        print(f"[{ts()}] loading {args.n_fineweb} fineweb activations "
              f"(seed={args.seed})", flush=True)
        fw = actlib.fineweb_acts_n(
            model, tokenizer, device, args.target, args.layer,
            n=args.n_fineweb, seed=args.seed, batch_size=args.fwd_batch)
        acts = fw["activations"]
        rank_label = "sae_fineweb"
        rank_meta = {
            "rank_set": "fineweb",
            "n_rank": int(acts.shape[0]),
            "fineweb_seed": args.seed,
        }
    print(f"  acts: {tuple(acts.shape)}", flush=True)

    W_enc = _to_t(W_enc_np).to(device)
    b_enc = _to_t(b_enc_np).to(device)
    b_dec = _to_t(b_dec_np).to(device)
    threshold = _to_t(threshold_np).to(device) if threshold_np is not None else None

    print(f"[{ts()}] encoding through SAE "
          f"(batch={args.encode_batch}, JumpReLU={threshold is not None})",
          flush=True)
    mean_abs = _encode_chunked(acts, W_enc, b_enc, b_dec, threshold,
                                device, batch_size=args.encode_batch)

    n_active = int((mean_abs > 0).sum())
    print(f"[{ts()}] mean_abs: nonzero={n_active}/{n_latents}  "
          f"max={mean_abs.max():.4f}  med_active="
          f"{mean_abs[mean_abs > 0].median().item() if n_active else float('nan'):.4f}",
          flush=True)
    if n_active < args.n:
        print(f"  WARNING: only {n_active} active features (< {args.n}); "
              f"ranking will include zero-activation features.",
              flush=True)

    # Top-N indices (descending by mean |activation|).
    top_vals, top_idx = torch.topk(mean_abs, args.n, largest=True)
    indices = sorted(top_idx.tolist())
    print(f"[{ts()}] top-{args.n} indices: {indices[:5]} ... {indices[-3:]}",
          flush=True)
    print(f"  rank scores (top {args.n}): "
          f"min={top_vals.min():.4f}  max={top_vals.max():.4f}  "
          f"med={top_vals.median():.4f}", flush=True)

    raw = torch.from_numpy(W_dec_np[indices].copy()).float()  # (n, d_in)
    dirs = F.normalize(raw, dim=-1)

    sae_id = f"layer_{args.layer}/width_{args.width}/average_l0_{l0}"
    names = [f"sae_{k:06d}" for k in indices]
    out = {
        "directions": {n: dirs[i].clone() for i, n in enumerate(names)},
        "family": rank_label,
        "signs": {n: "sae_decoder" for n in names},
        "prompt_set_hashes": {n: {"family": rank_label,
                                   "sae_repo": args.repo,
                                   "sae_id": sae_id,
                                   "latent_index": int(idx),
                                   "rank_score": float(
                                       mean_abs[idx].item()),
                                   **rank_meta}
                              for n, idx in zip(names, indices)},
        "sae_repo": args.repo,
        "sae_id": sae_id,
        "width": args.width,
        "average_l0": l0,
        "latent_indices": indices,
        "rank_scores": [float(mean_abs[i].item()) for i in indices],
        "rank_meta": rank_meta,
        "seed": args.seed,
        "layer": args.layer,
        "model": args.target,
        "model_full": args.model,
        "d_model": d_model,
        "n_latents": n_latents,
        "n_dirs": args.n,
    }
    out_path = os.path.join(
        OUT_DIR, f"dirs_{args.target}_L{args.layer}_{rank_label}.pkl")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, out_path)
    print(f"[{ts()}] saved -> {out_path}", flush=True)
    print(f"  ||d|| range: "
          f"[{dirs.norm(dim=-1).min():.6f}, {dirs.norm(dim=-1).max():.6f}]")
    print(f"  pairwise |cos| max (off-diag): "
          f"{(dirs @ dirs.T).fill_diagonal_(0).abs().max():.4f}")
    # Quick overlap-distribution print so we know upfront whether the
    # |cos| < 0.10 family-internal filter will be meaningful.
    cos = (dirs @ dirs.T).fill_diagonal_(0).abs()
    iu, ju = torch.triu_indices(args.n, args.n, offset=1)
    pair_cos = cos[iu, ju]
    over = (pair_cos > 0.10).sum().item()
    print(f"  |cos| > 0.10 pairs: {over}/{pair_cos.numel()}  "
          f"({100*over/pair_cos.numel():.1f}%)")


if __name__ == "__main__":
    main()
