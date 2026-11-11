"""E5 (R2 W1, R1 Q4): MLP-neuron directions as a direction family.

Reviewer 2's alternative: p > 2 could come from a privileged basis created by
an elementwise nonlinearity (MLP neurons), without superposition. If so, the
input directions of MLP neurons are where p > 2 should be strongest.

We perturb the residual stream after block `layer` (default 2; activations are
hidden_states[layer + 1]), so the first MLP the perturbation reaches is in
block `layer + 1` (block 3 for the default). Gemma-2's MLP is gated:
    act = gelu(W_gate x_n) * (W_up x_n),   x_n = RMSNorm_pre_ff(x) = x / rms(x) * (1 + w)
The nonlinearity acts on the gate pre-activation, so a neuron's direction in
residual-stream coordinates is W_gate[i] * (1 + w), L2-normalised.

Neurons are ranked by mean |act| at the last token over the same 10k-prompt
FineWeb sample used for the PCA / SAE / random-difference families (seed 42,
5-token prompts), and the top-33 kept, mirroring sae_directions_top.py.

Output: results/directions/dirs_gemma_L2_mlpneuron.pkl  (family 'mlpneuron')
Usage (GPU): python -u scripts/rebuttal/mlp_neuron_directions.py
"""
from __future__ import annotations
import os, pickle, argparse, sys, time
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from src.data import load_fineweb_fixed_length
from scripts.lib.activations import CACHE_ROOT, SEQ_LEN, SEED

OUT_DIR = "results/directions"


def ts(): return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--model", default="gemma-2-9b")
    ap.add_argument("--layer", type=int, default=2,
                    help="perturbation layer; neurons come from block layer+1")
    ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--n_fineweb", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--select", choices=["top", "random"], default="top",
                    help="top: top-n by mean |act| on FineWeb (family 'mlpneuron'); "
                         "random: n neurons uniformly at random (family 'mlpneuronrand')")
    ap.add_argument("--seed", type=int, default=0, help="seed for --select random")
    args = ap.parse_args()

    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    block_idx = args.layer + 1
    block = _get_blocks(model)[block_idx]
    mlp, norm = block.mlp, block.pre_feedforward_layernorm
    d_mlp = mlp.gate_proj.weight.shape[0]
    print(f"[{ts()}] block {block_idx}: d_mlp={d_mlp} "
          f"act_fn={type(mlp.act_fn).__name__}", flush=True)

    class _Stop(Exception):
        pass

    sum_abs = torch.zeros(d_mlp, dtype=torch.float64, device=device)
    captured = {}

    def hook(mod, inp):
        # neuron activations act = gelu(gate) * up, last token; stop the
        # forward pass here since later blocks aren't needed
        captured["act"] = inp[0][:, -1, :]
        raise _Stop

    h = mlp.down_proj.register_forward_pre_hook(hook)
    cache_dir = f"{CACHE_ROOT}/fineweb_cache_{args.target}"
    os.makedirs(cache_dir, exist_ok=True)
    toks = torch.stack(load_fineweb_fixed_length(
        args.n_fineweb, tokenizer, seq_len=SEQ_LEN, seed=SEED, cache_dir=cache_dir))
    for s in range(0, len(toks), args.batch_size):
        try:
            with torch.no_grad():
                model(toks[s:s + args.batch_size].to(device))
        except _Stop:
            pass
        sum_abs += captured["act"].abs().double().sum(0)
    h.remove()
    mean_abs = (sum_abs / len(toks)).float().cpu()
    print(f"[{ts()}] ranked {d_mlp} neurons over {len(toks)} prompts", flush=True)

    if args.select == "top":
        top_vals, top_idx = torch.topk(mean_abs, args.n)
        family = "mlpneuron"
    else:
        g = torch.Generator().manual_seed(args.seed)
        top_idx = torch.randperm(d_mlp, generator=g)[:args.n]
        top_vals = mean_abs[top_idx]
        family = "mlpneuronrand"
    idx = sorted(top_idx.tolist())
    scale = 1.0 + norm.weight.float()
    raw = mlp.gate_proj.weight[idx].float() * scale.unsqueeze(0)
    dirs = F.normalize(raw, dim=-1).cpu()

    names = [f"mlp{block_idx}_{i:05d}" for i in idx]
    cos = (dirs @ dirs.T).fill_diagonal_(0).abs()
    iu = torch.triu_indices(args.n, args.n, 1)
    pc = cos[iu[0], iu[1]]
    out = {
        "directions": {n: dirs[k].clone() for k, n in enumerate(names)},
        "family": family,
        "signs": {n: "gate_row" for n in names},
        "neuron_indices": idx, "block": block_idx,
        "rank_scores": [float(mean_abs[i]) for i in idx],
        "rank_meta": {"rank_set": "fineweb", "n_rank": len(toks), "seed": SEED,
                      "statistic": "mean |gelu(gate)*up| at last token",
                      "select": args.select, "select_seed": args.seed},
        "layer": args.layer, "model": args.target,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/dirs_{args.target}_L{args.layer}_{family}.pkl"
    with open(path + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(path + ".tmp", path)
    print(f"[{ts()}] saved {path}; rank scores max {top_vals.max():.3f} "
          f"median {top_vals.median():.3f} min {top_vals.min():.3f}; |cos|>0.1 pairs: "
          f"{int((pc > 0.1).sum())}/{pc.numel()} (max {pc.max():.3f})", flush=True)


if __name__ == "__main__":
    main()
