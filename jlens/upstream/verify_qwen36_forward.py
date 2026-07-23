#!/usr/bin/env python3
"""Sanity checks for Qwen3.6-27B support before running any sweeps.

Checks:
  1. Module tree: _get_blocks / _get_final_norm / _get_lm_head resolve, and
     layer count / d_model match the config (64 layers, 5120).
  2. Forward consistency: run the full model with output_hidden_states, then
     re-run blocks manually from L0 and from L36 via the same code path the
     sweeps use (forward_from_layer_to_layer) and compare hidden states at
     several layers. The hybrid linear-attention blocks are the risk here.
  3. Timing: one sweep-sized batch (64 points, 5-token seq) L36 -> L62,
     extrapolated to a per-pair estimate.

Writes a report to results/jlens/qwen3.6-27b/verify_forward.txt.
"""
import os, sys, time
sys.path.insert(0, '.')
import torch

from src.model import (load_model, _get_blocks, _get_final_norm, _get_lm_head,
                       forward_from_layer_to_layer)

MODEL_NAME = 'qwen3.6-27b'
LAYER = 36
MEASURE_LAYER = 62
OUT_DIR = 'results/jlens/qwen3.6-27b'
REPORT = os.path.join(OUT_DIR, 'verify_forward.txt')

TEST_PROMPT = "The quick brown fox jumps over the lazy dog near the river"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.bfloat16)
    log(f"  loaded in {time.time() - t0:.0f}s")

    # --- 1. module tree ---
    blocks = _get_blocks(model)
    norm = _get_final_norm(model)
    head = _get_lm_head(model)
    d_model = model._plateau_info['d_model']
    log(f"  n_blocks={len(blocks)}  d_model={d_model}")
    log(f"  final_norm={type(norm).__name__}  lm_head={tuple(head.weight.shape)}")
    log(f"  block types: {[getattr(b, 'block_type', '?') for b in blocks[:8]]}...")
    assert len(blocks) == 64 and d_model == 5120

    # --- 2. forward consistency ---
    tokens = tokenizer(TEST_PROMPT, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)
    hs = out.hidden_states
    log(f"\nFull forward: {len(hs)} hidden states, seq_len={hs[0].shape[1]}")

    # Manual run from layer k: context = hs[k+1][:, :-1], point = hs[k+1][:, -1]
    # then forward_from_layer_to_layer to target layer, compare vs hs[target+1].
    for start, target in [(0, 8), (0, MEASURE_LAYER), (LAYER, LAYER + 4),
                          (LAYER, MEASURE_LAYER)]:
        ctx = hs[start + 1][:, :-1, :]
        pt = hs[start + 1][:, -1:, :].unsqueeze(0).squeeze(1)  # [1, 1, d]
        with torch.no_grad():
            manual = forward_from_layer_to_layer(
                model, ctx, pt, start, target)          # [1, d]
        ref = hs[target + 1][:, -1, :]
        diff = (manual.float() - ref.float()).abs().max().item()
        rel = diff / ref.float().abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            manual.float(), ref.float(), dim=-1).item()
        log(f"  L{start}->L{target}: max|diff|={diff:.4g}  rel={rel:.3g}  "
            f"cos={cos:.6f}")
        # bf16 accumulation over ~60 blocks: allow loose tolerance, but
        # cosine must be essentially 1.
        assert cos > 0.999, f"forward mismatch L{start}->L{target}"

    # --- 3. timing (sweep-shaped workload) ---
    seq5 = tokenizer("The capital of France is", return_tensors='pt').to(device)
    with torch.no_grad():
        out5 = model(**seq5, output_hidden_states=True)
    ctx5 = out5.hidden_states[LAYER + 1][:, :-1, :]
    a5 = out5.hidden_states[LAYER + 1][:, -1, :].float()
    batch = a5.unsqueeze(1).expand(64, 1, -1).to(torch.bfloat16)  # [64, 1, d]
    with torch.no_grad():  # warmup
        forward_from_layer_to_layer(model, ctx5, batch, LAYER, MEASURE_LAYER)
    torch.cuda.synchronize()
    t0 = time.time()
    n_rep = 10
    with torch.no_grad():
        for _ in range(n_rep):
            forward_from_layer_to_layer(model, ctx5, batch, LAYER, MEASURE_LAYER)
    torch.cuda.synchronize()
    per_batch = (time.time() - t0) / n_rep
    n_batches_pair = 30 * ((41 * 41 + 63) // 64)   # 30 anchors x ceil(1681/64)
    log(f"\nTiming: {per_batch * 1e3:.1f} ms / 64-pt batch (L{LAYER}->L{MEASURE_LAYER})")
    log(f"  est. per J-Lens pair (30 anchors, 41x41): "
        f"{per_batch * n_batches_pair / 60:.1f} min")
    log(f"  est. 20 pairs: {per_batch * n_batches_pair * 20 / 3600:.2f} h")

    with open(REPORT, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved: {REPORT}")


if __name__ == '__main__':
    main()
