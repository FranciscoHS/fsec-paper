"""Quick steering sanity check for compositional steering (e.g. Wealth x Gender).

Applies the paper's norm-matched rotation (exp_map) at the perturbation layer
to EVERY position during generation, rotating each activation by a fixed angle
toward a target direction (single direction or the orthonormalized composite of
two). Tries both signs of each direction so we can read off which pole is which.

Dumps all completions to a text file for human review. Not a figure generator.

Usage:
  python -u scripts/steer_test.py --target gemma --layer 2 \
      --dirs results/directions/dirs_gemma_L2.pkl \
      --d1 Wealth --d2 Gender --angles 20 30 --max_new_tokens 60 \
      --out logs/steer_wealth_gender.txt
"""
from __future__ import annotations
import os, sys, argparse, pickle, math
sys.path.insert(0, ".")
import torch
from src.model import load_model


def _blocks(model):
    # Gemma-2 / Llama-style: model.model.layers
    return model.model.layers


def make_rotation_hook(target_unit: torch.Tensor, alpha_rad: float):
    """Forward hook that rotates the block's output hidden state at every
    position by `alpha_rad` toward `target_unit` (norm-matched, per position)."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out      # (B, T, D)
        dt = h.dtype
        hf = h.float()
        d = target_unit.to(hf.device, torch.float32)
        R = hf.norm(dim=-1, keepdim=True)                  # (B, T, 1)
        a_hat = hf / R.clamp_min(1e-8)
        # component of target orthogonal to each position's activation
        dp = d - (a_hat * d).sum(-1, keepdim=True) * a_hat
        dp_hat = dp / dp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ca, sa = math.cos(alpha_rad), math.sin(alpha_rad)
        new = R * (ca * a_hat + sa * dp_hat)
        new_h = new.to(dt)
        if isinstance(out, tuple):
            return (new_h,) + tuple(out[1:])
        return new_h
    return hook


def composite_unit(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """Orthonormalized-sum composite, direction-only (not tangent to a
    specific activation -- the per-position orthogonalization happens in the
    hook). Good enough for a steering target."""
    v = d1 / d1.norm() + d2 / d2.norm()
    return v / v.norm()


@torch.no_grad()
def generate(model, tokenizer, device, prompt, max_new_tokens,
             do_sample=False, temperature=0.8, seed=0):
    if do_sample:
        torch.manual_seed(seed)
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    kw = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.3,
              pad_token_id=tokenizer.eos_token_id)
    if do_sample:
        kw.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kw.update(do_sample=False)
    out = model.generate(**ids, **kw)
    text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text.replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--model", default="gemma-2-9b")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--dirs", default="results/directions/dirs_gemma_L2.pkl")
    ap.add_argument("--d1", default="Wealth")
    ap.add_argument("--d2", default="Gender")
    ap.add_argument("--angles", type=float, nargs="+", default=[20, 30])
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--out", default="logs/steer_test.txt")
    ap.add_argument("--n_samples", type=int, default=1,
                    help="samples per config; >1 enables sampling")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--only_combos", action="store_true",
                    help="restrict to +d1, +d2, and the +d1&+d2 composite")
    ap.add_argument("--prompts", nargs="+", default=[
        "Once upon a time, the prince",
        "The other day I met someone who",
        "Walking down the street, I saw a person who",
    ])
    args = ap.parse_args()

    print(f"loading {args.model}", flush=True)
    model, tokenizer, device = load_model(args.model, dtype=torch.bfloat16)
    model.eval()

    with open(args.dirs, "rb") as f:
        dd = pickle.load(f)
    dirs = dd["directions"] if "directions" in dd else dd
    signs = dd.get("signs", {})
    d1 = dirs[args.d1].to(device, torch.float32)
    d2 = dirs[args.d2].to(device, torch.float32)
    d1 = d1 / d1.norm(); d2 = d2 / d2.norm()
    print(f"sign convention recorded: {args.d1}={signs.get(args.d1)}, "
          f"{args.d2}={signs.get(args.d2)}", flush=True)

    block = _blocks(model)[args.layer]

    # Steering configurations: label -> target unit direction.
    # Both signs of each single direction, and all four composite sign combos.
    if args.only_combos:
        configs = [
            (f"+{args.d1}", d1),
            (f"+{args.d2}", d2),
            (f"+{args.d1} & +{args.d2}", composite_unit(d1, d2)),
        ]
    else:
        configs = []
        configs.append((f"+{args.d1}", d1))
        configs.append((f"-{args.d1}", -d1))
        configs.append((f"+{args.d2}", d2))
        configs.append((f"-{args.d2}", -d2))
        for s1, n1 in [(1, f"+{args.d1}"), (-1, f"-{args.d1}")]:
            for s2, n2 in [(1, f"+{args.d2}"), (-1, f"-{args.d2}")]:
                configs.append((f"{n1} & {n2}", composite_unit(s1 * d1, s2 * d2)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = [f"Steering test: {args.d1} x {args.d2}  (model={args.model}, "
             f"L={args.layer}, greedy+rep_penalty=1.3)\n"
             f"Norm-matched rotation at layer {args.layer}, all positions.\n"
             + "=" * 78 + "\n"]

    do_sample = args.n_samples > 1
    for prompt in args.prompts:
        lines.append(f"\nPROMPT: {prompt!r}\n" + "-" * 78)
        for s in range(args.n_samples):
            base = generate(model, tokenizer, device, prompt,
                            args.max_new_tokens, do_sample=do_sample,
                            temperature=args.temperature, seed=s)
            lines.append(f"[{'unsteered':>24}  s{s}] {base}")
        for label, tgt in configs:
            for ang in args.angles:
                for s in range(args.n_samples):
                    h = block.register_forward_hook(
                        make_rotation_hook(tgt, math.radians(ang)))
                    try:
                        txt = generate(model, tokenizer, device, prompt,
                                       args.max_new_tokens, do_sample=do_sample,
                                       temperature=args.temperature, seed=s)
                    finally:
                        h.remove()
                    lines.append(f"[{label:>24} {int(ang):>2}deg s{s}] {txt}")
                print(f"done: {prompt[:25]!r} {label} {ang}", flush=True)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
