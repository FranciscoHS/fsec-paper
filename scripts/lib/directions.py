"""DoM direction extraction with disk caching.

Cache path: results/directions/dirs_<target>_L<layer>.pkl
{
  'directions': {name: torch.Tensor[D]},  # unit vector, sign per registry
  'signs':      {name: 'side_a->side_b'},
  'layer':      int,
  'model':      str,
  'prompt_set_hashes': {name: sha256 hex},
}
"""
from __future__ import annotations
import os, pickle, hashlib, time
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
from scripts.lib import registry

OUT_DIR = "results/directions"
os.makedirs(OUT_DIR, exist_ok=True)


def _hash_list(lst: list[str]) -> str:
    h = hashlib.sha256()
    for s in lst:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _last_token_act(model, tokenizer, device, layer: int, prompt: str) -> torch.Tensor:
    tok = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outs = model(**tok, output_hidden_states=True)
    return outs.hidden_states[layer + 1][:, -1, :].squeeze(0).float().cpu()


def compute_dom(model, tokenizer, device, layer, prompts_a, prompts_b) -> torch.Tensor:
    """Return unit vector pointing a -> b: + dir = move toward b."""
    A = torch.stack([_last_token_act(model, tokenizer, device, layer, p)
                     for p in prompts_a])
    B = torch.stack([_last_token_act(model, tokenizer, device, layer, p)
                     for p in prompts_b])
    return F.normalize((B - A).mean(0), dim=0)


def extract_all(model, tokenizer, device, target: str, layer: int,
                names: list[str] | None = None, refresh: bool = False) -> dict:
    """Build (or extend) the per-(target, layer) DoM direction cache.

    Default `names`: all semantic directions (preserves existing E1 behavior).
    Pass `names=registry.all_direction_names()` to include language + code.

    If the cache already exists, only the missing names are computed and
    merged in. `refresh=True` forces a full recompute.
    """
    pkl = os.path.join(OUT_DIR, f"dirs_{target}_L{layer}.pkl")
    names = names or list(registry.DIRECTIONS)

    if os.path.exists(pkl) and not refresh:
        with open(pkl, "rb") as f:
            blob = pickle.load(f)
        missing = [n for n in names if n not in blob["directions"]]
        if not missing:
            return blob
        print(f"[directions] {target} L{layer}: extending cache with "
              f"{len(missing)} new directions", flush=True)
        dirs = blob["directions"]; signs = blob["signs"]
        hashes = blob.get("prompt_set_hashes", {})
        to_compute = missing
    else:
        print(f"[directions] {target} L{layer}: building cache with "
              f"{len(names)} directions", flush=True)
        dirs, signs, hashes = {}, {}, {}
        to_compute = names

    for n in to_compute:
        a, b = registry.get_prompts(n)
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        t0 = time.time()
        dirs[n] = compute_dom(model, tokenizer, device, layer, a, b)
        signs[n] = registry.get_sign(n)
        hashes[n] = {"a_sha": _hash_list(a), "b_sha": _hash_list(b),
                     "n_pairs": m, "family": registry.family(n)}
        print(f"  {n:14s}  {time.time()-t0:.1f}s  ||d||={dirs[n].norm().item():.3f}",
              flush=True)
    out = {"directions": dirs, "signs": signs, "layer": layer,
           "model": target, "prompt_set_hashes": hashes}
    tmp = pkl + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, pkl)
    return out
