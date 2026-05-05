"""Anchor activation extraction with disk caching.

Source kinds:
  fineweb:  English web crawl via load_fineweb_fixed_length (default).
  wiki_en:  English Wikipedia (wikimedia/wikipedia 20231101.en).
  wiki_zh:  Mandarin Wikipedia (wikimedia/wikipedia 20231101.zh).
  code:     Python code (bigcode/the-stack-smol).
  holdout:  registry-defined holdout list, last-token activation.

Cache path: results/exp_map/data/activations/acts_<target>_L<layer>_<source>.pkl
{
  'contexts':   list[Tensor[1, T-1, D]] in float32 (CPU)
  'activations': Tensor[N, D] (last-token, float32, CPU)
  'source':     str
  'layer':      int
  'seed':       int
  'prompt_set_hash':  str (None for streaming-text sources)
}
"""
from __future__ import annotations
import os, pickle, hashlib, time
import torch

import sys
sys.path.insert(0, ".")
from src.data import load_fineweb_fixed_length

OUT_DIR = "results/exp_map/data/activations"
os.makedirs(OUT_DIR, exist_ok=True)

N_ANCHORS = 30
SEQ_LEN = 5
SEED = 42


def _hash_list(lst):
    h = hashlib.sha256()
    for s in lst:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _tokenize_fixed_length(texts_iter, n, tokenizer, seq_len, cache_file=None,
                            add_special_tokens=True):
    """Stream text strings; collect n tokenized sequences of exactly seq_len.
    Mirrors load_fineweb_fixed_length's tokenization pattern (batched encode,
    truncation+padding=False) so anchor protocol stays consistent across
    sources."""
    if cache_file and os.path.exists(cache_file):
        return torch.load(cache_file, weights_only=True)
    token_ids_list = []
    text_batch = []
    batch_size = 1000
    t0 = time.time()
    for text in texts_iter:
        text_batch.append(text)
        if len(text_batch) >= batch_size:
            batch_tokens = tokenizer(text_batch, truncation=True,
                                     max_length=seq_len,
                                     add_special_tokens=add_special_tokens,
                                     padding=False)
            for ids in batch_tokens["input_ids"]:
                if len(ids) >= seq_len:
                    token_ids_list.append(torch.tensor(ids[:seq_len]))
                    if len(token_ids_list) >= n:
                        break
            text_batch = []
            if len(token_ids_list) >= n:
                break
    if text_batch and len(token_ids_list) < n:
        batch_tokens = tokenizer(text_batch, truncation=True,
                                 max_length=seq_len,
                                 add_special_tokens=add_special_tokens,
                                 padding=False)
        for ids in batch_tokens["input_ids"]:
            if len(ids) >= seq_len:
                token_ids_list.append(torch.tensor(ids[:seq_len]))
                if len(token_ids_list) >= n:
                    break
    elapsed = time.time() - t0
    print(f"  tokenized {len(token_ids_list)} sequences "
          f"(seq_len={seq_len}) in {elapsed:.0f}s", flush=True)
    if cache_file and len(token_ids_list) == n:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(token_ids_list, cache_file)
        print(f"  cached to {cache_file}", flush=True)
    return token_ids_list


def _anchor_acts(model, tokenizer, device, target: str, layer: int,
                  source: str, token_loader, n: int, seq_len: int,
                  seed: int) -> dict:
    """Shared core: load token lists via token_loader(), forward through
    model, cache last-token residual at `layer`. Activation pkl path is
    `acts_<target>_L<layer>_<source>.pkl`."""
    cache_pkl = os.path.join(OUT_DIR, f"acts_{target}_L{layer}_{source}.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)
    token_lists = token_loader()
    contexts, acts = [], []
    for tids in token_lists:
        tids = tids.unsqueeze(0).to(device)
        with torch.no_grad():
            outs = model(tids, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        acts.append(h[:, -1, :].squeeze(0).float().cpu())
    out = {"contexts": contexts, "activations": torch.stack(acts),
           "source": source, "layer": layer, "seed": seed,
           "n": n, "seq_len": seq_len, "prompt_set_hash": None}
    with open(cache_pkl, "wb") as f:
        pickle.dump(out, f)
    return out


def fineweb_acts(model, tokenizer, device, target: str, layer: int,
                 n: int = N_ANCHORS, seq_len: int = SEQ_LEN,
                 seed: int = SEED) -> dict:
    cache_dir = f"/workspace/fineweb_cache_{target}"
    os.makedirs(cache_dir, exist_ok=True)
    def token_loader():
        return load_fineweb_fixed_length(
            n, tokenizer, seq_len=seq_len, seed=seed, cache_dir=cache_dir)
    return _anchor_acts(model, tokenizer, device, target, layer, "fineweb",
                         token_loader, n, seq_len, seed)


def fineweb_acts_n(model, tokenizer, device, target: str, layer: int,
                    n: int, seq_len: int = SEQ_LEN, seed: int = SEED,
                    batch_size: int = 32) -> dict:
    """Like ``fineweb_acts`` but for a custom ``n`` (e.g. 10k for
    SAE-FineWeb / PCA-FineWeb baselines). Cache file name is
    ``acts_<target>_L<layer>_fineweb_<n>.pkl`` so it does not clobber the
    canonical 30-anchor cache. Forwards through the model in batches and
    only stores the last-token activation (the per-prompt context tensor
    that ``_anchor_acts`` saves for the 30-anchor case is too large at
    n=10k).

    Returned dict matches the shape of the 30-anchor cache where
    relevant: ``activations`` (Tensor[N, D] last-token, float32, CPU),
    plus the usual metadata. ``contexts`` is omitted.
    """
    cache_pkl = os.path.join(
        OUT_DIR, f"acts_{target}_L{layer}_fineweb_{n}.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)
    cache_dir = f"/workspace/fineweb_cache_{target}"
    os.makedirs(cache_dir, exist_ok=True)
    token_lists = load_fineweb_fixed_length(
        n, tokenizer, seq_len=seq_len, seed=seed, cache_dir=cache_dir)
    # Stack into a single (N, seq_len) batch for batched forwarding.
    tok_tensor = torch.stack(token_lists, dim=0)
    acts = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = tok_tensor[start:start + batch_size].to(device)
        with torch.no_grad():
            outs = model(chunk, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]   # (B, T, D)
        acts.append(h[:, -1, :].float().cpu())
        if (start // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            done = start + chunk.shape[0]
            print(f"  fineweb_acts_n: {done}/{n}  "
                  f"({elapsed:.0f}s, {done/max(elapsed,1e-6):.1f}/s)",
                  flush=True)
    activations = torch.cat(acts, dim=0)
    out = {"activations": activations, "source": f"fineweb_{n}",
           "layer": layer, "seed": seed, "n": n, "seq_len": seq_len,
           "prompt_set_hash": None}
    tmp = cache_pkl + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, cache_pkl)
    print(f"  fineweb_acts_n: cached -> {cache_pkl}", flush=True)
    return out


def wiki_acts(model, tokenizer, device, target: str, layer: int,
              n: int = N_ANCHORS, seq_len: int = SEQ_LEN,
              seed: int = SEED, language: str = "en") -> dict:
    """Wikipedia anchors. language in {'en', 'zh', ...} — passed straight
    through to the wikimedia/wikipedia 20231101 config name."""
    from datasets import load_dataset
    source = f"wiki_{language}"
    cache_dir = f"/workspace/{source}_cache_{target}"
    cache_file = os.path.join(
        cache_dir, f"{source}_n{n}_seq{seq_len}_seed{seed}.pt")

    def token_loader():
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{language}",
                          split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10000)
        return _tokenize_fixed_length(
            (s["text"] for s in ds), n, tokenizer, seq_len, cache_file)

    return _anchor_acts(model, tokenizer, device, target, layer, source,
                         token_loader, n, seq_len, seed)


def code_acts(model, tokenizer, device, target: str, layer: int,
              n: int = N_ANCHORS, seq_len: int = SEQ_LEN,
              seed: int = SEED) -> dict:
    """Python code anchors from bigcode/the-stack-smol (default subset)."""
    from datasets import load_dataset
    source = "code"
    cache_dir = f"/workspace/{source}_cache_{target}"
    cache_file = os.path.join(
        cache_dir, f"{source}_n{n}_seq{seq_len}_seed{seed}.pt")

    def token_loader():
        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python",
                          split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10000)
        return _tokenize_fixed_length(
            (s["content"] for s in ds), n, tokenizer, seq_len, cache_file)

    return _anchor_acts(model, tokenizer, device, target, layer, source,
                         token_loader, n, seq_len, seed)


def holdout_acts(model, tokenizer, device, target: str, layer: int,
                 source_name: str, prompts: list[str],
                 n: int = N_ANCHORS) -> dict:
    """source_name: short label used in filename, e.g. 'male_holdout'.
    prompts: full list; we take the first n.
    """
    cache_pkl = os.path.join(OUT_DIR, f"acts_{target}_L{layer}_{source_name}.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)
    pp = prompts[:n]
    contexts, acts = [], []
    for p in pp:
        tok = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            outs = model(**tok, output_hidden_states=True)
        h = outs.hidden_states[layer + 1]
        contexts.append(h[:, :-1, :].float().cpu())
        acts.append(h[:, -1, :].squeeze(0).float().cpu())
    out = {"contexts": contexts, "activations": torch.stack(acts),
           "source": source_name, "layer": layer, "n": len(pp),
           "prompt_set_hash": _hash_list(pp)}
    with open(cache_pkl, "wb") as f:
        pickle.dump(out, f)
    return out
