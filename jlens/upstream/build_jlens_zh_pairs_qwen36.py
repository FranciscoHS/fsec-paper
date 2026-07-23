#!/usr/bin/env python3
"""Build a large set of ZH J-Lens direction pairs on Qwen3.6-27B L36.

Scales the ZH leg up from 10 to ~N_PAIRS pairs by scanning the vocab for
single-token CJK words (2-4 Han characters) instead of the hand-picked 28
candidates, then greedily matching disjoint pairs under |cos|<=COS_MAX.

No model forward — loads only J and the lm_head rows. Run on the pod.
Writes a directions pkl (zh_pairs filled, en_pairs empty) in the same format
the sweep script consumes.
"""
import os, sys, json, argparse, pickle
sys.path.insert(0, '.')
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from scripts.jlens.expand_en_pairs_qwen36 import find_snapshot, matching_pairs

LAYER = 36
COS_MAX = 0.2
N_PAIRS = 100
MAX_POOL = 1200
OUT_DIR = 'results/jlens/qwen3.6-27b'


def is_cjk_word(s, min_len=2, max_len=4):
    if not (min_len <= len(s) <= max_len):
        return False
    return all('一' <= c <= '鿿' for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lens', default='/home/user/lens/qwen3.6-27b/jlens/'
                    'Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt')
    ap.add_argument('--hf_home', default=os.environ.get('HF_HOME',
                    '/home/user/hf_local'))
    ap.add_argument('--n_pairs', type=int, default=N_PAIRS)
    ap.add_argument('--cos_max', type=float, default=COS_MAX)
    ap.add_argument('--max_pool', type=int, default=MAX_POOL)
    args = ap.parse_args()

    snap = find_snapshot(args.hf_home)
    tokenizer = AutoTokenizer.from_pretrained(snap)

    # Scan vocab for single-token CJK words (2-4 Han chars).
    cands = []
    for tid in range(len(tokenizer)):
        s = tokenizer.decode([tid])
        if is_cjk_word(s):
            # confirm it really is a single token (decode round-trip)
            if tokenizer(s, add_special_tokens=False)['input_ids'] == [tid]:
                cands.append((tid, s))
    cands.sort(key=lambda x: x[0])   # lowest ids first (more frequent)
    if len(cands) > args.max_pool:
        cands = cands[:args.max_pool]
    ids = [c[0] for c in cands]
    names = [c[1] for c in cands]
    print(f"vocab-scanned single-token CJK words: {len(names)} "
          f"(cap {args.max_pool})", flush=True)

    lens = torch.load(args.lens, map_location='cpu', weights_only=False)
    J = lens['J'][LAYER].float()
    with open(os.path.join(snap, 'model.safetensors.index.json')) as f:
        shard = json.load(f)['weight_map']['lm_head.weight']
    with safe_open(os.path.join(snap, shard), framework='pt') as f:
        rows = f.get_tensor('lm_head.weight').float()[ids]

    vecs = (J.T @ rows.T).T
    V = vecs / vecs.norm(dim=-1, keepdim=True)
    C = (V @ V.T).numpy()

    m = V.mean(0)
    a = np.abs(C[np.triu_indices(len(V), k=1)])
    report = [f"ZH J-Lens pair scale-up — qwen3.6-27b L{LAYER}",
              f"pool: {len(names)} single-token CJK words",
              f"shared-direction: ||mean_vec||={m.norm():.3f}",
              f"pairwise |cos|: median={np.median(a):.3f} "
              f"frac<=0.2={(a <= 0.2).mean():.4f}"]

    pairs = matching_pairs(names, C, args.cos_max, args.n_pairs)
    pairs = pairs[:args.n_pairs]
    report.append(f"|cos|<={args.cos_max}: {len(pairs)} disjoint pairs "
                  f"(target {args.n_pairs})")

    name2vec = {names[i]: V[i].numpy() for i in range(len(names))}
    zh_pairs = [{'tokens': (a_, b_),
                 'vecs': (name2vec[a_], name2vec[b_]),
                 'cosine': c_} for a_, b_, c_ in pairs]
    for k, (a_, b_, c_) in enumerate(pairs):
        report.append(f"  pair {k:3d}: {a_!r} x {b_!r}  cos={c_:+.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pkl = os.path.join(OUT_DIR, f'jlens_zh{args.n_pairs}_pairs_L{LAYER}.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({'layer': LAYER, 'model': 'qwen3.6-27b',
                     'en_pairs': [], 'zh_pairs': zh_pairs,
                     'cos_max': args.cos_max, 'lens_path': args.lens,
                     'source': 'vocab_scan_cjk'}, f)
    rep_path = os.path.join(OUT_DIR, f'jlens_zh{args.n_pairs}_pairs_L{LAYER}.txt')
    with open(rep_path, 'w') as f:
        f.write('\n'.join(report))
    print('\n'.join(report[:6]))
    print(f"...\nSaved: {out_pkl}\n       {rep_path}")


if __name__ == '__main__':
    main()
