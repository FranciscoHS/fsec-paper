#!/usr/bin/env python3
"""Can we form EN J-Lens pairs at all on Qwen3.6-27B L36?

The original 30-token EN candidate list yielded 0 pairs under |cos|<=0.2
because the EN J-Lens vectors share a dominant common direction. This script
tests whether that's fundamental or just small-sample: it scans the whole
vocab for clean lowercase English word tokens (leading space, [a-z]{4,12}),
builds their J-Lens vectors at L36, and reports how many disjoint pairs a
max-cardinality matching finds at several |cos| thresholds. If >=N_PAIRS form
at a threshold, it writes a directions pkl (same format as
build_jlens_directions_qwen36.py) so the pairs can go straight into the sweep.

No model forward pass — loads only J and the lm_head rows (from safetensors).
Run on the pod (weights + lens are there).
"""
import os, sys, json, glob, argparse, pickle, re
sys.path.insert(0, '.')
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

LAYER = 36
N_PAIRS = 10
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.40]
WORD_RE = re.compile(r'^ [a-z]{4,12}$')
OUT_DIR = 'results/jlens/qwen3.6-27b'


def find_snapshot(hf_home):
    pats = glob.glob(os.path.join(
        hf_home, 'hub/models--Qwen--Qwen3.6-27B/snapshots/*'))
    assert pats, "model snapshot not found under HF_HOME"
    return pats[0]


def matching_pairs(names, C, cos_max, n_pairs):
    """Greedy disjoint matching, lowest |cos| edges first. For the existence
    question ("can we form n_pairs disjoint EN pairs under cos_max?") greedy
    low-cos-first is sufficient and near-instant, unlike blossom max-matching
    on a dense 3000-node graph. C is the precomputed cosine matrix."""
    iu = np.triu_indices(len(names), k=1)
    a = np.abs(C[iu])
    keep = a <= cos_max
    ii, jj = iu[0][keep], iu[1][keep]
    order = np.argsort(a[keep])          # low |cos| first
    used = np.zeros(len(names), dtype=bool)
    pairs = []
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        pairs.append((names[i], names[j], float(C[i, j])))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lens', default='/home/user/lens/qwen3.6-27b/jlens/'
                    'Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt')
    ap.add_argument('--hf_home', default=os.environ.get('HF_HOME',
                    '/home/user/hf_local'))
    ap.add_argument('--max_pool', type=int, default=3000,
                    help='cap on candidate tokens (lowest ids first = more '
                         'frequent BPE merges)')
    ap.add_argument('--write_threshold', type=float, default=0.20,
                    help='if a matching at this threshold yields >=N_PAIRS, '
                         'write the pkl of EN pairs')
    args = ap.parse_args()

    snap = find_snapshot(args.hf_home)
    tokenizer = AutoTokenizer.from_pretrained(snap)

    # Scan vocab for clean single-token lowercase English words.
    vocab = tokenizer.get_vocab()  # token_str -> id
    cands = []
    for tid in range(len(tokenizer)):
        s = tokenizer.decode([tid])
        if WORD_RE.match(s):
            cands.append((tid, s))
    cands.sort(key=lambda x: x[0])
    if len(cands) > args.max_pool:
        cands = cands[:args.max_pool]
    ids = [c[0] for c in cands]
    names = [c[1] for c in cands]
    print(f"vocab-scanned EN word tokens: {len(names)} "
          f"(cap {args.max_pool})", flush=True)

    # Load J[LAYER] and the needed lm_head rows (no model forward).
    lens = torch.load(args.lens, map_location='cpu', weights_only=False)
    J = lens['J'][LAYER].float()
    with open(os.path.join(snap, 'model.safetensors.index.json')) as f:
        shard = json.load(f)['weight_map']['lm_head.weight']
    with safe_open(os.path.join(snap, shard), framework='pt') as f:
        W = f.get_tensor('lm_head.weight').float()  # [vocab, d]
    rows = W[ids]                                    # [n, d]
    del W

    vecs = (J.T @ rows.T).T                           # [n, d]
    V = vecs / vecs.norm(dim=-1, keepdim=True)

    # Shared-direction stat over the big pool.
    m = V.mean(0)
    lines = [f"EN J-Lens pair expansion — qwen3.6-27b L{LAYER}",
             f"pool: {len(names)} vocab-scanned EN word tokens",
             f"shared-direction: ||mean_vec||={m.norm():.3f} "
             f"(1.0 identical, ~{1/np.sqrt(len(V)):.3f} random)"]
    C = (V @ V.T).numpy()
    a = np.abs(C[np.triu_indices(len(V), k=1)])
    lines.append(f"pairwise |cos|: median={np.median(a):.3f} "
                 f"min={a.min():.3f} p5={np.percentile(a, 5):.3f}; "
                 f"frac<=0.2={ (a <= 0.2).mean():.4f} "
                 f"frac<=0.3={ (a <= 0.3).mean():.4f}")

    written = None
    for thr in THRESHOLDS:
        pairs = matching_pairs(names, C, thr, N_PAIRS)
        lines.append(f"\n|cos|<={thr}: {len(pairs)} disjoint pairs "
                     f"(matching); showing up to {N_PAIRS}:")
        for a_, b_, c_ in pairs[:N_PAIRS]:
            lines.append(f"    {a_!r} x {b_!r}  cos={c_:+.3f}")
        if (thr == args.write_threshold and len(pairs) >= N_PAIRS
                and written is None):
            # build pkl in the sweep-compatible format (EN pairs only)
            name2vec = {names[i]: V[i].numpy() for i in range(len(names))}
            en_pairs = [{'tokens': (a_, b_),
                         'vecs': (name2vec[a_], name2vec[b_]),
                         'cosine': c_} for a_, b_, c_ in pairs[:N_PAIRS]]
            written = os.path.join(
                OUT_DIR, f'jlens_en_pairs_expanded_L{LAYER}.pkl')
            with open(written, 'wb') as f:
                pickle.dump({'layer': LAYER, 'model': 'qwen3.6-27b',
                             'en_pairs': en_pairs, 'zh_pairs': [],
                             'cos_max': thr, 'lens_path': args.lens,
                             'source': 'vocab_scan'}, f)

    txt = '\n'.join(lines)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f'en_pairs_expansion_L{LAYER}.txt'),
              'w') as f:
        f.write(txt)
    print(txt)
    if written:
        print(f"\nWrote EN pairs pkl: {written}")
    else:
        print(f"\nNo pkl written (no threshold >= write_threshold "
              f"reached {N_PAIRS} pairs).")


if __name__ == '__main__':
    main()
