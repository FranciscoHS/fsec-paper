#!/usr/bin/env python3
"""Do concrete English NOUNS yield J-Lens pairs on Qwen3.6-27B L36?

The vocab-scan expansion (expand_en_pairs_qwen36.py) found 227 pairs under
|cos|<=0.2, but they were dominated by code/technical tokens; concrete nouns
cluster on the shared "English default" direction. This script restricts the
pool to a large curated list of concrete, imageable nouns and asks: within
that register, how many disjoint pairs clear each |cos| threshold?

No model forward — loads only J and the lm_head rows. Run on the pod.
Outputs report + (if >=N_PAIRS at write_threshold) a sweep-ready pkl.
"""
import os, sys, json, glob, argparse, pickle
sys.path.insert(0, '.')
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from scripts.jlens.expand_en_pairs_qwen36 import (
    find_snapshot, matching_pairs, LAYER, N_PAIRS, THRESHOLDS)

OUT_DIR = 'results/jlens/qwen3.6-27b'

# ~300 concrete, imageable English nouns across many semantic fields. Leading
# space (natural mid-sentence form); filtered to single tokens at runtime.
CONCRETE_NOUNS = [
    # animals
    ' dog', ' cat', ' horse', ' lion', ' tiger', ' bear', ' wolf', ' fox',
    ' deer', ' rabbit', ' mouse', ' rat', ' snake', ' frog', ' fish',
    ' shark', ' whale', ' dolphin', ' eagle', ' owl', ' hawk', ' crow',
    ' duck', ' goose', ' chicken', ' cow', ' pig', ' sheep', ' goat',
    ' horse', ' camel', ' elephant', ' monkey', ' zebra', ' bee', ' ant',
    ' spider', ' butterfly', ' dragon', ' whale',
    # plants / nature
    ' tree', ' flower', ' rose', ' grass', ' leaf', ' forest', ' mountain',
    ' river', ' lake', ' ocean', ' sea', ' beach', ' desert', ' island',
    ' valley', ' hill', ' cliff', ' cave', ' volcano', ' glacier',
    ' storm', ' cloud', ' rain', ' snow', ' ice', ' fire', ' wind',
    ' thunder', ' lightning', ' rainbow', ' sun', ' moon', ' star',
    ' planet', ' comet', ' galaxy',
    # food / drink
    ' bread', ' cheese', ' meat', ' rice', ' soup', ' salad', ' cake',
    ' pie', ' sugar', ' salt', ' pepper', ' honey', ' butter', ' milk',
    ' coffee', ' tea', ' wine', ' beer', ' water', ' juice', ' apple',
    ' orange', ' banana', ' grape', ' lemon', ' cherry', ' peach',
    ' potato', ' tomato', ' onion', ' carrot', ' corn', ' egg', ' fish',
    # body
    ' hand', ' foot', ' head', ' eye', ' ear', ' nose', ' mouth', ' tooth',
    ' hair', ' skin', ' bone', ' heart', ' brain', ' blood', ' arm',
    ' leg', ' finger', ' knee', ' shoulder', ' chest', ' face', ' neck',
    # buildings / places
    ' house', ' home', ' church', ' castle', ' tower', ' bridge', ' road',
    ' street', ' city', ' town', ' village', ' school', ' hospital',
    ' library', ' museum', ' factory', ' farm', ' garden', ' park',
    ' station', ' airport', ' harbor', ' market', ' shop', ' store',
    ' hotel', ' prison', ' palace', ' temple', ' wall', ' gate', ' door',
    ' window', ' roof', ' floor', ' room', ' kitchen', ' bedroom',
    # tools / objects
    ' knife', ' fork', ' spoon', ' plate', ' cup', ' bottle', ' bowl',
    ' pot', ' pan', ' hammer', ' nail', ' saw', ' drill', ' wrench',
    ' rope', ' chain', ' wheel', ' engine', ' motor', ' gear', ' spring',
    ' clock', ' watch', ' lamp', ' candle', ' mirror', ' brush', ' comb',
    ' soap', ' towel', ' blanket', ' pillow', ' bag', ' box', ' basket',
    ' bucket', ' barrel', ' ladder', ' fence', ' key', ' lock',
    # vehicles
    ' car', ' truck', ' bus', ' train', ' plane', ' ship', ' boat',
    ' bike', ' rocket', ' tank', ' wagon', ' sled',
    # materials / substances
    ' gold', ' silver', ' iron', ' steel', ' copper', ' stone', ' rock',
    ' sand', ' clay', ' glass', ' wood', ' paper', ' cloth', ' cotton',
    ' silk', ' wool', ' leather', ' rubber', ' plastic', ' oil', ' coal',
    ' diamond',
    # clothing
    ' shirt', ' coat', ' hat', ' shoe', ' boot', ' glove', ' scarf',
    ' dress', ' skirt', ' sock', ' belt', ' button', ' pocket',
    # instruments / culture
    ' piano', ' guitar', ' violin', ' drum', ' flute', ' trumpet', ' bell',
    ' book', ' pen', ' pencil', ' paint', ' camera', ' phone', ' radio',
    # people / roles (concrete)
    ' king', ' queen', ' soldier', ' doctor', ' teacher', ' farmer',
    ' sailor', ' hunter', ' baker', ' priest', ' judge', ' nurse',
    # misc concrete
    ' ball', ' coin', ' flag', ' crown', ' sword', ' shield', ' arrow',
    ' bow', ' gun', ' bomb', ' net', ' cage', ' nest', ' web', ' seed',
    ' root', ' branch', ' feather', ' shell', ' horn', ' claw', ' wing',
    ' tail', ' fur', ' scale', ' egg', ' bone',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lens', default='/home/user/lens/qwen3.6-27b/jlens/'
                    'Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt')
    ap.add_argument('--hf_home', default=os.environ.get('HF_HOME',
                    '/home/user/hf_local'))
    ap.add_argument('--write_threshold', type=float, default=0.20)
    args = ap.parse_args()

    snap = find_snapshot(args.hf_home)
    tokenizer = AutoTokenizer.from_pretrained(snap)

    # dedupe, keep only single-token nouns
    seen, ids, names = set(), [], []
    for s in CONCRETE_NOUNS:
        if s in seen:
            continue
        seen.add(s)
        toks = tokenizer(s, add_special_tokens=False)['input_ids']
        if len(toks) == 1:
            ids.append(toks[0])
            names.append(s)
    print(f"concrete nouns: {len(seen)} unique, "
          f"{len(names)} single-token", flush=True)

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
    lines = [f"Concrete-noun EN J-Lens pairs — qwen3.6-27b L{LAYER}",
             f"pool: {len(names)} single-token concrete nouns",
             f"shared-direction: ||mean_vec||={m.norm():.3f} "
             f"(1.0 identical, ~{1/np.sqrt(len(V)):.3f} random)",
             f"pairwise |cos|: median={np.median(a):.3f} min={a.min():.3f} "
             f"p5={np.percentile(a, 5):.3f}; frac<=0.2={(a <= 0.2).mean():.4f} "
             f"frac<=0.3={(a <= 0.3).mean():.4f}"]

    written = None
    for thr in THRESHOLDS:
        pairs = matching_pairs(names, C, thr, N_PAIRS)
        lines.append(f"\n|cos|<={thr}: {len(pairs)} disjoint pairs; "
                     f"showing up to {N_PAIRS}:")
        for a_, b_, c_ in pairs[:N_PAIRS]:
            lines.append(f"    {a_!r} x {b_!r}  cos={c_:+.3f}")
        if (thr == args.write_threshold and len(pairs) >= N_PAIRS
                and written is None):
            name2vec = {names[i]: V[i].numpy() for i in range(len(names))}
            en_pairs = [{'tokens': (a_, b_),
                         'vecs': (name2vec[a_], name2vec[b_]),
                         'cosine': c_} for a_, b_, c_ in pairs[:N_PAIRS]]
            written = os.path.join(
                OUT_DIR, f'jlens_en_concrete_pairs_L{LAYER}.pkl')
            with open(written, 'wb') as f:
                pickle.dump({'layer': LAYER, 'model': 'qwen3.6-27b',
                             'en_pairs': en_pairs, 'zh_pairs': [],
                             'cos_max': thr, 'lens_path': args.lens,
                             'source': 'concrete_nouns'}, f)

    txt = '\n'.join(lines)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f'en_concrete_pairs_L{LAYER}.txt'),
              'w') as f:
        f.write(txt)
    print(txt)
    print(f"\n{'Wrote ' + written if written else 'No pkl written'}")


if __name__ == '__main__':
    main()
