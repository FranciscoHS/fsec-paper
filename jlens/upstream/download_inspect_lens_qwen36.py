#!/usr/bin/env python3
"""Download the pre-fitted Jacobian Lens for Qwen3.6-27B from
neuronpedia/jacobian-lens and inspect its format (keys, shapes, dtypes).

Lens file: qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt
(3.3 GB). Writes an inspection report next to the downloaded files.
"""
import argparse
import os
import torch
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument('--out_dir', default='results/jlens/qwen3.6-27b',
                help='where to put the lens (use local disk if the network '
                     'volume is full) — the inspection report goes here too')
OUT_DIR = ap.parse_args().out_dir
REPORT = os.path.join(OUT_DIR, 'lens_inspection.txt')
LENS_SUBDIR = 'qwen3.6-27b/jlens/Salesforce-wikitext'
LENS_FILE = 'Qwen3.6-27B_jacobian_lens_n1000.pt'


def describe(obj, name, lines, depth=0):
    pad = '  ' * depth
    if isinstance(obj, torch.Tensor):
        lines.append(f"{pad}{name}: Tensor {tuple(obj.shape)} {obj.dtype} "
                     f"norm={obj.float().norm():.4g}")
    elif isinstance(obj, dict):
        lines.append(f"{pad}{name}: dict with {len(obj)} keys")
        for k, v in obj.items():
            describe(v, str(k), lines, depth + 1)
    elif isinstance(obj, (list, tuple)):
        lines.append(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[:3]):
            describe(v, f"[{i}]", lines, depth + 1)
        if len(obj) > 3:
            lines.append(f"{pad}  ... ({len(obj) - 3} more)")
    else:
        r = repr(obj)
        lines.append(f"{pad}{name}: {type(obj).__name__} = "
                     f"{r if len(r) < 200 else r[:200] + '...'}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = snapshot_download(
        'neuronpedia/jacobian-lens',
        allow_patterns=['qwen3.6-27b/**'],
        local_dir=OUT_DIR,
    )
    print(f"Downloaded to: {path}")

    lens_dir = os.path.join(OUT_DIR, LENS_SUBDIR)
    lines = [f"Files in {lens_dir}:"]
    for f in sorted(os.listdir(lens_dir)):
        full = os.path.join(lens_dir, f)
        if os.path.isfile(full):
            lines.append(f"  {f}  ({os.path.getsize(full) / 1e6:.1f} MB)")

    pt = os.path.join(lens_dir, LENS_FILE)
    lines.append(f"\n--- {LENS_FILE} structure ---")
    obj = torch.load(pt, map_location='cpu', weights_only=False)
    describe(obj, 'root', lines)

    report = '\n'.join(lines)
    with open(REPORT, 'w') as f:
        f.write(report)
    print(report)
    print(f"\nReport saved: {REPORT}")


if __name__ == '__main__':
    main()
