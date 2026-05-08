"""E2: 2D iso-plateau pair sweeps in the exp-map chart.

Per (model, pair=(d1, d2)): 31x31 grid in (alpha_1, alpha_2) up to 60 deg,
with 30 FineWeb anchors per pair (Plan D2/D3/D4).

Output: results/sweeps_2d/sweep2d_<target>_L<layer>_<d1>__<d2>_fineweb_60deg.pkl
{
  'angles_deg':   (n,),
  'l2':           (n_anchors, n, n),
  ... full PKL header
}

Usage:
  python scripts/sweep_2d.py --target gemma --layer 2          # all 20 Tier-1 pairs
  python scripts/sweep_2d.py --target gemma --layer 2 \
      --pairs Gender,Tense                                              # one pair
"""
from __future__ import annotations
import os, sys, pickle, argparse, time, subprocess
sys.path.insert(0, ".")
import numpy as np
import torch

from src.model import load_model, _get_blocks
from scripts.lib import registry, directions as dirlib, activations as actlib
from scripts.lib.sweep_core import run_2d_per_anchors
from scripts.lib.parametrize import eps_thresholds, CHART

OUT_DIR = "results/sweeps_2d"
os.makedirs(OUT_DIR, exist_ok=True)

# locked params (Plan E2, D2/D3/D4)
N_STEPS_2D = 31
MAX_ANGLE_2D = 60.0
N_ANCHORS = 30


def _git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%H", "--", path],
            text=True).strip()
    except Exception:
        return "unknown"


def ts(): return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(registry.TARGETS))
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--pairs", default=None,
                    help="optional override; comma-separated single pair "
                         "like 'Gender,Tense' OR semicolon-separated multiple "
                         "pairs like 'Gender,Tense;Gender,Sentiment'")
    ap.add_argument("--pairs_file", default=None,
                    help="optional path to a CSV/text file of pairs. Each "
                         "line is 'a,b' (any extra columns ignored). "
                         "Overrides --pairs / --pair_set.")
    ap.add_argument("--pair_set", default="all",
                    choices=["all", "tier1"],
                    help="default 'all' = C(N,2) over semantic+lang+code; "
                         "'tier1' = the 20-pair v1 list")
    ap.add_argument("--shard", default=None,
                    help="optional 'i/N' shard selector; e.g. '0/4' runs the "
                         "first quarter of the pair list (after sorting). "
                         "Use to parallelize a model across multiple pods.")
    ap.add_argument("--max_angle", type=float, default=MAX_ANGLE_2D)
    ap.add_argument("--n_steps", type=int, default=N_STEPS_2D)
    ap.add_argument("--dense_max", type=float, default=None,
                    help="if set, build a non-uniform angle list: dense "
                         "region [0, dense_max] at --dense_step, then coarse "
                         "region (dense_max, max_angle] at --coarse_step. "
                         "Overrides --n_steps. Filename gets a "
                         "_d{dense_max}-{dense_step}-{coarse_step} suffix.")
    ap.add_argument("--dense_step", type=float, default=None)
    ap.add_argument("--coarse_step", type=float, default=None)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32", "float64"],
                    help="model dtype. fp32 needs ~36 GB just for weights; "
                         "fp64 needs ~72 GB. Use fp64 only on >=80GB cards "
                         "and only when probing extremely small "
                         "perturbations where fp32 roundoff matters. "
                         "Filename gets a _fp64/_fp32/_fp16 suffix when "
                         "not bf16.")
    ap.add_argument("--measure_offset", type=int, default=-2,
                    help="measure layer = n_layers + offset; default -2 "
                         "(penultimate). Use e.g. -7 for penult-5, -12 for "
                         "penult-10.")
    ap.add_argument("--metrics", default="l2",
                    help="comma-separated subset of {l2,cos,kl}; l2 always "
                         "stored. KL adds a final-layer forward per grid "
                         "point and is the slowest.")
    ap.add_argument("--mode", default="geodesic",
                    choices=["geodesic", "additive"],
                    help="geodesic: exp-map (norm-matched). additive: "
                         "a + sin(alpha) R d_perp. Same chart axes either "
                         "way.")
    ap.add_argument("--directions_pkl", default=None,
                    help="optional path to a prebuilt direction cache "
                         "(e.g. dirs_gemma_L2_sae.pkl). When set, the "
                         "script ignores DoM extraction + --pair_set / "
                         "--pairs and instead pairs every direction in "
                         "the cache against every other (C(N, 2)). "
                         "Filename gets a _dir<family> suffix so files "
                         "don't collide with the canonical run.")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--single_batch_ref", action="store_true",
                    help="run all n_steps^2 grid points in one forward and "
                         "use pf[0]=alpha=(0,0) as the within-batch "
                         "reference. Eliminates the bf16 batch-shape "
                         "rounding floor at alpha=(0,0) but needs more VRAM. "
                         "Filename gets a _sbr suffix.")
    ap.add_argument("--anchor_source", default="fineweb",
                    choices=["fineweb", "wiki_en", "wiki_zh", "code"],
                    help="which distribution the 30 anchor prompts come from. "
                         "Default 'fineweb' keeps the existing filename. Other "
                         "sources append a _src<label> suffix so files don't "
                         "collide with the canonical run.")
    ap.add_argument("--perturb_pos", type=int, default=-1,
                    help="position at which to splice perturbations and read "
                         "out L^2. -1 (default) = last token (canonical fast "
                         "path). Negative offsets allowed; pos != -1 forces "
                         "fineweb anchors and appends a _pos<k> suffix to the "
                         "filename. Direction is reused from the canonical "
                         "last-token DoM cache (see PLAN_position_ablation.md).")
    args = ap.parse_args()

    cfg = registry.TARGETS[args.target]
    perturb_layer = args.layer
    metric_set = {m.strip() for m in args.metrics.split(",") if m.strip()}
    if not metric_set <= {"l2", "cos", "kl"}:
        raise ValueError(f"unknown metric in {args.metrics}")
    record_cos = "cos" in metric_set
    record_kl = "kl" in metric_set
    print(f"[{ts()}] target={args.target} model={cfg['model']} "
          f"L={perturb_layer} mode={args.mode} metrics={sorted(metric_set)}",
          flush=True)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32, "float64": torch.float64}
    model, tokenizer, device = load_model(cfg["model"],
                                          dtype=dtype_map[args.dtype])
    n_layers = len(_get_blocks(model))
    measure_layer = n_layers + args.measure_offset
    print(f"  n_layers={n_layers} measure_layer={measure_layer} "
          f"(offset={args.measure_offset})", flush=True)

    if args.directions_pkl:
        with open(args.directions_pkl, "rb") as f:
            dirs_blob = pickle.load(f)
        all_dirs = dirs_blob["directions"]
        signs = dirs_blob.get("signs", {n: "ext" for n in all_dirs})
        family = dirs_blob.get("family", "ext")
        names = sorted(all_dirs.keys())
        pair_list = [(names[i], names[j])
                     for i in range(len(names))
                     for j in range(i + 1, len(names))]
        if args.shard:
            i, N = (int(x) for x in args.shard.split("/"))
            pair_list = pair_list[i::N]
        print(f"[{ts()}] {len(pair_list)} pairs from "
              f"{args.directions_pkl} (family={family}, "
              f"N_dirs={len(all_dirs)}, shard={args.shard})", flush=True)
    else:
        if args.pairs_file:
            pair_list = []
            with open(args.pairs_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue
                    # 'a,b' header line (optionally with extra cols)
                    if tuple(p.lower() for p in parts[:2]) == ("a", "b"):
                        continue
                    a, b = parts[0], parts[1]   # ignore any extra cols
                    pair_list.append((a, b))
            print(f"[{ts()}] loaded {len(pair_list)} pairs from "
                  f"{args.pairs_file}", flush=True)
        elif args.pairs:
            pair_list = []
            for chunk in args.pairs.split(";"):
                a, b = chunk.split(",")
                pair_list.append((a.strip(), b.strip()))
        elif args.pair_set == "tier1":
            pair_list = registry.TIER1_PAIRS
        else:
            pair_list = registry.all_pairs()
        if args.shard:
            i, N = (int(x) for x in args.shard.split("/"))
            pair_list = sorted(pair_list)[i::N]
        print(f"[{ts()}] {len(pair_list)} pairs (set={args.pair_set}, "
              f"shard={args.shard})", flush=True)

        # Decide which directions we actually need based on pair_list
        # (semantic + language + code as needed); extends an existing
        # cache rather than rebuilding it from scratch.
        needed = sorted({n for ab in pair_list for n in ab})
        dirs_blob = dirlib.extract_all(
            model, tokenizer, device, args.target, perturb_layer,
            names=needed)
        all_dirs = dirs_blob["directions"]
        signs = dirs_blob["signs"]
        family = None

    # Anchor activations (shared across pairs). Source defaults to FineWeb;
    # plan exp_map_anchor_source.md adds wiki_en, wiki_zh, code as ablations.
    # Token-position ablation (--perturb_pos != -1) takes a separate code
    # path that stores the full residual stream + position-`pos` activation
    # and is fineweb-only (see PLAN_position_ablation.md).
    full_hidden_list = None
    if args.perturb_pos != -1:
        if args.anchor_source != "fineweb":
            raise ValueError(
                "--perturb_pos != -1 currently requires --anchor_source=fineweb")
        print(f"[{ts()}] loading anchors (fineweb, pos={args.perturb_pos})",
              flush=True)
        fw = actlib.fineweb_acts_at_pos(
            model, tokenizer, device, args.target, perturb_layer,
            pos=args.perturb_pos, n=N_ANCHORS)
        full_hidden_list = fw["full_hidden"]
        activations = torch.stack(fw["pos_act"])
        contexts = None
        print(f"  N={len(activations)}, pos={args.perturb_pos}, "
              f"||pos_act||={activations.norm(dim=-1).mean():.2f}",
              flush=True)
    else:
        print(f"[{ts()}] loading anchors (source={args.anchor_source})",
              flush=True)
        if args.anchor_source == "fineweb":
            fw = actlib.fineweb_acts(model, tokenizer, device, args.target,
                                      perturb_layer, n=N_ANCHORS)
        elif args.anchor_source == "wiki_en":
            fw = actlib.wiki_acts(model, tokenizer, device, args.target,
                                   perturb_layer, n=N_ANCHORS, language="en")
        elif args.anchor_source == "wiki_zh":
            fw = actlib.wiki_acts(model, tokenizer, device, args.target,
                                   perturb_layer, n=N_ANCHORS, language="zh")
        elif args.anchor_source == "code":
            fw = actlib.code_acts(model, tokenizer, device, args.target,
                                   perturb_layer, n=N_ANCHORS)
        else:
            raise ValueError(f"unknown anchor_source {args.anchor_source}")
        contexts = fw["contexts"]
        activations = fw["activations"]
        print(f"  N={len(activations)}, ||a||={activations.norm(dim=-1).mean():.2f}",
              flush=True)

    if args.dense_max is not None:
        if args.dense_step is None or args.coarse_step is None:
            raise ValueError("--dense_max requires --dense_step and --coarse_step")
        # dense [0, dense_max] inclusive at dense_step, then coarse
        # (dense_max, max_angle] at coarse_step starting at dense_max+coarse_step
        n_dense = int(round(args.dense_max / args.dense_step)) + 1
        dense = np.linspace(0.0, args.dense_max, n_dense)
        coarse = np.arange(args.dense_max + args.coarse_step,
                           args.max_angle + 1e-9, args.coarse_step)
        angles_deg = np.concatenate([dense, coarse])
        print(f"  non-uniform angles: {len(dense)} dense + {len(coarse)} "
              f"coarse = {len(angles_deg)} total. "
              f"first 5: {angles_deg[:5]}, last 5: {angles_deg[-5:]}",
              flush=True)
    else:
        angles_deg = np.linspace(0, args.max_angle, args.n_steps)
    angles_rad = np.deg2rad(angles_deg)
    parametrize_sha = _git_sha("scripts/lib/parametrize.py")

    # filename suffix encodes non-default config
    variant_suffix = ""
    if args.measure_offset != -2:
        variant_suffix += f"_M{args.measure_offset}"
    if args.mode != "geodesic":
        variant_suffix += f"_{args.mode}"
    if metric_set != {"l2"}:
        variant_suffix += f"_{'+'.join(sorted(metric_set))}"
    if args.directions_pkl:
        variant_suffix += f"_dir{family}"
    if args.single_batch_ref:
        variant_suffix += "_sbr"
    if args.dense_max is not None:
        # encode dense+coarse spec in filename so we don't clobber
        variant_suffix += (f"_d{args.dense_max:g}-{args.dense_step:g}"
                           f"-{args.coarse_step:g}")
    if args.dtype != "bfloat16":
        variant_suffix += f"_{args.dtype.replace('float', 'fp')}"
    if args.anchor_source != "fineweb":
        # Note: the literal `fineweb` token stays in the filename for
        # back-compat with existing tooling (plot_robustness_beeswarm.py
        # and friends parse it as a fixed delimiter). The actual anchor
        # distribution is encoded by this `_src<label>` suffix.
        variant_suffix += f"_src{args.anchor_source}"
    if args.perturb_pos != -1:
        variant_suffix += f"_pos{args.perturb_pos}"

    for (a, b) in pair_list:
        if a not in all_dirs or b not in all_dirs:
            print(f"  miss {a}x{b}"); continue
        out_path = os.path.join(
            OUT_DIR,
            f"sweep2d_{args.target}_L{perturb_layer}_{a}__{b}_fineweb_"
            f"{int(args.max_angle)}deg{variant_suffix}.pkl")
        if os.path.exists(out_path) and args.skip_existing:
            print(f"  skip {a}x{b} (exists)"); continue

        d1 = all_dirs[a].float()
        d2 = all_dirs[b].float()
        t0 = time.time()
        r = run_2d_per_anchors(model, contexts, activations, d1, d2,
                                perturb_layer, measure_layer, device,
                                angles_rad, mode=args.mode,
                                record_cos=record_cos, record_kl=record_kl,
                                single_batch_ref=args.single_batch_ref,
                                pos=args.perturb_pos,
                                full_hidden_list=full_hidden_list)
        l2 = r["l2"]
        print(f"  {a:>10s} x {b:<10s} {time.time()-t0:.1f}s "
              f"l2_max={l2.max():.2f}", flush=True)

        hashes_blob = dirs_blob.get("prompt_set_hashes", {})
        out = {
            "chart": CHART,
            "parametrize_git_sha": parametrize_sha,
            "prompt_set_hashes": {
                a: hashes_blob.get(a),
                b: hashes_blob.get(b),
                "source": None,  # FineWeb
            },
            "direction_family": family,
            "seed_anchors": fw.get("seed", 42),
            "seed_random_dirs": None,
            "direction_signs": {a: signs.get(a), b: signs.get(b)},
            "eps_thresholds": eps_thresholds(),
            "model": args.target,
            "model_full": cfg["model"],
            "perturb_layer": perturb_layer,
            "measure_layer": measure_layer,
            "measure_offset": args.measure_offset,
            "perturb_pos": args.perturb_pos,
            "mode": args.mode,
            "metrics": sorted(metric_set),
            "direction_labels": (a, b),
            "source_name": "fineweb",
            "anchor_source": args.anchor_source,
            "angles_deg": angles_deg,
            "n_anchors": len(activations),
            "anchor_norm_mean": float(activations.norm(dim=-1).mean()),
            **{k: r[k] for k in r},   # l2 + optional cos, kl
        }
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f: pickle.dump(out, f)
        os.replace(tmp, out_path)

    print(f"[{ts()}] done", flush=True)


if __name__ == "__main__":
    main()
