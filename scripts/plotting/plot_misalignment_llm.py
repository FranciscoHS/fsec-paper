"""LLM misalignment sweep: fitted exponent p vs off-axis tilt theta.

The LLM counterpart of the toy ``plot_misalignment_lp`` figure (item 1 of
the baseline-strengthening plan). For each tilt theta the contrastive pair
directions are rotated off their true axes (see
``rotated_contrastive_directions.py``), re-swept, and fit; this script
aggregates the per-pair exponents into a median +/- IQR band over the
pair x ensemble-seed population at each theta.

  * x-axis: cos(theta) (left = aligned/real features, right = orthogonal)
  * y-axis: fitted superellipse exponent p
  * band:   IQR over the overlap-filtered pairs x ensemble seeds
  * p = 2 reference line; secondary top axis in misalignment degrees

Reads the per-cell fits written by ``fit_misalignment_thrfixed.py``:
  results/fits/fits_<target>_L<layer>_dirrotcontrastive_th<deg>_s<seed>
      [_thrfixed].pkl

Usage:
  python -m scripts.plotting.plot_misalignment_llm --target gemma --layer 2
"""
from __future__ import annotations
import os, sys, re, glob, pickle, argparse
sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt

from scripts.plotting.plot_robustness_beeswarm import median_p_for_pair

FITS_DIR = "results/fits"
OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def _collect(target: str, layer: int, thr: str):
    """Return {theta_deg: list[p]} pooled over pairs and ensemble seeds."""
    pat = os.path.join(
        FITS_DIR,
        f"fits_{target}_L{layer}_dirrotcontrastive_th*_s*{thr}.pkl")
    rx = re.compile(rf"_dirrotcontrastive_th(\d+)_s(\d+){re.escape(thr)}\.pkl$")
    by_theta: dict[int, list[float]] = {}
    n_files = 0
    for fp in sorted(glob.glob(pat)):
        m = rx.search(os.path.basename(fp))
        if not m:
            continue
        theta = int(m.group(1))
        with open(fp, "rb") as f:
            full = pickle.load(f)
        ps = [median_p_for_pair(fit) for fit in full.values()]
        ps = [p for p in ps if np.isfinite(p)]
        by_theta.setdefault(theta, []).extend(ps)
        n_files += 1
    print(f"  {n_files} fit files matched {os.path.basename(pat)}")
    return by_theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--use_thrpair", action="store_true", default=True,
                    help="load the _thrpair fits (default; current per-pair "
                         "threshold protocol).")
    ap.add_argument("--canonical", dest="use_thrpair", action="store_false",
                    help="load canonical (non-threshold) fits instead.")
    ap.add_argument("--exact", action="store_true",
                    help="load the exact-geodesic fits (..._thrpair_exact), "
                         "matching the paper's protocol; appends '_exact' to "
                         "the output filename.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    thr = "_thrpair" if args.use_thrpair else ""
    if args.exact:
        thr += "_exact"
    by_theta = _collect(args.target, args.layer, thr)
    if not by_theta:
        print("no rotated-contrastive fits found; run the sweep + "
              "fit_misalignment_thrfixed.py first")
        return

    thetas = np.array(sorted(by_theta))
    x = np.cos(np.radians(thetas))
    med = np.array([np.median(by_theta[t]) for t in thetas])
    q25 = np.array([np.percentile(by_theta[t], 25) for t in thetas])
    q75 = np.array([np.percentile(by_theta[t], 75) for t in thetas])
    ns = [len(by_theta[t]) for t in thetas]
    for t, m, n in zip(thetas, med, ns):
        print(f"  theta={t:>3d}deg  cos={np.cos(np.radians(t)):.3f}  "
              f"median p={m:.3f}  (n={n})")

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    color = "tab:blue"
    ax.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, med, color=color, marker="o", linewidth=1.6,
            label="contrastive (rotated)")
    ax.axhline(2, color="k", ls="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("cosine similarity to contrastive directions")
    ax.set_ylabel(r"fitted superellipse exponent $p$")
    ax.set_xlim(-0.02, 1.02)
    ax.invert_xaxis()   # left = aligned (cos=1), right = orthogonal (cos=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=10)

    def _cos_to_deg(c):
        return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))

    def _deg_to_cos(deg):
        return np.cos(np.radians(deg))

    secax = ax.secondary_xaxis("top", functions=(_cos_to_deg, _deg_to_cos))
    secax.set_xlabel(r"misalignment angle $\theta$ (deg)")
    secax.set_xticks([15, 30, 45, 60, 75, 90])

    fig.tight_layout()
    out_png = args.out or os.path.join(
        OUT_DIR, f"misalignment_llm_{args.target}_L{args.layer}{thr}.png")
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
