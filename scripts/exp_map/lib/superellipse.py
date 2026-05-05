"""Iso-L^2 contour extraction and superellipse fit |x|^p + |y|^p = 1.

In the exp-map chart, the iso-L^2 contour at a chosen threshold is a
closed curve in (alpha_1, alpha_2). Normalize each axis by its 1D
intercept (the alpha at which the 1D L^2 curve crosses the threshold)
and fit p.

Robustness protocol (per Plans/exp_map_workspace.md):
- threshold: {0.50, 0.75} * min(max(L2_grid[:, 0]), max(L2_grid[0, :]))
- angle range: alpha_i <= {30, 45, 60} degrees
- bootstrap: 200 resamples of the anchor index, median + [5, 95] CI

Headline statement target: "p in [low, high] across threshold x range x
bootstrap, for >= X% of pairs across 6 models."
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- low-level fit ----------

def fit_superellipse(x: np.ndarray, y: np.ndarray,
                     bounds: tuple = (0.5, 10.0)) -> dict:
    """Fit p in |x|^p + |y|^p = 1 to normalized contour points.

    Args:
      x, y: 1D arrays, both already normalized by their axis intercepts.
      bounds: search interval for p.
    Returns dict with p, mean_radial_frac, max_radial_frac, n_pts. p is
    np.nan if there are fewer than 3 usable points.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (x > 0.05) & (y > 0.05) & (x < 1.5) & (y < 1.5)
    x = x[mask]; y = y[mask]
    if len(x) < 3:
        return {"p": np.nan, "mean_radial_frac": np.nan,
                "max_radial_frac": np.nan, "n_pts": int(len(x))}
    def res(p):
        return float(np.sum((np.abs(x) ** p + np.abs(y) ** p - 1.0) ** 2))
    r = minimize_scalar(res, bounds=bounds, method="bounded")
    p_fit = float(r.x)
    lp = (np.abs(x) ** p_fit + np.abs(y) ** p_fit) ** (1.0 / p_fit)
    frac = np.abs(lp - 1.0)
    return {"p": p_fit, "mean_radial_frac": float(frac.mean()),
            "max_radial_frac": float(frac.max()), "n_pts": int(len(x))}


def superellipse_curve(p: float, n_pts: int = 200):
    t = np.linspace(0.0, np.pi / 2.0, n_pts)
    x = np.abs(np.cos(t)) ** (2.0 / p) * np.sign(np.cos(t))
    y = np.abs(np.sin(t)) ** (2.0 / p) * np.sign(np.sin(t))
    return x, y


# ---------- contour extraction ----------

def extract_contour(angles_deg: np.ndarray, l2_grid: np.ndarray,
                    threshold: float) -> np.ndarray:
    """Iso-threshold contour points from a 2D L^2 grid.

    angles_deg has shape (n,), l2_grid has shape (n, n) with
    l2_grid[i, j] = L2 at (alpha_1=angles_deg[i], alpha_2=angles_deg[j]).

    Returns array of shape (m, 2) with columns (alpha_1_deg, alpha_2_deg).
    """
    fig, ax = plt.subplots()
    cs = ax.contour(angles_deg, angles_deg, l2_grid.T, levels=[threshold])
    pts = []
    if hasattr(cs, "allsegs"):
        for seg_list in cs.allsegs:
            for seg in seg_list:
                pts.append(seg)
    else:
        for coll in cs.collections:
            for path in coll.get_paths():
                pts.append(path.vertices)
    plt.close(fig)
    if not pts:
        return np.zeros((0, 2))
    return np.concatenate(pts, axis=0)


def axis_intercept(angles_deg: np.ndarray, l2_curve: np.ndarray,
                   threshold: float) -> float:
    """Linear-interpolated angle (deg) at which the 1D L2 curve hits threshold.

    l2_curve has shape (n,). Returns nan if threshold not reached.
    """
    if l2_curve.max() < threshold:
        return float("nan")
    for i in range(1, len(l2_curve)):
        if (l2_curve[i - 1] - threshold) * (l2_curve[i] - threshold) <= 0:
            x0, x1 = angles_deg[i - 1], angles_deg[i]
            y0, y1 = l2_curve[i - 1], l2_curve[i]
            if y1 == y0:
                return float(x0)
            t = (threshold - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return float("nan")


# ---------- robustness protocol ----------

def threshold_levels(l2_grid_mean: np.ndarray) -> dict:
    """Compute the two anchor-thresholds per the protocol.

    base = min(max(L2[:, 0]), max(L2[0, :])) so both axes can reach it.
    """
    base = min(l2_grid_mean[:, 0].max(), l2_grid_mean[0, :].max())
    return {"50": 0.50 * base, "75": 0.75 * base, "base": base}


def fit_p_one_pass(angles_deg: np.ndarray, l2_grid_mean: np.ndarray,
                    threshold: float, max_alpha_deg: float,
                    exact_geodesic: bool = False) -> dict:
    """One (threshold, range) fit on the mean grid.

    If exact_geodesic is False (default), normalize contour points by
    (alpha_1/t1, alpha_2/t2) — the sin alpha ~ alpha small-angle form.
    If True, use the exact geodesic form derived from
        sin^n(alpha(phi)) [w_1 cos^n phi + w_2 sin^n phi] = const,
    i.e. (sin alpha cos phi / sin alpha_1, sin alpha sin phi / sin alpha_2)
    where alpha_1, alpha_2 are the 1D plateau-breaking angles. The fit
    routine itself is unchanged; only the normalization differs.
    """
    raw = extract_contour(angles_deg, l2_grid_mean, threshold)
    if raw.size == 0:
        return {"p": np.nan, "n_pts": 0, "mean_radial_frac": np.nan,
                "max_radial_frac": np.nan, "intercepts": (np.nan, np.nan)}
    # exclude grid-boundary clips: drop points within 1 deg of grid max
    grid_max = float(angles_deg.max())
    edge_eps = 1.0
    mask = (raw[:, 0] < grid_max - edge_eps) & (raw[:, 1] < grid_max - edge_eps)
    raw = raw[mask]
    # range cut
    mask = (raw[:, 0] <= max_alpha_deg) & (raw[:, 1] <= max_alpha_deg)
    raw = raw[mask]
    if len(raw) < 3:
        return {"p": np.nan, "n_pts": int(len(raw)),
                "mean_radial_frac": np.nan, "max_radial_frac": np.nan,
                "intercepts": (np.nan, np.nan)}
    # axis intercepts (using mean grid)
    t1 = axis_intercept(angles_deg, l2_grid_mean[:, 0], threshold)
    t2 = axis_intercept(angles_deg, l2_grid_mean[0, :], threshold)
    if not np.isfinite(t1) or not np.isfinite(t2) or t1 <= 0 or t2 <= 0:
        return {"p": np.nan, "n_pts": int(len(raw)),
                "mean_radial_frac": np.nan, "max_radial_frac": np.nan,
                "intercepts": (t1, t2)}
    if exact_geodesic:
        # raw[:, 0] = alpha(phi) cos phi, raw[:, 1] = alpha(phi) sin phi
        # (in degrees). Use sin alpha exactly; small-r safe div via clamp.
        r_deg = np.hypot(raw[:, 0], raw[:, 1])
        r_safe = np.where(r_deg > 1e-12, r_deg, 1.0)
        r_rad = np.deg2rad(r_deg)
        t1_rad = np.deg2rad(t1)
        t2_rad = np.deg2rad(t2)
        xn = (np.sin(r_rad) / np.sin(t1_rad)) * (raw[:, 0] / r_safe)
        yn = (np.sin(r_rad) / np.sin(t2_rad)) * (raw[:, 1] / r_safe)
    else:
        xn = raw[:, 0] / t1
        yn = raw[:, 1] / t2
    fit = fit_superellipse(xn, yn)
    fit["intercepts"] = (t1, t2)
    return fit


def robust_p_fit_fixed_l2(angles_deg: np.ndarray,
                           l2_grid_per_anchor: np.ndarray,
                           thresh_l2: float,
                           threshold_factors: tuple = (0.5, 1.0, 2.0),
                           max_alphas: tuple = (30.0, 45.0, 60.0),
                           n_bootstrap: int = 0,
                           seed: int = 42,
                           exact_geodesic: bool = False) -> dict:
    """Fixed-L^2 robustness protocol.

    Like robust_p_fit but uses thresh = factor * thresh_l2 for each
    factor in threshold_factors (rather than {50%, 75%} of axis max).
    Output structure is identical so downstream code (median_p_for_pair,
    plot_robustness_beeswarm) doesn't need changes.
    """
    rng = np.random.RandomState(seed)
    n_a = l2_grid_per_anchor.shape[0]
    mean_grid = np.median(l2_grid_per_anchor, axis=0)
    levels = {f"{f}xT": float(f * thresh_l2) for f in threshold_factors}
    out = {}
    for f in threshold_factors:
        thresh = float(f * thresh_l2)
        tl = f"{f}xT"
        for ma in max_alphas:
            base = fit_p_one_pass(angles_deg, mean_grid, thresh, ma,
                                   exact_geodesic=exact_geodesic)
            if n_bootstrap == 0:
                p = base["p"] if np.isfinite(base["p"]) else np.nan
                ps = np.array([p]) if np.isfinite(p) else np.array([])
            else:
                ps = []
                for _ in range(n_bootstrap):
                    idx = rng.randint(0, n_a, size=n_a)
                    grid_b = np.median(l2_grid_per_anchor[idx], axis=0)
                    fb = fit_p_one_pass(angles_deg, grid_b, thresh, ma,
                                         exact_geodesic=exact_geodesic)
                    if np.isfinite(fb["p"]):
                        ps.append(fb["p"])
                ps = np.array(ps)
            out[(tl, ma)] = {
                "p_point": base["p"],
                "p_median": float(np.median(ps)) if len(ps) else np.nan,
                "p_lo": float(np.percentile(ps, 5)) if len(ps) else np.nan,
                "p_hi": float(np.percentile(ps, 95)) if len(ps) else np.nan,
                "n_bootstrap_ok": int(len(ps)),
                "n_pts": base["n_pts"],
                "mean_radial_frac": base["mean_radial_frac"],
                "intercepts": base["intercepts"],
                "threshold": float(thresh),
                "threshold_label": tl,
                "max_alpha_deg": float(ma),
            }
    out["__levels__"] = levels
    return out


def robust_p_fit(angles_deg: np.ndarray, l2_grid_per_anchor: np.ndarray,
                  thresholds: tuple = ("50", "75"),
                  max_alphas: tuple = (30.0, 45.0, 60.0),
                  n_bootstrap: int = 200, seed: int = 42,
                  exact_geodesic: bool = False) -> dict:
    """Full robustness protocol.

    l2_grid_per_anchor has shape (n_anchors, n, n). Mean is taken across
    anchors to build the "data" grid; bootstrap resamples the anchor index
    with replacement to compute (5, 50, 95) percentile of p per
    (threshold, max_alpha) cell.

    Returns nested dict: {(thresh_label, max_alpha): {p_median, p_lo, p_hi,
                                                       n_pts, mean_radial_frac}}.
    """
    rng = np.random.RandomState(seed)
    n_a = l2_grid_per_anchor.shape[0]
    # Median over anchors (more robust to anchor heterogeneity than mean;
    # matches the older fig4 pipeline.)
    mean_grid = np.median(l2_grid_per_anchor, axis=0)
    levels = threshold_levels(mean_grid)
    out = {}
    for tl in thresholds:
        thresh = levels[tl]
        for ma in max_alphas:
            base = fit_p_one_pass(angles_deg, mean_grid, thresh, ma,
                                   exact_geodesic=exact_geodesic)
            if n_bootstrap == 0:
                # Fast path: use the point estimate directly. CI fields
                # are filled with the point estimate so downstream code
                # that reads p_lo / p_hi still works.
                p = base["p"] if np.isfinite(base["p"]) else np.nan
                ps = np.array([p]) if np.isfinite(p) else np.array([])
            else:
                ps = []
                for _ in range(n_bootstrap):
                    idx = rng.randint(0, n_a, size=n_a)
                    grid_b = np.median(l2_grid_per_anchor[idx], axis=0)
                    f = fit_p_one_pass(angles_deg, grid_b, thresh, ma,
                                        exact_geodesic=exact_geodesic)
                    if np.isfinite(f["p"]):
                        ps.append(f["p"])
                ps = np.array(ps)
            cell = {
                "p_point": base["p"],
                "p_median": float(np.median(ps)) if len(ps) else np.nan,
                "p_lo": float(np.percentile(ps, 5)) if len(ps) else np.nan,
                "p_hi": float(np.percentile(ps, 95)) if len(ps) else np.nan,
                "n_bootstrap_ok": int(len(ps)),
                "n_pts": base["n_pts"],
                "mean_radial_frac": base["mean_radial_frac"],
                "intercepts": base["intercepts"],
                "threshold": float(thresh),
                "threshold_label": tl,
                "max_alpha_deg": float(ma),
            }
            out[(tl, ma)] = cell
    out["__levels__"] = levels
    return out
