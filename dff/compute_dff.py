"""
Standardized dF/F computation for Najafi Lab 2P imaging pipeline.

Takes Suite2p outputs from a single session and produces:
    - dff.h5              : dF/F traces with metadata
    - dff_diagnostics.png : validation figure
    - dff_cell_summary.csv: per-cell statistics
    - dff_metadata.json   : pipeline parameters and run info

Usage:
    python compute_dff.py --suite2p_dir /path/to/suite2p/plane0 --output_dir /path/to/output

Pipeline:
    1. Load F.npy, Fneu.npy, iscell.npy, ops.npy
    2. Filter ROIs by iscell probability threshold
    3. Estimate per-ROI neuropil coef r via Allen's cross-validated method;
       neuropil-correct: F_corr = F - r * Fneu
    4. Light Gaussian pre-smoothing (1s sigma)
    5. Rolling 8th percentile baseline (30s window)
    6. Floor F0 at 0.1 * |median(F_corr)| per ROI
    7. dF/F = (F_corr - F0_floored) / F0_floored
    8. Save outputs

Design note on step 7:
    The dF/F numerator uses unsmoothed F_corr rather than F_smoothed.
    Smoothing is applied only to stabilize the F0 baseline estimate;
    the signal itself is intentionally kept at full temporal resolution.
    The floor is anchored to |median(F_corr)| — absolute value of the
    unsmoothed signal — so that the floor is always non-negative and
    shares the same reference scale as the numerator.

"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.optimize import minimize_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

PIPELINE_VERSION = "0.4.1"

# ============================================================================
# Default parameters
# ============================================================================

DEFAULT_PARAMS = {
    # Per-ROI neuropil coefficient estimation (default). None triggers Allen
    # Brain Observatory's cross-validated regression method (Visual Behavior
    # 2P whitepaper, Section F). Set to a float to override with a fixed
    # global value (e.g. 0.7 for the Suite2p convention).
    "neuropil_coef": None,
    "presmooth_sigma_sec": 1.0,
    "baseline_percentile": 8,
    "baseline_window_sec": 30.0,
    "f0_floor_epsilon": 0.1,
    "iscell_threshold": 0.3,
    # Post-filter: drop ROIs where F0 is pinned to the floor for more than
    # this fraction of the session. Such ROIs have dim or contaminated signal
    # and produce extreme negative dF/F. Exposed as a parameter so it is
    # logged to metadata and adjustable without touching source.
    "post_filter_floor_frac": 0.05,
}


# ============================================================================
# Neuropil coefficient estimation (Allen Visual Behavior 2P whitepaper, §F)
# ============================================================================


def _solve_F_C(target: np.ndarray, lam: float) -> np.ndarray:
    """
    Closed-form F_C minimizing
        sum_t (F_C[t] - target[t])^2 + lam * sum_t (F_C[t+1] - F_C[t])^2.

    Setting dE/dF_C = 0 gives a tridiagonal system. Boundary rows have only
    one neighbor, so the diagonal there is (1 + lam) instead of (1 + 2*lam).
    """
    T = target.shape[0]
    main = np.full(T, 1.0 + 2.0 * lam)
    main[0] = 1.0 + lam
    main[-1] = 1.0 + lam
    off = np.full(T - 1, -lam)
    A = diags([off, main, off], [-1, 0, 1], format="csc")
    return spsolve(A, target)


def _compute_E(F_M: np.ndarray, F_N: np.ndarray, r: float, lam: float) -> float:
    """E (time-mean) for a given r, with F_C taken as its closed-form minimizer."""
    target = F_M - r * F_N
    F_C = _solve_F_C(target, lam)
    residual_mse = np.mean((F_C - target) ** 2)
    deriv_mse = np.mean(np.diff(F_C) ** 2)
    return float(residual_mse + lam * deriv_mse)


def _fit_r_one_cell(
    F_M: np.ndarray, F_N: np.ndarray, lam: float = 0.05
) -> tuple[float | None, str]:
    """
    Estimate r for one cell using the Allen Visual Behavior 2P whitepaper's
    cross-validated regression with smoothness regularization (page 36).

    The whitepaper specifies gradient descent with a particular learning
    rate and stopping rule. We use bracketed Brent's method (scipy's
    minimize_scalar with method='bounded') instead — same objective
    function, same cross-validation, but ~200x faster (10-20 function
    evaluations vs. 400+ gradient steps near a flat minimum).

    The optimization is run on the second half of the trace (the eval half),
    matching the whitepaper's cross-validation strategy.

    Returns (r, status). r is None if all attempts failed.
    Status is one of:
      - 'converged'             : optimum found cleanly in [0, 1]
      - 'converged_widened'     : found at boundary, widened bracket and clipped
      - 'failed_E_too_high'     : E exceeds whitepaper threshold of 2*|<F_M>|
      - 'flagged_constant_fneu' : F_N has no variation
    """
    F_N = np.asarray(F_N, dtype=np.float64)
    F_M = np.asarray(F_M, dtype=np.float64)

    fneu_min = F_N.min()
    fneu_range = F_N.max() - fneu_min
    if fneu_range < 1e-9:
        return None, "flagged_constant_fneu"

    # Per-ROI normalization: F_N to (0, 1); F_M scaled by the same factor.
    # This standardizes the optimization across cells of varying brightness
    # without changing the optimum r (verified by E-landscape sweeps).
    F_N_norm = (F_N - fneu_min) / fneu_range
    F_M_norm = F_M / fneu_range

    half = F_M_norm.shape[0] // 2
    F_M_eval = F_M_norm[half:]
    F_N_eval = F_N_norm[half:]

    # Convergence threshold from whitepaper: E < 2 * |<F_M>|.
    E_thresh = 2.0 * abs(np.mean(F_M_norm))

    def _minimize(bracket):
        result = minimize_scalar(
            lambda r: _compute_E(F_M_eval, F_N_eval, r, lam),
            bounds=bracket,
            method="bounded",
            options={"xatol": 1e-4},
        )
        return float(result.x), float(result.fun)

    # Attempt 1: bounded search in [0, 1].
    r, E = _minimize((0.0, 1.0))

    # If the optimum is at the boundary, the true minimum may lie outside.
    # Widen and re-check; clip to [0, 1] for the final answer.
    at_boundary = (r < 0.01) or (r > 0.99)
    if at_boundary:
        r_wide, E_wide = _minimize((-0.5, 1.5))
        if E_wide < E:
            r, E = r_wide, E_wide
        if r < 0.0 or r > 1.0:
            r_clipped = float(np.clip(r, 0.0, 1.0))
            E_clipped = _compute_E(F_M_eval, F_N_eval, r_clipped, lam)
            if E_clipped < E_thresh:
                return r_clipped, "converged_widened"
            return None, "failed_E_too_high"

    if E < E_thresh:
        return float(r), "converged"
    return None, "failed_E_too_high"


def estimate_neuropil_coefs(
    F: np.ndarray, Fneu: np.ndarray, lam: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-ROI neuropil contamination ratio r.

    Implements Allen Brain Observatory's per-cell r estimation (Visual
    Behavior 2P Technical Whitepaper, Section F, page 36). For each ROI,
    the algorithm jointly estimates r and the cleaned cell trace F_C by
    minimizing
        E = <(F_C - (F_M - r * F_N))^2 + lambda * (dF_C/dt)^2>_t
    with lambda = 0.05. F_C has a closed form for any r (tridiagonal
    smoothing); cross-validated scalar minimization on r minimizes E.

    The Allen whitepaper reports mean r ≈ 0.68 (SD ≈ 0.38) on benchmark
    GCaMP6f data — substantial cell-to-cell heterogeneity that a fixed
    coefficient cannot capture.

    ROIs where the optimization fails (e.g. constant F_N, optimum outside
    plausible bounds) receive the median r of converged ROIs as a fallback,
    matching the whitepaper's strategy for non-converged cells.

    Validated against synthetic ground truth (100 cells, r drawn from
    Beta(2, 1.5)): bias ≈ 0, slope ≈ 1, Pearson r ≈ 0.999. See
    prototype_allen_neuropil.py for the validation harness.

    Parameters
    ----------
    F, Fneu : np.ndarray, shape (n_rois, n_timepoints)
    lam : float, default 0.05
        Smoothness weight on the cleaned trace.

    Returns
    -------
    r_per_roi : np.ndarray, shape (n_rois,), dtype float32
        Per-ROI contamination ratios in [0, 1]. Failed cells receive the
        population median r as a fallback.
    converged : np.ndarray, shape (n_rois,), dtype bool
        True where the optimization converged cleanly (excluding fallbacks).
    """
    n_rois = F.shape[0]
    r_vals = np.full(n_rois, np.nan)
    converged = np.zeros(n_rois, dtype=bool)

    for i in range(n_rois):
        r, status = _fit_r_one_cell(F[i], Fneu[i], lam=lam)
        if r is not None:
            r_vals[i] = r
            converged[i] = True

    if converged.any():
        fallback_r = float(np.median(r_vals[converged]))
    else:
        fallback_r = 0.7  # last-ditch
    r_vals[~converged] = fallback_r

    return r_vals.astype(np.float32), converged


# ============================================================================
# Core computation
# ============================================================================


def compute_dff(
    F: np.ndarray,
    Fneu: np.ndarray,
    fs: float,
    params: dict,
    r_per_roi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dF/F for a set of ROI traces.

    Parameters
    ----------
    F, Fneu : np.ndarray, shape (n_rois, n_timepoints)
        Raw fluorescence and neuropil traces from Suite2p.
    fs : float
        Imaging frame rate (Hz).
    params : dict
        Pipeline parameters; see DEFAULT_PARAMS.
    r_per_roi : np.ndarray, shape (n_rois,), optional
        Per-ROI neuropil contamination ratios from estimate_neuropil_coefs().
        If None, falls back to params["neuropil_coef"] as a fixed scalar.

    Returns
    -------
    dff : np.ndarray, shape (n_rois, n_timepoints)
        dF/F traces.
    F0 : np.ndarray, shape (n_rois, n_timepoints)
        Floored baseline used for division.
    floor_mask : np.ndarray, shape (n_rois, n_timepoints), dtype bool
        True where F0 was clipped to the floor.
    """
    # Step 1: Neuropil correction — per-ROI r if provided, else fixed scalar.
    if r_per_roi is None:
        coef = float(params.get("neuropil_coef") or 0.7)
        r_per_roi = np.full(F.shape[0], coef, dtype=np.float32)
    F_corr = F - r_per_roi[:, np.newaxis] * Fneu

    # Step 2: Light pre-smoothing — used only to stabilize F0 estimation.
    # The dF/F numerator uses unsmoothed F_corr (see module docstring).
    presmooth_sigma_frames = params["presmooth_sigma_sec"] * fs
    F_smoothed = gaussian_filter1d(F_corr, sigma=presmooth_sigma_frames, axis=1)

    # Step 3: Rolling percentile baseline.
    # NOTE: percentile_filter is called in a Python loop over ROIs. This is
    # intentional for clarity, but becomes a bottleneck at large cell counts
    # (>500 ROIs). Consider a Numba or C extension if runtime is a concern.
    window_frames = (
        int(params["baseline_window_sec"] * fs) | 1
    )  # ensure odd for symmetric centering
    F0 = np.array(
        [
            percentile_filter(
                trace, percentile=params["baseline_percentile"], size=window_frames
            )
            for trace in F_smoothed
        ]
    )

    # Step 4: Per-ROI floor anchored to |median(F_corr)| — absolute value
    # ensures the floor is always non-negative even when neuropil over-subtraction
    # pushes median(F_corr) below zero. Without abs, a negative floor_per_roi can
    # propagate to a negative F0_floored, inverting the sign of dF/F.
    median_F_per_roi = np.median(F_corr, axis=1, keepdims=True)
    floor_per_roi = params["f0_floor_epsilon"] * np.abs(median_F_per_roi)
    floor_mask = F0 < floor_per_roi
    F0_floored = np.where(floor_mask, floor_per_roi, F0)

    # Step 5: dF/F — numerator uses unsmoothed F_corr (full temporal resolution).
    dff = (F_corr - F0_floored) / F0_floored

    return dff.astype(np.float32), F0_floored.astype(np.float32), floor_mask


# ============================================================================
# Diagnostic outputs
# ============================================================================


def make_diagnostic_figure(
    dff: np.ndarray,
    F0: np.ndarray,
    floor_mask: np.ndarray,
    fs: float,
    output_path: Path,
) -> None:
    """Save a 6-panel diagnostic figure for the session."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Panel 1: dF/F histogram (clipped for visibility)
    ax = axes[0, 0]
    clip_range = (-3, 5)
    bins = np.linspace(*clip_range, 100)
    ax.hist(
        np.clip(dff.ravel(), *clip_range),
        bins=bins,
        density=True,
        color="steelblue",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("dF/F")
    ax.set_ylabel("density")
    ax.set_title(f"dF/F distribution (n_cells={dff.shape[0]})")
    ax.grid(alpha=0.3)

    # Panel 2: Per-ROI maxima
    ax = axes[0, 1]
    per_roi_max = dff.max(axis=1)
    ax.hist(
        np.clip(per_roi_max, 0, 30),
        bins=np.linspace(0, 30, 60),
        color="darkgreen",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel("per-ROI max dF/F")
    ax.set_ylabel("count")
    ax.set_title(f"Per-ROI maxima (median={np.median(per_roi_max):.2f})")
    ax.grid(alpha=0.3)

    # Panel 3: F0 floor diagnostic
    ax = axes[0, 2]
    floor_frac_per_roi = floor_mask.mean(axis=1)
    ax.hist(
        floor_frac_per_roi, bins=30, color="firebrick", edgecolor="white", linewidth=0.3
    )
    ax.set_xlabel("fraction of timepoints at F0 floor")
    ax.set_ylabel("count")
    n_problem = int((floor_frac_per_roi > 0.05).sum())
    ax.set_title(f"F0 floor activations (n>5%: {n_problem}/{dff.shape[0]})")
    ax.grid(alpha=0.3)

    # Panel 4: F0 trajectory over time for a few example ROIs
    ax = axes[1, 0]
    rng = np.random.default_rng(42)
    n_examples = 5
    example_idx = rng.choice(
        F0.shape[0], size=min(n_examples, F0.shape[0]), replace=False
    )
    t_minutes = np.arange(F0.shape[1]) / fs / 60
    for roi in example_idx:
        ax.plot(t_minutes, F0[roi], lw=0.8, alpha=0.8)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("F0")
    ax.set_title(f"F0 trajectories ({n_examples} example ROIs)")
    ax.grid(alpha=0.3)

    # Panel 5: Example dF/F traces (60s from middle of session)
    ax = axes[1, 1]
    duration_sec = 60
    n_frames = int(duration_sec * fs)
    start = dff.shape[1] // 2
    t_seconds = np.arange(n_frames) / fs
    n_traces = 4
    trace_idx = rng.choice(
        dff.shape[0], size=min(n_traces, dff.shape[0]), replace=False
    )
    offset = 0
    for roi in trace_idx:
        trace = dff[roi, start : start + n_frames]
        ax.plot(t_seconds, trace + offset, lw=0.8)
        offset += max(4, trace.max() - trace.min() + 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dF/F (offset)")
    ax.set_title(f"Example dF/F traces (60s from middle of session)")
    ax.grid(alpha=0.3)

    # Panel 6: Percentile breakdown
    ax = axes[1, 2]
    percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    values = [np.percentile(dff, p) for p in percentiles]
    ax.barh(range(len(percentiles)), values, color="slategray")
    ax.set_yticks(range(len(percentiles)))
    ax.set_yticklabels([f"{p}%" for p in percentiles])
    ax.set_xlabel("dF/F")
    ax.set_title("dF/F distribution percentiles")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def make_cell_summary(
    dff: np.ndarray,
    F0: np.ndarray,
    floor_mask: np.ndarray,
    F: np.ndarray,
    cell_indices: np.ndarray,
    iscell_prob: np.ndarray,
    r_per_roi: np.ndarray,
    output_path: Path,
) -> None:
    """Save per-cell summary statistics as CSV."""
    df = pd.DataFrame(
        {
            "row_index": np.arange(dff.shape[0]),
            "suite2p_roi_index": cell_indices,
            "iscell_prob": iscell_prob,
            "neuropil_coef": r_per_roi,
            "median_F": np.median(F, axis=1),
            "median_F0": np.median(F0, axis=1),
            "dff_max": dff.max(axis=1),
            "dff_min": dff.min(axis=1),
            "dff_median": np.median(dff, axis=1),
            "frac_at_floor": floor_mask.mean(axis=1),
        }
    )
    df.to_csv(output_path, index=False, float_format="%.4f")


def save_metadata(
    params: dict,
    fs: float,
    n_timepoints: int,
    n_rois_total: int,
    n_cells_after_iscell: int,
    n_cells_final: int,
    r_per_roi: np.ndarray,
    n_r_fallback: int,
    output_path: Path,
) -> None:
    """Save run metadata as JSON.

    Records both the post-iscell count and the final count after all
    pre/post filters, so the two numbers can be cross-referenced against
    dff.h5 without ambiguity.
    """
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "fs": float(fs),
        "n_timepoints": int(n_timepoints),
        "session_length_sec": float(n_timepoints / fs),
        "n_rois_total": int(n_rois_total),
        # n_cells_after_iscell: passed iscell threshold, before pre/post filters.
        "n_cells_after_iscell": int(n_cells_after_iscell),
        # n_cells_final: written to dff.h5 — after all filtering steps.
        "n_cells_final": int(n_cells_final),
        "neuropil_coef_per_roi": {
            "mean": float(r_per_roi.mean()),
            "std": float(r_per_roi.std()),
            "min": float(r_per_roi.min()),
            "max": float(r_per_roi.max()),
            "n_fallback": int(n_r_fallback),
        },
        "parameters": params,
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_dff_h5(
    dff: np.ndarray,
    F0: np.ndarray,
    cell_indices: np.ndarray,
    iscell_prob: np.ndarray,
    r_per_roi: np.ndarray,
    params: dict,
    fs: float,
    output_path: Path,
) -> None:
    """Save dF/F and metadata to HDF5."""
    with h5py.File(output_path, "w") as f:
        f.create_dataset("dff", data=dff, compression="gzip", compression_opts=4)
        f.create_dataset("F0", data=F0, compression="gzip", compression_opts=4)
        f.create_dataset("cell_indices", data=cell_indices)
        f.create_dataset("iscell_prob", data=iscell_prob)
        f.create_dataset("neuropil_coef", data=r_per_roi)
        meta = f.create_group("metadata")
        meta.attrs["fs"] = fs
        meta.attrs["neuropil_coef_method"] = "per_roi_regression"
        meta.attrs["neuropil_coef_mean"] = float(r_per_roi.mean())
        meta.attrs["neuropil_coef_std"] = float(r_per_roi.std())
        meta.attrs["baseline_method"] = (
            f"rolling_percentile_{params['baseline_percentile']}"
        )
        meta.attrs["baseline_window_sec"] = params["baseline_window_sec"]
        meta.attrs["presmooth_sigma_sec"] = params["presmooth_sigma_sec"]
        meta.attrs["f0_floor_epsilon"] = params["f0_floor_epsilon"]
        meta.attrs["iscell_threshold"] = params["iscell_threshold"]
        meta.attrs["post_filter_floor_frac"] = params["post_filter_floor_frac"]
        meta.attrs["pipeline_version"] = PIPELINE_VERSION
        meta.attrs["timestamp_utc"] = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )


# ============================================================================
# Main
# ============================================================================


def run(suite2p_dir: Path, output_dir: Path, params: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Suite2p outputs from: {suite2p_dir}")
    F = np.load(suite2p_dir / "F.npy")
    Fneu = np.load(suite2p_dir / "Fneu.npy")
    iscell = np.load(suite2p_dir / "iscell.npy")
    ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])

    n_rois_total = F.shape[0]
    print(
        f"  F shape: {F.shape}, frame rate: {fs} Hz, "
        f"session length: {F.shape[1] / fs / 60:.1f} min"
    )

    # Cell selection by iscell probability
    cell_mask = iscell[:, 1] > params["iscell_threshold"]
    n_cells_after_iscell = int(cell_mask.sum())
    print(
        f"  Cells kept (iscell prob > {params['iscell_threshold']}): "
        f"{n_cells_after_iscell} / {n_rois_total}"
    )

    if n_cells_after_iscell == 0:
        raise RuntimeError(
            f"No ROIs passed iscell threshold of {params['iscell_threshold']}. "
            "Check the iscell.npy file or lower the threshold."
        )

    F_kept = F[cell_mask]
    Fneu_kept = Fneu[cell_mask]
    cell_indices = np.where(cell_mask)[0]
    iscell_prob_kept = iscell[cell_mask, 1].astype(np.float32)

    # Estimate per-ROI neuropil coefficient r (Allen whitepaper, Section F).
    # If params["neuropil_coef"] is a float, skip estimation and use it directly.
    fixed_coef = params.get("neuropil_coef")
    if fixed_coef is not None:
        r_per_roi = np.full(F_kept.shape[0], float(fixed_coef), dtype=np.float32)
        r_converged = np.ones(F_kept.shape[0], dtype=bool)
        print(f"  Neuropil coef: fixed={fixed_coef}")
    else:
        r_per_roi, r_converged = estimate_neuropil_coefs(F_kept, Fneu_kept)
        n_fallback = int((~r_converged).sum())
        print(
            f"  Neuropil coef: mean={r_per_roi.mean():.3f} "
            f"± {r_per_roi.std():.3f} "
            f"[{r_per_roi.min():.3f}, {r_per_roi.max():.3f}]"
            + (f", {n_fallback} fallback" if n_fallback else "")
        )

    # No pre-filter step (removed in v0.4.0). The per-cell r estimator's
    # boundary-widening logic handles cells where neuropil dominates by
    # flagging them and assigning the population median r as fallback.

    print(
        f"Computing dF/F (baseline=rolling_p{params['baseline_percentile']}, "
        f"window={params['baseline_window_sec']}s, "
        f"presmooth_sigma={params['presmooth_sigma_sec']}s, "
        f"f0_floor_epsilon={params['f0_floor_epsilon']})"
    )
    dff, F0, floor_mask = compute_dff(F_kept, Fneu_kept, fs, params, r_per_roi)

    # Post-filter: drop cells with high floor-fraction.
    post_filter_floor_frac = params["post_filter_floor_frac"]
    floor_frac_per_cell = floor_mask.mean(axis=1)
    post_filter_mask = floor_frac_per_cell <= post_filter_floor_frac
    n_post_dropped = int((~post_filter_mask).sum())
    if n_post_dropped > 0:
        print(
            f"  [Post-filter] Dropping {n_post_dropped} ROIs with "
            f">{post_filter_floor_frac:.0%} of timepoints at F0 floor"
        )
        F_kept = F_kept[post_filter_mask]
        Fneu_kept = Fneu_kept[post_filter_mask]
        cell_indices = cell_indices[post_filter_mask]
        iscell_prob_kept = iscell_prob_kept[post_filter_mask]
        r_per_roi = r_per_roi[post_filter_mask]
        r_converged = r_converged[post_filter_mask]
        dff = dff[post_filter_mask]
        F0 = F0[post_filter_mask]
        floor_mask = floor_mask[post_filter_mask]

    n_cells_final = F_kept.shape[0]
    print(f"  Final cell count: {n_cells_final}")

    if n_cells_final == 0:
        raise RuntimeError(
            "No ROIs survived all filtering steps. "
            "Check the upstream Suite2p output and filter parameters."
        )

    # Distribution check
    print(
        f"  dF/F median: {np.median(dff):.4f}, "
        f"99th percentile: {np.percentile(dff, 99):.3f}, "
        f"0.1th percentile: {np.percentile(dff, 0.1):.3f}, "
        f"max: {dff.max():.3f}, min: {dff.min():.3f}"
    )
    n_extreme = int((np.abs(dff).max(axis=1) > 50).sum())
    print(f"  ROIs with |value| > 50 anywhere: {n_extreme} / {dff.shape[0]}")
    floor_frac = floor_mask.mean()
    print(f"  Fraction of timepoints at F0 floor: {floor_frac:.4f}")

    # Save outputs
    n_r_fallback = int((~r_converged).sum())
    print(f"\nSaving outputs to: {output_dir}")
    save_dff_h5(
        dff,
        F0,
        cell_indices,
        iscell_prob_kept,
        r_per_roi,
        params,
        fs,
        output_dir / "dff.h5",
    )
    print(f"  dff.h5")

    make_diagnostic_figure(dff, F0, floor_mask, fs, output_dir / "dff_diagnostics.png")
    print(f"  dff_diagnostics.png")

    make_cell_summary(
        dff,
        F0,
        floor_mask,
        F_kept,
        cell_indices,
        iscell_prob_kept,
        r_per_roi,
        output_dir / "dff_cell_summary.csv",
    )
    print(f"  dff_cell_summary.csv")

    save_metadata(
        params,
        fs,
        F.shape[1],
        n_rois_total,
        n_cells_after_iscell,
        n_cells_final,
        r_per_roi,
        n_r_fallback,
        output_dir / "dff_metadata.json",
    )
    print(f"  dff_metadata.json")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute standardized dF/F from Suite2p outputs."
    )
    parser.add_argument(
        "--suite2p_dir",
        type=Path,
        required=True,
        help="Path to suite2p/plane0 directory containing F.npy etc.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write dff.h5 and diagnostic outputs.",
    )
    parser.add_argument(
        "--neuropil_coef",
        type=float,
        default=None,
        help=(
            "Fixed neuropil subtraction coefficient for all ROIs. "
            "Default: estimate per ROI via Allen's cross-validated regression "
            "(Visual Behavior 2P whitepaper, Section F)."
        ),
    )
    parser.add_argument(
        "--presmooth_sigma_sec",
        type=float,
        default=DEFAULT_PARAMS["presmooth_sigma_sec"],
    )
    parser.add_argument(
        "--baseline_percentile", type=int, default=DEFAULT_PARAMS["baseline_percentile"]
    )
    parser.add_argument(
        "--baseline_window_sec",
        type=float,
        default=DEFAULT_PARAMS["baseline_window_sec"],
    )
    parser.add_argument(
        "--f0_floor_epsilon", type=float, default=DEFAULT_PARAMS["f0_floor_epsilon"]
    )
    parser.add_argument(
        "--iscell_threshold", type=float, default=DEFAULT_PARAMS["iscell_threshold"]
    )
    parser.add_argument(
        "--post_filter_floor_frac",
        type=float,
        default=DEFAULT_PARAMS["post_filter_floor_frac"],
        help=(
            "Drop ROIs where F0 is at the floor for more than this fraction "
            "of the session (default: %(default)s)."
        ),
    )
    args = parser.parse_args()

    params = {
        "neuropil_coef": args.neuropil_coef,
        "presmooth_sigma_sec": args.presmooth_sigma_sec,
        "baseline_percentile": args.baseline_percentile,
        "baseline_window_sec": args.baseline_window_sec,
        "f0_floor_epsilon": args.f0_floor_epsilon,
        "iscell_threshold": args.iscell_threshold,
        "post_filter_floor_frac": args.post_filter_floor_frac,
    }

    run(args.suite2p_dir, args.output_dir, params)


if __name__ == "__main__":
    main()
