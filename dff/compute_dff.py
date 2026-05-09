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
    3. Neuropil-correct: F_corr = F - 0.7 * Fneu
    4. Light Gaussian pre-smoothing (1s sigma)
    5. Rolling 8th percentile baseline (30s window)
    6. Floor F0 at 0.1 * median(F_corr) per ROI
    7. dF/F = (F_corr - F0_floored) / F0_floored
    8. Save outputs

Design note on step 7:
    The dF/F numerator uses unsmoothed F_corr rather than F_smoothed.
    Smoothing is applied only to stabilize the F0 baseline estimate;
    the signal itself is intentionally kept at full temporal resolution.
    The floor is anchored to median(F_corr) — the unsmoothed signal —
    so that the floor and the numerator share the same reference scale.

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

PIPELINE_VERSION = "0.2.0"

# ============================================================================
# Default parameters
# ============================================================================

DEFAULT_PARAMS = {
    # Empirically tuned to 0.5 for SA-line lab data (N=5 sessions across
    # SA11 and SA17). The Suite2p convention is 0.7, but on this lab's V1
    # GCaMP imaging that value over-subtracts, dropping ~20-27% of
    # iscell-passing ROIs via the pre-filter. At 0.5 the pre-filter drops
    # ~5-7% with no other adverse effects (no floor activations, stable
    # bulk distribution, comparable transient amplitudes). See the dff/
    # README "Step 2: Neuropil correction" section for the full rationale.
    "neuropil_coef": 0.5,
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
# Core computation
# ============================================================================


def compute_dff(
    F: np.ndarray, Fneu: np.ndarray, fs: float, params: dict
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

    Returns
    -------
    dff : np.ndarray, shape (n_rois, n_timepoints)
        dF/F traces.
    F0 : np.ndarray, shape (n_rois, n_timepoints)
        Floored baseline used for division.
    floor_mask : np.ndarray, shape (n_rois, n_timepoints), dtype bool
        True where F0 was clipped to the floor.
    """
    # Step 1: Neuropil correction
    F_corr = F - params["neuropil_coef"] * Fneu

    # Step 2: Light pre-smoothing — used only to stabilize F0 estimation.
    # The dF/F numerator uses unsmoothed F_corr (see module docstring).
    presmooth_sigma_frames = params["presmooth_sigma_sec"] * fs
    F_smoothed = gaussian_filter1d(F_corr, sigma=presmooth_sigma_frames, axis=1)

    # Step 3: Rolling percentile baseline.
    # NOTE: percentile_filter is called in a Python loop over ROIs. This is
    # intentional for clarity, but becomes a bottleneck at large cell counts
    # (>500 ROIs). Consider a Numba or C extension if runtime is a concern.
    window_frames = int(params["baseline_window_sec"] * fs)
    F0 = np.array(
        [
            percentile_filter(
                trace, percentile=params["baseline_percentile"], size=window_frames
            )
            for trace in F_smoothed
        ]
    )

    # Step 4: Per-ROI floor anchored to median(F_corr) — unsmoothed — so the
    # floor shares the same reference scale as the dF/F numerator.
    median_F_per_roi = np.median(F_corr, axis=1, keepdims=True)
    floor_per_roi = params["f0_floor_epsilon"] * median_F_per_roi
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
    for i, roi in enumerate(example_idx):
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
    output_path: Path,
) -> None:
    """Save per-cell summary statistics as CSV."""
    df = pd.DataFrame(
        {
            "row_index": np.arange(dff.shape[0]),
            "suite2p_roi_index": cell_indices,
            "iscell_prob": iscell_prob,
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
        "parameters": params,
    }
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_dff_h5(
    dff: np.ndarray,
    F0: np.ndarray,
    cell_indices: np.ndarray,
    iscell_prob: np.ndarray,
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
        meta = f.create_group("metadata")
        meta.attrs["fs"] = fs
        meta.attrs["neuropil_coef"] = params["neuropil_coef"]
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

    # Pre-filter: drop ROIs where F_corrected dips substantially negative.
    # These ROIs have neuropil contamination exceeding the cell signal during
    # parts of the session, and produce extreme dF/F regardless of the F0
    # floor. They are typically not real cells (partial cells, out-of-plane
    # ROIs, or contamination-dominated detections). Using the 5th percentile
    # rather than the median catches cells that are mostly positive but dip
    # negative for some fraction of the session.
    F_corr_check = F_kept - params["neuropil_coef"] * Fneu_kept
    pre_filter_mask = np.percentile(F_corr_check, 5, axis=1) > 0
    n_pre_dropped = int((~pre_filter_mask).sum())
    if n_pre_dropped > 0:
        print(
            f"  [Pre-filter] Dropping {n_pre_dropped} ROIs with negative "
            f"5th-percentile F_corrected (neuropil dominates in tail)"
        )
        F_kept = F_kept[pre_filter_mask]
        Fneu_kept = Fneu_kept[pre_filter_mask]
        cell_indices = cell_indices[pre_filter_mask]
        iscell_prob_kept = iscell_prob_kept[pre_filter_mask]

    if F_kept.shape[0] == 0:
        raise RuntimeError(
            "No ROIs passed both iscell and F_corrected filters. "
            "Check the upstream Suite2p output."
        )

    print(
        f"Computing dF/F (neuropil_coef={params['neuropil_coef']}, "
        f"baseline=rolling_p{params['baseline_percentile']}, "
        f"window={params['baseline_window_sec']}s, "
        f"presmooth_sigma={params['presmooth_sigma_sec']}s, "
        f"f0_floor_epsilon={params['f0_floor_epsilon']})"
    )
    dff, F0, floor_mask = compute_dff(F_kept, Fneu_kept, fs, params)

    # Post-filter: drop cells with high floor-fraction. Cells where F0 is
    # pinned to the floor for > post_filter_floor_frac of the session indicate
    # that the rolling baseline can't track the trace cleanly — typically
    # because of dim signal, transient artifacts, or contamination not caught
    # by the pre-filter. Including them produces extreme negative dF/F outliers.
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
    print(f"\nSaving outputs to: {output_dir}")
    save_dff_h5(
        dff, F0, cell_indices, iscell_prob_kept, params, fs, output_dir / "dff.h5"
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
        "--neuropil_coef", type=float, default=DEFAULT_PARAMS["neuropil_coef"]
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
