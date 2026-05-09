"""
Compare compute_dff.py output against Allen Brain Observatory's published
dF/F traces, as a validation step.

Our pipeline and Allen's pipeline are not identical — Allen uses a per-cell
neuropil correction r value while we use a fixed 0.7 coefficient, and the
baseline estimation methods differ. So expect numerical differences. What
this script checks is bulk distributional agreement:

    - Median dF/F should be near zero in both (sparse activity).
    - 99th percentile dF/F should be in a similar range.
    - Per-cell dF/F maxima should correlate strongly between the two.
    - Visually, transients should appear at the same timepoints with
      comparable amplitudes.

If any of those break, that's a signal that our parameters are off in a way
worth investigating before applying to real Najafi Lab data.

Usage:
    python compare_to_allen.py \
        --our_dff /path/to/dff_output/dff.h5 \
        --allen_dff /path/to/allen_dff.npy \
        --output_fig /path/to/comparison.png

The script prints summary stats to stdout and writes a 4-panel comparison
figure.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_our_dff(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load our pipeline's dF/F + the cell indices we kept + frame rate."""
    with h5py.File(path, "r") as f:
        dff = f["dff"][:]
        cell_indices = f["cell_indices"][:]
        fs = float(f["metadata"].attrs["fs"])
    return dff, cell_indices, fs


def print_summary_comparison(our_dff: np.ndarray, allen_dff: np.ndarray) -> None:
    """Print bulk distribution stats side-by-side."""
    print("=" * 64)
    print("Bulk distribution comparison")
    print("=" * 64)
    print(f"{'metric':<28} {'ours':>15} {'allen':>15}")
    print("-" * 64)
    for label, fn in [
        ("median", np.median),
        ("mean", np.mean),
        ("99th percentile", lambda x: np.percentile(x, 99)),
        ("99.9th percentile", lambda x: np.percentile(x, 99.9)),
        ("0.1th percentile", lambda x: np.percentile(x, 0.1)),
        ("max", np.max),
        ("min", np.min),
    ]:
        ours = fn(our_dff)
        allen = fn(allen_dff)
        print(f"{label:<28} {ours:>15.4f} {allen:>15.4f}")
    print()


def print_per_cell_correlation(our_dff: np.ndarray, allen_dff: np.ndarray) -> None:
    """Per-cell dF/F maxima should correlate strongly across the two pipelines."""
    our_max = our_dff.max(axis=1)
    allen_max = allen_dff.max(axis=1)
    n = min(len(our_max), len(allen_max))
    r = np.corrcoef(our_max[:n], allen_max[:n])[0, 1]
    print(f"Per-cell max-dF/F correlation (Pearson r): {r:.3f}")
    if r < 0.5:
        print(
            "  WARNING: correlation is lower than expected. "
            "The two pipelines should rank cells by activity similarly even "
            "with different dF/F numerical values."
        )
    print()


def make_comparison_figure(
    our_dff: np.ndarray,
    allen_dff: np.ndarray,
    fs: float,
    output_path: Path,
) -> None:
    """4-panel figure comparing the two pipelines."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: distribution histograms (overlaid, clipped for visibility)
    ax = axes[0, 0]
    clip_range = (-2, 4)
    bins = np.linspace(*clip_range, 80)
    ax.hist(
        np.clip(our_dff.ravel(), *clip_range),
        bins=bins,
        density=True,
        alpha=0.5,
        label=f"ours (n={our_dff.shape[0]})",
        color="steelblue",
    )
    ax.hist(
        np.clip(allen_dff.ravel(), *clip_range),
        bins=bins,
        density=True,
        alpha=0.5,
        label=f"allen (n={allen_dff.shape[0]})",
        color="firebrick",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("dF/F")
    ax.set_ylabel("density")
    ax.set_title("dF/F distribution (clipped to [-2, 4])")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: per-cell max scatter
    ax = axes[0, 1]
    n = min(our_dff.shape[0], allen_dff.shape[0])
    our_max = our_dff[:n].max(axis=1)
    allen_max = allen_dff[:n].max(axis=1)
    ax.scatter(allen_max, our_max, alpha=0.5, s=12, color="slategray")
    lim = max(our_max.max(), allen_max.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, label="y = x")
    r = np.corrcoef(our_max, allen_max)[0, 1]
    ax.set_xlabel("Allen per-cell max dF/F")
    ax.set_ylabel("our per-cell max dF/F")
    ax.set_title(f"Per-cell maxima (Pearson r = {r:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: example trace overlay (one cell, 60s window)
    ax = axes[1, 0]
    rng = np.random.default_rng(0)
    cell_idx = rng.choice(min(our_dff.shape[0], allen_dff.shape[0]))
    duration_sec = 60
    n_frames = int(duration_sec * fs)
    start = our_dff.shape[1] // 2
    t = np.arange(n_frames) / fs
    ax.plot(
        t,
        our_dff[cell_idx, start : start + n_frames],
        lw=0.8,
        label="ours",
        color="steelblue",
    )
    ax.plot(
        t,
        allen_dff[cell_idx, start : start + n_frames],
        lw=0.8,
        label="allen",
        color="firebrick",
        alpha=0.7,
    )
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dF/F")
    ax.set_title(f"Example trace: cell index {cell_idx} (60s from middle)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: percentile-percentile plot
    ax = axes[1, 1]
    percentiles = np.linspace(0.1, 99.9, 100)
    our_pct = np.percentile(our_dff, percentiles)
    allen_pct = np.percentile(allen_dff, percentiles)
    ax.plot(allen_pct, our_pct, color="darkgreen", lw=1.5)
    lim_lo = min(our_pct.min(), allen_pct.min())
    lim_hi = max(our_pct.max(), allen_pct.max())
    ax.plot(
        [lim_lo, lim_hi],
        [lim_lo, lim_hi],
        "k--",
        linewidth=0.8,
        alpha=0.5,
        label="y = x",
    )
    ax.set_xlabel("Allen dF/F percentile values")
    ax.set_ylabel("our dF/F percentile values")
    ax.set_title("Q-Q plot (0.1 to 99.9 percentiles)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote comparison figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare compute_dff.py output against Allen's published dF/F."
    )
    parser.add_argument(
        "--our_dff", type=Path, required=True, help="Path to our dff.h5 output."
    )
    parser.add_argument(
        "--allen_dff",
        type=Path,
        required=True,
        help="Path to allen_dff.npy from adapt_allen_to_suite2p.py.",
    )
    parser.add_argument(
        "--output_fig",
        type=Path,
        default=Path("comparison.png"),
        help="Output path for the comparison figure.",
    )
    args = parser.parse_args()

    our_dff, cell_indices, fs = load_our_dff(args.our_dff)
    allen_dff = np.load(args.allen_dff)

    # Our pipeline drops cells via the iscell threshold and pre/post filters,
    # so we may have fewer cells than Allen. Subset Allen's traces to match
    # the cells we kept, so the comparison is apples-to-apples.
    allen_dff_matched = allen_dff[cell_indices]

    print(f"Our dF/F shape:   {our_dff.shape}")
    print(f"Allen dF/F shape: {allen_dff_matched.shape} (matched to our cells)")
    print()

    print_summary_comparison(our_dff, allen_dff_matched)
    print_per_cell_correlation(our_dff, allen_dff_matched)
    make_comparison_figure(our_dff, allen_dff_matched, fs, args.output_fig)


if __name__ == "__main__":
    main()
