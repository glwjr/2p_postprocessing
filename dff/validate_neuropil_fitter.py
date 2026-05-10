"""
Validate the per-cell neuropil r fitting implementation against synthetic
data with known ground truth.

The question this answers: when we generate F_M = F_C + r_true * F_N + noise
with r_true drawn from a known distribution, does the fitter recover values
close to r_true?

If yes: the SA11 result (mean r = 0.374, half of Allen's 0.68) reflects
real properties of the lab's data — lower neuropil contamination than
Allen's V1 imaging.

If no: there's a bug in the fitter that biases r estimates low. The lab
data result is suspect and needs the fitter fixed before applying to
production.

USAGE:
    1. Edit the IMPORT line below to point to your fitter.
    2. python validate_neuropil_fitter.py

The script generates 100 synthetic cells with r_true sampled from
Beta(2, 1.5) — broadly distributed across [0, 1] with mean ~0.57. Realistic
calcium transients are injected into F_C; F_N has slow drift plus noise.
The fitter runs on each cell, and we compare estimated vs true r.
"""

# ============================================================================
# IMPORT YOUR FITTER HERE
# ============================================================================
# The fitter should take (F_M, F_N, fs) and return at minimum the estimated
# r value for one cell. If your function has a different signature, wrap it
# with a small adapter — see fit_one_cell() below.
#
# Example imports to try, in rough order of likely names:
#   from compute_dff import fit_neuropil_r
#   from compute_dff import estimate_r
#   from compute_dff import fit_r_one_cell
# ============================================================================

import sys
from pathlib import Path

# Allow running from anywhere; assume compute_dff.py is in the parent dir.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from compute_dff import estimate_neuropil_coefs as _fitter


def fit_one_cell(F_M, F_N, fs):
    # estimate_neuropil_coefs operates on (n_rois, n_timepoints) arrays
    F_M_2d = F_M[np.newaxis, :]
    F_N_2d = F_N[np.newaxis, :]
    r_array, _ = _fitter(F_M_2d, F_N_2d)
    return float(r_array[0])


# ============================================================================
# Synthetic data generator
# ============================================================================

import numpy as np


def generate_synthetic_cell(rng, n_frames, fs, r_true):
    """One synthetic cell with calcium transients, noisy neuropil, known r."""
    # F_C: baseline + photon noise + sparse calcium transients
    F_C = 100.0 + rng.normal(0, 2.0, n_frames)
    # ~1 transient per 10 seconds, exponential decay with tau=1s
    n_transients = int(n_frames / fs / 10)
    decay = np.exp(-np.arange(60) / (fs * 1.0))  # 60-frame tail at fs Hz
    for _ in range(n_transients):
        onset = rng.integers(0, n_frames - len(decay))
        amplitude = rng.uniform(20, 80)
        F_C[onset : onset + len(decay)] += amplitude * decay

    # F_N: noisier, slower-varying
    F_N = 40.0 + rng.normal(0, 5.0, n_frames)
    # Slow drift component, ~30s timescale
    F_N += 15 * np.sin(np.arange(n_frames) * 2 * np.pi / (fs * 30))
    # Higher-frequency neuropil activity
    F_N += rng.normal(0, 3.0, n_frames)

    # Measured signal
    F_M = F_C + r_true * F_N + rng.normal(0, 1.0, n_frames)
    return F_M, F_N


def main():
    rng = np.random.default_rng(0)
    n_cells = 100
    n_frames = 9000  # 5 min at 30 Hz; long enough for stable estimation
    fs = 30.0

    # Draw r_true from Beta(2, 1.5) — covers [0, 1] with mean 4/7 ~ 0.57.
    # Comparable to Allen's reported mean 0.68 with substantial spread.
    r_true_array = rng.beta(2.0, 1.5, n_cells)

    print(f"Validating fitter on {n_cells} synthetic cells, {n_frames / fs:.0f}s each")
    print(
        f"r_true distribution: mean={r_true_array.mean():.3f}, "
        f"std={r_true_array.std():.3f}, "
        f"range=[{r_true_array.min():.3f}, {r_true_array.max():.3f}]"
    )
    print()

    r_hat_array = np.full(n_cells, np.nan)
    for i, r_true in enumerate(r_true_array):
        F_M, F_N = generate_synthetic_cell(rng, n_frames, fs, r_true)
        try:
            r_hat = fit_one_cell(F_M, F_N, fs)
            if r_hat is not None:
                r_hat_array[i] = r_hat
        except Exception as e:
            print(f"  Cell {i}: fitter raised {type(e).__name__}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  Fit {i + 1}/{n_cells}")

    # Drop any NaNs (cells where the fitter returned None / flagged)
    valid = ~np.isnan(r_hat_array)
    n_valid = valid.sum()
    print(f"\nValid fits: {n_valid}/{n_cells}")

    r_true_v = r_true_array[valid]
    r_hat_v = r_hat_array[valid]

    # Bias and correlation
    bias = (r_hat_v - r_true_v).mean()
    abs_err = np.abs(r_hat_v - r_true_v).mean()
    pearson_r = np.corrcoef(r_true_v, r_hat_v)[0, 1]
    # Linear fit r_hat vs r_true to detect proportional bias
    slope, intercept = np.polyfit(r_true_v, r_hat_v, 1)

    print()
    print("=" * 60)
    print("Recovery diagnostics")
    print("=" * 60)
    print(f"  Mean bias (r_hat - r_true):       {bias:+.4f}")
    print(f"  Mean absolute error:              {abs_err:.4f}")
    print(f"  Pearson r(r_true, r_hat):         {pearson_r:.4f}")
    print(f"  Linear fit r_hat = {slope:.3f} * r_true + {intercept:+.3f}")
    print()
    print(f"  r_true mean: {r_true_v.mean():.3f}  |  r_hat mean: {r_hat_v.mean():.3f}")
    print(f"  r_true std:  {r_true_v.std():.3f}  |  r_hat std:  {r_hat_v.std():.3f}")

    # Verdict
    print()
    print("=" * 60)
    print("Verdict")
    print("=" * 60)
    issues = []
    if abs(bias) > 0.05:
        issues.append(
            f"Mean bias of {bias:+.3f} exceeds ±0.05 — the fitter is "
            f"systematically {'over' if bias > 0 else 'under'}-estimating r."
        )
    if pearson_r < 0.85:
        issues.append(
            f"Pearson r of {pearson_r:.3f} is below 0.85 — the fitter is not "
            f"tracking ground-truth r reliably even in rank order."
        )
    if abs(slope - 1.0) > 0.15:
        issues.append(
            f"Slope of {slope:.3f} indicates proportional bias — the fitter "
            f"compresses ({'toward zero' if slope < 1 else 'away from zero'}) "
            f"the true r distribution by ~{abs(1 - slope) * 100:.0f}%."
        )

    if not issues:
        print("  PASS: Fitter recovers ground-truth r within tolerances.")
        print("  Implication: SA11 mean r=0.374 reflects real lab-data property,")
        print("  not a bug. Lab imaging produces less neuropil contamination")
        print("  than Allen's V1 imaging on average.")
    else:
        print("  FAIL: Fitter has issues:")
        for issue in issues:
            print(f"    - {issue}")
        print()
        print("  Implication: SA11 result is suspect. Fix the fitter before")
        print("  applying to the rest of SA11_LG batch 1.")


if __name__ == "__main__":
    main()
