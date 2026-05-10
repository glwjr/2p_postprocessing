"""
Unit tests for compute_dff().

These tests use synthetic, deterministic inputs where the correct output
is known analytically. They are pure unit tests — no file I/O — and run
fast enough to use as pre-commit checks.
"""

import numpy as np

from compute_dff import compute_dff, estimate_neuropil_coefs

# ============================================================================
# Neuropil correction
# ============================================================================


class TestNeuropilCorrection:
    """Verify F - neuropil_coef * Fneu is applied correctly."""

    def test_neuropil_correction_constant_signal(self, default_params):
        """Constant F and Fneu → F_corr is constant at F - coef*Fneu."""
        n_rois, n_frames = 3, 3000
        F_value, Fneu_value = 10.0, 2.0
        F = np.full((n_rois, n_frames), F_value, dtype=np.float32)
        Fneu = np.full((n_rois, n_frames), Fneu_value, dtype=np.float32)

        # Pin a fixed coef. Default params now trigger the per-cell Allen
        # estimator, which is degenerate on constant inputs (no F_N variance
        # to fit against). For testing the arithmetic of neuropil correction,
        # we want a known scalar coefficient.
        params = dict(default_params)
        params["neuropil_coef"] = 0.7

        dff, F0, _ = compute_dff(F, Fneu, fs=30.0, params=params)

        expected_F_corr = F_value - params["neuropil_coef"] * Fneu_value
        # F_corr is constant, so F0 should match it (rolling percentile of a
        # constant is the constant). dF/F should be ~0.
        np.testing.assert_allclose(F0, expected_F_corr, rtol=1e-3)
        np.testing.assert_allclose(dff, 0.0, atol=1e-3)

    def test_custom_neuropil_coefficient(self, default_params):
        """A different neuropil_coef should change F_corr accordingly."""
        n_rois, n_frames = 2, 2000
        F = np.full((n_rois, n_frames), 100.0, dtype=np.float32)
        Fneu = np.full((n_rois, n_frames), 50.0, dtype=np.float32)

        params = dict(default_params)
        params["neuropil_coef"] = 0.5
        # Expected F_corr = 100 - 0.5*50 = 75.

        _, F0, _ = compute_dff(F, Fneu, fs=30.0, params=params)
        np.testing.assert_allclose(F0, 75.0, rtol=1e-3)


# ============================================================================
# Baseline behavior
# ============================================================================


class TestBaseline:
    """Verify the F0 baseline tracks slow drift and ignores transients."""

    def test_flat_signal_yields_near_zero_dff(self, default_params):
        """A perfectly flat signal should produce dF/F ≈ 0."""
        n_rois, n_frames = 5, 3000
        F = np.full((n_rois, n_frames), 100.0, dtype=np.float32)
        Fneu = np.full((n_rois, n_frames), 10.0, dtype=np.float32)

        dff, _, _ = compute_dff(F, Fneu, fs=30.0, params=default_params)
        # Allow tiny float noise; everything should be at zero.
        assert np.abs(dff).max() < 1e-4

    def test_baseline_tracks_slow_drift(self, default_params):
        """A slow linear drift should be absorbed into F0, not show up as dF/F."""
        n_rois, n_frames = 1, 6000  # 200s at 30Hz
        # Linear drift from 100 to 110 over the full session.
        drift = np.linspace(100.0, 110.0, n_frames, dtype=np.float32)
        F = drift[None, :].copy()
        Fneu = np.zeros_like(F)

        dff, F0, _ = compute_dff(F, Fneu, fs=30.0, params=default_params)

        # Most of the trace should be near zero. Edges of percentile_filter
        # are noisier, so check the middle 80%.
        lo, hi = int(0.1 * n_frames), int(0.9 * n_frames)
        assert np.abs(dff[0, lo:hi]).max() < 0.05

    def test_step_transient_recovered(self, default_params):
        """A known transient should produce dF/F near its true magnitude."""
        n_rois, n_frames = 1, 6000
        baseline = 100.0
        F = np.full((n_rois, n_frames), baseline, dtype=np.float32)
        # 1-second pulse at 200% of baseline (so dF/F should be ~1.0 at peak).
        pulse_start = 3000
        pulse_len = 30  # 1 second at 30 Hz
        F[0, pulse_start : pulse_start + pulse_len] = baseline * 2

        Fneu = np.zeros_like(F)

        dff, _, _ = compute_dff(F, Fneu, fs=30.0, params=default_params)

        # Peak dF/F in the pulse window. The Gaussian pre-smoothing will
        # spread the pulse slightly and reduce its peak; expect 0.7–1.1.
        peak = dff[0, pulse_start : pulse_start + pulse_len].max()
        assert 0.7 < peak < 1.1, f"Expected peak ~1.0, got {peak}"


# ============================================================================
# F0 floor behavior
# ============================================================================


class TestF0Floor:
    """Verify the floor clamp activates on dim signal and not otherwise."""

    def test_floor_does_not_activate_on_healthy_signal(self, default_params):
        """A bright stable signal should never hit the floor."""
        n_rois, n_frames = 3, 3000
        F = np.full((n_rois, n_frames), 200.0, dtype=np.float32)
        Fneu = np.full((n_rois, n_frames), 30.0, dtype=np.float32)

        _, _, floor_mask = compute_dff(F, Fneu, fs=30.0, params=default_params)
        assert floor_mask.sum() == 0

    def test_floor_activates_on_dim_dropout(self, default_params):
        """If F drops below f0_floor_epsilon * median, floor_mask should fire."""
        n_rois, n_frames = 1, 6000
        F = np.full((n_rois, n_frames), 100.0, dtype=np.float32)
        # Drop the middle of the trace far below the floor (epsilon=0.1, so
        # median*0.1 = 10). Setting it to 1 should trip the floor.
        F[0, 2000:4000] = 1.0
        Fneu = np.zeros_like(F)

        _, _, floor_mask = compute_dff(F, Fneu, fs=30.0, params=default_params)
        # Floor should fire somewhere in the dropout region.
        assert floor_mask[0, 2500:3500].any()


# ============================================================================
# Output shape and dtype
# ============================================================================


class TestOutputShapes:
    """Sanity checks on output shapes and dtypes."""

    def test_shapes_match_input(self, default_params):
        n_rois, n_frames = 7, 1500
        F = np.full((n_rois, n_frames), 100.0, dtype=np.float32)
        Fneu = np.full((n_rois, n_frames), 20.0, dtype=np.float32)

        dff, F0, floor_mask = compute_dff(F, Fneu, fs=30.0, params=default_params)
        assert dff.shape == (n_rois, n_frames)
        assert F0.shape == (n_rois, n_frames)
        assert floor_mask.shape == (n_rois, n_frames)

    def test_dtypes(self, default_params):
        F = np.full((2, 1500), 100.0, dtype=np.float32)
        Fneu = np.full((2, 1500), 20.0, dtype=np.float32)

        dff, F0, floor_mask = compute_dff(F, Fneu, fs=30.0, params=default_params)
        assert dff.dtype == np.float32
        assert F0.dtype == np.float32
        assert floor_mask.dtype == np.bool_


# ============================================================================
# Per-ROI independence
# ============================================================================


class TestPerROIIndependence:
    """Computation on one ROI should not depend on other ROIs."""

    def test_roi_independence(self, default_params):
        """Adding an ROI should not change the dF/F of the others."""
        rng = np.random.default_rng(123)
        n_frames = 3000

        F1 = rng.uniform(80, 120, (3, n_frames)).astype(np.float32)
        Fneu1 = rng.uniform(15, 30, (3, n_frames)).astype(np.float32)

        # Add a fourth ROI with very different statistics.
        F_extra = rng.uniform(500, 700, (1, n_frames)).astype(np.float32)
        Fneu_extra = rng.uniform(100, 200, (1, n_frames)).astype(np.float32)

        F2 = np.vstack([F1, F_extra])
        Fneu2 = np.vstack([Fneu1, Fneu_extra])

        dff1, _, _ = compute_dff(F1, Fneu1, fs=30.0, params=default_params)
        dff2, _, _ = compute_dff(F2, Fneu2, fs=30.0, params=default_params)

        # First three rows of the larger run should match the smaller run exactly.
        np.testing.assert_allclose(dff1, dff2[:3], rtol=1e-5, atol=1e-6)


# ============================================================================
# Determinism
# ============================================================================


class TestDeterminism:
    """Same input → same output, every time."""

    def test_repeated_calls_are_identical(self, default_params):
        rng = np.random.default_rng(7)
        F = rng.uniform(80, 120, (5, 3000)).astype(np.float32)
        Fneu = rng.uniform(15, 30, (5, 3000)).astype(np.float32)

        dff_a, F0_a, mask_a = compute_dff(F, Fneu, fs=30.0, params=default_params)
        dff_b, F0_b, mask_b = compute_dff(F, Fneu, fs=30.0, params=default_params)

        np.testing.assert_array_equal(dff_a, dff_b)
        np.testing.assert_array_equal(F0_a, F0_b)
        np.testing.assert_array_equal(mask_a, mask_b)


# ============================================================================
# Per-cell r estimator (Allen Visual Behavior 2P whitepaper, Section F)
# ============================================================================


class TestNeuropilEstimator:
    """Verify estimate_neuropil_coefs recovers ground-truth r."""

    @staticmethod
    def _synthesize(rng, n_frames, fs, r_true):
        """One synthetic cell with calcium transients and known r."""
        F_C = 100.0 + rng.normal(0, 2.0, n_frames)
        n_transients = int(n_frames / fs / 10)
        decay = np.exp(-np.arange(60) / fs)
        for _ in range(n_transients):
            onset = rng.integers(0, n_frames - len(decay))
            F_C[onset : onset + len(decay)] += rng.uniform(20, 80) * decay
        F_N = 40.0 + rng.normal(0, 5.0, n_frames)
        F_N += 15 * np.sin(np.arange(n_frames) * 2 * np.pi / (fs * 30))
        F_N += rng.normal(0, 3.0, n_frames)
        F_M = F_C + r_true * F_N + rng.normal(0, 1.0, n_frames)
        return F_M, F_N

    def test_recovers_known_r_distribution(self):
        """Bias <0.05, slope within 15% of 1, Pearson > 0.85 over 30 cells.

        Tighter bounds (bias < 0.01, Pearson > 0.99) are checked in the
        full validate_neuropil_fitter.py harness with 100 cells. This is a
        quick smoke test for CI; it's small enough to run in a few seconds.
        """
        rng = np.random.default_rng(0)
        n_cells, n_frames, fs = 30, 9000, 30.0
        r_true = rng.beta(2.0, 1.5, n_cells)
        F = np.zeros((n_cells, n_frames))
        Fneu = np.zeros((n_cells, n_frames))
        for i, r in enumerate(r_true):
            F[i], Fneu[i] = self._synthesize(rng, n_frames, fs, r)

        r_hat, converged = estimate_neuropil_coefs(F, Fneu)

        assert (
            converged.all()
        ), f"Expected all converged, got {converged.sum()}/{n_cells}"
        bias = (r_hat - r_true).mean()
        slope, _ = np.polyfit(r_true, r_hat, 1)
        pearson = np.corrcoef(r_true, r_hat)[0, 1]

        assert abs(bias) < 0.05, f"Bias {bias:+.4f} exceeds ±0.05"
        assert abs(slope - 1.0) < 0.15, f"Slope {slope:.3f} too far from 1.0"
        assert pearson > 0.85, f"Pearson {pearson:.3f} below 0.85"

    def test_returns_correct_shapes_and_dtypes(self):
        """estimate_neuropil_coefs returns float32 r and bool converged."""
        rng = np.random.default_rng(1)
        F = rng.uniform(80, 120, (5, 3000)).astype(np.float32)
        Fneu = rng.uniform(15, 30, (5, 3000)).astype(np.float32)
        r_hat, converged = estimate_neuropil_coefs(F, Fneu)

        assert r_hat.shape == (5,)
        assert converged.shape == (5,)
        assert r_hat.dtype == np.float32
        assert converged.dtype == np.bool_

    def test_r_values_in_unit_interval(self):
        """All estimated r values must lie in [0, 1] (clipped if necessary)."""
        rng = np.random.default_rng(2)
        F = rng.uniform(80, 120, (10, 3000)).astype(np.float32)
        Fneu = rng.uniform(15, 30, (10, 3000)).astype(np.float32)
        r_hat, _ = estimate_neuropil_coefs(F, Fneu)

        assert (r_hat >= 0.0).all() and (r_hat <= 1.0).all()

    def test_constant_fneu_falls_back_to_population_median(self):
        """A cell with zero-variance Fneu should not converge; falls back."""
        rng = np.random.default_rng(3)
        n_cells, n_frames = 5, 3000

        # 4 normal cells + 1 with constant Fneu (degenerate).
        F = rng.uniform(80, 120, (n_cells, n_frames)).astype(np.float32)
        Fneu = rng.uniform(15, 30, (n_cells, n_frames)).astype(np.float32)
        Fneu[0] = 25.0  # constant — fneu_range = 0

        r_hat, converged = estimate_neuropil_coefs(F, Fneu)

        # The degenerate cell should be flagged as not converged.
        assert not converged[0]
        # The remaining cells converged.
        assert converged[1:].all()
        # The flagged cell got the median of the converged cells.
        np.testing.assert_allclose(r_hat[0], np.median(r_hat[1:]), rtol=1e-5)


# ============================================================================
# F0 floor behavior — negative-median path (v0.4.1)
# ============================================================================


class TestF0FloorAbs:
    """The floor is anchored to |median(F_corr)|, robust to negative medians."""

    def test_negative_median_F_corr_yields_positive_floor(self, default_params):
        """If F_corr's median is negative, the floor must still be non-negative.

        This guards the v0.4.1 np.abs() fix. Without it, a negative
        median(F_corr) produces a negative floor_per_roi, which inverts the
        sign of dF/F where the floor activates.
        """
        n_rois, n_frames = 1, 6000
        # Construct F, Fneu, and a fixed r such that F - r*Fneu has a
        # negative median. F=10, Fneu=100, r=0.7 → F_corr = -60.
        F = np.full((n_rois, n_frames), 10.0, dtype=np.float32)
        Fneu = np.full((n_rois, n_frames), 100.0, dtype=np.float32)
        params = dict(default_params)
        params["neuropil_coef"] = 0.7

        _, F0, floor_mask = compute_dff(F, Fneu, fs=30.0, params=params)

        # F0 must be non-negative everywhere — even on a constant
        # negative-median F_corr (where the percentile baseline would
        # naturally be negative without flooring).
        assert (F0 >= 0).all(), (
            f"F0 has negative entries when median(F_corr) < 0; "
            f"min F0 = {F0.min()}. The np.abs() floor fix may be missing."
        )


# ============================================================================
# Default params trigger the per-cell estimator
# ============================================================================


class TestDefaultParams:
    """Guard the v0.4.0 default of neuropil_coef=None."""

    def test_default_neuropil_coef_is_none(self, default_params):
        """neuropil_coef=None signals 'use the per-cell Allen estimator' in run()."""
        assert default_params["neuropil_coef"] is None
