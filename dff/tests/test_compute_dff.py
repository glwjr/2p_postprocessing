"""
Unit tests for compute_dff().

These tests use synthetic, deterministic inputs where the correct output
is known analytically. They are pure unit tests — no file I/O — and run
fast enough to use as pre-commit checks.
"""

import numpy as np
import pytest

from compute_dff import compute_dff


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

        dff, F0, _ = compute_dff(F, Fneu, fs=30.0, params=default_params)

        # Derive expected from params so the test follows the default.
        expected_F_corr = F_value - default_params["neuropil_coef"] * Fneu_value
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
