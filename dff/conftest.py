"""Shared pytest fixtures for compute_dff tests."""

import numpy as np
import pytest


@pytest.fixture
def default_params():
    """Default pipeline parameters, importable into any test."""
    from compute_dff import DEFAULT_PARAMS

    return dict(DEFAULT_PARAMS)


@pytest.fixture
def fake_suite2p_dir(tmp_path):
    """
    Build a minimal Suite2p output directory in tmp_path.

    Returns the directory path. The fixture creates a 20-ROI, 5-minute
    session at 30 Hz with realistic fluorescence ranges and a mix of
    iscell probabilities so cell-selection logic gets exercised.
    """
    rng = np.random.default_rng(0)
    n_rois = 20
    fs = 30.0
    duration_sec = 300.0
    n_frames = int(fs * duration_sec)

    # Baseline fluorescence with mild drift; nothing pathological.
    drift = np.linspace(1.0, 0.95, n_frames)[None, :]
    F = (rng.uniform(80, 150, (n_rois, 1)) * drift).astype(np.float32)
    F += rng.normal(0, 2.0, (n_rois, n_frames)).astype(np.float32)

    Fneu = rng.uniform(20, 40, (n_rois, 1)).astype(np.float32) * np.ones(
        (1, n_frames), dtype=np.float32
    )
    Fneu += rng.normal(0, 1.0, (n_rois, n_frames)).astype(np.float32)

    # iscell probabilities: ~70% above the default 0.3 threshold.
    iscell_probs = rng.uniform(0.0, 1.0, n_rois)
    iscell = np.column_stack([np.ones(n_rois), iscell_probs]).astype(np.float32)

    ops = {"fs": fs}

    np.save(tmp_path / "F.npy", F)
    np.save(tmp_path / "Fneu.npy", Fneu)
    np.save(tmp_path / "iscell.npy", iscell)
    np.save(tmp_path / "ops.npy", np.array(ops, dtype=object))

    return tmp_path


@pytest.fixture
def fake_suite2p_dir_with_transients(tmp_path):
    """
    Suite2p directory with injected calcium transients.

    Each ROI gets ~10 transients of known amplitude. Useful for asserting
    that the pipeline actually surfaces signal rather than just running
    without errors.
    """
    rng = np.random.default_rng(42)
    n_rois = 10
    fs = 30.0
    duration_sec = 300.0
    n_frames = int(fs * duration_sec)

    # Stable baseline.
    F_base = rng.uniform(100, 120, (n_rois, 1)).astype(np.float32)
    F = np.tile(F_base, (1, n_frames))
    F += rng.normal(0, 1.5, (n_rois, n_frames)).astype(np.float32)

    # Inject transients: 100% amplitude, ~30-frame decay.
    decay_kernel = np.exp(-np.arange(60) / 15.0).astype(np.float32)
    for roi in range(n_rois):
        n_transients = 10
        onsets = rng.integers(100, n_frames - 100, n_transients)
        for onset in onsets:
            amplitude = F_base[roi, 0] * 1.0
            end = min(onset + len(decay_kernel), n_frames)
            F[roi, onset:end] += amplitude * decay_kernel[: end - onset]

    Fneu = (rng.uniform(20, 35, (n_rois, 1)) * np.ones((1, n_frames))).astype(
        np.float32
    )
    Fneu += rng.normal(0, 1.0, (n_rois, n_frames)).astype(np.float32)

    # All ROIs pass iscell.
    iscell = np.column_stack([np.ones(n_rois), np.full(n_rois, 0.9)]).astype(np.float32)
    ops = {"fs": fs}

    np.save(tmp_path / "F.npy", F)
    np.save(tmp_path / "Fneu.npy", Fneu)
    np.save(tmp_path / "iscell.npy", iscell)
    np.save(tmp_path / "ops.npy", np.array(ops, dtype=object))

    return tmp_path
