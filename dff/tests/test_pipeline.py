"""
Integration tests for the full pipeline (run()).

These tests exercise the end-to-end pipeline against synthetic Suite2p
output directories (built by fixtures in conftest.py) and verify that:

    - All four output files are produced
    - Cross-file consistency holds (dff.h5 ↔ dff_metadata.json ↔ dff_cell_summary.csv)
    - Parameters are correctly threaded through to outputs
    - Pre/post filters fire when they should
    - Pipeline produces sensible dF/F values on signal-bearing data
"""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from compute_dff import run

# ============================================================================
# Output file production
# ============================================================================


class TestOutputFiles:
    """All four expected outputs should be created."""

    def test_all_four_outputs_exist(self, fake_suite2p_dir, default_params, tmp_path):
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir, out_dir, default_params)

        for filename in [
            "dff.h5",
            "dff_diagnostics.png",
            "dff_cell_summary.csv",
            "dff_metadata.json",
        ]:
            assert (out_dir / filename).exists(), f"Missing output: {filename}"

    def test_output_dir_is_created_if_missing(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        """run() should create output_dir even if it doesn't exist yet."""
        out_dir = tmp_path / "deep" / "nested" / "out"
        assert not out_dir.exists()
        run(fake_suite2p_dir, out_dir, default_params)
        assert out_dir.exists()
        assert (out_dir / "dff.h5").exists()


# ============================================================================
# Cross-file consistency
# ============================================================================


class TestCrossFileConsistency:
    """The cell counts and shapes across outputs must agree."""

    def test_dff_h5_shape_matches_metadata_n_cells_final(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        """
        Regression test for the metadata fix: n_cells_final in the JSON
        must equal dff.shape[0] in the HDF5 file. Catches the original bug
        where n_cells_kept was recorded pre-filter while dff.h5 held the
        post-filter count.
        """
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir, out_dir, default_params)

        with h5py.File(out_dir / "dff.h5", "r") as f:
            dff_n_rows = f["dff"].shape[0]

        with open(out_dir / "dff_metadata.json") as f:
            meta = json.load(f)

        assert meta["n_cells_final"] == dff_n_rows

    def test_csv_row_count_matches_dff_h5(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir, out_dir, default_params)

        with h5py.File(out_dir / "dff.h5", "r") as f:
            dff_n_rows = f["dff"].shape[0]

        df = pd.read_csv(out_dir / "dff_cell_summary.csv")
        assert len(df) == dff_n_rows

    def test_dff_and_F0_have_matching_shapes(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir, out_dir, default_params)

        with h5py.File(out_dir / "dff.h5", "r") as f:
            assert f["dff"].shape == f["F0"].shape

    def test_cell_indices_length_matches_dff(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir, out_dir, default_params)

        with h5py.File(out_dir / "dff.h5", "r") as f:
            assert len(f["cell_indices"]) == f["dff"].shape[0]
            assert len(f["iscell_prob"]) == f["dff"].shape[0]

    def test_n_cells_after_iscell_geq_n_cells_final(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        """Pre/post filters can only reduce, never grow, the cell count."""
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir, out_dir, default_params)

        with open(out_dir / "dff_metadata.json") as f:
            meta = json.load(f)

        assert meta["n_cells_after_iscell"] >= meta["n_cells_final"]
        assert meta["n_rois_total"] >= meta["n_cells_after_iscell"]


# ============================================================================
# Parameter propagation
# ============================================================================


class TestParameterPropagation:
    """Parameters passed to run() must show up correctly in saved metadata."""

    def test_all_params_in_metadata_json(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir, out_dir, default_params)

        with open(out_dir / "dff_metadata.json") as f:
            meta = json.load(f)

        for key, value in default_params.items():
            assert key in meta["parameters"], f"Missing parameter in metadata: {key}"
            assert (
                meta["parameters"][key] == value
            ), f"Parameter {key} mismatch: {meta['parameters'][key]} != {value}"

    def test_post_filter_floor_frac_propagated(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        """
        Regression test for the post_filter_floor_frac fix: the parameter
        must reach both the JSON metadata and the HDF5 attrs, not be a
        magic constant in the source.
        """
        params = dict(default_params)
        params["post_filter_floor_frac"] = 0.123  # distinctive value
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir, out_dir, params)

        with open(out_dir / "dff_metadata.json") as f:
            meta = json.load(f)
        assert meta["parameters"]["post_filter_floor_frac"] == 0.123

        with h5py.File(out_dir / "dff.h5", "r") as f:
            assert f["metadata"].attrs["post_filter_floor_frac"] == 0.123

    def test_iscell_threshold_changes_cell_count(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        """A stricter threshold should reduce cell yield."""
        params_loose = dict(default_params, iscell_threshold=0.0)
        params_strict = dict(default_params, iscell_threshold=0.8)

        run(fake_suite2p_dir, tmp_path / "loose", params_loose)
        run(fake_suite2p_dir, tmp_path / "strict", params_strict)

        meta_loose = json.loads((tmp_path / "loose" / "dff_metadata.json").read_text())
        meta_strict = json.loads(
            (tmp_path / "strict" / "dff_metadata.json").read_text()
        )

        assert meta_loose["n_cells_after_iscell"] >= meta_strict["n_cells_after_iscell"]


# ============================================================================
# Filter behavior
# ============================================================================


class TestFilters:
    """Pre- and post-filters should fire on appropriate inputs."""

    def test_zero_iscell_threshold_keeps_all_rois(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        params = dict(default_params, iscell_threshold=-0.001)
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir, out_dir, params)

        with open(out_dir / "dff_metadata.json") as f:
            meta = json.load(f)
        # All 20 fixture ROIs should pass.
        assert meta["n_cells_after_iscell"] == meta["n_rois_total"]

    def test_impossible_iscell_threshold_raises(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        """An iscell threshold > 1.0 should drop everything and raise."""
        params = dict(default_params, iscell_threshold=1.5)
        out_dir = tmp_path / "out"

        with pytest.raises(RuntimeError, match="iscell threshold"):
            run(fake_suite2p_dir, out_dir, params)

    def test_strict_post_filter_drops_more_cells(self, tmp_path):
        """
        With aggressive post-filter (0% floor tolerance), fewer cells survive
        than with the default 5% tolerance.

        This builds a noisier session than the default fixture, with some
        ROIs that legitimately hit the floor occasionally.
        """
        rng = np.random.default_rng(2)
        n_rois, fs, n_frames = 30, 30.0, 6000
        # Mix of bright and dim ROIs to force some floor activations.
        baselines = rng.uniform(20, 200, (n_rois, 1)).astype(np.float32)
        F = (baselines * np.ones((1, n_frames))).astype(np.float32)
        F += rng.normal(0, 5.0, (n_rois, n_frames)).astype(np.float32)
        # Force some dim ROIs into the floor occasionally.
        F[:5, 1000:2000] *= 0.05

        Fneu = (rng.uniform(15, 30, (n_rois, 1)) * np.ones((1, n_frames))).astype(
            np.float32
        )
        iscell = np.column_stack([np.ones(n_rois), np.full(n_rois, 0.9)]).astype(
            np.float32
        )

        s2p_dir = tmp_path / "s2p"
        s2p_dir.mkdir()
        np.save(s2p_dir / "F.npy", F)
        np.save(s2p_dir / "Fneu.npy", Fneu)
        np.save(s2p_dir / "iscell.npy", iscell)
        np.save(s2p_dir / "ops.npy", np.array({"fs": fs}, dtype=object))

        from compute_dff import DEFAULT_PARAMS

        loose = dict(DEFAULT_PARAMS, post_filter_floor_frac=0.5)
        strict = dict(DEFAULT_PARAMS, post_filter_floor_frac=0.0)

        run(s2p_dir, tmp_path / "loose", loose)
        run(s2p_dir, tmp_path / "strict", strict)

        meta_loose = json.loads((tmp_path / "loose" / "dff_metadata.json").read_text())
        meta_strict = json.loads(
            (tmp_path / "strict" / "dff_metadata.json").read_text()
        )

        assert meta_strict["n_cells_final"] <= meta_loose["n_cells_final"]


# ============================================================================
# Signal recovery on transient-bearing data
# ============================================================================


class TestSignalRecovery:
    """The pipeline should surface real signal, not just run without errors."""

    def test_transients_produce_above_noise_dff(
        self, fake_suite2p_dir_with_transients, default_params, tmp_path
    ):
        """
        The transients fixture injects ~100% amplitude calcium events. The
        99th percentile dF/F should clearly exceed the median by a substantial
        margin (well above what you'd see on noise alone).
        """
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir_with_transients, out_dir, default_params)

        with h5py.File(out_dir / "dff.h5", "r") as f:
            dff = f["dff"][:]

        p99 = np.percentile(dff, 99)
        median = np.median(dff)

        # With 100% amplitude transients, p99 should be well above 0.3.
        # The median should sit near zero. If we see p99 ~ 0.1 or median far
        # from zero, the pipeline isn't recovering the signal we injected.
        assert p99 > 0.3, f"99th percentile dF/F suspiciously low: {p99}"
        assert abs(median) < 0.1, f"Median dF/F should be near zero, got {median}"

    def test_csv_summary_dff_max_matches_h5(
        self, fake_suite2p_dir_with_transients, default_params, tmp_path
    ):
        """Per-cell dff_max in the CSV should match max along axis=1 of dff.h5."""
        out_dir = tmp_path / "out"
        run(fake_suite2p_dir_with_transients, out_dir, default_params)

        with h5py.File(out_dir / "dff.h5", "r") as f:
            dff = f["dff"][:]

        df = pd.read_csv(out_dir / "dff_cell_summary.csv")
        # CSV is float-formatted to %.4f, so allow that tolerance.
        np.testing.assert_allclose(df["dff_max"].values, dff.max(axis=1), atol=1e-3)


# ============================================================================
# Determinism at the pipeline level
# ============================================================================


class TestPipelineDeterminism:
    """Same Suite2p inputs + same params → identical dff.h5."""

    def test_two_runs_produce_identical_dff(
        self, fake_suite2p_dir, default_params, tmp_path
    ):
        run(fake_suite2p_dir, tmp_path / "a", default_params)
        run(fake_suite2p_dir, tmp_path / "b", default_params)

        with h5py.File(tmp_path / "a" / "dff.h5", "r") as fa:
            dff_a = fa["dff"][:]
            F0_a = fa["F0"][:]

        with h5py.File(tmp_path / "b" / "dff.h5", "r") as fb:
            dff_b = fb["dff"][:]
            F0_b = fb["F0"][:]

        np.testing.assert_array_equal(dff_a, dff_b)
        np.testing.assert_array_equal(F0_a, F0_b)
