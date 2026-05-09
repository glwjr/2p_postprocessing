"""
Regression test against an Allen-derived fixture.

The fixture under tests/fixtures/allen_slice/ was generated once locally by
running build_regression_fixture.py against a real Allen Visual Coding 2P
experiment, then sliced to ~30 cells × 5 minutes. This test re-runs
compute_dff.py against that committed slice and asserts the output matches
what was generated when the fixture was built.

Why this matters: synthetic-data tests catch logic errors but don't catch
issues that only surface on real noise structure, neuropil contamination,
and bleaching. This test guards against silent regressions during future
refactors that pass the synthetic suite but break on real data shape.

How to update: if you intentionally change the pipeline (e.g., bump the
version, change a filter rule), re-run build_regression_fixture.py and
commit the new expected output.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from compute_dff import DEFAULT_PARAMS, run

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "allen_slice"
SLICE_S2P_DIR = FIXTURE_DIR / "suite2p" / "plane0"
EXPECTED_DFF_PATH = FIXTURE_DIR / "expected_dff_output" / "dff.h5"


# Skip the entire module if the fixture isn't present. This means
# developers who haven't generated the fixture locally don't get failing
# tests — they just get a skip, with a clear reason.
pytestmark = pytest.mark.skipif(
    not EXPECTED_DFF_PATH.exists(),
    reason=(
        f"Allen regression fixture not found at {FIXTURE_DIR}. "
        "Run `python build_regression_fixture.py` to generate it."
    ),
)


@pytest.fixture(scope="module")
def regression_run(tmp_path_factory):
    """Run the pipeline once against the fixture; share the output across tests."""
    out_dir = tmp_path_factory.mktemp("allen_regression_out")
    run(SLICE_S2P_DIR, out_dir, dict(DEFAULT_PARAMS))
    return out_dir


def test_dff_matches_expected(regression_run):
    """The dF/F traces must match the golden output to float32 precision."""
    with h5py.File(regression_run / "dff.h5", "r") as f:
        actual_dff = f["dff"][:]
    with h5py.File(EXPECTED_DFF_PATH, "r") as f:
        expected_dff = f["dff"][:]

    assert actual_dff.shape == expected_dff.shape, (
        f"dF/F shape changed: {actual_dff.shape} vs expected {expected_dff.shape}. "
        "If this is intentional, re-run build_regression_fixture.py."
    )
    np.testing.assert_array_equal(
        actual_dff,
        expected_dff,
        err_msg=(
            "dF/F values changed against the Allen fixture. "
            "If this is intentional (e.g., you changed the algorithm), "
            "re-run build_regression_fixture.py and commit the new expected output."
        ),
    )


def test_F0_matches_expected(regression_run):
    """The F0 baseline must also match — catches changes that cancel in dF/F."""
    with h5py.File(regression_run / "dff.h5", "r") as f:
        actual_F0 = f["F0"][:]
    with h5py.File(EXPECTED_DFF_PATH, "r") as f:
        expected_F0 = f["F0"][:]

    np.testing.assert_array_equal(actual_F0, expected_F0)


def test_cell_indices_match_expected(regression_run):
    """Pre/post filters must drop the same cells they did when the fixture was built."""
    with h5py.File(regression_run / "dff.h5", "r") as f:
        actual_indices = f["cell_indices"][:]
    with h5py.File(EXPECTED_DFF_PATH, "r") as f:
        expected_indices = f["cell_indices"][:]

    np.testing.assert_array_equal(
        actual_indices,
        expected_indices,
        err_msg=(
            "Cell-selection result changed. The pre/post filters are dropping "
            "different cells than they did when the fixture was built. "
            "If intentional, re-run build_regression_fixture.py."
        ),
    )
