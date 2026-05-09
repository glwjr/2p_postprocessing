"""
One-time helper to build a small Allen-derived regression fixture.

This script is a developer tool, not part of the regular pipeline. Run it
once locally to generate the fixture files that get committed to the repo
under tests/fixtures/. Subsequent regression test runs consume those files
without needing AllenSDK or network access.

What it does:
    1. Pulls one Allen Visual Coding 2P experiment (uses adapt_allen_to_suite2p.py)
    2. Slices it down to ~30 cells × 5 minutes (≈1 MB) for git-friendliness
    3. Runs compute_dff.py on the slice to generate a golden dff.h5
    4. Writes everything to tests/fixtures/allen_slice/

After running, commit the contents of tests/fixtures/allen_slice/ to the
repo. Re-run only when the pipeline's expected output legitimately changes
(e.g., bumping the pipeline version with intentional algorithm changes).

Usage:
    python build_regression_fixture.py
    # writes to tests/fixtures/allen_slice/

The fixture is small enough for git: with 30 cells × 9000 frames × 4 bytes,
F.npy and Fneu.npy are ~1MB each, iscell.npy is trivial, and the golden
dff.h5 with gzip compression is ~600KB. Total ~3 MB committed.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# Default slice parameters chosen to balance fixture size against
# meaningful coverage. 30 cells exercises pre/post filters, 5 minutes is
# long enough that the 30-second baseline window has room to operate.
N_CELLS_KEEP = 30
DURATION_MINUTES = 5.0


def slice_suite2p_outputs(
    full_s2p_dir: Path,
    sliced_s2p_dir: Path,
    n_cells: int,
    duration_minutes: float,
) -> None:
    """Copy a small slice of a full Suite2p output directory."""
    sliced_s2p_dir.mkdir(parents=True, exist_ok=True)

    F = np.load(full_s2p_dir / "F.npy")
    Fneu = np.load(full_s2p_dir / "Fneu.npy")
    iscell = np.load(full_s2p_dir / "iscell.npy")
    ops = np.load(full_s2p_dir / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])

    # Slice cells: take the top n_cells by max F, so we get reasonably
    # active cells (more useful for regression than random selection).
    cell_max_F = F.max(axis=1)
    top_cells = np.argsort(cell_max_F)[-n_cells:]
    top_cells = np.sort(top_cells)  # preserve original order

    # Slice time: take the first duration_minutes worth of frames.
    n_frames = int(duration_minutes * 60 * fs)
    n_frames = min(n_frames, F.shape[1])

    F_sliced = F[top_cells][:, :n_frames]
    Fneu_sliced = Fneu[top_cells][:, :n_frames]
    iscell_sliced = iscell[top_cells]

    np.save(sliced_s2p_dir / "F.npy", F_sliced)
    np.save(sliced_s2p_dir / "Fneu.npy", Fneu_sliced)
    np.save(sliced_s2p_dir / "iscell.npy", iscell_sliced)
    np.save(sliced_s2p_dir / "ops.npy", np.array(ops, dtype=object))

    print(f"  Sliced from {F.shape} to {F_sliced.shape}")
    print(
        f"  F.npy: {(sliced_s2p_dir / 'F.npy').stat().st_size / 1024:.0f} KB, "
        f"Fneu.npy: {(sliced_s2p_dir / 'Fneu.npy').stat().st_size / 1024:.0f} KB"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build the Allen regression fixture (one-time)."
    )
    parser.add_argument(
        "--experiment_id",
        type=int,
        default=569407590,
        help="Allen ophys_experiment_id (default: tutorial experiment).",
    )
    parser.add_argument(
        "--fixture_dir",
        type=Path,
        default=Path(__file__).parent / "tests" / "fixtures" / "allen_slice",
        help="Output directory for the committed fixture files.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path.home() / "allen_brain_observatory",
        help="AllenSDK cache (avoids re-downloading on subsequent runs).",
    )
    args = parser.parse_args()

    # Use a temp directory for the full-size Allen output; only the sliced
    # version goes to the committed fixture path.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        full_output_dir = tmp / "allen_full"

        # Step 1: pull the Allen experiment via the adapter.
        print(f"Step 1: Adapting Allen experiment {args.experiment_id}...")
        subprocess.run(
            [
                sys.executable,
                "adapt_allen_to_suite2p.py",
                "--experiment_id",
                str(args.experiment_id),
                "--output_dir",
                str(full_output_dir),
                "--cache_dir",
                str(args.cache_dir),
            ],
            check=True,
        )

        # Step 2: slice it down.
        print(f"\nStep 2: Slicing to {N_CELLS_KEEP} cells × {DURATION_MINUTES} min...")
        args.fixture_dir.mkdir(parents=True, exist_ok=True)
        sliced_s2p_dir = args.fixture_dir / "suite2p" / "plane0"
        slice_suite2p_outputs(
            full_output_dir / "suite2p" / "plane0",
            sliced_s2p_dir,
            N_CELLS_KEEP,
            DURATION_MINUTES,
        )

        # Step 3: run compute_dff on the slice to produce the golden output.
        print("\nStep 3: Running compute_dff.py on the slice to build golden output...")
        golden_output_dir = args.fixture_dir / "expected_dff_output"
        if golden_output_dir.exists():
            shutil.rmtree(golden_output_dir)
        subprocess.run(
            [
                sys.executable,
                "compute_dff.py",
                "--suite2p_dir",
                str(sliced_s2p_dir),
                "--output_dir",
                str(golden_output_dir),
            ],
            check=True,
        )

    # Summarize.
    print(f"\nFixture written to: {args.fixture_dir}")
    print("Files committed to repo:")
    for p in sorted(args.fixture_dir.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            rel = p.relative_to(args.fixture_dir)
            print(f"  {rel}  ({size_kb:.0f} KB)")
    total_kb = sum(
        p.stat().st_size for p in args.fixture_dir.rglob("*") if p.is_file()
    ) / 1024
    print(f"\nTotal: {total_kb:.0f} KB")
    print("\nNext: git add tests/fixtures/allen_slice/ && commit.")


if __name__ == "__main__":
    main()
