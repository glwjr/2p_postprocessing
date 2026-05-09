"""
Convert an Allen Brain Observatory (Visual Coding 2P) experiment into the
Suite2p output format expected by compute_dff.py.

This script reads Allen's published NWB files directly via h5py rather than
using AllenSDK. AllenSDK pins old dependencies that don't build cleanly on
Python 3.12, and we only need a small fraction of its functionality
(reading raw F, Fneu, and dF/F from the NWB file). Direct h5py access is
faster, more transparent, and doesn't require a heavy dependency.

The NWB files are just HDF5 files. The dataset paths used here come from
the AllenSDK source — see brain_observatory_nwb_data_set.py for the
authoritative reference.

Allen does its own neuropil correction with a per-cell r value, not the
fixed 0.7 coefficient that Suite2p uses. So expect numerical differences
when comparing dF/F traces — what you're checking is bulk distributional
agreement, not exact numerical match.

Usage:
    python adapt_allen_to_suite2p.py --experiment_id 569407590 \
        --output_dir /path/to/output

Files are downloaded into a local cache (default: ~/allen_brain_observatory)
and reused on subsequent runs. A typical experiment is ~few hundred MB.

Output directory layout (matches what Suite2p produces):
    <output_dir>/
    ├── suite2p/plane0/
    │   ├── F.npy
    │   ├── Fneu.npy
    │   ├── iscell.npy
    │   └── ops.npy
    └── allen_dff.npy        # Allen's published dF/F for comparison
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np

# Allen's S3 bucket pattern for Visual Coding 2P NWB files.
S3_URL_TEMPLATE = (
    "https://allen-brain-observatory.s3.us-west-2.amazonaws.com/"
    "visual-coding-2p/ophys_experiment_data/{experiment_id}.nwb"
)

# HDF5 dataset paths inside the NWB file. From AllenSDK's
# brain_observatory_nwb_data_set.py.
PIPELINE_DATASET = "brain_observatory_pipeline"
FLUORESCENCE_PATH = f"processing/{PIPELINE_DATASET}/Fluorescence/imaging_plane_1"
NEUROPIL_V2_PATH = (
    f"processing/{PIPELINE_DATASET}/Fluorescence/imaging_plane_1_neuropil_response"
)
DFF_PATH = f"processing/{PIPELINE_DATASET}/DfOverF/imaging_plane_1"
PIPELINE_VERSION_PATH = "general/generated_by"


def download_nwb(experiment_id: int, cache_dir: Path) -> Path:
    """Download the NWB file for an experiment if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    nwb_path = cache_dir / f"{experiment_id}.nwb"

    if nwb_path.exists():
        size_mb = nwb_path.stat().st_size / (1024 * 1024)
        print(f"  Using cached NWB file ({size_mb:.0f} MB): {nwb_path}")
        return nwb_path

    url = S3_URL_TEMPLATE.format(experiment_id=experiment_id)
    print(f"  Downloading {url}")
    print(f"  → {nwb_path}")
    print("  (this may take several minutes for a few hundred MB)")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            sys.stdout.write(
                f"\r  {pct:3d}% ({downloaded / 1024 / 1024:.0f} MB"
                f" / {total_size / 1024 / 1024:.0f} MB)"
            )
            sys.stdout.flush()

    try:
        urlretrieve(url, nwb_path, reporthook=progress)
    except Exception:
        # Don't leave a partial download in the cache.
        if nwb_path.exists():
            nwb_path.unlink()
        raise
    print()  # newline after progress bar
    return nwb_path


def detect_pipeline_version(nwb: h5py.File) -> str:
    """
    Return the Allen pipeline version that processed this file.

    Allen reorganized the neuropil dataset path between v1.x and v2.x; the
    AllenSDK source checks pipeline_version >= "2.0" to decide. We do the
    same. If the version metadata isn't readable, default to assuming v2+
    since that covers the modern data — older data is rare.
    """
    try:
        # general/generated_by is a small dataset of strings like
        # ["pipeline", "ophys_pipeline", "version", "2.5", ...].
        if PIPELINE_VERSION_PATH in nwb:
            entries = [
                s.decode() if isinstance(s, bytes) else s
                for s in nwb[PIPELINE_VERSION_PATH][()]
            ]
            for i, entry in enumerate(entries):
                if entry.lower() == "version" and i + 1 < len(entries):
                    return entries[i + 1]
    except Exception:
        pass
    return "2.0"  # safe default


def load_traces(nwb_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Pull F, Fneu, Allen's dF/F, and frame rate from the NWB file.

    Returns
    -------
    F, Fneu, allen_dff : np.ndarray, shape (n_cells, n_timepoints)
    fs : float, frame rate in Hz derived from timestamps
    """
    with h5py.File(nwb_path, "r") as f:
        version = detect_pipeline_version(f)
        is_v2_or_later = tuple(int(x) for x in version.split(".")[:2]) >= (2, 0)

        # Raw fluorescence and timestamps.
        F = f[f"{FLUORESCENCE_PATH}/data"][()]
        timestamps = f[f"{FLUORESCENCE_PATH}/timestamps"][()]

        # Neuropil traces. Path differs by pipeline version.
        if is_v2_or_later:
            Fneu = f[f"{NEUROPIL_V2_PATH}/data"][()]
        else:
            Fneu = f[f"{FLUORESCENCE_PATH}/neuropil_traces"][()]

        # Allen's published dF/F.
        allen_dff = f[f"{DFF_PATH}/data"][()]

    fs = float(1.0 / np.median(np.diff(timestamps)))
    print(f"  Pipeline version: {version}")
    return F, Fneu, allen_dff, fs


def adapt_experiment(
    experiment_id: int,
    output_dir: Path,
    cache_dir: Path,
) -> None:
    """Pull an Allen Visual Coding 2P experiment and write Suite2p-format outputs."""
    print(f"Loading Allen experiment {experiment_id}")
    print(f"  (cache directory: {cache_dir})")

    nwb_path = download_nwb(experiment_id, cache_dir)

    print("Extracting fluorescence traces from NWB file...")
    F, Fneu, allen_dff, fs = load_traces(nwb_path)

    n_cells, n_frames = F.shape
    print(
        f"  Loaded {n_cells} cells, {n_frames} timepoints, "
        f"{n_frames / fs / 60:.1f} min at {fs:.2f} Hz"
    )

    # Build the Suite2p directory layout.
    s2p_dir = output_dir / "suite2p" / "plane0"
    s2p_dir.mkdir(parents=True, exist_ok=True)

    # Cast to float32 to match Suite2p's dtype (Allen's NWB stores float64).
    np.save(s2p_dir / "F.npy", F.astype(np.float32))
    np.save(s2p_dir / "Fneu.npy", Fneu.astype(np.float32))

    # iscell.npy: Allen's pipeline already curated cells. Mark all as cells
    # with high probability. Suite2p convention: column 0 = binary, column 1
    # = probability.
    iscell = np.column_stack(
        [
            np.ones(n_cells, dtype=np.float32),
            np.full(n_cells, 0.95, dtype=np.float32),
        ]
    )
    np.save(s2p_dir / "iscell.npy", iscell)

    # ops.npy: compute_dff.py only needs `fs`. The full Suite2p ops dict has
    # many more fields, but they're not used by our pipeline.
    ops = {"fs": fs}
    np.save(s2p_dir / "ops.npy", np.array(ops, dtype=object))

    # Save Allen's dF/F outside the suite2p/ tree for the comparison step.
    np.save(output_dir / "allen_dff.npy", allen_dff.astype(np.float32))

    print(f"\nWrote Suite2p-format outputs to: {s2p_dir}")
    print(f"Wrote Allen's reference dF/F to:  {output_dir / 'allen_dff.npy'}")
    print()
    print("Next step:")
    print(f"  python compute_dff.py \\")
    print(f"      --suite2p_dir {s2p_dir} \\")
    print(f"      --output_dir {output_dir / 'dff_output'}")
    print()
    print("Then run compare_to_allen.py to validate against Allen's reference.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Allen Brain Observatory data to Suite2p format."
    )
    parser.add_argument(
        "--experiment_id",
        type=int,
        default=569407590,
        help=(
            "Allen ophys_experiment_id. Default 569407590 is the experiment "
            "used in the Allen brain_observatory tutorial notebook."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write Suite2p-format outputs to.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path.home() / "allen_brain_observatory",
        help=(
            "NWB file cache directory (default: ~/allen_brain_observatory). "
            "First run downloads ~few hundred MB; subsequent runs reuse it."
        ),
    )
    args = parser.parse_args()

    adapt_experiment(args.experiment_id, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
