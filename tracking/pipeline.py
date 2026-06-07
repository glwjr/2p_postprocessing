"""
pipeline.py
===========
Helpers for the ROICaT tracking pipeline.

The main entry point for a notebook run is :func:`filter_sessions_by_overlap`,
which handles the load → screen → filter → reload cycle so the notebook never
needs to re-run a cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import roicat
from scipy.sparse.csgraph import connected_components


def filter_sessions_by_overlap(
    paths_stat: list[str],
    paths_ops: list[str],
    um_per_pixel: float | list[float] = 1.0,
    device: str = "cpu",
    z_threshold: float = 4.0,
    radius_in: float = 4.0,
    radius_out: float = 20.0,
) -> tuple[roicat.data_importing.Data_suite2p, list[int]]:
    """Load all sessions, run a silent geometric-alignment screening pass, and
    return a Data_suite2p containing only the largest co-registerable group.

    This replaces the manual ``keep = [...]`` pattern and the subsequent
    data-rebuild cell.  Call it once; the returned ``data`` object is ready to
    hand straight to ``aligner.augment_FOV_images``.

    Parameters
    ----------
    paths_stat, paths_ops:
        Parallel lists of stat.npy / ops.npy paths, one per session.
    um_per_pixel:
        Scalar applied to all sessions, or a per-session list.
    device:
        Torch device string (``'cpu'``, ``'cuda'``, …).  The screening pass
        only uses DISK_LightGlue which runs on CPU regardless, so ``'cpu'``
        is fine here even if you plan to use a GPU for ROInet later.
    z_threshold:
        Alignment z-score threshold.  Session pairs whose score falls below
        this are treated as unaligned.
    radius_in, radius_out:
        Aligner inner / outer radii in micrometers.

    Returns
    -------
    data : Data_suite2p
        Loaded with only the sessions in the largest co-registerable group.
    keep : list[int]
        Indices into the original path lists of the retained sessions.
    """
    n = len(paths_stat)
    umpp = [um_per_pixel] * n if not isinstance(um_per_pixel, list) else list(um_per_pixel)

    # ── 1. Load all sessions ──────────────────────────────────────────────────
    data_all = roicat.data_importing.Data_suite2p(
        paths_statFiles=paths_stat,
        paths_opsFiles=paths_ops,
        um_per_pixel=umpp,
        new_or_old_suite2p="new",
        type_meanImg="meanImgE",
        verbose=False,
    )

    # ── 2. Silent geometric screening pass ───────────────────────────────────
    _aligner = roicat.tracking.alignment.Aligner(
        use_match_search=True,
        all_to_all=False,
        radius_in=radius_in,
        radius_out=radius_out,
        z_threshold=z_threshold,
        um_per_pixel=data_all.um_per_pixel[0],
        device=device,
        verbose=False,
    )
    fovs = _aligner.augment_FOV_images(
        FOV_images=data_all.FOV_images,
        spatialFootprints=data_all.spatialFootprints,
        normalize_FOV_intensities=True,
        roi_FOV_mixing_factor=0.5,
        use_CLAHE=True,
        CLAHE_grid_block_size=10,
        CLAHE_clipLimit=1.0,
        CLAHE_normalize=True,
    )
    _aligner.fit_geometric(
        template=0.5,
        ims_moving=fovs,
        template_method="sequential",
        mask_borders=(0, 0, 0, 0),
        method="DISK_LightGlue",
        kwargs_method={
            "DISK_LightGlue": {
                "num_features": 3000,
                "threshold_confidence": 0.0,
                "window_nms": 7,
            }
        },
        constraint="affine",
        kwargs_RANSAC={"inl_thresh": 3.0, "max_iter": 100, "confidence": 0.99},
        verbose=False,
    )

    # ── 3. Find the largest co-registerable group ─────────────────────────────
    aligned = _aligner.results_geometric["final"]["alignment_all_to_all"].copy()
    np.fill_diagonal(aligned, True)
    aligned = (aligned | aligned.T).astype(bool)  # symmetrise

    _, group_ids = connected_components(aligned, directed=False)
    keep = sorted(
        np.where(group_ids == np.argmax(np.bincount(group_ids)))[0].tolist()
    )
    dropped = [i for i in range(n) if i not in keep]

    print(f"Session filter: {len(keep)}/{n} sessions kept")
    for i in keep:
        print(f"  keep  [{i}]  {Path(paths_stat[i]).parts[-3]}")
    for i in dropped:
        print(f"  drop  [{i}]  {Path(paths_stat[i]).parts[-3]}  (poor overlap)")

    # ── 4. Rebuild with only the kept sessions ────────────────────────────────
    if len(keep) == n:
        # Nothing was dropped — avoid an unnecessary reload
        return data_all, keep

    data = roicat.data_importing.Data_suite2p(
        paths_statFiles=[paths_stat[i] for i in keep],
        paths_opsFiles=[paths_ops[i] for i in keep],
        um_per_pixel=[umpp[i] for i in keep],
        new_or_old_suite2p="new",
        type_meanImg="meanImgE",
        verbose=False,
    )
    assert data.check_completeness(verbose=False)["tracking"]
    return data, keep
