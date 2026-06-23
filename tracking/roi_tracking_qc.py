"""
roi_tracking_qc.py
==================
Per-UCID cross-session QC figures for ROICaT tracking output.

Figure layout
-------------
Rows (top to bottom):
  0  raw / unregistered whole FOV  (only when fovs_raw is supplied)
  1  aligned whole FOV
  2  zoomed crop around the ROI

Columns (left to right):
  0  superimposed projection across all sessions
  1…N  individual per-session panels

Visual conventions
------------------
- ROI outline: red contour drawn on the smoothed footprint.
  Gaussian smoothing (σ = 1.5 px) is applied before contouring to merge the
  small disconnected fragments that arise from sparse suite2p masks after
  nonrigid warping.
- Sessions with no detection for this UCID are left blank (no marker).
- Crop box: a thin dashed yellow rectangle drawn on full-FOV rows (raw and
  aligned) to show where the zoomed row sits spatially.

Two exporters share the same figure builder:
  export_pdf(...)  → one multipage PDF, one UCID per page
  export_html(...) → one self-contained HTML file with a UCID picker
"""

import io
import base64

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter

try:
    import scipy.sparse as sp

    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _footprint_image(rois_aligned, session, roi_idx, H, W):
    """Return the (H, W) aligned footprint of one ROI, or None if absent."""
    if roi_idx is None:
        return None
    block = rois_aligned[session]
    if _HAVE_SCIPY and sp.issparse(block):
        row = np.asarray(block[roi_idx].todense()).ravel()
        return row.reshape(H, W)
    block = np.asarray(block)
    if block.ndim == 2:
        return block[roi_idx].reshape(H, W)
    if block.ndim == 3:
        return block[roi_idx]
    raise ValueError(f"Unexpected ROIs_aligned[{session}] shape: {block.shape}")


def _weighted_centroid(fp):
    """Intensity-weighted (row, col) centroid of a footprint image."""
    total = fp.sum()
    if total <= 0:
        r, c = np.array(fp.shape) / 2.0
        return float(r), float(c)
    rr, cc = np.indices(fp.shape)
    return float((rr * fp).sum() / total), float((cc * fp).sum() / total)


def _norm(img, p=(1, 99)):
    """Percentile-clip an image to [0, 1] for display."""
    img = np.asarray(img, dtype=float)
    lo, hi = np.percentile(img, p)
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _session_roi_index(labels_bySession, session, ucid):
    """Index of the ROI belonging to ucid in session, or None if absent."""
    hits = np.where(np.asarray(labels_bySession[session]) == ucid)[0]
    return int(hits[0]) if len(hits) else None


def _draw(
    ax,
    bg,
    fp,
    centroid,
    crop_hw,
    color,
    present,
    label,
    zoom_box_hw=None,
    zoom_box_centroids=None,
):
    """Render one panel of the QC figure.

    Parameters
    ----------
    bg : (H, W) float array, already normalised to [0, 1]
    fp : (H, W) footprint array or None
    centroid : (row, col) of the consensus centroid
    crop_hw : int | None
        None → show full image; int → zoom axes to ±crop_hw around centroid.
    color : matplotlib colour for the ROI contour / n/d marker
    present : bool — whether this session detected the ROI
    label : str | None — panel title (shown above the axes)
    zoom_box_hw : int | None
        If set, draw thin dashed yellow rectangles of ±zoom_box_hw to indicate
        where the zoomed row sits.
    zoom_box_centroids : list of (row, col) or None
        Centres for each yellow box.  When None, a single box is drawn at
        `centroid`.  Pass a per-session list to draw one box per session
        (used on the superimposed raw-FOV panel).
    """
    fH, fW = bg.shape
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    r, c = centroid

    if present and fp is not None and fp.max() > 0:
        # Gaussian smoothing merges the small disconnected fragments that arise
        # from nonrigid warping of sparse suite2p footprint masks, producing a
        # single smooth closed contour instead of many disjoint segments.
        fp_s = gaussian_filter(fp.astype(float), sigma=1.5)
        if fp_s.max() > 0:
            ax.contour(fp_s, levels=[fp_s.max() * 0.5], colors=[color], linewidths=1.1)
    else:
        ax.text(
            0.5,
            0.04,
            "n/d",
            transform=ax.transAxes,
            ha="center",
            color=color,
            fontsize=8,
            weight="bold",
        )

    if crop_hw is not None:
        r0, r1 = max(0, r - crop_hw), min(fH, r + crop_hw)
        c0, c1 = max(0, c - crop_hw), min(fW, c + crop_hw)
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)  # inverted y for image coordinates

    if zoom_box_hw is not None:
        centers = zoom_box_centroids if zoom_box_centroids is not None else [(r, c)]
        for rb, cb in centers:
            r0b = max(0, rb - zoom_box_hw)
            r1b = min(fH, rb + zoom_box_hw)
            c0b = max(0, cb - zoom_box_hw)
            c1b = min(fW, cb + zoom_box_hw)
            ax.add_patch(
                Rectangle(
                    (c0b, r0b),
                    c1b - c0b,
                    r1b - r0b,
                    linewidth=0.8,
                    edgecolor="yellow",
                    facecolor="none",
                    linestyle="--",
                    zorder=5,
                )
            )

    # pixel-coordinate tick labels (sparse, small)
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.tick_params(labelsize=6)

    if label:
        ax.set_title(label, fontsize=8)


# ---------------------------------------------------------------------------
# core figure builder
# ---------------------------------------------------------------------------


def build_ucid_figure(
    ucid,
    fovs_aligned,
    rois_aligned,
    labels_bySession,
    H,
    W,
    fovs_raw=None,
    rois_raw=None,
    mouse_name=None,
    crop_halfwidth=40,
    superimpose="mean",
    roi_color="red",
    cs_sil_value=None,
    session_names=None,
    dpi=110,
):
    """Build and return a matplotlib Figure for one UCID.

    Parameters
    ----------
    fovs_aligned : list of (H, W) arrays
        Registered / nonrigid-aligned FOV images, one per session.
    fovs_raw : list of (H, W) arrays or None
        Unregistered FOV images (e.g. data.FOV_images).  When supplied, a
        third row is added showing the raw FOVs with a crop-box overlay.
    rois_raw : same structure as rois_aligned or None
        Pre-alignment ROI footprints.  When supplied alongside fovs_raw, the
        raw-FOV row uses these footprints (and their centroid) so that the ROI
        contour and yellow crop-box reflect the true pre-alignment position.
    mouse_name : str or None
        E.g. "SA11_LG".  Prepended to the figure suptitle.
    session_names : list of str or None
        Column headings, e.g. ["SA11_20250806 (VG)", …].
    """
    n = len(fovs_aligned)
    use_raw = fovs_raw is not None
    n_rows = 3 if use_raw else 2

    if session_names is None:
        session_names = [f"S{s}" for s in range(n)]

    # row indices: raw row is always 0 when present (guarded by use_raw checks)
    r_aligned = 1 if use_raw else 0
    r_zoom = 2 if use_raw else 1

    # ── normalise backgrounds ────────────────────────────────────────────────
    bg_aln = [_norm(fovs_aligned[s]) for s in range(n)]
    stk_aln = np.stack(bg_aln, 0)
    super_aln = stk_aln.max(0) if superimpose == "max" else stk_aln.mean(0)

    # initialised here so they are always bound (filled below when use_raw)
    bg_raw: list[np.ndarray] = []
    super_raw: np.ndarray = np.zeros_like(super_aln)
    if use_raw:
        assert fovs_raw is not None  # guaranteed by use_raw; helps type checker
        bg_raw = [_norm(fovs_raw[s]) for s in range(n)]
        stk_raw = np.stack(bg_raw, 0)
        super_raw = stk_raw.max(0) if superimpose == "max" else stk_raw.mean(0)

    # ── per-session footprints + consensus ──────────────────────────────────
    idxs = [_session_roi_index(labels_bySession, s, ucid) for s in range(n)]
    fps = [_footprint_image(rois_aligned, s, idxs[s], H, W) for s in range(n)]
    present = [i is not None for i in idxs]

    present_fps = [f for f, p in zip(fps, present) if p and f is not None]
    fp_con = np.sum(present_fps, 0) if present_fps else np.zeros((H, W))
    centroid = _weighted_centroid(fp_con)

    # Pre-alignment footprints and per-session centroids for the raw FOV row
    fps_raw: list = []
    fp_con_raw: np.ndarray = fp_con
    centroid_raw = centroid
    centroids_raw_per_session: list = []  # one (r,c) per session, or None if absent
    if use_raw and rois_raw is not None:
        fps_raw = [_footprint_image(rois_raw, s, idxs[s], H, W) for s in range(n)]
        present_fps_raw = [f for f, p in zip(fps_raw, present) if p and f is not None]
        fp_con_raw = np.sum(present_fps_raw, 0) if present_fps_raw else np.zeros((H, W))
        centroid_raw = _weighted_centroid(fp_con_raw)
        centroids_raw_per_session = [
            (
                _weighted_centroid(fps_raw[s])
                if (present[s] and fps_raw[s] is not None)
                else None
            )
            for s in range(n)
        ]

    # ── figure layout ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_rows,
        n + 1,
        figsize=(2.0 * (n + 1), 2.1 * n_rows),
        dpi=dpi,
    )
    if n == 0:
        axes = np.array(axes).reshape(n_rows, 1)

    # ── column 0: superimposed ───────────────────────────────────────────────
    if use_raw:
        # Superimposed raw: one yellow box per session at each session's raw centroid
        valid_raw_centroids = [
            c for c in centroids_raw_per_session if c is not None
        ] or None
        _draw(
            axes[0, 0],
            super_raw,
            fp_con_raw,
            centroid_raw,
            None,
            roi_color,
            fp_con_raw.max() > 0,
            "Superimposed",
            zoom_box_hw=crop_halfwidth,
            zoom_box_centroids=valid_raw_centroids,
        )

    _draw(
        axes[r_aligned, 0],
        super_aln,
        fp_con,
        centroid,
        None,
        roi_color,
        fp_con.max() > 0,
        None if use_raw else "Superimposed",
        zoom_box_hw=crop_halfwidth,
    )

    _draw(
        axes[r_zoom, 0],
        super_aln,
        fp_con,
        centroid,
        crop_halfwidth,
        roi_color,
        fp_con.max() > 0,
        None,
    )

    # ── columns 1..n: per session ────────────────────────────────────────────
    for s in range(n):
        col_title = session_names[s]
        if use_raw:
            fp_raw_s = fps_raw[s] if fps_raw else fps[s]
            # Use this session's own raw centroid so the box tracks the pre-alignment position;
            # fall back to the consensus raw centroid for absent sessions (n/d marker placement).
            cen_raw_s = (
                centroids_raw_per_session[s]
                if (
                    centroids_raw_per_session
                    and centroids_raw_per_session[s] is not None
                )
                else centroid_raw
            )
            _draw(
                axes[0, s + 1],
                bg_raw[s],
                fp_raw_s,
                cen_raw_s,
                None,
                roi_color,
                present[s],
                col_title,
                zoom_box_hw=crop_halfwidth if present[s] else None,
            )
        _draw(
            axes[r_aligned, s + 1],
            bg_aln[s],
            fps[s],
            centroid,
            None,
            roi_color,
            present[s],
            col_title if not use_raw else None,
            zoom_box_hw=crop_halfwidth if present[s] else None,
        )
        _draw(
            axes[r_zoom, s + 1],
            bg_aln[s],
            fps[s],
            centroid,
            crop_halfwidth,
            roi_color,
            present[s],
            None,
        )

    # ── row labels (left-column y-axes) ──────────────────────────────────────
    # axes[row, col] is typed as ndarray by stubs; cast to Axes for attribute access
    import matplotlib.axes as maxes

    def _ax(row: int, col: int) -> maxes.Axes:
        return axes[row, col]  # type: ignore[return-value]

    if use_raw:
        _ax(0, 0).set_ylabel("raw FOV", fontsize=9, labelpad=4)
    _ax(r_aligned, 0).set_ylabel("aligned FOV", fontsize=9, labelpad=4)
    _ax(r_zoom, 0).set_ylabel("zoom", fontsize=9, labelpad=4)

    # ── x / y pixel labels on every panel ────────────────────────────────────
    for row in range(n_rows):
        for col in range(n + 1):
            ax = _ax(row, col)
            ax.set_xlabel("x (px)", fontsize=7, labelpad=1)
            if col > 0:
                # col 0 already carries the row label as its ylabel
                ax.set_ylabel("y (px)", fontsize=7, labelpad=1)

    # ── suptitle ─────────────────────────────────────────────────────────────
    k = sum(present)
    parts = []
    if mouse_name:
        parts.append(mouse_name)
    parts += [f"UCID {ucid}", f"detected {k}/{n} sessions"]
    if cs_sil_value is not None:
        parts.append(f"cs_sil = {cs_sil_value:.3f}")
    fig.suptitle("   |   ".join(parts), fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


def order_ucids_by_quality(
    labels_bySession, cs_sil=None, ascending=True, drop_unclustered=True
):
    """Return UCIDs ordered by cs_sil (worst-first by default)."""
    all_u = np.unique(np.concatenate([np.asarray(x) for x in labels_bySession]))
    if drop_unclustered:
        all_u = all_u[all_u >= 0]
    if cs_sil is None:
        return list(all_u)
    cs_sil = np.asarray(cs_sil)
    keyed = [(u, cs_sil[u] if 0 <= u < len(cs_sil) else np.nan) for u in all_u]
    keyed.sort(
        key=lambda t: (np.nan_to_num(t[1], nan=np.inf), t[0]), reverse=not ascending
    )
    return [u for u, _ in keyed]


# ---------------------------------------------------------------------------
# exporters
# ---------------------------------------------------------------------------


def export_pdf(
    path,
    ucids,
    fovs_aligned,
    rois_aligned,
    labels_bySession,
    H,
    W,
    cs_sil=None,
    **fig_kwargs,
):
    """Write one multipage PDF, one UCID per page."""
    cs_sil = np.asarray(cs_sil) if cs_sil is not None else None
    with PdfPages(path) as pdf:
        for u in ucids:
            score = (
                float(cs_sil[u])
                if (cs_sil is not None and 0 <= u < len(cs_sil))
                else None
            )
            fig = build_ucid_figure(
                u,
                fovs_aligned,
                rois_aligned,
                labels_bySession,
                H,
                W,
                cs_sil_value=score,
                **fig_kwargs,
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {len(ucids)} pages -> {path}")


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>ROICaT cross-session QC</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a}}
 .bar{{display:flex;gap:10px;align-items:center;margin-bottom:14px;flex-wrap:wrap}}
 select,button{{font-size:15px;padding:5px 9px}}
 #meta{{color:#555;font-size:14px}}
 img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
</style></head><body>
<div class="bar">
 <strong>UCID</strong>
 <select id="pick"></select>
 <button id="prev">&larr; prev</button>
 <button id="next">next &rarr;</button>
 <span id="meta"></span>
</div>
<img id="fig" alt="ROI figure">
<script>
const DATA = {data_json};
const pick = document.getElementById('pick');
const fig  = document.getElementById('fig');
const meta = document.getElementById('meta');
DATA.forEach((d,i)=>{{
  const o=document.createElement('option');
  o.value=i;
  o.text='UCID '+d.ucid+(d.score!=null?'  (cs_sil '+d.score.toFixed(3)+')':'');
  pick.appendChild(o);
}});
function show(i){{
  i=Math.max(0,Math.min(DATA.length-1,i));
  pick.value=i;
  fig.src='data:image/png;base64,'+DATA[i].png;
  meta.textContent=(i+1)+' / '+DATA.length;
}}
pick.onchange=()=>show(+pick.value);
document.getElementById('prev').onclick=()=>show(+pick.value-1);
document.getElementById('next').onclick=()=>show(+pick.value+1);
show(0);
</script></body></html>"""


def export_html(
    path,
    ucids,
    fovs_aligned,
    rois_aligned,
    labels_bySession,
    H,
    W,
    cs_sil=None,
    max_ucids=400,
    dpi=90,
    **fig_kwargs,
):
    """Write one self-contained HTML file with a UCID picker.

    Every figure is pre-rendered and base64-embedded.  For large runs, pass a
    cs_sil-sorted subset rather than all clusters — max_ucids is a safety cap.
    """
    import json

    if len(ucids) > max_ucids:
        raise ValueError(
            f"{len(ucids)} UCIDs exceeds max_ucids={max_ucids}; the HTML would be "
            f"huge.  Pass a worst-first subset, e.g. "
            f"order_ucids_by_quality(...)[:{max_ucids}]."
        )
    cs_sil = np.asarray(cs_sil) if cs_sil is not None else None
    records = []
    for u in ucids:
        score = (
            float(cs_sil[u]) if (cs_sil is not None and 0 <= u < len(cs_sil)) else None
        )
        fig = build_ucid_figure(
            u,
            fovs_aligned,
            rois_aligned,
            labels_bySession,
            H,
            W,
            cs_sil_value=score,
            dpi=dpi,
            **fig_kwargs,
        )
        records.append({"ucid": int(u), "score": score, "png": _fig_to_b64(fig)})
    html = _HTML.replace("{data_json}", json.dumps(records))
    with open(path, "w") as f:
        f.write(html)
    print(f"wrote {len(records)} UCIDs -> {path}")


if __name__ == "__main__":
    pass
