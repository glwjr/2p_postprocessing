"""
roi_tracking_qc.py
==================
Per-UCID cross-session QC figures for ROICaT tracking output.

Implements the following spec: for each tracked neuron (UCID), a 2-row figure where
  - row 0 = whole aligned FOV, row 1 = the same cropped tight around the ROI
  - column 0 = superimposed (projection across all sessions)
  - columns 1..N = individual per-session aligned FOVs
The ROI is outlined in every panel by contouring its *aligned* footprint; in
sessions where the neuron was not detected, a dashed marker is drawn at the
consensus centroid and labelled "n/d".

Two exporters share the same figure builder:
  - export_pdf(...)  -> one multipage PDF, all (or a subset of) UCIDs
  - export_html(...) -> one self-contained HTML file with a UCID picker
"""

import io
import base64
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import scipy.sparse as sp

    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _footprint_image(rois_aligned, session, roi_idx, H, W):
    """Return the (H, W) aligned footprint of one ROI, or None if roi_idx is None."""
    if roi_idx is None:
        return None
    block = rois_aligned[session]
    if _HAVE_SCIPY and sp.issparse(block):
        row = np.asarray(block[roi_idx].todense()).ravel()
        return row.reshape(H, W)
    block = np.asarray(block)
    if block.ndim == 2:  # (n_ROIs, H*W) flattened
        return block[roi_idx].reshape(H, W)
    if block.ndim == 3:  # (n_ROIs, H, W)
        return block[roi_idx]
    raise ValueError(f"Unexpected ROIs_aligned[{session}] shape: {block.shape}")


def _weighted_centroid(fp):
    """(row, col) intensity-weighted centroid of a footprint image."""
    total = fp.sum()
    if total <= 0:
        r, c = np.array(fp.shape) / 2.0
        return float(r), float(c)
    rr, cc = np.indices(fp.shape)
    return float((rr * fp).sum() / total), float((cc * fp).sum() / total)


def _norm(img, p=(1, 99)):
    """Percentile-clip an image to 0..1 for display."""
    img = np.asarray(img, dtype=float)
    lo, hi = np.percentile(img, p)
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _session_roi_index(labels_bySession, session, ucid):
    """Index of the ROI in `session` belonging to `ucid`, or None if absent.
    If a session somehow has >1 member of a cluster, take the first and let the
    figure title flag it via the membership count."""
    hits = np.where(np.asarray(labels_bySession[session]) == ucid)[0]
    return int(hits[0]) if len(hits) else None


def _draw(ax, bg, fp, centroid, crop_hw, color, present, label):
    """Render one panel: background FOV + ROI contour (or n/d marker)."""
    H, W = bg.shape
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    r, c = centroid
    if present and fp is not None and fp.max() > 0:
        ax.contour(fp, levels=[0.5 * fp.max()], colors=[color], linewidths=1.1)
    else:
        ax.plot(c, r, marker="o", mfc="none", mec=color, mew=1.2, ms=14, ls="--")
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
        r0, r1 = max(0, r - crop_hw), min(H, r + crop_hw)
        c0, c1 = max(0, c - crop_hw), min(W, c + crop_hw)
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)  # inverted y for image coords
    ax.set_xticks([])
    ax.set_yticks([])
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
    crop_halfwidth=40,
    superimpose="mean",
    roi_color="red",
    cs_sil_value=None,
    session_names=None,
    dpi=110,
):
    """Build and return a matplotlib Figure for a single UCID (2 x (1+n_sessions))."""
    n = len(fovs_aligned)
    if session_names is None:
        session_names = [f"S{s}" for s in range(n)]

    # normalized per-session backgrounds
    bg = [_norm(fovs_aligned[s]) for s in range(n)]

    # superimposed background
    stack = np.stack(bg, axis=0)
    bg_super = stack.max(axis=0) if superimpose == "max" else stack.mean(axis=0)

    # per-session membership + footprints
    idxs = [_session_roi_index(labels_bySession, s, ucid) for s in range(n)]
    fps = [_footprint_image(rois_aligned, s, idxs[s], H, W) for s in range(n)]
    present = [i is not None for i in idxs]

    # consensus footprint / centroid from the sessions where it was detected
    present_fps = [f for f, p in zip(fps, present) if p and f is not None]
    if present_fps:
        fp_consensus = np.sum(present_fps, axis=0)
    else:
        fp_consensus = np.zeros((H, W))
    centroid = _weighted_centroid(fp_consensus)

    fig, axes = plt.subplots(2, n + 1, figsize=(2.0 * (n + 1), 4.2), dpi=dpi)
    if n == 0:  # degenerate guard
        axes = np.array(axes).reshape(2, 1)

    # column 0: superimposed (whole + zoom)
    _draw(
        axes[0, 0],
        bg_super,
        fp_consensus,
        centroid,
        None,
        roi_color,
        fp_consensus.max() > 0,
        "Superimposed",
    )
    _draw(
        axes[1, 0],
        bg_super,
        fp_consensus,
        centroid,
        crop_halfwidth,
        roi_color,
        fp_consensus.max() > 0,
        None,
    )

    # columns 1..n: individual sessions (whole + zoom)
    for s in range(n):
        _draw(
            axes[0, s + 1],
            bg[s],
            fps[s],
            centroid,
            None,
            roi_color,
            present[s],
            session_names[s],
        )
        _draw(
            axes[1, s + 1],
            bg[s],
            fps[s],
            centroid,
            crop_halfwidth,
            roi_color,
            present[s],
            None,
        )

    axes[0, 0].set_ylabel("whole FOV", fontsize=9)
    axes[1, 0].set_ylabel("zoom", fontsize=9)

    k = sum(present)
    title = f"UCID {ucid}   |   detected in {k}/{n} sessions"
    if cs_sil_value is not None:
        title += f"   |   cs_sil = {cs_sil_value:.3f}"
    fig.suptitle(title, fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
            score = cs_sil[u] if (cs_sil is not None and 0 <= u < len(cs_sil)) else None
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
const DATA = {data_json};      // [{{ucid, score, png}}]
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

    Every figure is pre-rendered and base64-embedded, so file size grows with
    the number of UCIDs (roughly 40-120 KB each). For large runs, pass a
    cs_sil-sorted subset rather than all clusters. max_ucids is a safety cap."""
    import json

    if len(ucids) > max_ucids:
        raise ValueError(
            f"{len(ucids)} UCIDs exceeds max_ucids={max_ucids}; the HTML would be "
            f"huge. Pass a worst-first subset, e.g. "
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


# ---------------------------------------------------------------------------
# example wiring (uncomment in your notebook after a tracking run)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pass
    # results, run_data, params = roicat.pipelines.pipeline_tracking(params)
    #
    # labels_bySession = results['clusters']['labels_bySession']
    # rois_aligned     = results['ROIs']['ROIs_aligned']
    # H = results['ROIs']['frame_height']; W = results['ROIs']['frame_width']
    # cs_sil = results['clusters']['quality_metrics']['cs_sil']
    # fovs_aligned = aligner.ims_registered_nonrigid   # grab from the alignment cell
    #
    # order = order_ucids_by_quality(labels_bySession, cs_sil, ascending=True)
    #
    # # (a) full PDF, worst matches first:
    # export_pdf("tracking_qc.pdf", order, fovs_aligned, rois_aligned,
    #            labels_bySession, H, W, cs_sil=cs_sil, crop_halfwidth=40)
    #
    # # (b) interactive HTML for the borderline clusters only:
    # export_html("tracking_qc.html", order[:200], fovs_aligned, rois_aligned,
    #             labels_bySession, H, W, cs_sil=cs_sil, crop_halfwidth=40)
