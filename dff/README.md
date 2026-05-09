# Standardized dF/F Computation

**Status:** Draft for review (Najafi Lab)

**Author:** Gary White

This module computes dF/F from Suite2p outputs as a standardized post-processing step in the lab pipeline. It produces an HDF5 file with the dF/F traces, a diagnostic figure, a per-cell summary CSV, and a metadata JSON.

## Why this exists

The current Suite2p pipeline (via `run_suite2p_pipeline.py` in the `2p_imaging` repo) produces per-session ROI traces but does not include a documented dF/F step. The `dff.h5` files that exist on disk appear to have been produced by ad-hoc post-processing that varied across sessions and produced occasional artifacts.

Three concrete issues with the existing dF/F outputs informed this proposal:

1. **Division blowups.** Inspection of the QC'd `dff.h5` for SA11_LG session 20250828 showed 22 of 768 ROIs (~2.9%) with dF/F values exceeding ±50, with extremes at ±300+. The bulk distribution was normal (median −0.075, 99th percentile 2.78), but the outlier tail makes downstream analysis brittle.

2. **Aggressive pre-smoothing in the existing config.** `config_neuron.json` sets `sig_baseline: 300.0`, which is a 10-second Gaussian smoothing kernel at 30 Hz. For fast calcium indicators with `tau: 0.25s`, this is 30× longer than typical calcium transients and would smear across them before baseline estimation. The Suite2p default is 10 frames (~333 ms).

3. **Cell selection is undocumented.** The Suite2p config sets `use_builtin_classifier: false` with no custom classifier path. The `iscell.npy` files appear to have been populated by an unknown mechanism — for the 20250828 session, the QC'd dF/F file contains 768 ROIs while the current `iscell.npy` contains 1,081 ROIs with only 42 flagged as cells. There is no rule that maps one to the other.

## The pipeline

For a given session, the script reads `F.npy`, `Fneu.npy`, `iscell.npy`, and `ops.npy` from a Suite2p output directory and runs the following steps.

### Step 1: Cell selection (three-pass filter)

Filter ROIs in three layered passes:

1. **iscell threshold.** Keep ROIs where `iscell[:, 1] > 0.3` (the second column is Suite2p's anatomical detection probability).

2. **F_corrected pre-filter.** After computing `F - 0.7 * Fneu` for each remaining ROI, drop ROIs where the 5th percentile of `F_corrected` is non-positive. These ROIs have neuropil contamination exceeding the cell signal during parts of the session — even if the median is positive, substantial negative dips produce extreme dF/F outliers regardless of the F0 floor. Using the 5th percentile (rather than the median) catches cells that are mostly positive but dip negative for some fraction of the session.

3. **F0 floor-fraction post-filter.** After computing F0 and dF/F, drop ROIs where F0 is pinned to the floor for more than 5% of the session. Cells that hit the floor frequently indicate that the rolling baseline can't track the trace cleanly — typically because of dim signal, transient artifacts, or contamination not caught by the pre-filter. Including them produces extreme negative dF/F outliers in their tails.

The iscell threshold is the most defensible deterministic rule for the first pass given the current state of the pipeline. The Suite2p classifier was disabled in the existing config, so the binary first column reflects whatever ad-hoc curation produced it (often only a handful of cells). The probability column is populated by Suite2p's anatomical sparse detection regardless of the classifier setting, and a threshold of 0.3 produces cell counts in a range typical for cortical 2P imaging (e.g. 70 cells for the 20250828 session, vs. 42 from the binary column). The two F_corrected filters typically drop a small additional number (e.g. 2–6 cells per session combined) and are necessary to prevent extreme dF/F outliers in the final output.

**The 0.3 iscell threshold is the part of this proposal most worth discussing.** Alternatives include (a) training a Suite2p classifier on a few manually-curated sessions and applying it consistently, which is more rigorous but requires upfront curation work, or (b) per-session manual curation, which is highest quality but doesn't scale. The 0.3 threshold is a pragmatic immediate path; (a) is the recommended longer-term solution if cell selection becomes a bottleneck.

### Step 2: Neuropil correction

```
F_corrected(t) = F(t) - 0.7 * F_neu(t)
```

The 0.7 coefficient is the Suite2p / Pachitariu lab convention and matches the existing config (`neucoeff: 0.7`). Unchanged.

### Step 3: Light pre-smoothing

Apply a Gaussian filter to `F_corrected` along the time axis with `sigma = 1.0 seconds` (30 frames at 30 Hz). This smooths shot noise (high-frequency) while preserving calcium event shape (transients have time constants of 0.5–2 seconds for fast indicators). Conservative compared to the existing config's 10-second smoothing.

### Step 4: Baseline estimation (F₀)

For each ROI, compute F₀ as a rolling 8th-percentile of the smoothed trace within a 30-second window centered on each timepoint.

The rolling percentile approach is widely used in published 2P calcium imaging work. The 8th percentile is a standard choice for sparse activity — neurons spend most of their time at baseline, so the 8th percentile of any window is dominated by quiet periods. The 30-second window is long enough to be robust against single events but short enough to track baseline drift across the session.

This differs from the `maximin` method specified in the existing config. Maximin is faster but less interpretable, and its parameters in the existing config over-smooth transients. Rolling percentile with light pre-smoothing produces qualitatively similar baselines to a properly-tuned maximin while being more standard in the field.

### Step 5: Baseline floor

```
F0_floored = max(F0, 0.1 * median(F_corrected_per_roi))
```

This directly addresses the division blowups in the QC'd file. Without a floor, ROIs with low baseline fluorescence produce extreme dF/F values when F₀ approaches zero. The floor is per-ROI (not absolute) so legitimately dim cells aren't disproportionately affected, and the 0.1 multiplier means the floor only activates when F₀ is unusually low for that specific cell. Cells that consistently hit the floor are flagged in the diagnostic output.

### Step 6: dF/F

```
dff(t) = (F_corrected(t) - F0_floored(t)) / F0_floored(t)
```

## Differences from the current state

|                      | Current                                   | Proposed                                                                          |
| -------------------- | ----------------------------------------- | --------------------------------------------------------------------------------- |
| dF/F step location   | Undocumented post-processing              | Standardized script (`compute_dff.py`)                                            |
| Neuropil coefficient | 0.7                                       | 0.7 (unchanged)                                                                   |
| Baseline method      | `maximin` (config) or unknown (actual)    | Rolling 8th percentile, 30s window                                                |
| Pre-smoothing        | 10s Gaussian (config) or unknown (actual) | 1s Gaussian                                                                       |
| F₀ floor             | None (causes blowups)                     | Per-ROI, 0.1 × median(F_corrected)                                                |
| Cell selection       | Ad-hoc, varies across sessions            | iscell[:, 1] > 0.3, then F_corrected 5th percentile > 0, then floor-fraction ≤ 5% |
| Output metadata      | None                                      | Full parameter and version metadata                                               |
| Diagnostic outputs   | None                                      | Figure + CSV + metadata JSON                                                      |
| Reproducibility      | Low                                       | High                                                                              |

## Output format

The primary output is `dff.h5`:

```
dff.h5
├── /dff                  shape: (n_cells, n_timepoints), dtype: float32
├── /F0                   shape: (n_cells, n_timepoints), dtype: float32
├── /cell_indices         shape: (n_cells,), dtype: int64
│                         Suite2p ROI indices for each row of /dff
├── /iscell_prob          shape: (n_cells,), dtype: float32
│                         Suite2p iscell probabilities for selected cells
└── /metadata (group attrs)
    ├── fs                       float
    ├── neuropil_coef            float
    ├── baseline_method          string
    ├── baseline_window_sec      float
    ├── presmooth_sigma_sec      float
    ├── f0_floor_epsilon         float
    ├── iscell_threshold         float
    ├── pipeline_version         string
    └── timestamp_utc            string (ISO 8601)
```

This replaces the existing `dff.h5` format (single `name` dataset with no metadata or cell-mapping). The `cell_indices` field is the key addition: it tells you exactly which Suite2p ROI each row of `dff` corresponds to, which is necessary for matching dF/F to ROICaT UCID assignments downstream.

The script also produces:

- `dff_diagnostics.png` — 6-panel validation figure (dF/F distribution, per-ROI maxima, F₀ floor activations, F₀ trajectories, example traces, percentile breakdown).
- `dff_cell_summary.csv` — per-cell stats: row index, original Suite2p ROI index, iscell probability, median F, median F₀, dF/F max/min/median, fraction of timepoints at F₀ floor.
- `dff_metadata.json` — parameters used, frame rate, session length, ROI counts at each filtering step, timestamp, script version.

These catch problems at the dF/F step rather than letting them propagate to downstream analyses.

## Setup

Use the conda environment defined at the repo root:

```bash
conda env create -f ../environment.yml
conda activate 2p_postprocessing
```

See the [top-level README](../README.md) for full setup details.

## Running on a single session

```bash
python compute_dff.py \
    --suite2p_dir /path/to/SA11_LG/SA11_20250828/suite2p/plane0 \
    --output_dir /path/to/output/SA11_20250828
```

## Tuning parameters

Defaults are set in `DEFAULT_PARAMS` at the top of `compute_dff.py`. Override at the command line:

```bash
python compute_dff.py \
    --suite2p_dir /path/to/suite2p/plane0 \
    --output_dir /path/to/output \
    --baseline_percentile 10 \
    --baseline_window_sec 60
```

Available flags: `--neuropil_coef`, `--presmooth_sigma_sec`, `--baseline_percentile`, `--baseline_window_sec`, `--f0_floor_epsilon`, `--iscell_threshold`.

## Running on a batch of sessions

For SA11_LG batch 1 (sessions 1–10), wrap in a shell loop:

```bash
SA11_DIR=/path/to/SA11_LG
OUT_DIR=/path/to/dff_output

for session in $(ls $SA11_DIR | head -10); do
    python compute_dff.py \
        --suite2p_dir $SA11_DIR/$session/suite2p/plane0 \
        --output_dir $OUT_DIR/$session
done
```

After all sessions complete, review the diagnostic figures and summary CSVs before moving to downstream similarity analysis.

## Reading the output in downstream analysis

```python
import h5py

with h5py.File("dff.h5", "r") as f:
    dff = f["dff"][:]                     # (n_cells, n_timepoints)
    cell_indices = f["cell_indices"][:]   # Suite2p ROI indices for each row
    iscell_prob = f["iscell_prob"][:]
    fs = f["metadata"].attrs["fs"]
```

## Validation

Before applying to all batch 1 sessions, validate on session 20250828 by:

1. Running the script and inspecting `dff_diagnostics.png`.
2. Confirming the bulk distribution shape matches expectations: median near 0, 99th percentile in the 2–4 range, fraction-negative around 0.3–0.5.
3. Confirming the new method eliminates the extreme outliers seen in the QC'd file (no ROIs with `|value| > 50`).
4. Confirming example traces show clean calcium transients without distortion from over-smoothing.
5. Optionally running a parameter sensitivity check: vary `f0_floor_epsilon` (0.05, 0.1, 0.2), `baseline_percentile` (5, 8, 10), and `baseline_window_sec` (20, 30, 60). Output should be robust to these choices.

If anything looks off, parameter tuning or a different cell-selection rule may be needed before applying to all 10 batch 1 sessions.

## Open questions

1. **Cell selection threshold (0.3).** Pragmatic choice based on inspection of one session. May need adjustment for other mice or sessions. Worth revisiting once we see how it performs across all of SA11_LG batch 1.

2. **Save deconvolved spikes alongside dF/F?** Suite2p's `spikedetect` was disabled in the existing config. If the lab wants the option to switch downstream similarity analysis from dF/F-based to spike-rate-based later, we could enable deconvolution in the runner and save `spks.npy` alongside `dff.h5`. Adds runtime but provides flexibility.

3. **Long-term cell selection.** The 0.3 probability threshold is a stopgap. A trained Suite2p classifier on 5–10 manually-curated sessions would produce more reliable selection across the lab's data. Worth scoping as a separate task.

4. **Backwards compatibility.** Existing analyses using the old `dff.h5` format will need to be updated to read the new format. The `cell_indices` field is the main addition that enables better downstream use.
