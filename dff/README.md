# Standardized dF/F Computation

This module computes dF/F from Suite2p outputs as a standardized post-processing step in the lab pipeline. It produces an HDF5 file with the dF/F traces, a diagnostic figure, a per-cell summary CSV, and a metadata JSON.

## Why this exists

The current Suite2p pipeline (via `run_suite2p_pipeline.py` in the `2p_imaging` repo) produces per-session ROI traces but does not include a documented dF/F step. The `dff.h5` files that exist on disk appear to have been produced by ad-hoc post-processing that varied across sessions and produced occasional artifacts.

Three concrete issues with the existing dF/F outputs informed this proposal:

1. **Division blowups.** Inspection of the QC'd `dff.h5` for SA11_LG session 20250828 showed 22 of 768 ROIs (~2.9%) with dF/F values exceeding ±50, with extremes at ±300+. The bulk distribution was normal (median −0.075, 99th percentile 2.78), but the outlier tail makes downstream analysis brittle.

2. **Aggressive pre-smoothing in the existing config.** `config_neuron.json` sets `sig_baseline: 300.0`, which is a 10-second Gaussian smoothing kernel at 30 Hz. For fast calcium indicators with `tau: 0.25s`, this is 30× longer than typical calcium transients and would smear across them before baseline estimation. The Suite2p default is 10 frames (~333 ms).

3. **Cell selection is undocumented.** The Suite2p config sets `use_builtin_classifier: false` with no custom classifier path. The `iscell.npy` files appear to have been populated by an unknown mechanism — for the 20250828 session, the QC'd dF/F file contains 768 ROIs while the current `iscell.npy` contains 1,081 ROIs with only 42 flagged as cells. There is no rule that maps one to the other.

## The pipeline

For a given session, the script reads `F.npy`, `Fneu.npy`, `iscell.npy`, and `ops.npy` from a Suite2p output directory and runs the following steps.

### Step 1: Cell selection (two-pass filter)

Filter ROIs in two layered passes:

1. **iscell threshold.** Keep ROIs where `iscell[:, 1] > 0.3` (the second column is Suite2p's anatomical detection probability).

2. **F0 floor-fraction post-filter.** After computing F0 and dF/F, drop ROIs where F0 is pinned to the floor for more than 5% of the session (default; configurable via `--post_filter_floor_frac`). Cells that hit the floor frequently indicate that the rolling baseline can't track the trace cleanly — typically because of dim signal, transient artifacts, or pathological neuropil contamination. Including them produces extreme negative dF/F outliers in their tails.

Earlier versions of the pipeline (v0.1–v0.3) ran an additional pre-filter on `F - r * Fneu` that dropped cells with negative 5th-percentile corrected fluorescence. This was removed in v0.4.0: the per-cell `r` estimator (Step 2 below) handles pathological cells natively by detecting when the optimum r lies outside [0, 1] and assigning the population median r as fallback. The pre-filter was a workaround for a fixed-coefficient method's failure mode, no longer needed.

The iscell threshold is the most defensible deterministic rule for the first pass given the current state of the pipeline. The Suite2p classifier was disabled in the existing config, so the binary first column reflects whatever ad-hoc curation produced it (often only a handful of cells). The probability column is populated by Suite2p's anatomical sparse detection regardless of the classifier setting, and a threshold of 0.3 produces cell counts in a range typical for cortical 2P imaging (e.g. 70 cells for the 20250828 session, vs. 42 from the binary column). The post-filter typically drops 0–2 cells per session.

**The 0.3 iscell threshold is the part of this proposal most worth discussing.** Alternatives include (a) training a Suite2p classifier on a few manually-curated sessions and applying it consistently, which is more rigorous but requires upfront curation work, or (b) per-session manual curation, which is highest quality but doesn't scale. The 0.3 threshold is a pragmatic immediate path; (a) is the recommended longer-term solution if cell selection becomes a bottleneck.

### Step 2: Neuropil correction

```
F_corrected(t) = F(t) - r_i * F_neu(t)
```

where `r_i` is estimated **per ROI** rather than using a fixed global coefficient.

**Method.** Implements the Allen Brain Observatory Visual Behavior 2P pipeline's per-cell `r` estimation (Section F of the Technical Whitepaper, page 36). For each ROI, the algorithm jointly estimates `r` and a smoothed cell trace `F_C` by minimizing

```
E = <(F_C - (F_M - r * F_N))² + λ * (dF_C/dt)²>_t
```

with `λ = 0.05`. `F_C` has a closed-form solution for any fixed `r` (a tridiagonal regularized least-squares problem); the optimization on `r` is cross-validated, with `r` chosen on the second half of the trace to minimize `E`.

**Optimizer.** The whitepaper specifies gradient descent. We use scipy's `minimize_scalar` (Brent's method, bounded) instead — same objective function, same cross-validation, ~200x faster (10–20 function evaluations vs. ~400+ gradient steps near the flat minimum). This is a faithful implementation of Allen's mathematical objective with a different solver.

**Validation.** The estimator was validated against synthetic ground truth (100 cells with `r_true` drawn from `Beta(2, 1.5)`, 5-min traces with realistic calcium transients): bias ≈ 0.0002, slope ≈ 1.003, Pearson r ≈ 0.999, all 100 cells converged. The validation harness is `validate_neuropil_fitter.py`, runnable on any future change to the estimator.

**Validation against Allen's published dF/F.** Running `compute_dff.py` on one Allen Visual Coding 2P experiment, the resulting dF/F correlates with Allen's published dF/F at Pearson r = 0.98 (per-cell maxima). See the Cross-pipeline validation section below.

**Fallback.** ROIs where the optimization fails (e.g. constant `F_N`, optimum `E` exceeds the whitepaper's threshold of `2 * |<F_M>|`) receive the median `r` of converged ROIs as a fallback — the same strategy Allen uses for non-converged cells. The number of fallback ROIs is logged and recorded in `dff_metadata.json`.

**Override.** To force a fixed global coefficient (e.g. for comparison with Suite2p convention), pass `--neuropil_coef 0.7` at the command line. Per-ROI estimation is the default.

**Note on the population mean of `r` for this lab's data.** Across SA-line sessions tested so far (SA11, SA17, SA18), the per-session mean `r` falls in the 0.2–0.3 range with std 0.2–0.3 — substantially lower than Allen's reported mean of 0.68 on benchmark V1 GCaMP6f data. The algorithm is identical to Allen's; the gap likely reflects genuinely lower neuropil contamination in this preparation (possible contributors: Mesoscope vs. Scientifica imaging, surgical preparation, indicator brightness). This hypothesis should be verified with senior lab members before being asserted. The validity of the *algorithm* is independent of this observation, since synthetic ground-truth recovery is unbiased.

### Step 3: Light pre-smoothing

Apply a Gaussian filter to `F_corrected` along the time axis with `sigma = 1.0 seconds` (30 frames at 30 Hz). This smooths shot noise (high-frequency) while preserving calcium event shape (transients have time constants of 0.5–2 seconds for fast indicators). Conservative compared to the existing config's 10-second smoothing.

### Step 4: Baseline estimation (F₀)

For each ROI, compute F₀ as a rolling 8th-percentile of the smoothed trace within a 30-second window centered on each timepoint.

The rolling percentile approach is widely used in published 2P calcium imaging work. The 8th percentile is a standard choice for sparse activity — neurons spend most of their time at baseline, so the 8th percentile of any window is dominated by quiet periods. The 30-second window is long enough to be robust against single events but short enough to track baseline drift across the session.

This differs from the `maximin` method specified in the existing config. Maximin is faster but less interpretable, and its parameters in the existing config over-smooth transients. Rolling percentile with light pre-smoothing produces qualitatively similar baselines to a properly-tuned maximin while being more standard in the field.

### Step 5: Baseline floor

```
F0_floored = max(F0, 0.1 * |median(F_corrected_per_roi)|)
```

This directly addresses the division blowups in the QC'd file. Without a floor, ROIs with low baseline fluorescence produce extreme dF/F values when F₀ approaches zero. The floor is per-ROI (not absolute) so legitimately dim cells aren't disproportionately affected, and the 0.1 multiplier means the floor only activates when F₀ is unusually low for that specific cell.

The floor is anchored to the *absolute value* of `median(F_corrected)` (added in v0.4.1). If neuropil over-subtraction pushes the corrected signal mostly negative, an unsigned median produces a negative floor — which inverts the sign of dF/F where the floor activates. Taking the absolute value ensures the floor is always non-negative, regardless of how aggressively neuropil was subtracted. Cells that consistently hit the floor are flagged in the diagnostic output and dropped by the post-filter (Step 1).

### Step 6: dF/F

```
dff(t) = (F_corrected(t) - F0_floored(t)) / F0_floored(t)
```

## Differences from the current state

|                      | Current                                   | Proposed                                                                                                                       |
| -------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| dF/F step location   | Undocumented post-processing              | Standardized script (`compute_dff.py`)                                                                                         |
| Neuropil coefficient | 0.7                                       | Per-ROI Allen cross-validated regression (Visual Behavior 2P whitepaper §F); validated unbiased against synthetic ground truth |
| Baseline method      | `maximin` (config) or unknown (actual)    | Rolling 8th percentile, 30s window                                                                                             |
| Pre-smoothing        | 10s Gaussian (config) or unknown (actual) | 1s Gaussian                                                                                                                    |
| F₀ floor             | None (causes blowups)                     | Per-ROI, 0.1 × \|median(F_corrected)\|                                                                                         |
| Cell selection       | Ad-hoc, varies across sessions            | iscell[:, 1] > 0.3, then floor-fraction ≤ 5%                                                                                   |
| Output metadata      | None                                      | Full parameter and version metadata                                                                                            |
| Diagnostic outputs   | None                                      | Figure + CSV + metadata JSON                                                                                                   |
| Reproducibility      | Low                                       | High                                                                                                                           |

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
├── /neuropil_coef        shape: (n_cells,), dtype: float32
│                         Per-ROI neuropil contamination ratio r used for subtraction
└── /metadata (group attrs)
    ├── fs                         float
    ├── neuropil_coef_method       string  ("per_roi_regression" for Allen-style cross-validated regression, or "fixed")
    ├── neuropil_coef_mean         float
    ├── neuropil_coef_std          float
    ├── baseline_method            string
    ├── baseline_window_sec        float
    ├── presmooth_sigma_sec        float
    ├── f0_floor_epsilon           float
    ├── iscell_threshold           float
    ├── pipeline_version           string
    └── timestamp_utc              string (ISO 8601)
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

Available flags: `--neuropil_coef`, `--presmooth_sigma_sec`, `--baseline_percentile`, `--baseline_window_sec`, `--f0_floor_epsilon`, `--iscell_threshold`, `--post_filter_floor_frac`.

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
2. Confirming the bulk distribution shape matches expectations: median near 0, 99th percentile in the 1–3 range, fraction-negative around 0.3–0.5.
3. Confirming the new method eliminates the extreme outliers seen in the QC'd file (no ROIs with `|value| > 50` outside of sessions with documented imaging artifacts; see SA18 caveat below).
4. Confirming example traces show clean calcium transients without distortion from over-smoothing.
5. Reviewing the neuropil coefficient distribution in the metadata: mean `r` for SA-line sessions has been falling in 0.2–0.3 with std 0.2–0.3 and zero fallback cells. Wildly different values are a signal worth investigating.

If anything looks off, parameter tuning or a different cell-selection rule may be needed before applying to all 10 batch 1 sessions.

**Known recording-artifact caveat:** session SA18_LG/20251226 produced 4 of 183 cells with `|dff| > 50`, traced to step discontinuities in F at ~7 min and ~28 min visible in the F0 trajectory panel — a recording-level event (focus shift, file concatenation, or motion correction failure), not a pipeline bug. The same mouse's 12/27 session was clean. The pipeline correctly handles steady-state portions of the recording; the discontinuity affects only cells active across the step. This kind of issue should be surfaced to whoever runs the imaging side rather than worked around in dF/F computation.

## Testing

The pipeline has an automated test suite covering both the core `compute_dff()` function and the full end-to-end `run()` pipeline. Tests use synthetic Suite2p outputs built by pytest fixtures, so no real session data is required to run them.

```bash
# Full suite (~30 seconds)
pytest

# Unit tests only — fast enough for a pre-commit hook
pytest test_compute_dff.py
```

Test files:

- `test_compute_dff.py` — unit tests on `compute_dff()` and `estimate_neuropil_coefs()` with deterministic synthetic inputs. Covers neuropil correction math, baseline tracking on slow drift, transient recovery, F0 floor activation (including the v0.4.1 negative-median path), output shapes/dtypes, per-ROI independence, determinism, and per-cell `r` recovery against synthetic ground truth.
- `test_pipeline.py` — integration tests on `run()` against synthetic Suite2p directories. Covers output file production, cross-file consistency (e.g. `dff.h5` shape matches `n_cells_final` in metadata), parameter propagation, filter behavior, signal recovery on transient-bearing data, and pipeline determinism.
- `test_regression_allen.py` — regression test against a small slice of real Allen Brain Observatory data committed under `tests/fixtures/allen_slice/`. Re-running the pipeline against the fixture must produce byte-identical dF/F, F0, and cell indices to what was saved when the fixture was built. Skipped automatically if the fixture isn't present (fresh clone).
- `conftest.py` — shared pytest fixtures, including a clean 20-ROI synthetic session and a 10-ROI session with injected calcium transients.
- `validate_neuropil_fitter.py` — standalone validation harness for the per-cell `r` estimator. Runs 100 synthetic cells with `r_true` drawn from `Beta(2, 1.5)` and reports bias, slope, Pearson r. Use this when changing the estimator, not on every CI run (the unit-test version uses 30 cells for speed).

To generate or update the regression fixture, run `python build_regression_fixture.py` once locally. This pulls one Allen experiment via direct NWB download (no AllenSDK dependency), slices it to 30 cells × 5 minutes, runs `compute_dff.py` on the slice, and writes the inputs and expected output to `tests/fixtures/allen_slice/`. Commit the fixture and the regression test runs against it on every subsequent test invocation. Re-run the helper only when the pipeline's expected output legitimately changes (intentional algorithm changes, version bumps).

The tests verify pipeline correctness on synthetic data; they don't replace the manual validation step above on real sessions. Both layers are needed.

## Cross-pipeline validation against Allen Brain Observatory

Beyond manual inspection on lab sessions and automated tests on synthetic data, this module includes scripts to validate `compute_dff.py` against the Allen Brain Observatory Visual Coding 2P dataset. Allen publishes raw fluorescence (`F`), neuropil traces (`Fneu`), and their own pipeline's dF/F traces, all derived from awake mouse visual cortex 2P imaging — the same domain the lab works in. Comparing our dF/F against Allen's published dF/F is a direct sanity check.

The validation reads Allen's NWB files directly via `h5py` (already in `environment.yml`), avoiding the AllenSDK dependency, which has Python 3.12 compatibility issues.

```bash
# Pull one Allen experiment, write Suite2p-format outputs + Allen's reference dF/F
python adapt_allen_to_suite2p.py --output_dir ~/allen_validation

# Run our pipeline on it
python compute_dff.py \
    --suite2p_dir ~/allen_validation/suite2p/plane0 \
    --output_dir ~/allen_validation/dff_output

# Compare our dF/F against Allen's published dF/F
python compare_to_allen.py \
    --our_dff ~/allen_validation/dff_output/dff.h5 \
    --allen_dff ~/allen_validation/allen_dff.npy \
    --output_fig ~/allen_validation/comparison.png
```

**Expected agreement.** As of v0.4.0+, our neuropil correction is identical to Allen's algorithm (cross-validated regression with smoothness regularization). The remaining algorithmic difference is in dF/F itself: Allen applies a 3.33-second median filter and clips trends to ±2.5σ as a post-dF/F detrending step; we deliberately omit this to preserve transient timing for downstream cross-session similarity work.

Empirical comparison on Allen experiment 569407590 (128 cells, ~1 hour):

| metric                     | ours         | Allen        |
| -------------------------- | ------------ | ------------ |
| median dF/F                | 0.017        | 0.000        |
| 99th percentile            | 0.314        | 0.237        |
| 99.9th percentile          | 1.587        | 1.290        |
| 0.1th percentile           | -0.202       | -0.214       |
| max                        | 6.21         | 5.26         |
| min                        | -0.40        | -0.45        |
| Per-cell max-dF/F Pearson r | 0.98         | —            |

The negative tail nearly matches Allen's exactly, indicating the unbiased per-cell `r` is doing the right work. The remaining ~30% gap on positive percentiles is consistent with the absence of post-dF/F median smoothing in our pipeline.

What to look for when running on a new Allen experiment:

- **Per-cell max-dF/F correlation > 0.95** (Pearson r). The two pipelines should rank cells by activity nearly identically. Below ~0.85 is a signal worth investigating.
- **Median dF/F within 0.05 of zero** in both. Sparse activity should leave the bulk of the distribution at baseline.
- **99th percentile within ~50% of Allen's.** A 2× discrepancy points to parameter or implementation issues.
- **Q-Q plot roughly along y = x for the body of the distribution.** Divergence in the top 1% is fine (post-dF/F smoothing); the bulk should track.

Important limitation: this validates that the pipeline produces output consistent with Allen's published method on Allen's data. It does **not** validate that the parameters are right for Najafi Lab data specifically — different mice, indicators, and imaging conditions may require different choices. Manual inspection of `dff_diagnostics.png` on a real session (the Validation section above) remains essential.

## Open questions

1. **Cell selection threshold (0.3).** Pragmatic choice based on inspection of one session. May need adjustment for other mice or sessions. Worth revisiting once we see how it performs across all of SA11_LG batch 1.

2. **Save deconvolved spikes alongside dF/F?** Suite2p's `spikedetect` was disabled in the existing config. If the lab wants the option to switch downstream similarity analysis from dF/F-based to spike-rate-based later, we could enable deconvolution in the runner and save `spks.npy` alongside `dff.h5`. Adds runtime but provides flexibility.

3. **Long-term cell selection.** The 0.3 probability threshold is a stopgap. A trained Suite2p classifier on 5–10 manually-curated sessions would produce more reliable selection across the lab's data. Worth scoping as a separate task.

4. **Backwards compatibility.** Existing analyses using the old `dff.h5` format will need to be updated to read the new format. The `cell_indices` field is the main addition that enables better downstream use.

5. **Population mean of `r` is lower than Allen's reported benchmark.** Across SA-line sessions tested so far, mean `r` per session is ~0.2–0.3, while Allen reports ~0.68 on benchmark V1 GCaMP6f data. The algorithm matches Allen's exactly and is unbiased on synthetic ground truth, so the gap likely reflects real preparation differences (Mesoscope vs. Scientifica imaging, surgical preparation, indicator brightness) rather than methodological ones. Worth verifying with senior lab members. Monitor the `neuropil_coef_per_roi` summary in `dff_metadata.json` (mean, std, n_fallback) when processing a new data type; an unusually high fallback count (>5%) or wildly different mean across sessions of the same mouse is worth investigating.
