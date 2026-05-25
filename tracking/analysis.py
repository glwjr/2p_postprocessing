"""
Analyze ROICaT tracking results for one or more datasets.

Usage:
  # Single dataset (by name, assumes lab's standard layout):
  python analysis.py SA11_LG

  # Single dataset (explicit path to .richfile.zip):
  python analysis.py /Volumes/Elements/Najafi/2P_Imaging/SA11_LG/results/SA11.tracking.results_all.richfile.zip

  # All datasets under the base directory:
  python analysis.py --all

  # Custom base directory:
  python analysis.py --all --base /some/other/path
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import roicat

# Default base directory — lab's 2P imaging root
DEFAULT_BASE = Path("/Volumes/Elements/Najafi/2P_Imaging")


def find_richfile(dataset_name: str, base: Path) -> Path:
    """Find the results_all.richfile.zip for a dataset under base/{dataset_name}/results/."""
    results_dir = base / dataset_name / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"No results directory at {results_dir}")

    candidates = list(results_dir.glob("*.tracking.results_all.richfile.zip"))
    if not candidates:
        raise FileNotFoundError(f"No richfile.zip found in {results_dir}")
    if len(candidates) > 1:
        print(
            f"  Warning: multiple richfiles in {results_dir}, using {candidates[0].name}"
        )
    return candidates[0]


def find_all_richfiles(base: Path) -> list[Path]:
    """Find all results_all.richfile.zip files under base/*/results/."""
    return sorted(base.glob("*/results/*.tracking.results_all.richfile.zip"))


def summarize_quality_metrics(qm: dict) -> dict:
    """Compute numerical summaries of ROICaT quality metrics."""
    summary = {}
    for key, vals in qm.items():
        try:
            arr = np.asarray(vals, dtype=float).ravel()
        except (ValueError, TypeError):
            continue  # skip non-numeric metrics

        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue

        summary[key] = {
            "n": int(len(valid)),
            "median": float(np.median(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "frac_above_0.2": float((valid > 0.2).mean()),  # Nguyen et al. threshold
        }
    return summary


def analyze_dataset(richfile_path: Path) -> dict:
    """Run the full matched-neuron + quality-metrics analysis on one dataset.

    Writes per-dataset CSVs alongside the input file and returns a dict of
    headline numbers for the cross-dataset summary.
    """
    output_dir = richfile_path.parent
    dataset_name = richfile_path.stem.replace(".tracking.results_all.richfile", "")

    print(f"\n{'='*60}")
    print(f"Analyzing: {dataset_name}")
    print(f"  Path: {richfile_path}")
    print(f"{'='*60}")

    results = roicat.util.RichFile_ROICaT(path=str(richfile_path)).load()

    # --- Matched-neuron counts ---
    labels = np.array(results["clusters"]["labels"])
    labels_bySession = [np.array(s) for s in results["clusters"]["labels_bySession"]]
    n_sessions = len(labels_bySession)

    ucid_to_sessions = {}
    for sess_idx, sess_labels in enumerate(labels_bySession):
        for ucid in np.unique(sess_labels):
            if ucid < 0:
                continue
            ucid_to_sessions.setdefault(int(ucid), set()).add(sess_idx)

    session_counts = np.array([len(s) for s in ucid_to_sessions.values()])

    n_total_rois = len(labels)
    n_discarded = int((labels == -1).sum())
    n_total_clusters = len(ucid_to_sessions)
    n_matched_2plus = int((session_counts >= 2).sum())
    n_matched_10plus = int((session_counts >= 10).sum())
    n_matched_15plus = int((session_counts >= 15).sum())
    n_in_all_sessions = int((session_counts == n_sessions).sum())

    print(f"\nSessions: {n_sessions}")
    print(f"Total ROIs: {n_total_rois}  (discarded: {n_discarded})")
    print(f"Total clusters: {n_total_clusters}")
    print(f"  Matched in ≥2 sessions:  {n_matched_2plus}")
    print(f"  Matched in ≥10 sessions: {n_matched_10plus}")
    print(f"  Matched in ≥15 sessions: {n_matched_15plus}")
    print(f"  Matched in all {n_sessions}: {n_in_all_sessions}")

    # --- Quality metrics summary ---
    qm = results["clusters"].get("quality_metrics") or {}
    qm_summary = summarize_quality_metrics(qm)

    if qm_summary:
        print(f"\nQuality metrics:")
        for key, stats in qm_summary.items():
            print(
                f"  {key:25s} median={stats['median']:+.3f}  "
                f"mean={stats['mean']:+.3f}  "
                f"frac>0.2={stats['frac_above_0.2']:.1%}  "
                f"(n={stats['n']})"
            )
    else:
        print("\nNo quality metrics found in results.")

    # --- Write per-dataset CSVs ---
    summary_path = output_dir / f"{dataset_name}.matched_neurons_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "n_sessions",
                "n_total_rois",
                "n_discarded_rois",
                "n_total_clusters",
                "n_matched_2plus_sessions",
                "n_matched_10plus_sessions",
                "n_matched_15plus_sessions",
                "n_matched_all_sessions",
            ]
        )
        writer.writerow(
            [
                dataset_name,
                n_sessions,
                n_total_rois,
                n_discarded,
                n_total_clusters,
                n_matched_2plus,
                n_matched_10plus,
                n_matched_15plus,
                n_in_all_sessions,
            ]
        )

    distribution_path = output_dir / f"{dataset_name}.matched_neurons_distribution.csv"
    with open(distribution_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_sessions_present", "n_neurons"])
        for n in range(1, n_sessions + 1):
            writer.writerow([n, int((session_counts == n).sum())])

    qm_path = output_dir / f"{dataset_name}.quality_metrics_summary.csv"
    with open(qm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["metric", "n", "median", "mean", "std", "min", "max", "frac_above_0.2"]
        )
        for key, stats in qm_summary.items():
            writer.writerow(
                [
                    key,
                    stats["n"],
                    stats["median"],
                    stats["mean"],
                    stats["std"],
                    stats["min"],
                    stats["max"],
                    stats["frac_above_0.2"],
                ]
            )

    print(f"\nWrote:")
    print(f"  {summary_path.name}")
    print(f"  {distribution_path.name}")
    print(f"  {qm_path.name}")

    # --- Return headline numbers for combined summary ---
    return {
        "dataset": dataset_name,
        "n_sessions": n_sessions,
        "n_total_rois": n_total_rois,
        "n_discarded_rois": n_discarded,
        "n_total_clusters": n_total_clusters,
        "n_matched_2plus_sessions": n_matched_2plus,
        "n_matched_10plus_sessions": n_matched_10plus,
        "n_matched_15plus_sessions": n_matched_15plus,
        "n_matched_all_sessions": n_in_all_sessions,
        "cluster_silhouette_median": qm_summary.get("cs_sil", {}).get("median"),
        "cluster_silhouette_frac_above_0.2": qm_summary.get("cs_sil", {}).get(
            "frac_above_0.2"
        ),
        "cluster_intra_means_median": qm_summary.get("cs_intra_means", {}).get(
            "median"
        ),
    }


def write_combined_summary(rows: list[dict], output_path: Path):
    """Write a combined CSV across all analyzed datasets."""
    if not rows:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCombined summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name (e.g., 'SA11_LG') or full path to richfile.zip",
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all datasets under --base"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Base 2P_Imaging directory (default: {DEFAULT_BASE})",
    )
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Provide a dataset name/path, or use --all")

    # Resolve which richfiles to process
    if args.all:
        richfiles = find_all_richfiles(args.base)
        if not richfiles:
            print(f"No richfiles found under {args.base}")
            sys.exit(1)
        print(f"Found {len(richfiles)} dataset(s) under {args.base}")
    else:
        # Single dataset: either a name or a full path
        as_path = Path(args.dataset)
        if as_path.exists() and as_path.suffix == ".zip":
            richfiles = [as_path]
        else:
            richfiles = [find_richfile(args.dataset, args.base)]

    # Analyze each, collect headline rows
    rows = []
    for rf in richfiles:
        try:
            row = analyze_dataset(rf)
            rows.append(row)
        except Exception as e:
            print(f"\n  ERROR analyzing {rf}: {e}")
            continue

    # Write combined summary if more than one
    if len(rows) > 1:
        combined_path = args.base / "matched_neurons_combined_summary.csv"
        write_combined_summary(rows, combined_path)


if __name__ == "__main__":
    main()
