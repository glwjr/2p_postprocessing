# 2p_postprocessing

Post-Suite2p processing tools for the Najafi Lab 2P imaging pipeline.

## Purpose

This repo provides standardized post-processing steps that run after Suite2p (cell detection and trace extraction) and produce inputs for downstream analysis. The goal is to keep methods explicit, reproducible, and consistent across lab members and across mice.

## Modules

| Module         | Status  | Description                                                                                  |
| -------------- | ------- | -------------------------------------------------------------------------------------------- |
| [`dff/`](dff/) | Draft   | Standardized dF/F computation from Suite2p F/Fneu traces                                     |
| `similarity/`  | Planned | Cross-session pairwise similarity analysis on dF/F traces (consumes ROICaT UCID assignments) |

## Pipeline context

```
Raw 2P imaging data
        │
        ▼
   Suite2p (existing: najafi-laboratory/2p_imaging)
        │  produces F.npy, Fneu.npy, iscell.npy, ops.npy per session
        ▼
   ROICaT (existing: najafi-laboratory/cell_matching_roicat)
        │  produces UCID assignments matching ROIs across sessions
        ▼
   This repo
        │  produces standardized dF/F + cross-session similarity analyses
        ▼
   Downstream science (plasticity, drift, etc.)
```

## Setup

ROICaT requires Python 3.11 or 3.12, so this repo pins Python 3.12 to match.

```bash
conda env create -f environment.yml
conda activate 2p_postprocessing
```

This creates an environment with the scientific Python stack (numpy, scipy, h5py, matplotlib, pandas) plus ROICaT. ROICaT is installed via pip inside the conda environment because it isn't packaged on conda-forge.

## Repository conventions

- Each module is a self-contained subdirectory with its own `README.md`.
- Default parameters are set at the top of each script and overridable via command line.
- Output files always include a metadata JSON or HDF5 attribute group recording the parameters used and the pipeline version.
