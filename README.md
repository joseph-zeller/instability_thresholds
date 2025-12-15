Collapse Thresholds

Reproducible code and data for identifying a convergent percentile instability threshold preceding societal collapse

This repository provides the full computational workflow required to reproduce the results reported in:

Collapse Beyond Extreme Internal Stress: Convergent Percentile Thresholds in Pre-modern Societies

All analyses, figures, tables, and robustness checks reported in the manuscript and Supplementary Information (SI) are generated programmatically from version-controlled source code contained in this repository.

Overview

The repository implements a unified analytical pipeline to identify and validate a convergent percentile threshold of internal stress preceding societal collapse across independent historical datasets.

Specifically, it reproduces:

Threshold estimation using cross-validated logistic models

Percentile-based instability thresholds

Robustness checks across weighting schemes, horizons, and exclusions

Cross-dataset permutation tests for threshold convergence

All manuscript and SI figures and tables

The workflow is designed for full computational reproducibility.

## Repository Structure

Key directories are organised as follows:

```text
collapse_thresholds/
├── config/                      # Configuration files (YAML)
│   ├── disputed_cases.yaml
│   ├── horizons.yaml
│   └── regions.yaml
├── data/
│   └── final/                   # Final analysis datasets (CSV)
│       ├── seshat_EI_collapse_panel_w100.csv
│       ├── seshat_EI_collapse_panel_w50.csv
│       ├── SPC1_collapse_panel_w100.csv
│       └── SPC1_collapse_panel_w100_horizons.csv
├── src/                         # All analysis code
│   ├── compute_thresholds.py
│   ├── run_all.py
│   ├── figures/
│   │   └── make_all_figures.py
│   ├── tables/
│   │   └── make_all_tables.py
│   └── robustness/
│       └── permutation_threshold_alignment.py
├── figures/                     # Generated figures
│   ├── manuscript/
│   └── si/
├── output/                      # Generated SI tables (CSV + Markdown)
├── results/                     # Model outputs and robustness results
├── environment.yml              # Reproducible Python environment
├── README.md
└── REPRODUCIBILITY.md

Software Environment

All analyses were executed using the following environment:

Python: 3.13.9

Operating system: OS-independent (tested on Windows)

A fully specified conda environment is provided:

environment.yml


This file pins all required dependencies and versions.

Reproducing the Results

A complete, step-by-step replication guide is provided in:

REPRODUCIBILITY.md


At a high level, reproduction proceeds as follows:

Create and activate the conda environment

Run the full pipeline via:

python src/run_all.py


Generated outputs will populate:

figures/ (manuscript + SI figures)

output/ (SI tables)

results/ (thresholds, robustness, permutation tests)

No manual intervention is required once the pipeline is launched.

Data Sources

The analysis draws on harmonised historical datasets including:

Seshat Global History Databank (internal stress and institutional variables)

MOROS (Mortality of States Database) for collapse event validation

All datasets used in the analyses are included in data/final/ in processed form suitable for replication.

License

This repository is released under the MIT License, permitting reuse with attribution.

Contact

For questions regarding the code or analyses, please refer to the manuscript or open an issue in this repository.