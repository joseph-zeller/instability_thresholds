Reproducibility Statement

This repository provides the complete computational workflow required to reproduce the results reported in:

Collapse Beyond Extreme Internal Stress: Convergent Percentile Thresholds in Pre-modern Societies

All analyses, figures, tables, and robustness checks reported in the manuscript and Supplementary Information (SI) are generated programmatically from version-controlled source code contained in this repository.

Computational Environment

All analyses were executed using:

Python version: 3.13.9

Operating system: Platform-independent (tested on Windows)

Random seeds: Fixed where applicable to ensure exact reproducibility

Required Python packages are standard scientific libraries (e.g., numpy, pandas, scikit-learn, matplotlib). No proprietary software is required.

Repository Structure

Key directories are organised as follows:

config/          # Analysis configuration files (horizons, regions, disputed cases)
data/final/      # Final, analysis-ready datasets used in all computations
src/             # All source code for analysis, figures, tables, and robustness checks
figures/         # Generated manuscript and SI figures
output/          # Generated SI tables and formatted table outputs
results/         # Intermediate and diagnostic results (thresholds, bootstrap, robustness)

Configuration Files (config/)

YAML files specifying analytic parameters used across robustness analyses:

horizons.yaml – temporal window definitions

regions.yaml – regional groupings

disputed_cases.yaml – historically disputed collapse cases

Data (data/final/)

Contains the harmonised, analysis-ready collapse panels derived from Seshat and MOROS data sources. These files constitute the sole data inputs used by the analysis scripts.

Reproducing the Results

All results reported in the manuscript and SI can be reproduced by running a single command from the repository root:

python src/run_all.py


This script executes, in sequence:

Threshold estimation (cross-validated logistic models)

Robustness analyses (bootstrap and permutation tests)

Table generation (SI Tables S1–S6)

Figure generation (manuscript and SI figures)

Outputs are written to the appropriate subdirectories (results/, output/, and figures/).

Threshold Estimation

Primary collapse thresholds are estimated using cross-validated logistic regression and expressed in empirical percentile space. Threshold outputs are written to:

results/thresholds/


For small-sample datasets (e.g., Seshat η-ratio, n = 21), percentile location rather than absolute threshold magnitude is emphasised, consistent with the manuscript and SI.

Robustness Analyses

Robustness checks include:

Alternative window lengths

Alternative variable transformations

Case exclusions

Bootstrap uncertainty analysis (SPC1 only)

Non-parametric permutation test of cross-dataset threshold convergence

Permutation test outputs supporting SI Section S1.10 are written to:

results/robustness/permutation/


The permutation test uses a fixed random seed and a one-tailed hypothesis testing unusually small percentile separation (convergence), as described in the SI.

Notes on Interpretation

Bootstrap resampling is reported only for SPC1, where sample size permits stable inference. Bootstrap results for the smaller Seshat dataset are intentionally excluded due to statistical non-informativeness, as documented in the Supplementary Information.

Citation and Use

This repository is provided to support transparency and reproducibility of the associated manuscript. Users are encouraged to cite the manuscript when using or adapting the code or results.