[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18489850.svg)](https://doi.org/10.5281/zenodo.18489850)

# Instability Thresholds in Large-Scale Human Systems

Reproducible code and data for identifying convergent instability thresholds and nonlinear regime transitions in large-scale human systems.

This repository accompanies the paper:

**Threshold Instability in Large-Scale Human Systems: Quantitative Evidence for Collapse Beyond Extreme Complexity**

It provides fully reproducible computational pipelines for estimating regime-transition thresholds associated with systemic breakdown across independent historical datasets.

---

## 📌 Overview

Complex societies accumulate internal structural and informational load as they grow.  
This project tests whether large-scale societal breakdown is reliably preceded by a universal high-stress instability regime.

Using independent representations of internal systemic stress, the analysis:

- Estimates nonlinear regime-transition thresholds via logistic models  
- Locates thresholds using scale-free percentile methods  
- Tests robustness across temporal horizons, stratifications, and exclusions  
- Evaluates cross-predictor convergence via permutation inference  

Results demonstrate consistent threshold convergence in the extreme upper tail of internal stress distributions, indicating a universal instability regime preceding collapse.

---

## 📂 Repository Structure

```text
instability_thresholds/
├── config/                 # Model configuration files
├── data/final/             # Cleaned datasets used in analysis
├── results/                # Model outputs and threshold estimates
├── figures/                # Generated figures for manuscript & SI
├── output/
│   └── tables/             # Reproducible SI tables (CSV + Markdown)
├── src/
│   ├── tables/             # Table generation scripts
│   └── run_all.py          # End-to-end reproducibility pipeline
├── README.md
├── REPRODUCIBILITY.md
└── environment.yml


📊 Data Sources

This analysis integrates independent historical datasets:

Seshat Equinox (2020) — institutional and social complexity indicators

MOROS — independent catalogues of large-scale political regime breakdown

SPC1 dataset — structural-demographic pressure proxy

All cleaned datasets used in the paper are provided in data/final/.

⚙️ Reproducibility

All results in the manuscript and Supplementary Information can be regenerated from raw inputs using the provided environment.

1. Create environment
conda env create -f environment.yml
conda activate instability_thresholds

2. Run full pipeline
python src/run_all.py


This will:

Recompute all model fits

Generate threshold estimates

Produce all robustness tables

Recreate manuscript figures

3. Generate SI tables only
python src/tables/make_all_tables.py


Outputs are written to:

output/tables/


in both CSV and Markdown format for direct manuscript inclusion.

📈 Core Methods

Logistic regime-transition modelling

Percentile-based threshold localisation

Cross-validated discrimination (AUC)

Robustness checks across:

Temporal horizons

Influential-case exclusions

Population stratification

Independent predictor convergence

Permutation inference for threshold alignment.

📜 License

MIT License — open for reuse and extension with attribution.

📖 Citation

If you use this code or data, please cite the accompanying paper and the Zenodo release (see DOI badge above).