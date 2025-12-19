Table S4. Robustness of η-ratio Thresholds Across Temporal Horizons
-------------------------------------------------------------------------------
Horizon (Model)                   |  n  |   θ*   | Percentile |   AUC
-------------------------------------------------------------------------------
η-ratio (100-year horizon)       |  21 |   2.12 |       95.2% |  0.72
η-ratio (50-year horizon)        |  21 |   9.01 |      100.0% |  0.47
η-ratio (150-year horizon)       |  21 |   1.47 |       81.0% |  0.69
-------------------------------------------------------------------------------
Notes:
- The 100-year η-ratio row uses the cross-validated threshold summary
  from threshold_cv_summary_w100.csv (as in Table 1).
- θ* is the logistic decision threshold where P(collapse)=0.5.
- Percentiles are computed on the 0–100 scale.
