# Robustness scripts

## Permutation test: threshold percentile convergence

Run from repo root:

python src/robustness/permutation_threshold_alignment.py

Expected (seed=42, n_perms=10000):
- eta n=21, spc1 n=102
- observed percentiles ~95.238 and ~96.078 (diff ~0.840)
- p_two_tailed ~0.0445
- p_one_tailed (convergence) ~0.02225
