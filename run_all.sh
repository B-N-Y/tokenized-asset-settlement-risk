#!/bin/bash
# Reproduce every result reported in the article, in order.
#
#   bash run_all.sh            # uses the python on PATH
#   PY=/path/to/python bash run_all.sh
#
# Expected runtime: about 4 minutes on a current desktop machine.
# See REPRODUCIBILITY.md for the verified environment and the expected values.
set -u
PY="${PY:-python}"
cd "$(dirname "$0")/scripts" || exit 1

SCRIPTS=(
  02_empirical_analysis.py          # empirical calibration, jump detection
  03_full_recalibration.py          # Bridge and Native simulation grids
  05_historical_validation.py       # Verra 2023 consistency check, optimal buffer
  06_endogeneity_analysis.py        # confidence-crisis dynamics
  07_robustness_price_models.py     # GBM, Merton, Student-t, GARCH
  08_sensitivity_elasticity.py      # trust-elasticity sweep
  09_statistical_tests.py           # hypothesis tests
  11_distributional_diagnostics.py  # goodness of fit, ARCH-LM, variance ratio
  12_cross_market_correlation.py    # auxiliary cross-market checks
  13_architecture_dominance.py      # Bridge versus Native threshold
  14_regenerate_summary_csvs.py     # summary CSVs used by the table generator
  15_consistency_reruns.py          # H6/TOST, variance decomposition
  16_convergence_seed_stability.py  # convergence and multi-seed stability
  04_generate_figures.py            # figures
  10_generate_latex_tables.py       # LaTeX tables
)

fail=0
for s in "${SCRIPTS[@]}"; do
  printf '\n=== %s\n' "$s"
  start=$(date +%s)
  if MPLBACKEND=Agg "$PY" "$s"; then
    printf '[done] %s in %ss\n' "$s" "$(( $(date +%s) - start ))"
  else
    printf '[FAILED] %s\n' "$s"
    fail=1
    break
  fi
done

if [ "$fail" -eq 0 ]; then
  printf '\nAll scripts completed. Outputs are in results/tables, results/grids and results/figures.\n'
else
  printf '\nPipeline stopped on an error.\n'
  exit 1
fi
