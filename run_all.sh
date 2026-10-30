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

if [ "$fail" -ne 0 ]; then
  printf '\nPipeline stopped on an error.\n'
  exit 1
fi

printf '\nAll scripts completed. Outputs are in results/tables, results/grids and results/figures.\n'

# Auxiliary check. It downloads benchmark series (BTC-USD, ETH-USD, KRBN) from
# Yahoo Finance at run time, so it needs network access and can be rate
# limited. The article does not report its output, and a failure here leaves
# every published result intact.
printf '\n=== 12_cross_market_correlation.py (optional, needs network access)\n'
if MPLBACKEND=Agg "$PY" 12_cross_market_correlation.py; then
  printf '[done] 12_cross_market_correlation.py\n'
else
  printf '[skipped] 12_cross_market_correlation.py could not download the benchmark series.\n'
  printf '          This is an auxiliary check and no published result depends on it.\n'
fi
