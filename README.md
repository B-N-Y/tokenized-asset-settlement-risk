# Carbon Risk Simulation Framework

Monte Carlo stress testing framework for tokenized carbon markets, comparing **Bridge** and **Native** tokenization architectures.

## Repository Structure

```
├── scripts/              # Python analysis pipeline (numbered modules; see Usage for execution order)
│   ├── 01_load_real_data.py             # Load & clean the CoinMarketCap BCT price export
│   ├── 02_empirical_analysis.py         # Empirical volatility, jumps, fat tails, calibration
│   ├── 03_full_recalibration.py         # Full Monte Carlo simulation grids (Bridge & Native)
│   ├── 04_generate_figures.py           # Main result figures
│   ├── 05_historical_validation.py      # Verra 2023 crisis consistency check + optimal buffer
│   ├── 06_endogeneity_analysis.py       # Confidence-crisis (trust elasticity) dynamics
│   ├── 07_robustness_price_models.py    # GBM / Merton / Student-t / GARCH robustness
│   ├── 08_sensitivity_elasticity.py     # Sensitivity to trust elasticity (beta sweep)
│   ├── 09_statistical_tests.py          # Hypothesis tests (Wilcoxon, Mann-Whitney, MC test)
│   ├── 10_generate_latex_tables.py      # LaTeX tables for the manuscript (run after 07 and 14)
│   ├── 11_distributional_diagnostics.py # Goodness-of-fit, ARCH-LM, variance-ratio tests
│   ├── 12_cross_market_correlation.py   # Auxiliary cross-market correlation checks
│   ├── 13_architecture_dominance.py     # Bridge-vs-Native dominance threshold (Prop. 4)
│   ├── 14_regenerate_summary_csvs.py    # Refresh summary CSVs consumed by Script 10
│   ├── 15_consistency_reruns.py         # H6/TOST, variance decomposition, Prop. 3 checks
│   └── 16_convergence_seed_stability.py # Monte Carlo sampling convergence & multi-seed stability (R3)
├── data/raw/             # Price data (CoinMarketCap export + CoinGecko cross-check)
└── results/              # Simulation outputs (regenerated end-to-end by the scripts)
```

## Key Findings

- **Bridge-type model:** a delayed USD mark-to-market settlement guarantee with a leading-order collateral benchmark, `c* ≈ 0.11 σ √τ`; the σ=87%, τ=1.5-day calibration yields about 12%. This is not a claim about conventional in-kind bridges or a regulatory requirement.
- **Native architecture:** integrity-verification risk governed by a residence-time law; "confidence crisis" feedback amplifies VaR₉₅ from −63.9% to −67.6% at trust elasticity β = 3
- **Architecture comparison:** an approximate, model-dependent threshold compares normalized daily loss proxies on the same baseline-capitalization basis; at the stated unmitigated calibration the crossover is near β ≈ 0.9
- **Consistency check:** the realized Verra 2023 crisis (64.5% price collapse) falls at the 3.8th percentile of the model's stress range simulated from pre-crisis information alone

## Requirements

- Python 3.11+
- NumPy 1.26+
- Pandas 2.1+
- Matplotlib 3.8+
- SciPy 1.11+
- Seaborn 0.13+
- scikit-learn 1.4+
- yfinance 0.2+

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# The analysis scripts resolve data/output paths relative to scripts/.
cd scripts

# 1. Load & clean the CoinMarketCap price export (already included in data/raw/)
python 01_load_real_data.py

# 2. Empirical analysis & calibration (writes bct_cleaned.csv, empirical_calibration.csv)
python 02_empirical_analysis.py

# 3. Full simulation grids (Bridge & Native)
python 03_full_recalibration.py

# 4. Historical (Verra 2023) consistency check + optimal buffer derivation
python 05_historical_validation.py

# 5. Endogenous confidence-crisis analysis
python 06_endogeneity_analysis.py

# 6. Robustness across price models (GBM, Merton, Student-t, GARCH)
python 07_robustness_price_models.py

# 7. Trust-elasticity sweep & statistical tests
python 08_sensitivity_elasticity.py
python 09_statistical_tests.py

# 8. Distributional diagnostics & architecture dominance
python 11_distributional_diagnostics.py
python 12_cross_market_correlation.py
python 13_architecture_dominance.py

# 9. Refresh summary CSVs consumed by the LaTeX table generator
python 14_regenerate_summary_csvs.py

# 10. Figures and LaTeX tables (run after the steps above)
python 04_generate_figures.py
python 10_generate_latex_tables.py

# 11. Consistency reruns for text statistics (H6/TOST, variance decomposition, Prop. 3)
python 15_consistency_reruns.py

# 12. Monte Carlo convergence & seed stability assessment across N=1000..10000 and 10 random seeds
python 16_convergence_seed_stability.py
```

Headline simulations use fixed seed 42. The convergence assessment uses ten independent, non-overlapping base-seed blocks listed in `16_convergence_seed_stability.py` and writes them with the results to `results/tables/convergence_stability.csv`. On a 16GB CPU-only workstation, the full pipeline takes approximately two hours.

## Data

Price data for the BCT (Base Carbon Tonne) token — the sole calibration source for the simulations — is a daily OHLC export from [CoinMarketCap](https://coinmarketcap.com/), covering 1,540 consecutive daily observations from October 21, 2021 to January 7, 2026 (`data/raw/bct_coinmarketcap_2021-10-21_2026-01-07.csv`). The processed series is **included** in `data/raw/bct_cleaned.csv`, so the pipeline can be run without re-fetching; `02_empirical_analysis.py` regenerates it from the raw export. An independent CoinGecko series (`data/raw/bct-usd-max.csv`) is included only as a cross-check (median absolute price deviation 1.5%).

> **Note:** The market data is derived from CoinMarketCap and remains subject to [CoinMarketCap's terms of use](https://coinmarketcap.com/terms/). The MIT License below applies to the **code** in this repository, not to the underlying price data.

## License

The code in this repository is released under the [MIT License](LICENSE). The market data under `data/raw/` is derived from CoinMarketCap and is subject to CoinMarketCap's terms of use rather than the MIT License.
