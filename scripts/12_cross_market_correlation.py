"""
12_cross_market_correlation.py
Cross-market correlation analysis for BCT (Base Carbon Tonne)

Purpose: Quantify the extent to which BCT price dynamics reflect
cryptocurrency market sentiment vs. carbon market fundamentals.

Benchmarks:
  - BTC-USD (Bitcoin): proxy for broad crypto market sentiment
  - ETH-USD (Ethereum): proxy for DeFi/smart-contract ecosystem
  - KRBN (KraneShares Global Carbon Strategy ETF): proxy for
    compliance carbon markets (EUA, CCA futures)

Methodology:
  Since BCT trades 24/7 (crypto) while KRBN trades only on
  NYSE business days, all series are aligned to common business-day
  dates to ensure methodological comparability.

  Robustness is verified via weekly (Friday-to-Friday) returns.

Output:
  results/tables/cross_market_correlation.csv
  results/tables/cross_market_rolling.csv
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BCT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "bct_cleaned.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Load BCT data
# ============================================================
print("Loading BCT data...")
bct = pd.read_csv(BCT_PATH)
bct["date"] = pd.to_datetime(bct["date"])
bct = bct.set_index("date").sort_index()
print(f"  BCT: {bct.index.min().date()} to {bct.index.max().date()}, N={len(bct)}")

# ============================================================
# 2. Download benchmark data from Yahoo Finance
# ============================================================
START = str(bct.index.min().date())
END = str((bct.index.max() + pd.Timedelta(days=1)).date())

benchmarks = {}
for ticker in ["BTC-USD", "ETH-USD", "KRBN"]:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START, end=END, progress=False)
    df.index = df.index.tz_localize(None)
    close = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]
    benchmarks[ticker] = close
    print(f"  {ticker}: {close.index.min().date()} to {close.index.max().date()}, N={len(close)}")

# ============================================================
# 3. Align all series to common business-day dates (KRBN dates)
# ============================================================
print("\nAligning to common business-day dates...")
common_dates = benchmarks["KRBN"].dropna().index

bct_bd = bct["price_usd"].reindex(common_dates).dropna()
btc_bd = benchmarks["BTC-USD"].reindex(common_dates).dropna()
eth_bd = benchmarks["ETH-USD"].reindex(common_dates).dropna()
krbn_bd = benchmarks["KRBN"].reindex(common_dates).dropna()

# Compute log returns
bct_ret = np.log(bct_bd / bct_bd.shift(1)).dropna()
btc_ret = np.log(btc_bd / btc_bd.shift(1)).dropna()
eth_ret = np.log(eth_bd / eth_bd.shift(1)).dropna()
krbn_ret = np.log(krbn_bd / krbn_bd.shift(1)).dropna()

# Merge on common dates
daily = pd.DataFrame({
    "BCT": bct_ret,
    "BTC": btc_ret,
    "ETH": eth_ret,
    "KRBN": krbn_ret
}).dropna()

print(f"  Common business-day observations: {len(daily)}")

# ============================================================
# 4. Compute correlations (daily, business-day aligned)
# ============================================================
print("\n=== DAILY CORRELATIONS (business-day aligned) ===")
results_daily = []
for name in ["BTC", "ETH", "KRBN"]:
    r_pearson, p_pearson = stats.pearsonr(daily["BCT"], daily[name])
    r_spearman, p_spearman = stats.spearmanr(daily["BCT"], daily[name])
    row = {
        "Benchmark": name,
        "Frequency": "Daily (BD-aligned)",
        "N": len(daily),
        "Pearson_r": round(r_pearson, 4),
        "Pearson_p": round(p_pearson, 6),
        "Spearman_rho": round(r_spearman, 4),
        "Spearman_p": round(p_spearman, 6),
    }
    results_daily.append(row)
    print(f"  BCT-{name}: r={r_pearson:.4f} (p={p_pearson:.6f}), "
          f"ρ={r_spearman:.4f} (p={p_spearman:.6f}), N={len(daily)}")

# ============================================================
# 5. Weekly returns (Friday-to-Friday) robustness check
# ============================================================
print("\n=== WEEKLY CORRELATIONS (Friday-to-Friday) ===")
bct_w = bct["price_usd"].resample("W-FRI").last().dropna()
btc_w = benchmarks["BTC-USD"].resample("W-FRI").last().dropna()
eth_w = benchmarks["ETH-USD"].resample("W-FRI").last().dropna()
krbn_w = benchmarks["KRBN"].resample("W-FRI").last().dropna()

weekly = pd.DataFrame({
    "BCT": np.log(bct_w / bct_w.shift(1)),
    "BTC": np.log(btc_w / btc_w.shift(1)),
    "ETH": np.log(eth_w / eth_w.shift(1)),
    "KRBN": np.log(krbn_w / krbn_w.shift(1)),
}).dropna()

results_weekly = []
for name in ["BTC", "ETH", "KRBN"]:
    r_pearson, p_pearson = stats.pearsonr(weekly["BCT"], weekly[name])
    r_spearman, p_spearman = stats.spearmanr(weekly["BCT"], weekly[name])
    row = {
        "Benchmark": name,
        "Frequency": "Weekly (Fri-Fri)",
        "N": len(weekly),
        "Pearson_r": round(r_pearson, 4),
        "Pearson_p": round(p_pearson, 6),
        "Spearman_rho": round(r_spearman, 4),
        "Spearman_p": round(p_spearman, 6),
    }
    results_weekly.append(row)
    print(f"  BCT-{name}: r={r_pearson:.4f} (p={p_pearson:.6f}), "
          f"ρ={r_spearman:.4f} (p={p_spearman:.6f}), N={len(weekly)}")

# ============================================================
# 6. Rolling 90-day correlations (business-day aligned)
# ============================================================
print("\n=== ROLLING 90-DAY CORRELATIONS ===")
rolling_results = []
for name in ["BTC", "ETH", "KRBN"]:
    rc = daily["BCT"].rolling(90).corr(daily[name]).dropna()
    row = {
        "Benchmark": name,
        "Rolling_mean": round(rc.mean(), 4),
        "Rolling_std": round(rc.std(), 4),
        "Rolling_min": round(rc.min(), 4),
        "Rolling_max": round(rc.max(), 4),
    }
    rolling_results.append(row)
    print(f"  BCT-{name}: mean={rc.mean():.4f}, std={rc.std():.4f}, "
          f"min={rc.min():.4f}, max={rc.max():.4f}")

# ============================================================
# 7. Sanity check: BTC-KRBN correlation
# ============================================================
print("\n=== SANITY CHECK: BTC-KRBN ===")
r, p = stats.pearsonr(daily["BTC"], daily["KRBN"])
print(f"  BTC-KRBN (daily BD): r={r:.4f}, p={p:.6f}")
r_w, p_w = stats.pearsonr(weekly["BTC"], weekly["KRBN"])
print(f"  BTC-KRBN (weekly):   r={r_w:.4f}, p={p_w:.6f}")

# ============================================================
# 8. Save results
# ============================================================
df_corr = pd.DataFrame(results_daily + results_weekly)
corr_path = os.path.join(OUTPUT_DIR, "cross_market_correlation.csv")
df_corr.to_csv(corr_path, index=False)
print(f"\nSaved: {corr_path}")

df_roll = pd.DataFrame(rolling_results)
roll_path = os.path.join(OUTPUT_DIR, "cross_market_rolling.csv")
df_roll.to_csv(roll_path, index=False)
print(f"Saved: {roll_path}")

print("\nDone.")
