"""
Empirical Analysis of BCT Historical Data
==========================================
Real data from CoinMarketCap (Aug 2021 - Jan 2026)
1540 daily observations

This script performs:
1. Descriptive statistics
2. Volatility estimation
3. Jump detection and calibration
4. Crisis period analysis (Jan 2023)
5. Comparison with our event-based estimates
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = "../results"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR = os.path.join(OUTPUT_DIR, "tables")
DATA_DIR = "../data/raw"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def load_coinmarketcap_data():
    """Load BCT data from CoinMarketCap CSV."""
    filepath = os.path.join(
        DATA_DIR, 
        "bct_coinmarketcap_2021-10-21_2026-01-07.csv"
    )
    
    # CSV uses semicolon delimiter
    df = pd.read_csv(filepath, delimiter=';')
    
    # Parse dates
    df['date'] = pd.to_datetime(df['timeClose']).dt.date
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract price columns
    df['price_usd'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # Sort chronologically
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate returns
    df['log_return'] = np.log(df['price_usd'] / df['price_usd'].shift(1))
    df['pct_return'] = df['price_usd'].pct_change()
    
    # Rolling volatility
    df['volatility_30d'] = df['log_return'].rolling(30).std() * np.sqrt(252)
    df['volatility_90d'] = df['log_return'].rolling(90).std() * np.sqrt(252)
    
    return df[['date', 'open', 'high', 'low', 'price_usd', 'volume', 
               'log_return', 'pct_return', 'volatility_30d', 'volatility_90d']]


def descriptive_statistics(df):
    """Calculate descriptive statistics."""
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS: BCT (Base Carbon Tonne)")
    print("=" * 70)
    
    print(f"\nData Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Observations: {len(df)} days")
    
    print("\nPrice Statistics:")
    print(f"  Mean:   ${df['price_usd'].mean():.4f}")
    print(f"  Median: ${df['price_usd'].median():.4f}")
    print(f"  Std:    ${df['price_usd'].std():.4f}")
    print(f"  Min:    ${df['price_usd'].min():.4f}")
    print(f"  Max:    ${df['price_usd'].max():.4f}")
    print(f"  ATH:    ${df['price_usd'].max():.2f} ({df.loc[df['price_usd'].idxmax(), 'date'].date()})")
    print(f"  ATL:    ${df['price_usd'].min():.4f} ({df.loc[df['price_usd'].idxmin(), 'date'].date()})")
    
    # Returns statistics
    returns = df['log_return'].dropna()
    print("\nLog Return Statistics:")
    print(f"  Mean (daily):  {returns.mean():.4f} ({returns.mean() * 252:.2%} annual)")
    print(f"  Std (daily):   {returns.std():.4f} ({returns.std() * np.sqrt(252):.2%} annual)")
    print(f"  Skewness:      {returns.skew():.4f}")
    print(f"  Kurtosis:      {returns.kurtosis():.4f} (excess)")
    print(f"  Min return:    {returns.min():.4f} ({returns.min() * 100:.1f}%)")
    print(f"  Max return:    {returns.max():.4f} ({returns.max() * 100:.1f}%)")
    
    stats = {
        "n_observations": len(df),
        "date_start": str(df['date'].min().date()),
        "date_end": str(df['date'].max().date()),
        "price_mean": df['price_usd'].mean(),
        "price_median": df['price_usd'].median(),
        "price_std": df['price_usd'].std(),
        "price_min": df['price_usd'].min(),
        "price_max": df['price_usd'].max(),
        "return_mean_daily": returns.mean(),
        "return_std_daily": returns.std(),
        "return_mean_annual": returns.mean() * 252,
        "return_std_annual": returns.std() * np.sqrt(252),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }
    
    return stats


def detect_jumps(df, threshold_sigma=3):
    """
    Detect jump events using threshold method.
    Jump = |return| > threshold * sigma
    """
    returns = df['log_return'].dropna()
    sigma = returns.std()
    threshold = threshold_sigma * sigma
    
    jumps = df[df['log_return'].abs() > threshold].copy()
    
    print("\n" + "=" * 70)
    print(f"JUMP DETECTION (Threshold: {threshold_sigma}σ = {threshold:.4f})")
    print("=" * 70)
    
    print(f"\nTotal jumps detected: {len(jumps)}")
    n_days = len(df)
    print(f"Jump frequency: {len(jumps) / n_days:.4f} per day ({len(jumps) / n_days * 252:.1f} per year)")
    
    if len(jumps) > 0:
        print(f"\nJump statistics:")
        print(f"  Mean jump size: {jumps['log_return'].mean():.4f}")
        print(f"  Median jump:    {jumps['log_return'].median():.4f}")
        print(f"  Std of jumps:   {jumps['log_return'].std():.4f}")
        
        # Negative vs positive jumps
        neg_jumps = jumps[jumps['log_return'] < 0]
        pos_jumps = jumps[jumps['log_return'] > 0]
        
        print(f"\n  Negative jumps: {len(neg_jumps)} (avg: {neg_jumps['log_return'].mean():.4f})")
        print(f"  Positive jumps: {len(pos_jumps)} (avg: {pos_jumps['log_return'].mean():.4f})")
        
        print("\nTop 10 largest drops:")
        top_drops = jumps.nsmallest(10, 'log_return')[['date', 'log_return', 'price_usd']]
        for _, row in top_drops.iterrows():
            print(f"  {row['date'].date()}: {row['log_return']:.4f} ({row['log_return']*100:.1f}%) | Price: ${row['price_usd']:.4f}")
    
    jump_params = {
        "n_jumps": len(jumps),
        "jump_frequency_daily": len(jumps) / n_days,
        "jump_frequency_annual": len(jumps) / n_days * 252,
        "jump_mean": jumps['log_return'].mean() if len(jumps) > 0 else 0,
        "jump_std": jumps['log_return'].std() if len(jumps) > 0 else 0,
        "threshold_sigma": threshold_sigma,
        "threshold_value": threshold,
    }
    
    return jumps, jump_params


def analyze_crisis_period(df):
    """
    Analyze Verra crisis period (Jan 2023).
    """
    print("\n" + "=" * 70)
    print("VERRA CRISIS ANALYSIS (January 2023)")
    print("=" * 70)
    
    # Define periods
    pre_crisis = df[(df['date'] >= '2023-01-01') & (df['date'] < '2023-01-17')]
    crisis = df[(df['date'] >= '2023-01-17') & (df['date'] <= '2023-01-31')]
    post_crisis = df[(df['date'] >= '2023-02-01') & (df['date'] <= '2023-02-28')]
    
    # Year 2023 full
    year_2023 = df[(df['date'] >= '2023-01-01') & (df['date'] < '2024-01-01')]
    
    if len(pre_crisis) > 0:
        pre_price = pre_crisis['price_usd'].iloc[0]
        print(f"\nPre-crisis (Jan 1-16):")
        print(f"  Starting price: ${pre_price:.4f}")
    else:
        pre_price = None
        print("\n  No pre-crisis data available")
    
    if len(crisis) > 0:
        crisis_max = crisis['price_usd'].max()
        crisis_min = crisis['price_usd'].min()
        crisis_end = crisis['price_usd'].iloc[-1]
        
        print(f"\nCrisis period (Jan 17-31):")
        print(f"  High: ${crisis_max:.4f}")
        print(f"  Low:  ${crisis_min:.4f}")
        print(f"  End:  ${crisis_end:.4f}")
        
        if pre_price:
            depeg = (crisis_min - pre_price) / pre_price
            print(f"\n  Depeg magnitude: {depeg:.1%}")
            print(f"  Log return:      {np.log(crisis_min / pre_price):.4f}")
    
    if len(post_crisis) > 0:
        print(f"\nPost-crisis (Feb 2023):")
        print(f"  Avg price: ${post_crisis['price_usd'].mean():.4f}")
        print(f"  Recovery:  {post_crisis['price_usd'].iloc[-1] / crisis_min - 1:.1%}")
    
    if len(year_2023) > 0:
        print(f"\n2023 Full Year:")
        print(f"  Avg price: ${year_2023['price_usd'].mean():.4f}")
        print(f"  Volatility (30d avg): {year_2023['volatility_30d'].mean():.1%}")
    
    crisis_data = {
        "pre_crisis_price": pre_price,
        "crisis_low": crisis['price_usd'].min() if len(crisis) > 0 else None,
        "crisis_high": crisis['price_usd'].max() if len(crisis) > 0 else None,
        "depeg_pct": depeg if pre_price and len(crisis) > 0 else None,
    }
    
    return crisis_data


def calibrate_parameters(df, jump_params):
    """
    Derive calibrated parameters for simulation.
    """
    print("\n" + "=" * 70)
    print("CALIBRATED PARAMETERS FOR SIMULATION")
    print("=" * 70)
    
    returns = df['log_return'].dropna()
    
    # Volatility (using different methods)
    sigma_simple = returns.std()
    sigma_annual = sigma_simple * np.sqrt(252)
    
    # Parkinson volatility (using high-low)
    parkinson_sum = np.log(df['high'] / df['low']) ** 2
    sigma_parkinson = np.sqrt(parkinson_sum.mean() / (4 * np.log(2))) * np.sqrt(252)
    
    print("\nVolatility Estimates:")
    print(f"  Simple σ (daily):    {sigma_simple:.4f}")
    print(f"  Simple σ (annual):   {sigma_annual:.1%}")
    print(f"  Parkinson σ (annual): {sigma_parkinson:.1%}")
    
    # Jump-diffusion parameters
    lambda_J = jump_params['jump_frequency_annual']
    mu_J = jump_params['jump_mean']
    sigma_J = jump_params['jump_std']
    
    print("\nJump-Diffusion Parameters:")
    print(f"  λ (jump intensity): {lambda_J:.2f} jumps/year")
    print(f"  μ_J (jump mean):    {mu_J:.4f}")
    print(f"  σ_J (jump std):     {sigma_J:.4f}")
    
    # Compare with our event-based estimates
    print("\n" + "-" * 40)
    print("COMPARISON: Event-Based vs Empirical")
    print("-" * 40)
    print(f"{'Parameter':<20} {'Event-Based':>15} {'Empirical':>15}")
    print("-" * 50)
    print(f"{'σ (daily)':<20} {'0.39':>15} {sigma_simple:>15.4f}")
    print(f"{'λ (jumps/year)':<20} {'1.0':>15} {lambda_J:>15.2f}")
    print(f"{'μ_J':<20} {'-1.099':>15} {mu_J:>15.4f}")
    print(f"{'σ_J':<20} {'0.30':>15} {sigma_J:>15.4f}")
    
    calibrated = {
        "sigma_daily": sigma_simple,
        "sigma_annual": sigma_annual,
        "sigma_parkinson": sigma_parkinson,
        "lambda_J": lambda_J,
        "mu_J": mu_J,
        "sigma_J": sigma_J,
        "drift_daily": returns.mean(),
        "drift_annual": returns.mean() * 252,
    }
    
    return calibrated


def generate_figures(df, jumps, calibrated):
    """Generate publication-ready figures."""
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Price series with key events
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['price_usd'], 'b-', linewidth=0.8)
    ax1.axvline(pd.Timestamp('2023-01-17'), color='red', linestyle='--', 
                label='Verra Crisis (Jan 2023)', alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.set_title('BCT Price History (2021-2026)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Returns distribution
    ax2 = axes[0, 1]
    returns = df['log_return'].dropna()
    ax2.hist(returns, bins=100, density=True, alpha=0.7, color='steelblue', label='Actual')
    
    # Overlay normal distribution
    x = np.linspace(returns.min(), returns.max(), 100)
    from scipy.stats import norm
    ax2.plot(x, norm.pdf(x, returns.mean(), returns.std()), 
             'r-', linewidth=2, label='Normal')
    ax2.set_xlabel('Log Return')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Return Distribution (Kurtosis: {returns.kurtosis():.1f})')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Rolling volatility
    ax3 = axes[1, 0]
    ax3.plot(df['date'], df['volatility_30d'] * 100, 'b-', linewidth=0.8, label='30-day')
    ax3.plot(df['date'], df['volatility_90d'] * 100, 'orange', linewidth=1, label='90-day')
    ax3.axvline(pd.Timestamp('2023-01-17'), color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Annualized Volatility (%)')
    ax3.set_title('Rolling Volatility')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Jump events
    ax4 = axes[1, 1]
    ax4.stem(jumps['date'], jumps['log_return'] * 100, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Return (%)')
    ax4.set_title(f'Detected Jumps (|r| > 3σ, n={len(jumps)})')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "bct_empirical_analysis.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"✓ Saved: {fig_path}")
    return fig_path


def main():
    print("=" * 70)
    print("BCT EMPIRICAL ANALYSIS: Real Data Calibration")
    print("Data source: CoinMarketCap (1540 daily observations)")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    df = load_coinmarketcap_data()
    print(f"   Loaded {len(df)} observations")
    
    # Descriptive statistics
    print("\n2. Calculating statistics...")
    stats = descriptive_statistics(df)
    
    # Jump detection
    print("\n3. Detecting jumps...")
    jumps, jump_params = detect_jumps(df, threshold_sigma=3)
    
    # Crisis analysis
    print("\n4. Analyzing Verra crisis...")
    crisis_data = analyze_crisis_period(df)
    
    # Calibrate parameters
    print("\n5. Calibrating parameters...")
    calibrated = calibrate_parameters(df, jump_params)
    
    # Generate figures
    print("\n6. Generating figures...")
    fig_path = generate_figures(df, jumps, calibrated)
    
    # Save results
    print("\n7. Saving results...")
    
    # Save cleaned data
    df.to_csv(os.path.join(DATA_DIR, "bct_cleaned.csv"), index=False)
    print(f"   ✓ {DATA_DIR}/bct_cleaned.csv")
    
    # Save calibrated parameters
    pd.DataFrame([calibrated]).to_csv(
        os.path.join(TAB_DIR, "empirical_calibration.csv"), index=False
    )
    print(f"   ✓ {TAB_DIR}/empirical_calibration.csv")
    
    # Save jump events
    jumps.to_csv(os.path.join(TAB_DIR, "jump_events.csv"), index=False)
    print(f"   ✓ {TAB_DIR}/jump_events.csv")
    
    print("\n" + "=" * 70)
    print("EMPIRICAL CALIBRATION COMPLETE")
    print("=" * 70)
    print("\nKey findings:")
    print(f"  • Data: {len(df)} daily observations ({df['date'].min().date()} to {df['date'].max().date()})")
    print(f"  • Price range: ${df['price_usd'].min():.4f} to ${df['price_usd'].max():.2f}")
    print(f"  • Annual volatility: {calibrated['sigma_annual']:.1%}")
    print(f"  • Jump frequency: {calibrated['lambda_J']:.1f} per year")
    print(f"  • Crisis depeg: {crisis_data['depeg_pct']:.1%}" if crisis_data['depeg_pct'] else "N/A")
    print("\n✓ Ready for simulation re-calibration!")


if __name__ == "__main__":
    main()
