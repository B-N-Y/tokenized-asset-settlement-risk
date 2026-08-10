"""
Data Loader for BCT/NCT Historical Price Data

Use this script to load CSV data from:
1. CoinMarketCap manual export
2. Refinitiv (if available)
3. Any other source

Expected CSV format:
- date (or Date): YYYY-MM-DD or timestamp
- price/close/Open/Close: USD price
- volume (optional): trading volume

After loading, run calibrate_from_real_data() to update model parameters.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = "data"


def load_coinmarketcap_csv(filepath):
    """
    Load BCT/NCT data from CoinMarketCap CSV export.
    
    CoinMarketCap format typically has:
    - timeOpen, timeClose, timeHigh, timeLow
    - open, high, low, close
    - volume, marketCap
    """
    df = pd.read_csv(filepath)
    
    # Detect column names (CMC uses various formats)
    date_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    price_cols = [c for c in df.columns if c.lower() in ['close', 'price', 'open']]
    
    if date_cols:
        df['date'] = pd.to_datetime(df[date_cols[0]])
    
    if price_cols:
        df['price_usd'] = df[price_cols[0]]
    elif 'close' in df.columns:
        df['price_usd'] = df['close']
    
    # Volume
    if 'volume' in df.columns:
        df['volume_24h'] = df['volume']
    
    # Clean and sort
    df = df[['date', 'price_usd']].dropna()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add derived columns
    df['log_return'] = np.log(df['price_usd'] / df['price_usd'].shift(1))
    df['volatility_30d'] = df['log_return'].rolling(30).std() * np.sqrt(252)
    
    return df


def load_refinitiv_csv(filepath):
    """
    Load data from Refinitiv export.
    
    Refinitiv format typically has:
    - Date
    - Close, High, Low, Open
    - Volume
    """
    df = pd.read_csv(filepath)
    
    # Standard column mapping
    col_map = {
        'Date': 'date',
        'Close': 'price_usd',
        'Volume': 'volume_24h',
    }
    
    df = df.rename(columns=col_map)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    df = df[['date', 'price_usd']].dropna()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Derived columns
    df['log_return'] = np.log(df['price_usd'] / df['price_usd'].shift(1))
    df['volatility_30d'] = df['log_return'].rolling(30).std() * np.sqrt(252)
    
    return df


def load_generic_csv(filepath):
    """
    Load from any CSV with flexible column detection.
    """
    df = pd.read_csv(filepath)
    print(f"Columns found: {list(df.columns)}")
    
    # Try to find date column
    for col in df.columns:
        if any(x in col.lower() for x in ['date', 'time', 'timestamp']):
            df['date'] = pd.to_datetime(df[col])
            break
    
    # Try to find price column
    for col in df.columns:
        if any(x in col.lower() for x in ['close', 'price', 'last']):
            df['price_usd'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    if 'date' not in df.columns or 'price_usd' not in df.columns:
        print("ERROR: Could not identify date and price columns")
        print("Please rename columns to 'date' and 'price_usd'")
        return pd.DataFrame()
    
    df = df[['date', 'price_usd']].dropna()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Derived
    df['log_return'] = np.log(df['price_usd'] / df['price_usd'].shift(1))
    df['volatility_30d'] = df['log_return'].rolling(30).std() * np.sqrt(252)
    
    return df


def calibrate_from_real_data(df):
    """
    Calibrate jump-diffusion parameters from actual price data.
    
    Returns dict of calibrated parameters to update 03_full_recalibration.py
    """
    if df.empty:
        print("ERROR: Empty dataframe, cannot calibrate")
        return {}
    
    print("=" * 60)
    print("CALIBRATING FROM REAL DATA")
    print("=" * 60)
    
    # Basic statistics
    print(f"\nData range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Observations: {len(df)}")
    print(f"Price range: ${df['price_usd'].min():.4f} to ${df['price_usd'].max():.2f}")
    
    # Volatility estimation
    daily_vol = df['log_return'].std()
    annual_vol = daily_vol * np.sqrt(252)
    
    print(f"\nVolatility:")
    print(f"  Daily σ: {daily_vol:.4f}")
    print(f"  Annual σ: {annual_vol:.2%}")
    
    # Jump detection (returns > 3 std devs)
    threshold = 3 * daily_vol
    jumps = df[df['log_return'].abs() > threshold]
    n_jumps = len(jumps)
    n_days = len(df)
    
    print(f"\nJumps detected (|return| > 3σ):")
    print(f"  Count: {n_jumps}")
    print(f"  Frequency: {n_jumps / n_days * 252:.1f} per year")
    
    if n_jumps > 0:
        jump_mean = jumps['log_return'].mean()
        jump_std = jumps['log_return'].std()
        print(f"  Mean jump: {jump_mean:.3f}")
        print(f"  Jump std: {jump_std:.3f}")
    else:
        jump_mean = -1.0
        jump_std = 0.3
    
    # Crisis period analysis (Jan 2023)
    crisis_data = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2023-02-28')]
    if not crisis_data.empty:
        crisis_return = np.log(crisis_data['price_usd'].iloc[-1] / crisis_data['price_usd'].iloc[0])
        print(f"\nCrisis period (Jan-Feb 2023):")
        print(f"  Return: {crisis_return:.1%}")
    
    # Calibrated parameters
    params = {
        "sigma": daily_vol,
        "annualized_vol": annual_vol,
        "lambda_J": n_jumps / n_days * 252,  # Jumps per year
        "mu_J": jump_mean if n_jumps > 0 else -1.0,
        "sigma_J": jump_std if n_jumps > 0 else 0.3,
        "n_observations": n_days,
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
    }
    
    print("\n" + "=" * 60)
    print("CALIBRATED PARAMETERS")
    print("=" * 60)
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Save
    output_path = os.path.join(DATA_DIR, "real_data_calibration.csv")
    pd.DataFrame([params]).to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    return params


def main():
    """
    Main function - looks for CSV files in data/ directory.
    """
    print("Looking for CSV files in data/ directory...")
    
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and 'bct' in f.lower()]
    
    if not csv_files:
        print("\nNo BCT CSV files found!")
        print("Please download data and save as:")
        print(f"  {DATA_DIR}/bct_coinmarketcap.csv")
        print(f"  or {DATA_DIR}/bct_refinitiv.csv")
        return
    
    for csv_file in csv_files:
        filepath = os.path.join(DATA_DIR, csv_file)
        print(f"\nLoading: {filepath}")
        
        df = load_generic_csv(filepath)
        
        if not df.empty:
            # Save cleaned version
            clean_path = os.path.join(DATA_DIR, f"bct_cleaned.csv")
            df.to_csv(clean_path, index=False)
            print(f"✓ Cleaned data saved to {clean_path}")
            
            # Calibrate
            params = calibrate_from_real_data(df)
            
            if params:
                print("\n✅ Real data calibration complete!")
                print("Next: Update 03_full_recalibration.py with these parameters")


if __name__ == "__main__":
    main()
