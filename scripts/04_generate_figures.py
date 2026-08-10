"""
Generate Manuscript Figures - Empirical Only
=============================================

All figures use ONLY empirical parameters:
- σ = 87% (annual volatility from 1,540 days BCT data)
- λ = 5.4 jumps/year
- μ_J = 3.7%, σ_J = 27%

No baseline comparisons - clean, single-scenario narrative.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../results")
DATA_DIR = os.path.join(SCRIPT_DIR, "../data/raw")
MANUSCRIPT_FIG = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(MANUSCRIPT_FIG, exist_ok=True)

# Empirical parameters (from BCT analysis)
SIGMA_EMPIRICAL = 0.87
LAMBDA_J = 5.4
MU_J = 0.037
SIGMA_J = 0.27

# Load simulation grids
bridge_grid = pd.read_csv(os.path.join(OUTPUT_DIR, "grids", "bridge_empirical_grid.csv"))
native_grid = pd.read_csv(os.path.join(OUTPUT_DIR, "grids", "native_empirical_grid.csv"))

# Publication settings
plt.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (7, 5),
})


# ===========================================================================
# FIGURE 1: BCT Empirical Data Analysis (4-panel)
# ===========================================================================

def generate_fig1_empirical_analysis():
    """
    Generate 4 separate atomic figures for empirical analysis.
    """
    # Load BCT data
    bct_file = os.path.join(DATA_DIR, "bct_cleaned.csv")
    if not os.path.exists(bct_file):
        bct_file = os.path.join(DATA_DIR, "bct_prices.csv")
    if not os.path.exists(bct_file):
        print("  [SKIP] BCT data file not found")
        return
    
    df = pd.read_csv(bct_file)
    df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['timeOpen'])
    df = df.sort_values('date')
    
    price_col = 'price_usd' if 'price_usd' in df.columns else 'close'
    df['price'] = df[price_col]
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna(subset=['log_return'])
    
    # Calculate rolling volatility
    df['vol_30d'] = df['log_return'].rolling(30).std() * np.sqrt(252)
    df['vol_90d'] = df['log_return'].rolling(90).std() * np.sqrt(252)
    
    # Detect jumps (|return| > 3σ)
    sigma = df['log_return'].std()
    df['is_jump'] = np.abs(df['log_return']) > 3 * sigma

    # (a) Price history
    plt.figure(figsize=(7, 4))
    plt.semilogy(df['date'], df['price'], color='steelblue', linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('BCT/USD (log scale)')
    # Title removed for ACM compliance
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "empirical_price_history_R2.png"), dpi=600)
    plt.close()

    # (b) Return distribution
    plt.figure(figsize=(7, 4))
    returns = df['log_return'].dropna()
    plt.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', label='Observed')
    x = np.linspace(returns.min(), returns.max(), 100)
    _mu, _sd = returns.mean(), returns.std()
    normal_pdf = np.exp(-0.5 * ((x - _mu) / _sd) ** 2) / (_sd * np.sqrt(2 * np.pi))
    plt.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal fit')
    plt.xlabel('Daily Log Return')
    plt.ylabel('Density')
    # Title removed for ACM compliance
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "empirical_return_dist_R2.png"), dpi=600)
    plt.close()

    # (c) Rolling volatility
    plt.figure(figsize=(7, 4))
    plt.plot(df['date'], df['vol_30d'] * 100, label='30-day', color='blue', alpha=0.7)
    plt.plot(df['date'], df['vol_90d'] * 100, label='90-day', color='red', linewidth=2)
    plt.axhline(87, color='darkred', linestyle='--', label='Full-sample σ=87%')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility (%)')
    # Title removed for ACM compliance
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "empirical_rolling_vol_R2.png"), dpi=600)
    plt.close()

    # (d) Jumps
    plt.figure(figsize=(7, 4))
    jumps = df[df['is_jump']]
    plt.scatter(jumps['date'], jumps['log_return'] * 100, 
               c=np.where(jumps['log_return'] > 0, 'green', 'red'),
               s=50, alpha=0.7)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(3 * sigma * 100, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(-3 * sigma * 100, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Log Return (%)')
    # Title removed for ACM compliance
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "empirical_jumps_R2.png"), dpi=600)
    plt.close()
    
    print(f"✓ Fig1: Empirical analysis (4 atomic files)")


# ===========================================================================
# FIGURE 2: Bridge VaR Distribution at σ=87%
# ===========================================================================

def generate_buffer_effectiveness():
    """
    Generate atomic plots for Buffer Effectiveness: VaR, ES, and Breach Probability.
    Replaces old Fig 2 (Bar) and Fig 3 (Sensitivity).
    """
    df = bridge_grid[
        (bridge_grid['sigma_annual'] == SIGMA_EMPIRICAL) & 
        (bridge_grid['delay_mean'] == 1.5)
    ].sort_values('buffer')
    
    # (a) VaR95 Curve
    plt.figure(figsize=(4, 3))
    plt.plot(df['buffer'] * 100, df['VaR95_median'], 'o-', 
            color='darkred', linewidth=2, markersize=8)
    plt.fill_between(df['buffer'] * 100, 0, df['VaR95_median'], alpha=0.2, color='red')
    plt.xlabel('Collateral Buffer (%)')
    plt.ylabel('Median VaR$_{95}$ (USD)')
    # Title removed for ACM compliance
    plt.grid(alpha=0.3)
    plt.axhline(0, color='green', linestyle='--', alpha=0.7, label='Zero Risk')
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "buffer_effectiveness_var_R2.png"), dpi=600)
    plt.close()

    # (b) ES95 Curve
    plt.figure(figsize=(4, 3))
    plt.plot(df['buffer'] * 100, df['ES95_median'], 's-', 
            color='darkblue', linewidth=2, markersize=8)
    plt.fill_between(df['buffer'] * 100, 0, df['ES95_median'], alpha=0.2, color='blue')
    plt.xlabel('Collateral Buffer (%)')
    plt.ylabel('Median ES$_{95}$ (USD)')
    # Title removed for ACM compliance
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "buffer_effectiveness_es_R2.png"), dpi=600)
    plt.close()

    # (c) Breach Probability Curve
    plt.figure(figsize=(4, 3))
    plt.plot(df['buffer'] * 100, df['P_breach_mean'] * 100, 'D-', 
            color='purple', linewidth=2, markersize=8)
    plt.xlabel('Collateral Buffer (%)')
    plt.ylabel('Breach Probability (%)')
    # Title removed for ACM compliance
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "buffer_effectiveness_prob_R2.png"), dpi=600)
    plt.close()

    print(f"✓ Buffer Effectiveness: 3 atomic files (VaR, ES, Prob)")


# ===========================================================================
# FIGURE 4: Delay-Buffer Heatmap
# ===========================================================================

def generate_fig4_heatmap():
    """
    VaR95 heatmap: Settlement Delay vs Buffer.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = bridge_grid[bridge_grid['sigma_annual'] == SIGMA_EMPIRICAL].copy()
    
    pivot = df.pivot_table(
        values='VaR95_median',
        index='delay_mean',
        columns='buffer'
    )
    
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', origin='lower')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in pivot.index])
    
    ax.set_xlabel('Collateral Buffer')
    ax.set_ylabel('Settlement Delay (days)')
    # Title removed for ACM compliance
    
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > 0.03 else 'black'
            ax.text(j, i, f'${val:.2f}', ha='center', va='center',
                    color=color, fontsize=8, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='VaR$_{95}$ (USD)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "heatmap_delay_buffer_R2.png"),
                dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig4: Delay-Buffer heatmap")


# ===========================================================================
# FIGURE 5: Native Model - Detection Efficiency
# ===========================================================================

def generate_fig5_native_detection():
    """
    Native model: Integrity loss vs detection probability.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    df = native_grid[native_grid['p_high_regime'] == 0.10].copy()
    
    # Group by detection probability
    for p_fraud in sorted(df['p_fraud_high'].unique()):
        sub = df[df['p_fraud_high'] == p_fraud].sort_values('p_detect')
        ax.plot(sub['p_detect'], sub['E_IL_NET_mean'] * 100, 
                'o-', linewidth=2, markersize=7,
                label=f'p_fraud={p_fraud:.2f}')
    
    ax.set_xlabel('Detection Probability')
    ax.set_ylabel('Mean Net Integrity Loss (%)')
    # Title removed for ACM compliance
    ax.legend(title='Fraud Intensity')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "Fig5_native_detection_R2.png"),
                dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig5: Native detection efficiency")


# ===========================================================================
# FIGURE 6: Bridge vs Native Comparison
# ===========================================================================

def generate_fig6_comparison():
    """
    Generate separate heatmaps for Bridge (VaR95) and Native (Integrity Loss).
    """
    # (a) Bridge heatmap
    plt.figure(figsize=(7, 5))
    df_bridge = bridge_grid[bridge_grid['buffer'] == 0.05].copy()
    pivot = df_bridge.pivot_table(
        values='VaR95_median',
        index='sigma_annual',
        columns='delay_mean'
    )
    plt.imshow(pivot.values, cmap='Reds', aspect='auto', origin='lower')
    plt.xticks(range(len(pivot.columns)), [f"{c:.1f}" for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{r:.0%}" for r in pivot.index])
    plt.xlabel('Settlement Delay (days)')
    plt.ylabel('Volatility (σ)')
    # Title removed for ACM compliance
    plt.colorbar(label='VaR$_{95}$')
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "bridge_heatmap_comparison_R2.png"), dpi=600)
    plt.close()
    
    # (b) Native heatmap
    plt.figure(figsize=(7, 5))
    df_native = native_grid[native_grid['p_high_regime'] == 0.10].copy()
    pivot2 = df_native.pivot_table(
        values='E_IL_NET_mean',
        index='p_fraud_high',
        columns='p_detect'
    )
    plt.imshow(pivot2.values * 100, cmap='Blues', aspect='auto', origin='lower')
    plt.xticks(range(len(pivot2.columns)), [f"{c:.1f}" for c in pivot2.columns])
    plt.yticks(range(len(pivot2.index)), [f"{r:.2f}" for r in pivot2.index])
    plt.xlabel('Detection Probability')
    plt.ylabel('Fraud Intensity')
    # Title removed for ACM compliance
    plt.colorbar(label='IL$_{NET}$ (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "native_heatmap_comparison_R2.png"), dpi=600)
    plt.close()

    print(f"✓ Fig6: Bridge vs Native heatmaps (2 atomic files)")


def generate_rank_correlations():
    """
    Generate Rank Correlations bar chart (H1-H6).
    Approximated by calculating Spearman corr over the grid results.
    """
    # Calculate Bridge Correlations (VaR95 vs sigma, delay)
    b_corr_sigma = bridge_grid['VaR95_median'].corr(bridge_grid['sigma_annual'], method='spearman')
    b_corr_delay = bridge_grid['VaR95_median'].corr(bridge_grid['delay_mean'], method='spearman')
    
    # Calculate Native Correlations (IL_NET vs p_fraud, p_detect)
    n_corr_fraud = native_grid['E_IL_NET_mean'].corr(native_grid['p_fraud_high'], method='spearman')
    n_corr_detect = native_grid['E_IL_NET_mean'].corr(native_grid['p_detect'], method='spearman')
    
    # Plot
    labels = ['Volatility (σ)', 'Settlement Delay (τ)', 'Fraud Intensity', 'Detection Prob']
    values = [b_corr_sigma, b_corr_delay, n_corr_fraud, n_corr_detect]
    colors = ['darkred', 'darkred', 'darkblue', 'darkblue']
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylim(-1, 1)
    plt.ylabel('Spearman Rank Correlation')
    # Title removed for ACM compliance
    plt.grid(axis='y', alpha=0.3)
    
    # Add values
    for bar, val in zip(bars, values):
        y_pos = val + 0.05 if val > 0 else val - 0.1
        plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:.2f}", 
                 ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "rank_correlations_R2.png"), dpi=600)
    plt.close()
    print("✓ Fig: Rank Correlations Bar Plot")


# ===========================================================================
# TABLE: Key Results Summary
# ===========================================================================

def generate_results_table():
    """
    Summary table of key empirical results.
    """
    # Extract key values
    emp_5pct = bridge_grid[
        (bridge_grid['sigma_annual'] == SIGMA_EMPIRICAL) &
        (bridge_grid['buffer'] == 0.05) &
        (bridge_grid['delay_mean'] == 1.5)
    ].iloc[0]
    
    emp_0pct = bridge_grid[
        (bridge_grid['sigma_annual'] == SIGMA_EMPIRICAL) &
        (bridge_grid['buffer'] == 0.00) &
        (bridge_grid['delay_mean'] == 1.5)
    ].iloc[0]
    
    data = {
        'Metric': ['VaR₉₅ (no buffer)', 'VaR₉₅ (5% buffer)', 
                   'ES₉₅ (no buffer)', 'ES₉₅ (5% buffer)',
                   'P(breach) no buffer', 'P(breach) 5% buffer'],
        'Value': [
            f"${emp_0pct['VaR95_median']:.3f}",
            f"${emp_5pct['VaR95_median']:.3f}",
            f"${emp_0pct['ES95_median']:.3f}",
            f"${emp_5pct['ES95_median']:.3f}",
            f"{emp_0pct['P_breach_mean']:.1%}",
            f"{emp_5pct['P_breach_mean']:.1%}",
        ]
    }
    
    df_table = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    
    table = ax.table(cellText=df_table.values,
                     colLabels=df_table.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df_table.columns)):
        table[(0, i)].set_facecolor('#2F5496')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Title removed for ACM compliance
    
    
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "Table_results_summary.png"),
                dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Table: Results summary")


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    print("=" * 70)
    print("GENERATING MANUSCRIPT FIGURES (EMPIRICAL ONLY)")
    print(f"σ = {int(SIGMA_EMPIRICAL*100)}%, λ = {LAMBDA_J}/year")
    print("=" * 70)
    print(f"Output: {MANUSCRIPT_FIG}\n")
    
    print("1. BCT empirical analysis...")
    generate_fig1_empirical_analysis()
    
    print("\n2. Buffer effectiveness curves...")
    generate_buffer_effectiveness()
    
    print("\n4. Delay-Buffer heatmap...")
    generate_fig4_heatmap()
    
    print("\n5. Native detection efficiency...")
    generate_fig5_native_detection()
    
    print("\n6. Bridge vs Native comparison...")
    generate_fig6_comparison()
    
    print("\n7. Results summary table...")
    generate_results_table()

    print("\n8. Rank Correlations...")
    generate_rank_correlations()
    
    print("\n" + "=" * 70)
    print("COMPLETE: 7 figures generated (empirical-only)")
    print("=" * 70)


if __name__ == "__main__":
    main()
