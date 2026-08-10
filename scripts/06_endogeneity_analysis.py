"""
Price Endogeneity and Feedback Loop Analysis
=============================================

Models the confidence-crisis feedback loop in which major verification failures trigger price collapses.

This script models the "Confidence Crisis" feedback loop:
1. Native System: Fraud events accumulate (Integrity Loss).
2. Detection Event: A portion of fraud is revealed.
3. Price Impact: Revealed fraud triggers a price jump (crash).
   - Magnitude of crash depends on "Elasticity of Trust" (beta).
   - Price(t) = Price(t-) * exp(- beta * Detected_Fraud_Size)

This creates a correlation between Integrity Failures and Market Crashes,
amplifying Tail Risk beyond the static model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
MANUSCRIPT_FIG = os.path.join(RESULTS_DIR, "figures")
os.makedirs(MANUSCRIPT_FIG, exist_ok=True)

# Settings
SEED = 42
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
})

def simulate_endogenous_crash(
    n_days=90,         # Crisis period ~3 months (realistic)
    sigma_base=0.87,   # Empirical volatility
    p_fraud_high=0.02, # ~5 major events/year (realistic)
    p_detect=0.4,
    elasticity_beta=3.0, # Price sensitivity: 1% fraud -> 3% drop (Verra-calibrated)
    n_paths=1000
):
    dt = 1/252
    
    # Storage for comparison
    static_losses = []  # Loss without price impact
    dynamic_losses = [] # Loss WITH price impact
    
    price_paths = []
    fraud_paths = []
    best_detected = -1.0  # track the most illustrative path (most detected fraud)

    for i in range(n_paths):
        # 1. Base Market Dynamics (GBM)
        # Generate standard shocks
        z = rng.normal(0, 1, n_days)
        
        # 2. Fraud Dynamics (Native)
        # Regime switching (simplified: always high risk for stress test)
        is_fraud = rng.random(n_days) < p_fraud_high
        fraud_size = rng.lognormal(-3, 1.0, n_days) # varying sizes
        fraud_size = np.clip(fraud_size, 0.001, 0.05) * is_fraud
        
        # Detection
        is_detected = rng.random(n_days) < p_detect
        detected_amount = fraud_size * is_detected
        
        # 3. Dynamic Price Path
        log_price = np.zeros(n_days)
        log_price[0] = np.log(10.0)
        
        cumulative_impact = 0.0
        
        for t in range(1, n_days):
            # Standard diffusion
            diffusion = (0 - 0.5*sigma_base**2)*dt + sigma_base*np.sqrt(dt)*z[t]
            
            # Endogenous Shock: Detection triggers crash
            # Shock ~ -beta * Detected_Amount
            endo_shock = -elasticity_beta * detected_amount[t]
            
            log_price[t] = log_price[t-1] + diffusion + endo_shock
            cumulative_impact += endo_shock
            
        prices = np.exp(log_price)
        
        # Calculate Terminal Loss
        # Static: Fraud * Original_Price_Trend (without shock)
        # Dynamic: Fraud * Crashed_Price
        
        # Reconstruct "No-Feedback" Price for comparison
        no_shock_log_price = log_price - np.cumsum(-elasticity_beta * detected_amount)
        prices_static = np.exp(no_shock_log_price)
        
        # Monetary Loss = Undetected_Fraud_Remaining * Price
        # (Simplified: assume we hold the bag on all fraud generated)
        total_fraud_vol = np.sum(fraud_size)
        
        loss_static = total_fraud_vol * prices_static[-1]
        
        # For dynamic loss, the holder suffers from:
        # 1. The fraud itself (dilution/invalidity)
        # 2. The price crash caused by the fraud
        # Total Value Destruction = Initial_Value - Final_Value_Adjusted
        
        # Better metric: Portfolio Value Drop
        # Portfolio = 100 tokens. 
        # Static Scenario: Ends with 100 tokens worth P_static. But X% are fake.
        #                  Real Value = 100 * (1 - Fraud%) * P_static
        # Dynamic Scenario: Ends with 100 tokens worth P_dynamic. But X% are fake.
        #                   Real Value = 100 * (1 - Fraud%) * P_dynamic
        
        fraud_ratio = np.sum(fraud_size) # Cumulative fraud as % of supply approx
        
        val_static = (1 - fraud_ratio) * prices_static[-1]
        val_dynamic = (1 - fraud_ratio) * prices[-1]
        
        # Normalize by initial price (10.0)
        ret_static = val_static / 10.0 - 1
        ret_dynamic = val_dynamic / 10.0 - 1
        
        static_losses.append(ret_static)
        dynamic_losses.append(ret_dynamic)
        
        # Save the most illustrative path (most detected fraud) so the panel
        # actually shows the static-vs-endogenous divergence and the fraud events.
        if detected_amount.sum() > best_detected:
            best_detected = detected_amount.sum()
            price_paths = pd.DataFrame({
                'Static Price': prices_static,
                'Dynamic Price': prices,
                'Detected Fraud': detected_amount
            })
            
    return np.array(static_losses), np.array(dynamic_losses), price_paths


def generate_endogeneity_figure():
    print("Simulating Endogenous Price Feedback...")
    
    # Run simulation
    # Beta=3 means: 1% fraud detection -> 3% price drop (Verra-calibrated)
    static_ret, dyn_ret, path_df = simulate_endogenous_crash(elasticity_beta=3.0, n_paths=2000)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Single Path Dynamics
    ax = axes[0]
    ax2 = ax.twinx()
    
    days = np.arange(len(path_df))
    l1 = ax.plot(days, path_df['Static Price'], 'b--', alpha=0.5, label='Exogenous Price (Static)')
    l2 = ax.plot(days, path_df['Dynamic Price'], 'r-', linewidth=1.5, label='Endogenous Price (Feedback)')
    
    # Plot detection events as bars
    l3 = ax2.bar(days, path_df['Detected Fraud']*100, color='red', alpha=0.3, label='Detected Fraud (%)', width=2.0)
    
    ax.set_xlabel('Days')
    ax.set_ylabel('Token Price (USD)')
    ax2.set_ylabel('Detected Fraud Magnitude (%)')
    # Title removed - moved to caption
    
    # Legend
    lines = l1 + l2 + [l3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.grid(alpha=0.3)
    
    # (b) Loss Distribution Comparison
    ax = axes[1]
    
    sns.kdeplot(static_ret, ax=ax, color='blue', fill=True, alpha=0.2, label='Static Model')
    sns.kdeplot(dyn_ret, ax=ax, color='red', fill=True, alpha=0.2, label='Dynamic Feedback')
    
    # VaR lines
    var95_static = np.percentile(static_ret, 5)
    var95_dyn = np.percentile(dyn_ret, 5)
    
    ax.axvline(var95_static, color='blue', linestyle='--')
    ax.axvline(var95_dyn, color='red', linestyle='--')
    
    # Stagger labels vertically to avoid overlap
    ax.text(var95_static + 0.05, ax.get_ylim()[1]*0.2, f'Static VaR: {var95_static:.1%}', 
            rotation=90, color='blue', ha='left', fontsize=9)
    ax.text(var95_dyn - 0.05, ax.get_ylim()[1]*0.5, f'Dynamic VaR: {var95_dyn:.1%}', 
            rotation=90, color='red', ha='right', fontsize=9)
    
    ax.set_xlabel('Portfolio Return')
    ax.set_ylabel('Density')
    # Title removed - moved to caption
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MANUSCRIPT_FIG, "Fig_endogeneity_impact_R2.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Endogeneity figure saved")
    print("\n--- RESULTS ---")
    print(f"Static VaR (5%):  {var95_static:.1%}")
    print(f"Dynamic VaR (5%): {var95_dyn:.1%}")
    print(f"Risk Amplification: {(var95_dyn/var95_static - 1)*100:.1f}%")

if __name__ == "__main__":
    generate_endogeneity_figure()
