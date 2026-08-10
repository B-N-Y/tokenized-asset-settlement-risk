r"""
Auto-Generate LaTeX Tables from Results CSV Files
==================================================

This script generates all manuscript tables directly from the results/tables
CSV files, ensuring consistency and eliminating manual entry errors.

Outputs:
- manuscript/tables/*.tex files (to be \input{} in manuscript)
"""

import os
import pandas as pd
import numpy as np

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results/tables")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../results/tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_bridge_baselines_table():
    """Generate Table: Bridge Baseline Comparison (0% vs 5% buffer)."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "summary_bridge_baselines.csv"))
        
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Bridge architecture at empirical volatility ($\sigma{=}0.87$, $\tau_s{=}1.5$) under 0 and 5\% collateral buffers. Medians computed over $N{=}3000$ simulation paths. A 5\% buffer roughly halves the tail metrics but does not eliminate under-collateralization; see Section~\ref{sec:optimal-buffer} for the $\sim$12\% buffer required at this volatility.}
\label{tab:bridge_baselines}
\begin{tabular}{l c c c}
\hline
Scenario & VaR$_{95}$ (USD) & ES$_{95}$ (USD) & $\widehat{\mathbb{P}}_{\mathrm{UC}}$ \\
\hline
"""
        for _, row in df.iterrows():
            scenario = row['Scenario'].replace('_', ' ').title().replace('%', r'\%')
            var95 = f"{row['VaR95_USD (median)']:.3f}"
            es95 = f"{row['ES95_USD (median)']:.3f}"
            puc = f"{row['Under-collateralization p (mean)']:.3f}"
            latex += f"{scenario} & {var95} & {es95} & {puc} \\\\\n"

        latex += r"""\hline
\end{tabular}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_bridge_baselines_R2.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_bridge_baselines_R2.tex")
    except Exception as e:
        print(f"✗ Bridge baselines: {e}")


def generate_breach_stats_table():
    """Generate Table: Breach Statistics."""
    try:
        df_nobuf = pd.read_csv(os.path.join(RESULTS_DIR, "bridge_breach_statistics_nobuf.csv"))
        df_5pct = pd.read_csv(os.path.join(RESULTS_DIR, "bridge_breach_statistics_5pct.csv"))
        
        def get_val(df, metric):
            row = df[df['Metric'] == metric]
            return float(row['Value'].values[0]) if len(row) > 0 else 0.0
        
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Breach statistics for the Bridge setup at empirical volatility ($\sigma{=}0.87$, $\tau_s{=}1.5$). Buffering mainly lowers breach frequency; conditional breach depth is similar across buffers.}
\label{tab:bridge_breach_stats}
\begin{tabular}{l c c c c}
\hline
Scenario & Breach Prob. & ES$\,|\,$Breach (USD) & Max Gap (USD) & Median Gap (USD) \\
\hline
"""
        # No buffer
        bp_nobuf = get_val(df_nobuf, 'Under-collateralization p (mean)')
        es_nobuf = get_val(df_nobuf, 'ES|breach (mean USD)')
        max_nobuf = get_val(df_nobuf, 'Max_gap (mean)')
        med_nobuf = get_val(df_nobuf, 'Median_gap (mean)')
        latex += f"No buffer & {bp_nobuf:.4f} & {es_nobuf:.3f} & {max_nobuf:.3f} & {med_nobuf:.3f} \\\\\n"

        # 5% buffer
        bp_5pct = get_val(df_5pct, 'Under-collateralization p (mean)')
        es_5pct = get_val(df_5pct, 'ES|breach (mean USD)')
        max_5pct = get_val(df_5pct, 'Max_gap (mean)')
        med_5pct = get_val(df_5pct, 'Median_gap (mean)')
        latex += f"5\\% buffer & {bp_5pct:.4f} & {es_5pct:.3f} & {max_5pct:.3f} & {med_5pct:.3f} \\\\\n"

        latex += r"""\hline
\end{tabular}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_bridge_breach_stats_R2.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_bridge_breach_stats_R2.tex")
    except Exception as e:
        print(f"✗ Breach stats: {e}")


def generate_robustness_models_table():
    """Generate Table: Robustness to Alternative Price Models."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "robustness_price_models.csv"))
        
        name_map = {"Student-t(5)": r"Student-$t$ ($\nu{=}5$)"}
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Optimal buffer requirements under alternative price models. All models are calibrated to $\sigma = 87\%$ and $\tau = 1.5$ days. The 2.5 pp variation across the tested specifications (range: 11.0\%--13.5\%) summarizes structural sensitivity within this grid; it is not external model validation.}
\label{tab:robustness_models}
\begin{tabular}{l c c}
\hline
Model & Optimal Buffer & Breach Prob \\
\hline
"""
        for _, row in df.iterrows():
            model = name_map.get(row['Model'], row['Model'])
            buffer = f"{row['Optimal_Buffer']*100:.1f}\\%"
            breach = f"{row['Breach_Prob']*100:.1f}\\%"
            latex += f"{model} & {buffer} & {breach} \\\\\n"

        latex += r"""\hline
\end{tabular}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_robustness_models_R2.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_robustness_models_R2.tex")
    except Exception as e:
        print(f"✗ Robustness models: {e}")


def generate_statistical_tests_table():
    """Generate Table: Statistical Hypothesis Tests."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "statistical_tests.csv"))
        
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{l l c c l}
\hline
Test & Method & Statistic & $p$-value & Conclusion \\
\hline
"""
        for _, row in df.iterrows():
            test = row['Test'][:35] + "..." if len(row['Test']) > 35 else row['Test']
            method = row['Method'][:20] + "..." if len(row['Method']) > 20 else row['Method']
            stat = f"{row['Statistic']:.2f}" if not pd.isna(row['Statistic']) else "N/A"
            pval = f"{row['p_value']:.3e}" if row['p_value'] < 0.001 else f"{row['p_value']:.3f}"
            concl = "Reject" if "Reject" in row['Conclusion'] else "Fail to reject"
            latex += f"{test} & {method} & {stat} & {pval} & {concl} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\caption{Summary of formal statistical hypothesis tests. All tests use $\alpha = 0.05$.}
\label{tab:statistical_tests}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_statistical_tests.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_statistical_tests.tex")
    except Exception as e:
        print(f"✗ Statistical tests: {e}")


def generate_crisis_params_table():
    """Generate Table: Verra Crisis Parameters."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "crisis_parameters.csv"))
        
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Market parameters before and during the Verra 2023 crisis. The crisis period exhibits lower average volatility but extreme tail behavior (kurtosis $>30$).}
\label{tab:crisis_params}
\begin{tabular}{lcccc}
\hline
Period & $\sigma_{ann}$ & Max Drawdown & Worst Day & Kurtosis \\
\hline
"""
        for _, row in df.iterrows():
            period = row['period']
            sigma = f"{row['sigma_annual']*100:.1f}\\%"
            mdd = f"{row['max_drawdown']*100:.0f}\\%"
            worst = f"{row['worst_daily_return']*100:.0f}\\%"
            kurt = f"{row['kurtosis']:.1f}"
            latex += f"{period} & {sigma} & {mdd} & {worst} & {kurt} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_crisis_params_R2.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_crisis_params_R2.tex")
    except Exception as e:
        print(f"✗ Crisis params: {e}")


def generate_sensitivity_elasticity_table():
    """Generate Table: Elasticity Sensitivity Analysis."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "sensitivity_elasticity.csv"))
        
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{c c c c}
\hline
$\beta$ & Static VaR$_{95}$ & Dynamic VaR$_{95}$ & Amplification \\
\hline
"""
        for _, row in df.iterrows():
            beta = int(row['Beta'])
            static = f"{row['VaR95_Static']*100:.1f}\\%"
            dynamic = f"{row['VaR95_Dynamic']*100:.1f}\\%"
            amp = f"{row['Risk_Amplification_pct']:.1f}\\%"
            latex += f"{beta} & {static} & {dynamic} & {amp} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\caption{Sensitivity of death spiral dynamics to elasticity parameter $\beta$. The amplification effect is robust for $\beta > 5$.}
\label{tab:sensitivity_elasticity}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_sensitivity_elasticity.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_sensitivity_elasticity.tex")
    except Exception as e:
        print(f"✗ Sensitivity elasticity: {e}")


def generate_empirical_calibration_table():
    """Generate Table: Empirical Calibration Parameters."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "empirical_calibration.csv"))
        
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{l c}
\hline
Parameter & Value \\
\hline
"""
        for _, row in df.iterrows():
            param = row['Parameter'].replace('_', ' ').title()
            val = row['Value']
            if isinstance(val, float):
                if abs(val) > 1:
                    latex += f"{param} & {val:.2f} \\\\\n"
                else:
                    latex += f"{param} & {val:.3f} \\\\\n"
            else:
                latex += f"{param} & {val} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\caption{Empirically calibrated parameters from 1,540 days of BCT/USD data.}
\label{tab:empirical_calibration}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_empirical_calibration.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_empirical_calibration.tex")
    except Exception as e:
        print(f"✗ Empirical calibration: {e}")


def generate_summary_metrics_table():
    """Generate common-basis summary comparison of Bridge-type vs Native models."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "summary_metrics_v3.csv"))
        
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Common-basis daily tail metrics. Losses are fractions of baseline capitalization $S_0P_0$ over the same 250-day path horizon; $P_0=1$.}
\label{tab:summary_metrics}
\begin{tabular}{l c c}
\hline
Model & Normalized VaR$_{95}$ & Normalized ES$_{95}$ \\
\hline
"""
        for _, row in df.iterrows():
            model = row['Model'].replace('Bridge', 'Bridge-type')
            var95 = f"{row['VaR95_USD (median)']:.3f}"
            es95 = f"{row['ES95_USD (median)']:.3f}"
            latex += f"{model} & {var95} & {es95} \\\\\n"

        latex += r"""\hline
\end{tabular}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_summary_metrics_R2.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_summary_metrics_R2.tex")
    except Exception as e:
        print(f"✗ Summary metrics: {e}")


def generate_optimal_buffer_table():
    """Generate Table: Optimal buffer grid across volatility and delay."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "optimal_buffer_grid.csv"))
        
        # Pivot to get grid format
        sigmas = sorted(df['sigma'].unique())
        delays = sorted(df['delay'].unique())
        
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Optimal collateral buffer $c^*$ to achieve $P(\text{breach}) \leq 5\%$ across volatility and settlement delay combinations. At the empirically observed volatility and $\tau = 1.5$ days, the optimal buffer is approximately 12\%.}
\label{tab:optimal_buffer}
\begin{tabular}{l""" + "c" * len(delays) + r"""}
\hline
$\sigma_{ann}$ & """ + " & ".join([f"$\\tau={d:.1f}$d" for d in delays]) + r""" \\
\hline
"""
        for sigma in sigmas:
            row_data = df[df['sigma'] == sigma].sort_values('delay')
            buffers = [f"{row['optimal_buffer']*100:.1f}\\%" for _, row in row_data.iterrows()]
            latex += f"{int(sigma*100)}\\% & " + " & ".join(buffers) + " \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_optimal_buffer_R2.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_optimal_buffer_R2.tex")
    except Exception as e:
        print(f"✗ Optimal buffer: {e}")


def generate_validation_vol_table():
    """Generate Table: Observed volatility across rolling windows."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "validation_price_scalecheck.csv"))
        # Filter to compare_windows rows
        df = df[df['type'] == 'compare_windows']
        
        latex = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Observed annualized volatility of BCT/USD using rolling windows.}
\label{tab:validation_vol}
\begin{tabular}{l c c c}
\hline
Window (days) & $\sigma_{\text{ann}}$ & Winsorized & MAD-based \\
\hline
"""
        for _, row in df.iterrows():
            window = int(row['window_days'])
            sigma = f"{row['sigma_ann']:.3f}"
            winsor = f"{row['sigma_ann_winsor_1_99']:.3f}"
            mad = f"{row['sigma_ann_mad']:.3f}"
            latex += f"{window} & {sigma} & {winsor} & {mad} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_validation_vol_R2.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_validation_vol_R2.tex")
    except Exception as e:
        print(f"✗ Validation vol: {e}")


def generate_analytic_validation_table():
    """Generate Table: cell-by-cell validation of the Bridge buffer law (Prop 1)."""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "optimal_buffer_grid.csv")).sort_values(["sigma", "delay"])
        latex = r"""\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.05}
\caption{Analytical validation of the Bridge buffer law (Proposition~\ref{prop:buffer}). For each volatility--delay cell, the simulated optimal buffer $c^*_{\mathrm{sim}}$ (from Table~\ref{tab:optimal_buffer}, $P(\text{breach})\le 5\%$) is compared with the leading-order analytical prediction $c^*_{\mathrm{pred}}=0.104\,\sigma\sqrt{\tau_s}$ ($\kappa=z_{0.95}/\sqrt{252}$). The implied coefficient $\hat\kappa=c^*_{\mathrm{sim}}/(\sigma\sqrt{\tau_s})$ is near-constant ($0.107$--$0.119$, mean $0.113$) across a $2.9\times$ range in $\sigma$ and a $2.5\times$ range in $\tau_s$, confirming the $\sigma\sqrt{\tau_s}$ scaling. The analytical value $0.104$ slightly underpredicts, with the mild upward drift in $\hat\kappa$ as $\sigma$ rises reflecting the Poisson-distributed settlement delay and the $-\tfrac{1}{2}\sigma^{2}$ It\^{o} drift term omitted from the leading-order pure-diffusion benchmark. The $c^*_{\mathrm{sim}}$ values are reproduced from Table~\ref{tab:optimal_buffer} for direct comparison.}
\label{tab:analytic_validation}
\begin{tabular}{ccccc}
\hline
$\sigma_{\mathrm{ann}}$ & $\tau_s$ (d) & $c^*_{\mathrm{sim}}$ (\%) & $c^*_{\mathrm{pred}}=0.104\,\sigma\sqrt{\tau_s}$ (\%) & $\hat\kappa=c^*_{\mathrm{sim}}/(\sigma\sqrt{\tau_s})$ \\
\hline
"""
        prev = None
        for _, row in df.iterrows():
            sig, tau = row["sigma"], row["delay"]
            if prev is not None and sig != prev:
                latex += "\\hline\n"
            csim = round(row["optimal_buffer"] * 100, 1)   # displayed simulated buffer (%)
            pred = 0.104 * sig * np.sqrt(tau) * 100         # leading-order analytical prediction
            khat = (csim / 100) / (sig * np.sqrt(tau))      # implied kappa from displayed c*_sim
            latex += f"{int(round(sig*100))}\\% & {tau:.1f} & {csim:.1f} & {pred:.1f} & {khat:.3f} \\\\\n"
            prev = sig
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        with open(os.path.join(OUTPUT_DIR, "tab_analytic_validation_R2.tex"), "w") as f:
            f.write(latex)
        print("✓ tab_analytic_validation_R2.tex")
    except Exception as e:
        print(f"✗ Analytic validation: {e}")


def main():
    print("=" * 60)
    print("AUTO-GENERATING LATEX TABLES FROM RESULTS")
    print("=" * 60)
    
    generate_bridge_baselines_table()
    generate_breach_stats_table()
    generate_robustness_models_table()
    generate_statistical_tests_table()
    generate_crisis_params_table()
    generate_sensitivity_elasticity_table()
    generate_empirical_calibration_table()
    generate_summary_metrics_table()
    generate_optimal_buffer_table()
    generate_validation_vol_table()
    generate_analytic_validation_table()
    
    print("\n" + "=" * 60)
    print(f"Tables written to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nTo use in manuscript, add:")
    print(r"  \input{tables/tab_bridge_baselines_R2.tex}")
    print(r"  \input{tables/tab_robustness_models_R2.tex}")
    print("  etc.")


if __name__ == "__main__":
    main()
