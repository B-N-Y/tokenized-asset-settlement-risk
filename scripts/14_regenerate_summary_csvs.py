"""
Regenerate the three summary CSVs that Script 10 consumes, from the CURRENT
engine (Script 03), replacing stale legacy artifacts.

- summary_bridge_baselines.csv   (Tab: bridge baselines)
- bridge_breach_statistics_nobuf.csv / _5pct.csv  (Tab: breach stats)
- summary_metrics_v3.csv         (Tab: summary metrics, Bridge vs Native)

Bridge tail metrics and breach-depth are recomputed by re-running the empirical
Bridge simulation at sigma=0.87, tau_s=1.5 for buffer in {0, 5%}, N=3000, SEED=42
(identical price paths to Script 03). Native tail is read from the authoritative
native grid. This also serves as the engine verification of the breach-depth
numbers reported in the manuscript.
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # Script 03 reads ../results/... with relative paths

# Import the engine (import-safe: main() is guarded)
from importlib import import_module
mod = import_module("03_full_recalibration") if False else None
import importlib.util
spec = importlib.util.spec_from_file_location("recal", os.path.join(SCRIPT_DIR, "03_full_recalibration.py"))
recal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recal)

simulate_price_empirical = recal.simulate_price_empirical
SEED = recal.SEED  # 42

RESULTS = os.path.join(SCRIPT_DIR, "../results")
TAB = os.path.join(RESULTS, "tables")
GRID = os.path.join(RESULTS, "grids")

N_PATHS = 3000
DAYS = 250
SIGMA = 0.87
DELAY_MEAN = 1.5
SIGMA_ORACLE = 0.02


def bridge_run(buffer):
    """Replicate Script 03 simulate_bridge_empirical loop, adding breach-depth."""
    var95s, es95s, pbreaches = [], [], []
    es_cond, max_gap, med_gap = [], [], []
    for i in range(N_PATHS):
        seed = SEED + i
        P_t = simulate_price_empirical(days=DAYS, seed=seed, sigma=SIGMA)  # use_jumps=False
        V_t = (1 + buffer) * P_t
        rng = np.random.default_rng(seed)
        tau = rng.poisson(lam=DELAY_MEAN, size=DAYS)
        eps = rng.normal(0, SIGMA_ORACLE, size=DAYS)
        T_t = np.empty(DAYS)
        for t in range(DAYS):
            t_lag = max(0, t - int(tau[t]))
            T_t[t] = P_t[t_lag] * (1 + eps[t])
        gap = T_t - V_t
        loss = np.maximum(0.0, gap)
        breach = gap > 0
        # tail metrics (as in Script 03)
        sl = np.sort(loss)
        idx = int(0.95 * len(sl)) - 1
        var95s.append(sl[max(0, idx)])
        es95s.append(sl[idx:].mean() if idx < len(sl) else 0.0)
        pbreaches.append(breach.mean())
        # breach-depth
        bl = loss[breach]
        es_cond.append(bl.mean() if bl.size else 0.0)
        max_gap.append(loss.max())
        med_gap.append(np.median(bl) if bl.size else 0.0)
    return dict(
        VaR95_median=np.median(var95s),
        ES95_median=np.median(es95s),
        Puc_mean=np.mean(pbreaches),
        Puc_median=np.median(pbreaches),
        ES_breach_mean=np.mean(es_cond),
        ES_breach_median=np.median(es_cond),
        MaxGap_mean=np.mean(max_gap),
        MaxGap_median=np.median(max_gap),
        MedGap_mean=np.mean(med_gap),
        MedGap_median=np.median(med_gap),
        Mass_at_zero=np.mean(np.array(var95s) == 0.0),
    )


print("Re-running Bridge empirical (sigma=0.87, tau=1.5, N=3000)...")
nb = bridge_run(0.0)
b5 = bridge_run(0.05)

for name, d in [("No buffer", nb), ("5% buffer", b5)]:
    print(f"\n== {name} ==")
    for k, v in d.items():
        print(f"  {k:18s} {v:.4f}")

# ---- Native baseline from authoritative grid ----
ng = pd.read_csv(os.path.join(GRID, "native_empirical_grid.csv"))
nb_row = ng[(ng.p_fraud_high == 0.06) & (ng.p_detect == 0.4) & (ng.p_high_regime == 0.10)].iloc[0]
native_var, native_es = float(nb_row.VaR95_median), float(nb_row.ES95_median)
print(f"\nNative baseline: VaR95={native_var:.4f}  ES95={native_es:.4f}")

# ============================================================
# WRITE CSVs (schemas consumed by Script 10)
# ============================================================
# 1) summary_bridge_baselines.csv
pd.DataFrame([
    {"Scenario": "No buffer", "VaR95_USD (median)": nb["VaR95_median"],
     "ES95_USD (median)": nb["ES95_median"], "Mass_at_zero_VaR": nb["Mass_at_zero"],
     "Under-collateralization p (mean)": nb["Puc_mean"]},
    {"Scenario": "5% buffer", "VaR95_USD (median)": b5["VaR95_median"],
     "ES95_USD (median)": b5["ES95_median"], "Mass_at_zero_VaR": b5["Mass_at_zero"],
     "Under-collateralization p (mean)": b5["Puc_mean"]},
]).to_csv(os.path.join(TAB, "summary_bridge_baselines.csv"), index=False)

# 2) bridge_breach_statistics_nobuf.csv / _5pct.csv  (Metric,Value long format)
def breach_csv(d, path):
    pd.DataFrame([
        {"Metric": "Under-collateralization p (mean)", "Value": d["Puc_mean"]},
        {"Metric": "Under-collateralization p (median)", "Value": d["Puc_median"]},
        {"Metric": "ES|breach (mean USD)", "Value": d["ES_breach_mean"]},
        {"Metric": "ES|breach (median USD)", "Value": d["ES_breach_median"]},
        {"Metric": "Max_gap (mean)", "Value": d["MaxGap_mean"]},
        {"Metric": "Max_gap (median)", "Value": d["MaxGap_median"]},
        {"Metric": "Median_gap (mean)", "Value": d["MedGap_mean"]},
        {"Metric": "Median_gap (median)", "Value": d["MedGap_median"]},
    ]).to_csv(path, index=False)

breach_csv(nb, os.path.join(TAB, "bridge_breach_statistics_nobuf.csv"))
breach_csv(b5, os.path.join(TAB, "bridge_breach_statistics_5pct.csv"))

# 3) summary_metrics_v3.csv
pd.DataFrame([
    {"Model": "Bridge (5\\% buffer)", "VaR95_USD (median)": b5["VaR95_median"],
     "ES95_USD (median)": b5["ES95_median"]},
    {"Model": "Native", "VaR95_USD (median)": native_var,
     "ES95_USD (median)": native_es},
]).to_csv(os.path.join(TAB, "summary_metrics_v3.csv"), index=False)

print("\n✓ Regenerated: summary_bridge_baselines.csv, bridge_breach_statistics_{nobuf,5pct}.csv, summary_metrics_v3.csv")
