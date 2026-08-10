"""
Consistency Reruns for R2 Text Claims
=====================================

Regenerates, from the CURRENT pipeline (same engines/params as scripts 03 and 06),
the numbers quoted in the manuscript text that previously came from stale runs:

  (A) Proposition 3 reconciliation: E[sum D_t], predicted vs simulated VaR shift
      (endogeneity engine of 06_endogeneity_analysis.py, seed 42, n_paths=2000).
  (B) H6a-H6b volatility-neutrality: Spearman correlations of E[IL_NET] and VaR95
      with sigma over the Bridge-grid sigma range, plus TOST equivalence tests
      (bounds +/- 0.05, Fisher-z).
  (C) Bridge variance decomposition: OLS on grid-cell medians of VaR95 over
      (sigma, tau_s) at zero buffer + permutation importance; oracle-noise share
      bounded via a sigma_eps sweep at baseline (sigma=0.35, tau=1.5).

Outputs: results/tables/prop3_reconciliation.csv, h6_tost.csv,
         variance_decomposition.csv  (relative to github/ root)
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "../results/tables")
os.makedirs(OUT, exist_ok=True)

SEED = 42

# ===========================================================================
# (A) Proposition 3 reconciliation (verbatim engine from 06_endogeneity_analysis)
# ===========================================================================
rng = np.random.default_rng(SEED)

def simulate_endogenous_crash(n_days=90, sigma_base=0.87, p_fraud_high=0.02,
                              p_detect=0.4, elasticity_beta=3.0, n_paths=1000):
    dt = 1 / 252
    static_losses, dynamic_losses, sumD = [], [], []
    for i in range(n_paths):
        z = rng.normal(0, 1, n_days)
        is_fraud = rng.random(n_days) < p_fraud_high
        fraud_size = rng.lognormal(-3, 1.0, n_days)
        fraud_size = np.clip(fraud_size, 0.001, 0.05) * is_fraud
        is_detected = rng.random(n_days) < p_detect
        detected_amount = fraud_size * is_detected

        log_price = np.zeros(n_days)
        log_price[0] = np.log(10.0)
        for t in range(1, n_days):
            diffusion = (0 - 0.5 * sigma_base ** 2) * dt + sigma_base * np.sqrt(dt) * z[t]
            endo_shock = -elasticity_beta * detected_amount[t]
            log_price[t] = log_price[t - 1] + diffusion + endo_shock
        prices = np.exp(log_price)
        no_shock_log_price = log_price - np.cumsum(-elasticity_beta * detected_amount)
        prices_static = np.exp(no_shock_log_price)

        fraud_ratio = np.sum(fraud_size)
        val_static = (1 - fraud_ratio) * prices_static[-1]
        val_dynamic = (1 - fraud_ratio) * prices[-1]
        static_losses.append(val_static / 10.0 - 1)
        dynamic_losses.append(val_dynamic / 10.0 - 1)
        sumD.append(detected_amount[1:].sum())  # shocks enter from t=1

    return np.array(static_losses), np.array(dynamic_losses), np.array(sumD)


def prop3_reconciliation():
    beta = 3.0
    st, dy, sD = simulate_endogenous_crash(elasticity_beta=beta, n_paths=2000)
    var_st = np.percentile(st, 5)
    var_dy = np.percentile(dy, 5)
    E_sD = sD.mean()
    # log-quantile shift (simulated)
    log_shift_sim = np.log(1 + var_st) - np.log(1 + var_dy)
    # first-order predictions
    log_shift_pred = beta * E_sD
    ret_shift_pred = (1 + var_st) * (1 - np.exp(-beta * E_sD))
    # conditional mean of sum D in the dynamic left tail
    tail = dy <= var_dy
    E_sD_tail = sD[tail].mean()
    ret_shift_pred_tail = (1 + var_st) * (1 - np.exp(-beta * E_sD_tail))

    df = pd.DataFrame([{
        "beta": beta,
        "VaR95_static": var_st,
        "VaR95_dynamic": var_dy,
        "shift_pp": (var_st - var_dy) * 100,
        "rel_amplification_pct": (var_st - var_dy) / abs(var_st) * 100,
        "E_sumD": E_sD,
        "E_sumD_tail5": E_sD_tail,
        "log_shift_sim_pp": log_shift_sim * 100,
        "log_shift_pred_pp": log_shift_pred * 100,
        "ret_shift_pred_uncond_pp": ret_shift_pred * 100,
        "ret_shift_pred_tailcond_pp": ret_shift_pred_tail * 100,
    }])
    df.to_csv(os.path.join(OUT, "prop3_reconciliation.csv"), index=False)
    print("\n[A] Proposition 3 reconciliation")
    print(df.T.to_string(header=False))
    return df


# ===========================================================================
# (B) H6 volatility neutrality with TOST (engines from 03_full_recalibration)
# ===========================================================================

def simulate_price(days, sigma, seed):
    lr = np.random.default_rng(seed)
    dt = 1 / 252
    diff = lr.normal((0 - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), size=days)
    return np.exp(np.cumsum(diff))  # S0=1


def simulate_native_paths(sigma, N_paths=3000, days=250, p_fraud_low=0.002,
                          p_fraud_high=0.06, p_high_regime=0.10, p_detect=0.4,
                          detect_delay=7, alpha=0.95, seed_offset=0):
    out_il, out_var = np.empty(N_paths), np.empty(N_paths)
    for i in range(N_paths):
        seed = SEED + seed_offset + i
        P_t = simulate_price(days, sigma, seed)
        lr = np.random.default_rng(seed)
        IL_net = np.zeros(days)
        for t in range(days):
            p_fraud = p_fraud_high if lr.random() < p_high_regime else p_fraud_low
            if lr.random() < p_fraud:
                magnitude = min(lr.lognormal(mean=np.log(0.02), sigma=0.6), 0.05)
                if lr.random() < p_detect:
                    end = t + int(lr.exponential(detect_delay))
                else:
                    end = days
                IL_net[t:min(end, days)] += magnitude
        loss_usd = IL_net * P_t
        s = np.sort(loss_usd)
        idx = int(alpha * len(s)) - 1
        out_il[i] = IL_net.mean()
        out_var[i] = s[max(0, idx)]
    return out_il, out_var


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return np.corrcoef(ra, rb)[0, 1]


def tost_rho(r, n, bound=0.05):
    """TOST for Spearman rho within (-bound, bound) via Fisher z."""
    se = 1.0 / np.sqrt(n - 3)
    z_lo = (np.arctanh(r) - np.arctanh(-bound)) / se   # H0: rho <= -bound
    z_hi = (np.arctanh(r) - np.arctanh(bound)) / se    # H0: rho >= +bound
    from math import erf, sqrt
    p_lo = 1 - 0.5 * (1 + erf(z_lo / sqrt(2)))         # P(Z > z_lo)
    p_hi = 0.5 * (1 + erf(z_hi / sqrt(2)))             # P(Z < z_hi)
    return max(p_lo, p_hi)


def h6_tost():
    sigmas = [0.35, 0.50, 0.65, 0.87]
    rows, il_all, var_all, sig_all = [], [], [], []
    for k, s in enumerate(sigmas):
        il, var = simulate_native_paths(s, seed_offset=100000 * k)
        il_all.append(il); var_all.append(var)
        sig_all.append(np.full_like(il, s))
        rows.append({"sigma": s, "E_IL_NET_median": np.median(il),
                     "VaR95_median": np.median(var)})
        print(f"  sigma={s:.2f}: median E[IL_NET]={np.median(il):.4f}, "
              f"median VaR95={np.median(var):.4f}")
    il_all = np.concatenate(il_all); var_all = np.concatenate(var_all)
    sig_all = np.concatenate(sig_all)
    n = len(sig_all)
    r_il = spearman(sig_all, il_all)
    r_var = spearman(sig_all, var_all)
    p_il = tost_rho(r_il, n)
    p_var = tost_rho(r_var, n)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "h6_sigma_sweep.csv"), index=False)
    res = pd.DataFrame([
        {"metric": "E_IL_NET", "spearman_r": r_il, "n": n, "tost_bound": 0.05, "tost_p": p_il},
        {"metric": "VaR95_USD", "spearman_r": r_var, "n": n, "tost_bound": 0.05, "tost_p": p_var},
    ])
    res.to_csv(os.path.join(OUT, "h6_tost.csv"), index=False)
    print("\n[B] H6 TOST")
    print(res.to_string(index=False))
    return res


# ===========================================================================
# (C) Bridge variance decomposition on grid medians + oracle-noise bound
# ===========================================================================

def simulate_bridge_paths(sigma, delay_mean, sigma_oracle=0.02, buffer=0.0,
                          N_paths=3000, days=250, alpha=0.95):
    out_var = np.empty(N_paths)
    tgrid = np.arange(days)
    for i in range(N_paths):
        seed = SEED + i
        P_t = simulate_price(days, sigma, seed)
        V_t = (1 + buffer) * P_t
        lr = np.random.default_rng(seed)
        tau = lr.poisson(lam=delay_mean, size=days)
        eps = lr.normal(0, sigma_oracle, size=days)
        lag_idx = np.maximum(0, tgrid - tau)
        T_t = P_t[lag_idx] * (1 + eps)
        loss = np.maximum(0, T_t - V_t)
        s = np.sort(loss)
        idx = int(alpha * len(s)) - 1
        out_var[i] = s[max(0, idx)]
    return out_var


def variance_decomposition():
    sigmas = [0.35, 0.50, 0.65, 0.87]
    delays = [1.0, 1.5, 2.5, 3.5]
    rows = []
    for s in sigmas:
        for d in delays:
            v = np.median(simulate_bridge_paths(s, d))
            rows.append({"sigma": s, "tau": d, "VaR95_median": v})
            print(f"  sigma={s:.2f}, tau={d:.1f}: median VaR95={v:.4f}")
    g = pd.DataFrame(rows)

    X = np.column_stack([np.ones(len(g)), g.sigma, g.tau])
    y = g.VaR95_median.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

    # permutation importance (drop in R2, averaged over 200 shuffles)
    rng_p = np.random.default_rng(0)
    drops = {}
    for j, name in [(1, "sigma"), (2, "tau")]:
        d_list = []
        for _ in range(200):
            Xp = X.copy()
            Xp[:, j] = rng_p.permutation(Xp[:, j])
            bp, *_ = np.linalg.lstsq(Xp, y, rcond=None)
            pp = Xp @ bp
            r2p = 1 - np.sum((y - pp) ** 2) / np.sum((y - y.mean()) ** 2)
            d_list.append(max(0.0, r2 - r2p))
        drops[name] = np.mean(d_list)
    tot = sum(drops.values())
    shares = {k: v / tot for k, v in drops.items()}

    # oracle-noise bound: sigma_eps sweep at baseline (sigma=0.35, tau=1.5, c=0)
    eps_rows = []
    for se in [0.02, 0.05, 0.10, 0.20]:
        v = np.median(simulate_bridge_paths(0.35, 1.5, sigma_oracle=se))
        eps_rows.append({"sigma_eps": se, "VaR95_median": v})
        print(f"  sigma_eps={se:.2f}: median VaR95={v:.4f}")
    e = pd.DataFrame(eps_rows)
    base = e.VaR95_median.iloc[0]
    eps_rel_change = (e.VaR95_median.iloc[1] - base) / base  # 0.02 -> 0.05

    res = pd.DataFrame([{
        "R2_linear_fit_cells": r2,
        "share_sigma": shares["sigma"],
        "share_tau": shares["tau"],
        "VaR_change_eps_002_to_005_pct": eps_rel_change * 100,
        "VaR_eps_002": base,
        "VaR_eps_005": e.VaR95_median.iloc[1],
        "VaR_eps_010": e.VaR95_median.iloc[2],
        "VaR_eps_020": e.VaR95_median.iloc[3],
    }])
    g.to_csv(os.path.join(OUT, "variance_decomposition_grid.csv"), index=False)
    e.to_csv(os.path.join(OUT, "oracle_noise_sweep.csv"), index=False)
    res.to_csv(os.path.join(OUT, "variance_decomposition.csv"), index=False)
    print("\n[C] Variance decomposition")
    print(res.T.to_string(header=False))
    return res


if __name__ == "__main__":
    prop3_reconciliation()
    print("\nRunning H6 sweep (4 sigmas x 3000 paths)...")
    h6_tost()
    print("\nRunning Bridge variance grid (16 cells x 3000 paths)...")
    variance_decomposition()
    print("\nDone.")
