#!/usr/bin/env python3
"""
13_architecture_dominance.py

Empirical counterpart of Proposition 4 (Architecture Dominance Threshold), evaluated
at the tokenized-carbon volatility (sigma = 0.87, tau_s = 1.5) as a function of the
trust elasticity beta.

The Bridge settlement-mismatch tail L_B = z_{0.95} * sigma * sqrt(tau_s / N_yr) is fixed
by the market (Proposition 1) and does NOT depend on beta. The confidence-amplified
Native integrity tail L_N = (1 + beta) * q^IL (Proposition 3) grows with the platform's
confidence fragility beta, where q^IL is the 0.95-quantile of net integrity loss at the
baseline Native configuration (p_high = 0.06, pi = 0.10, p_detect = 0.4).

The safer architecture is therefore regime-dependent: a well-safeguarded Native design
(low beta -- circuit breakers, insurance pools) carries the smaller architecture-induced
tail, while a fragile one (high beta) is overtaken by the Bridge design. The crossover
locates the beta at which the two tails are equal.

Output:
  - results/tables/architecture_dominance.csv
  - results/tables/tab_architecture_dominance_R2.tex
"""
import os
import numpy as np
import pandas as pd

SEED = 42
Z95 = 1.645          # z_{0.95}
N_YR = 252           # trading days per year
SIGMA = 0.87         # tokenized-carbon annualized volatility
TAU_S = 1.5          # baseline settlement delay (days)
BETAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
os.makedirs(TAB, exist_ok=True)


def native_qIL(N=2000, p_high=0.06, pi=0.10, p_low=0.002, p_detect=0.4, dd=7, days=250):
    """0.95-quantile of pooled net integrity loss at the baseline Native configuration."""
    pool = []
    for i in range(N):
        rng = np.random.default_rng(SEED + i)
        IL = np.zeros(days)
        events = []
        for t in range(days):
            p_fraud = p_high if rng.random() < pi else p_low
            if rng.random() < p_fraud:
                magnitude = min(rng.lognormal(np.log(0.02), 0.6), 0.05)  # truncated to [0, 0.05]
                detect_day = t + int(rng.exponential(dd)) if rng.random() < p_detect else None
                events.append((t, magnitude, detect_day))
            for (start, loss, dday) in events:
                if t >= start and (dday is None or t < dday):
                    IL[t] += loss
        pool.append(IL)
    return float(np.percentile(np.concatenate(pool), 95))


# --- Fixed Bridge settlement tail at the carbon volatility (Proposition 1) ---
L_B = Z95 * SIGMA * np.sqrt(TAU_S / N_YR)

# --- Native integrity quantile (Proposition 3 input) ---
q_IL = native_qIL()

rows = []
for beta in BETAS:
    L_N = (1.0 + beta) * q_IL
    rows.append({
        "beta": beta,
        "Bridge_L_B": L_B,
        "Native_L_N": L_N,
        "ratio_B_over_N": L_B / L_N,
        "safer": "Native" if L_N < L_B else "Bridge",
    })

beta_star = L_B / q_IL - 1.0   # crossover beta (L_B = L_N)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(TAB, "architecture_dominance.csv"), index=False)

# --- LaTeX table ---
caption = (
    r"\caption{Approximate common-basis comparison at $\sigma=0.87$, "
    r"$\tau_s=1.5$, $c=0$. Both columns are leading-order daily loss proxies "
    r"normalized by $S_0P_0$ for a representative unhedged platform. The crossover "
    r"near $\beta\approx " + f"{beta_star:.1f}" + r"$ is scenario- and model-dependent; "
    r"it is not a ranking of complete operating architectures.}"
)
lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\footnotesize",
    caption,
    r"\label{tab:architecture_dominance}",
    r"\begin{tabular}{ccccc}",
    r"\hline",
    r"$\beta$ & Bridge-type $\mathcal{L}_B$ & Native $\mathcal{L}_N$ & Ratio B/N & Smaller proxy \\",
    r"\hline",
]
for r in rows:
    lines.append(
        f"{r['beta']:.1f} & {r['Bridge_L_B']:.3f} & {r['Native_L_N']:.3f} & "
        f"{r['ratio_B_over_N']:.2f} & {r['safer']} \\\\"
    )
lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]

with open(os.path.join(TAB, "tab_architecture_dominance_R2.tex"), "w") as f:
    f.write("\n".join(lines))

print(f"Bridge L_B (fixed, carbon)      = {L_B:.3f}")
print(f"Native q^IL (baseline)          = {q_IL:.3f}")
print(f"Crossover beta (L_B = L_N)      = {beta_star:.2f}")
print("\nbeta  L_B    L_N    ratioB/N  safer")
for r in rows:
    print(f"{r['beta']:>4} {r['Bridge_L_B']:.3f}  {r['Native_L_N']:.3f}  "
          f"{r['ratio_B_over_N']:>6.2f}   {r['safer']}")
print(f"\nSaved: {TAB}/tab_architecture_dominance_R2.tex")
