#!/usr/bin/env python3
"""Monte Carlo convergence and independent-seed stability assessment.

This script deliberately reuses the production simulation functions in
``03_full_recalibration.py``.  It therefore tests the same price process,
path-level seeding, Poisson settlement delays, oracle noise, fraud-event
magnitudes, detection draws, and removal delays as the headline analysis.
For each independent base seed, the smaller samples are nested prefixes of
the N=10,000 run.  This isolates the effect of adding paths without silently
changing the data-generating process.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_SIZES = (1_000, 3_000, 5_000, 10_000)
# Non-overlapping blocks because the production engine uses base_seed + path.
BASE_SEEDS = tuple(42 + 100_000 * k for k in range(10))
MAX_N = max(SAMPLE_SIZES)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TAB_DIR = REPO_ROOT / "results" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)


def load_production_model():
    """Load the exact headline-model functions without running its pipeline."""
    module_path = HERE / "03_full_recalibration.py"
    spec = importlib.util.spec_from_file_location("full_recalibration", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load production model from {module_path}")
    module = importlib.util.module_from_spec(spec)
    previous_cwd = Path.cwd()
    try:
        # The production module defines output directories relative to scripts/.
        os.chdir(HERE)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


def summarize_across_seeds(values: list[float]) -> tuple[float, float, float, float]:
    """Return mean, sample SD, CV, and relative range across base seeds."""
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    cv = float(100.0 * std / mean) if mean else np.nan
    relative_range = float(100.0 * (array.max() - array.min()) / mean) if mean else np.nan
    return mean, std, cv, relative_range


def main() -> None:
    model = load_production_model()
    by_n = {n: {"Bridge": [], "Native": []} for n in SAMPLE_SIZES}

    for base_seed in BASE_SEEDS:
        model.SEED = base_seed
        bridge = model.simulate_bridge_empirical(
            N_paths=MAX_N,
            days=250,
            buffer=0.05,
            delay_mean=1.5,
            sigma_oracle=0.02,
            sigma=0.87,
            use_jumps=False,
        )
        native = model.simulate_native_empirical(
            N_paths=MAX_N,
            days=250,
            p_fraud_low=0.002,
            p_fraud_high=0.06,
            p_high_regime=0.10,
            p_detect=0.40,
            detect_delay=7,
            sigma=0.87,
            use_jumps=False,
        )

        for n in SAMPLE_SIZES:
            by_n[n]["Bridge"].append(float(bridge.iloc[:n]["VaR95"].median()))
            by_n[n]["Native"].append(float(native.iloc[:n]["VaR95"].median()))

    rows = []
    for n in SAMPLE_SIZES:
        b_mean, b_sd, b_cv, b_range = summarize_across_seeds(by_n[n]["Bridge"])
        n_mean, n_sd, n_cv, n_range = summarize_across_seeds(by_n[n]["Native"])
        rows.append(
            {
                "Sample_Size_N": n,
                "Bridge_Mean_Median_VaR95": b_mean,
                "Bridge_Seed_SD": b_sd,
                "Bridge_CV_Percent": b_cv,
                "Bridge_Relative_Range_Percent": b_range,
                "Native_Mean_Median_VaR95": n_mean,
                "Native_Seed_SD": n_sd,
                "Native_CV_Percent": n_cv,
                "Native_Relative_Range_Percent": n_range,
            }
        )

    results = pd.DataFrame(rows)
    output = TAB_DIR / "convergence_stability.csv"
    results.to_csv(output, index=False)
    print("Monte Carlo convergence using the exact production DGP:")
    print(results.to_string(index=False))
    print(f"\nIndependent base seeds: {BASE_SEEDS}")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
