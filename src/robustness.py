"""
Robustness Checks and Subsample Stability Module.
Evaluates pricing model stability across historical sub-eras, crisis windows, and econometric lag specifications.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.data_loader import load_master_dataset
from src.evaluation import compute_model_pricing_error_metrics
from src.grs import compute_grs_test


def run_subsample_robustness_analysis(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compare model pricing errors across historical eras:
    1. Early / Pre-GFC Era (1993-11 to 2008-12, N=182)
    2. Modern / Post-GFC Era (2009-01 to 2025-12, N=204)
    3. Global Financial Crisis (2007-10 to 2009-03, N=18)
    4. COVID-19 Shock (2020-01 to 2020-12, N=12)
    """
    if df is None:
        df = load_master_dataset()
        
    eras = {
        "Early Era (1993-2008)": (config.SUBSAMPLE_EARLY_START, config.SUBSAMPLE_EARLY_END),
        "Modern Era (2009-2025)": (config.SUBSAMPLE_MODERN_START, config.SUBSAMPLE_MODERN_END),
        "In-Sample (1993-2015)": (config.IN_SAMPLE_START, config.IN_SAMPLE_END),
        "Out-of-Sample (2016-2025)": (config.OUT_OF_SAMPLE_START, config.OUT_OF_SAMPLE_END),
    }
    
    models = ["CAPM", "FF3", "FF4"]
    rows = []
    
    for era_name, (start, end) in eras.items():
        for m in models:
            res = compute_model_pricing_error_metrics(
                model_name=m,
                df=df,
                sample_start=start,
                sample_end=end,
                portfolios=config.BALANCED_TEST_PORTFOLIOS,
            )
            res["Era"] = era_name
            res["Date Range"] = f"{start} to {end}"
            rows.append(res)
            
    rob_df = pd.DataFrame(rows)
    cols = ["Era", "Date Range", "Model", "MAA (%/mo)", "MAA Annualized (%)", "RMS Alpha (%/mo)", "Max |Alpha| (%/mo)", "Sig Alphas (p<0.05)", "Mean R2 (%)", "Mean Adj-R2 (%)"]
    rob_df = rob_df[cols]
    
    # Save table
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rob_df.to_csv(config.TABLES_DIR / "robustness_subsample_analysis.csv", index=False)
    
    with open(config.TABLES_DIR / "robustness_subsample_analysis.md", "w") as f:
        f.write("# Subsample and Era Robustness Analysis\n\n")
        f.write(rob_df.to_markdown(index=False))
        f.write("\n\n*Evaluated across 9 balanced test portfolios with Newey-West HAC standard errors.*\n")
        
    return rob_df


def run_hac_lag_sensitivity(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Test sensitivity of alpha significance to Newey-West lag truncation choices: L = 1, 3, 6, 12 lags.
    """
    if df is None:
        df = load_master_dataset()
        
    lags = [1, 3, 6, 12]
    models = ["CAPM", "FF3", "FF4"]
    rows = []
    
    for l in lags:
        for m in models:
            res = compute_model_pricing_error_metrics(
                model_name=m,
                df=df,
                sample_start=config.FULL_SAMPLE_START,
                sample_end=config.FULL_SAMPLE_END,
                portfolios=config.BALANCED_TEST_PORTFOLIOS,
                hac_lags=l,
            )
            res["HAC Lags (Months)"] = l
            rows.append(res)
            
    lag_df = pd.DataFrame(rows)
    cols = ["HAC Lags (Months)", "Model", "MAA (%/mo)", "RMS Alpha (%/mo)", "Sig Alphas (p<0.05)", "Sig Alphas (p<0.10)"]
    lag_df = lag_df[cols]
    
    lag_df.to_csv(config.TABLES_DIR / "robustness_hac_lag_sensitivity.csv", index=False)
    
    with open(config.TABLES_DIR / "robustness_hac_lag_sensitivity.md", "w") as f:
        f.write("# Newey-West HAC Lag Sensitivity Analysis\n\n")
        f.write(lag_df.to_markdown(index=False))
        f.write("\n\n*Tests sensitivity of statistical inference to lag length selection (1, 3, 6, 12 months).*\n")
        
    return lag_df


if __name__ == "__main__":
    r_sub = run_subsample_robustness_analysis()
    print("Subsample Robustness Analysis:")
    print(r_sub.to_string(index=False))
    
    r_lag = run_hac_lag_sensitivity()
    print("\nHAC Lag Sensitivity:")
    print(r_lag.to_string(index=False))
