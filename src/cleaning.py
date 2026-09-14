"""
Data validation and cleaning audit module.
Generates comprehensive missingness and sample-continuity tables.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.data_loader import load_master_dataset


def generate_data_quality_report() -> pd.DataFrame:
    """
    Produce missingness summary by variable across the full sample (386 months).
    """
    df = load_master_dataset()
    
    records = []
    for col in df.columns:
        total_obs = len(df)
        valid_obs = df[col].notna().sum()
        missing_obs = df[col].isna().sum()
        pct_missing = (missing_obs / total_obs) * 100
        first_valid = df[col].first_valid_index()
        last_valid = df[col].last_valid_index()
        
        # Categorize variable
        if col in ["MF", "SMB", "HML", "WML", "RF"]:
            category = "Factor / Benchmark"
        elif col in ["SV", "SN", "SG", "BV", "BN", "BG"]:
            category = "Size-Value Test Portfolio"
        else:
            category = "Size-Momentum Test Portfolio"
            
        records.append({
            "Variable": col,
            "Category": category,
            "Total Months": total_obs,
            "Valid Months": valid_obs,
            "Missing Months": missing_obs,
            "Missingness (%)": round(pct_missing, 2),
            "First Valid Month": str(first_valid),
            "Last Valid Month": str(last_valid),
        })
        
    report_df = pd.DataFrame(records)
    
    # Save table
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(config.TABLES_DIR / "data_quality_report.csv", index=False)
    
    # Also save Markdown version
    with open(config.TABLES_DIR / "data_quality_report.md", "w") as f:
        f.write("# Data Quality and Missingness Audit (1993-11 to 2025-12)\n\n")
        f.write(report_df.to_markdown(index=False))
        f.write("\n\n**Note:** Missing values in BV (Big-Value) are due to IIMA's minimum-5-stocks diversification constraint.\n")
        
    return report_df


def generate_subsample_breakdown() -> pd.DataFrame:
    """
    Compute observation counts and missingness across In-Sample vs Out-of-Sample periods.
    """
    df = load_master_dataset()
    
    in_sample = df.loc[config.IN_SAMPLE_START:config.IN_SAMPLE_END]
    oos = df.loc[config.OUT_OF_SAMPLE_START:config.OUT_OF_SAMPLE_END]
    
    breakdown = []
    for col in df.columns:
        breakdown.append({
            "Variable": col,
            "Full Sample Valid (N=386)": df[col].notna().sum(),
            "In-Sample Valid (N=266)": in_sample[col].notna().sum(),
            "In-Sample Missing (%)": round((in_sample[col].isna().sum() / len(in_sample)) * 100, 1),
            "OOS Valid (N=120)": oos[col].notna().sum(),
            "OOS Missing (%)": round((oos[col].isna().sum() / len(oos)) * 100, 1),
        })
        
    breakdown_df = pd.DataFrame(breakdown)
    breakdown_df.to_csv(config.TABLES_DIR / "subsample_missingness_breakdown.csv", index=False)
    
    with open(config.TABLES_DIR / "subsample_missingness_breakdown.md", "w") as f:
        f.write("# In-Sample vs Out-of-Sample Missingness Breakdown\n\n")
        f.write(breakdown_df.to_markdown(index=False))
        
    return breakdown_df


if __name__ == "__main__":
    rep = generate_data_quality_report()
    print("Data Quality Report:")
    print(rep.to_string(index=False))
    
    sub = generate_subsample_breakdown()
    print("\nSubsample Breakdown:")
    print(sub.to_string(index=False))
