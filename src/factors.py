"""
Factor analysis and descriptive statistics module.
Computes return moments, correlation structures, and rolling analytics.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.data_loader import load_master_dataset


def compute_factor_descriptive_stats(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for all factors.
    Includes mean, std, annualized figures, Sharpe, skew, kurtosis, autocorrelations, and t-stats.
    """
    if df is None:
        df = load_master_dataset()
        
    factor_cols = config.FACTOR_COLS + [config.RISK_FREE_COL]
    stats_list = []
    
    for col in factor_cols:
        series = df[col].dropna()
        n = len(series)
        mean_m = series.mean()
        std_m = series.std(ddof=1)
        
        # Annualized values
        mean_ann = mean_m * 12
        std_ann = std_m * np.sqrt(12)
        sharpe_ann = (mean_ann / std_ann) if std_ann > 0 else np.nan
        
        # Higher moments
        skew = stats.skew(series)
        kurt = stats.kurtosis(series)  # Fisher excess kurtosis
        
        min_val = series.min()
        max_val = series.max()
        
        # Autocorrelations
        ac1 = series.autocorr(lag=1)
        ac12 = series.autocorr(lag=12)
        
        # T-test for mean != 0
        t_stat, p_val = stats.ttest_1samp(series, 0.0)
        
        stats_list.append({
            "Factor": col,
            "N": n,
            "Monthly Mean (%)": round(mean_m, 2),
            "Monthly Std (%)": round(std_m, 2),
            "Annualized Mean (%)": round(mean_ann, 2),
            "Annualized Std (%)": round(std_ann, 2),
            "Annualized Sharpe": round(sharpe_ann, 2),
            "t-stat (Mean)": round(t_stat, 2),
            "p-value": round(p_val, 4),
            "Skewness": round(skew, 2),
            "Excess Kurtosis": round(kurt, 2),
            "Min (%)": round(min_val, 2),
            "Max (%)": round(max_val, 2),
            "Autocorr Lag 1": round(ac1, 2),
            "Autocorr Lag 12": round(ac12, 2),
        })
        
    stats_df = pd.DataFrame(stats_list)
    
    # Save tables
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(config.TABLES_DIR / "factor_descriptive_stats_full_sample.csv", index=False)
    
    with open(config.TABLES_DIR / "factor_descriptive_stats_full_sample.md", "w") as f:
        f.write("# Factor Descriptive Statistics (Full Sample: 1993-11 to 2025-12)\n\n")
        f.write(stats_df.to_markdown(index=False))
        
    return stats_df


def compute_factor_correlations(df: pd.DataFrame = None) -> dict:
    """
    Compute pairwise correlation matrices for factors across Full Sample, In-Sample, and Out-of-Sample.
    """
    if df is None:
        df = load_master_dataset()
        
    factor_cols = config.FACTOR_COLS
    
    full_corr = df[factor_cols].corr()
    in_sample_corr = df.loc[config.IN_SAMPLE_START:config.IN_SAMPLE_END, factor_cols].corr()
    oos_corr = df.loc[config.OUT_OF_SAMPLE_START:config.OUT_OF_SAMPLE_END, factor_cols].corr()
    
    # Save correlation tables
    full_corr.to_csv(config.TABLES_DIR / "factor_correlation_full_sample.csv")
    in_sample_corr.to_csv(config.TABLES_DIR / "factor_correlation_in_sample.csv")
    oos_corr.to_csv(config.TABLES_DIR / "factor_correlation_oos.csv")
    
    with open(config.TABLES_DIR / "factor_correlations.md", "w") as f:
        f.write("# Factor Pairwise Correlations\n\n")
        f.write("### Full Sample (1993-11 to 2025-12)\n\n")
        f.write(full_corr.round(3).to_markdown())
        f.write("\n\n### In-Sample (1993-11 to 2015-12)\n\n")
        f.write(in_sample_corr.round(3).to_markdown())
        f.write("\n\n### Out-of-Sample (2016-01 to 2025-12)\n\n")
        f.write(oos_corr.round(3).to_markdown())
        
    return {
        "full": full_corr,
        "in_sample": in_sample_corr,
        "oos": oos_corr,
    }


def compute_rolling_factor_stats(df: pd.DataFrame = None, window: int = 36) -> pd.DataFrame:
    """
    Compute rolling 36-month annualized returns and Sharpe ratios for each factor.
    """
    if df is None:
        df = load_master_dataset()
        
    factor_cols = config.FACTOR_COLS
    rolling_df = pd.DataFrame(index=df.index)
    
    for col in factor_cols:
        r_mean = df[col].rolling(window=window).mean() * 12
        r_std = df[col].rolling(window=window).std() * np.sqrt(12)
        r_sharpe = r_mean / r_std
        
        rolling_df[f"{col}_rolling_ret"] = r_mean
        rolling_df[f"{col}_rolling_vol"] = r_std
        rolling_df[f"{col}_rolling_sharpe"] = r_sharpe
        
    return rolling_df


if __name__ == "__main__":
    st = compute_factor_descriptive_stats()
    print("Factor Descriptive Statistics:")
    print(st.to_string(index=False))
    
    corrs = compute_factor_correlations()
    print("\nFull Sample Factor Correlations:")
    print(corrs["full"].round(3))
