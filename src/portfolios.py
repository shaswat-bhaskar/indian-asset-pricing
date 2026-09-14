"""
Test portfolio excess return construction and validation module.
Calculates excess returns over 91-day T-bill for all 10 characteristic portfolios.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.data_loader import load_master_dataset


def get_test_portfolio_excess_returns(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute excess returns (R_it - R_ft) for all 10 test portfolios:
    Size-Value: SV, SN, SG, BV, BN, BG
    Size-Momentum: WS, WB, LS, LB
    """
    if df is None:
        df = load_master_dataset()
        
    rf = df[config.RISK_FREE_COL]
    excess_df = pd.DataFrame(index=df.index)
    
    for port in config.ALL_TEST_PORTFOLIOS:
        if port in df.columns:
            excess_df[port] = df[port] - rf
            
    # Save processed excess returns
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    excess_df.to_csv(config.PROCESSED_DATA_DIR / "test_portfolios_excess_returns.csv")
    return excess_df


def compute_portfolio_summary_stats(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute descriptive statistics for test portfolio raw returns and excess returns.
    """
    if df is None:
        df = load_master_dataset()
        
    excess_df = get_test_portfolio_excess_returns(df)
    
    stats_list = []
    for port in config.ALL_TEST_PORTFOLIOS:
        raw = df[port].dropna()
        exc = excess_df[port].dropna()
        
        stats_list.append({
            "Portfolio": port,
            "Description": _get_portfolio_description(port),
            "N": len(raw),
            "Raw Mean (%/mo)": round(raw.mean(), 2),
            "Excess Mean (%/mo)": round(exc.mean(), 2),
            "Annualized Excess (%)": round(exc.mean() * 12, 2),
            "Annualized Vol (%)": round(exc.std(ddof=1) * np.sqrt(12), 2),
            "Sharpe Ratio": round((exc.mean() * 12) / (exc.std(ddof=1) * np.sqrt(12)), 2),
            "Skewness": round(raw.skew(), 2),
            "Kurtosis": round(raw.kurtosis(), 2),
            "Min (%)": round(raw.min(), 2),
            "Max (%)": round(raw.max(), 2),
        })
        
    summary_df = pd.DataFrame(stats_list)
    summary_df.to_csv(config.TABLES_DIR / "test_portfolios_summary_stats.csv", index=False)
    
    with open(config.TABLES_DIR / "test_portfolios_summary_stats.md", "w") as f:
        f.write("# Test Portfolios Return Characteristics (1993-11 to 2025-12)\n\n")
        f.write(summary_df.to_markdown(index=False))
        
    return summary_df


def _get_portfolio_description(port: str) -> str:
    mapping = {
        "SV": "Small-Cap / High B/M (Value)",
        "SN": "Small-Cap / Neutral B/M",
        "SG": "Small-Cap / Low B/M (Growth)",
        "BV": "Large-Cap / High B/M (Value)",
        "BN": "Large-Cap / Neutral B/M",
        "BG": "Large-Cap / Low B/M (Growth)",
        "WS": "Small-Cap / Winners (High Momentum)",
        "WB": "Large-Cap / Winners (High Momentum)",
        "LS": "Small-Cap / Losers (Low Momentum)",
        "LB": "Large-Cap / Losers (Low Momentum)",
    }
    return mapping.get(port, port)


if __name__ == "__main__":
    exc = get_test_portfolio_excess_returns()
    print("Excess returns shape:", exc.shape)
    st = compute_portfolio_summary_stats()
    print("\nTest Portfolios Summary:")
    print(st.to_string(index=False))
