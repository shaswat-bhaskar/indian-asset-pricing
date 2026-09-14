"""
Factor Strategy Backtesting and Transaction Cost Engine.
Simulates long-short factor portfolios and equal-weighted multi-factor combinations,
accounting for turnover friction across 0, 25, 50, and 100 bps transaction cost tiers.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.data_loader import load_master_dataset


# Empirical monthly turnover estimates for factor rebalancing in India
# Size & Value rebalance annually -> ~3% per month
# Momentum rebalances monthly -> ~15% per month (30% two-way)
FACTOR_MONTHLY_TURNOVER = {
    "MF": 0.02,     # Index/Market low turnover ~2% / mo
    "SMB": 0.05,    # Size ~5% / mo
    "HML": 0.05,    # Value ~5% / mo
    "WML": 0.18,    # Momentum high monthly turnover ~18% / mo
    "MultiFactor_3F": 0.08,  # Equal-weight MF + HML + WML
    "MultiFactor_4F": 0.075, # Equal-weight MF + SMB + HML + WML
}


def calculate_portfolio_metrics(
    returns_series: pd.Series,
    annual_rf: float = 0.0,
    annualization_factor: int = config.ANNUALIZATION_FACTOR
) -> dict:
    """
    Calculate CAGR, Annualized Vol, Sharpe, Sortino, Max Drawdown, Calmar, and Hit Rate.
    Input returns in decimal (e.g. 0.01 for 1%).
    """
    r = returns_series.dropna()
    if len(r) == 0:
        return {}
        
    n_months = len(r)
    n_years = n_months / annualization_factor
    
    # Cumulative wealth index
    wealth_index = (1.0 + r).cumprod()
    final_wealth = wealth_index.iloc[-1]
    
    # CAGR
    cagr = (final_wealth ** (1.0 / n_years)) - 1.0 if final_wealth > 0 else -1.0
    
    # Volatility
    monthly_vol = r.std(ddof=1)
    ann_vol = monthly_vol * np.sqrt(annualization_factor)
    
    # Sharpe Ratio (using zero cash benchmark for pure zero-cost long-short spread)
    ann_mean = r.mean() * annualization_factor
    sharpe = (ann_mean - annual_rf) / ann_vol if ann_vol > 0 else np.nan
    
    # Downside Volatility and Sortino
    downside_returns = r[r < 0]
    downside_vol = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(annualization_factor) if len(downside_returns) > 0 else 1e-6
    sortino = (ann_mean - annual_rf) / downside_vol if downside_vol > 0 else np.nan
    
    # Drawdowns
    running_max = wealth_index.cummax()
    drawdowns = (wealth_index - running_max) / running_max
    max_dd = drawdowns.min()  # Negative value
    
    # Calmar Ratio
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else np.nan
    
    # Hit Rate (% positive months)
    hit_rate = (r > 0).mean()
    
    return {
        "CAGR (%)": round(cagr * 100, 2),
        "Annualized Vol (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Calmar Ratio": round(calmar, 2),
        "Monthly Hit Rate (%)": round(hit_rate * 100, 1),
        "Final Wealth ($1 base)": round(final_wealth, 2),
    }


def run_factor_backtests(
    df: pd.DataFrame = None,
    cost_bps: int = 0
) -> pd.DataFrame:
    """
    Run backtest for all individual factors and multi-factor combinations under a specific transaction cost.
    Cost is specified in basis points (e.g. 25 bps = 0.0025 per unit of turnover).
    """
    if df is None:
        df = load_master_dataset()
        
    cost_rate = cost_bps / 10000.0
    
    # Convert % returns to decimal
    ret_df = pd.DataFrame(index=df.index)
    for f in config.FACTOR_COLS:
        turnover = FACTOR_MONTHLY_TURNOVER.get(f, 0.05)
        # Net monthly return = Gross return - (Turnover * Cost)
        ret_df[f] = (df[f] / 100.0) - (turnover * cost_rate)
        
    # Multi-Factor combinations (Equal-Weighted)
    turnover_3f = FACTOR_MONTHLY_TURNOVER["MultiFactor_3F"]
    ret_df["MultiFactor_3F (MF+HML+WML)"] = (
        (df["MF"] + df["HML"] + df["WML"]) / (3.0 * 100.0)
    ) - (turnover_3f * cost_rate)
    
    turnover_4f = FACTOR_MONTHLY_TURNOVER["MultiFactor_4F"]
    ret_df["MultiFactor_4F (All 4)"] = (
        (df["MF"] + df["SMB"] + df["HML"] + df["WML"]) / (4.0 * 100.0)
    ) - (turnover_4f * cost_rate)
    
    records = []
    for strategy in ret_df.columns:
        metrics = calculate_portfolio_metrics(ret_df[strategy])
        metrics["Strategy"] = strategy
        metrics["Cost (bps)"] = cost_bps
        records.append(metrics)
        
    res_df = pd.DataFrame(records)
    cols = ["Strategy", "Cost (bps)", "CAGR (%)", "Annualized Vol (%)", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)", "Calmar Ratio", "Monthly Hit Rate (%)", "Final Wealth ($1 base)"]
    res_df = res_df[cols]
    return res_df, ret_df


def run_transaction_cost_sensitivity(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Evaluate strategy performance across 0, 25, 50, and 100 bps transaction cost tiers.
    """
    if df is None:
        df = load_master_dataset()
        
    all_tiers = []
    for bps in config.TRANSACTION_COST_BPS:
        res_tier, _ = run_factor_backtests(df, cost_bps=bps)
        all_tiers.append(res_tier)
        
    master_backtest_df = pd.concat(all_tiers, ignore_index=True)
    
    # Save table
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    master_backtest_df.to_csv(config.TABLES_DIR / "transaction_cost_sensitivity.csv", index=False)
    
    with open(config.TABLES_DIR / "transaction_cost_sensitivity.md", "w") as f:
        f.write("# Strategy Performance and Transaction Cost Sensitivity (1993-11 to 2025-12)\n\n")
        f.write(master_backtest_df.to_markdown(index=False))
        f.write("\n\n*Turnover assumptions: Market 2%/mo, Size 5%/mo, Value 5%/mo, Momentum 18%/mo, Multi-Factor 8%/mo.*\n")
        
    return master_backtest_df


if __name__ == "__main__":
    bt_res, _ = run_factor_backtests(cost_bps=0)
    print("Gross Backtest Results (0 bps):")
    print(bt_res.to_string(index=False))
    
    sens = run_transaction_cost_sensitivity()
    print("\nTransaction Cost Sensitivity Summary (Momentum & Multi-Factor at 0 vs 100 bps):")
    print(sens[sens["Strategy"].isin(["WML", "MultiFactor_3F (MF+HML+WML)"])].to_string(index=False))
