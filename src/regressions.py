"""
Asset Pricing Time-Series Regression Engine.
Implements OLS regressions for CAPM, Fama-French 3-Factor, and Carhart 4-Factor models
with Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) robust inference.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.data_loader import load_master_dataset
from src.portfolios import get_test_portfolio_excess_returns


def run_time_series_regression(
    y: pd.Series,
    X: pd.DataFrame,
    hac_lags: int = config.HAC_LAGS_MONTHLY
) -> dict:
    """
    Run OLS time-series regression with Newey-West HAC standard errors.
    
    Parameters:
        y: Dependent variable (portfolio excess return)
        X: Independent variables (factor returns)
        hac_lags: Maximum lag for Bartlett kernel in Newey-West estimator
        
    Returns:
        dict containing coefficients, t-stats, p-values, R2, adj-R2, residuals, and diagnostics.
    """
    # Align data and drop missing observations pairwise
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    y_clean = data["y"]
    X_clean = sm.add_constant(data[X.columns])
    
    n_obs = len(y_clean)
    if n_obs < (len(X.columns) + 2):
        raise ValueError(f"Insufficient observations (N={n_obs}) for regression.")
        
    # Fit OLS model with HAC robust covariance
    model = sm.OLS(y_clean, X_clean)
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    
    # Extract coefficients and statistics
    params = results.params
    tvalues = results.tvalues
    pvalues = results.pvalues
    bse = results.bse
    
    residuals = results.resid
    fitted = results.fittedvalues
    
    # Diagnostics
    dw_stat = durbin_watson(residuals)
    lb_test = acorr_ljungbox(residuals, lags=[hac_lags], return_df=True)
    lb_pvalue = lb_test["lb_pvalue"].iloc[0]
    
    try:
        _, bp_pvalue, _, _ = het_breuschpagan(residuals, X_clean)
    except Exception:
        bp_pvalue = np.nan
        
    return {
        "params": params,
        "tvalues": tvalues,
        "pvalues": pvalues,
        "bse": bse,
        "r2": results.rsquared,
        "adj_r2": results.rsquared_adj,
        "resid": residuals,
        "fitted": fitted,
        "n_obs": n_obs,
        "dw_stat": dw_stat,
        "lb_pvalue": lb_pvalue,
        "bp_pvalue": bp_pvalue,
        "resid_std": residuals.std(ddof=len(X.columns)+1),
        "results_obj": results,
    }


def estimate_all_models_for_portfolio(
    portfolio_name: str,
    df: pd.DataFrame = None,
    sample_start: str = config.FULL_SAMPLE_START,
    sample_end: str = config.FULL_SAMPLE_END,
    hac_lags: int = config.HAC_LAGS_MONTHLY
) -> dict:
    """
    Estimate CAPM, FF3, and FF4 models for a given test portfolio over a specified period.
    """
    if df is None:
        df = load_master_dataset()
        
    sample_df = df.loc[sample_start:sample_end]
    excess_ports = get_test_portfolio_excess_returns(sample_df)
    
    y = excess_ports[portfolio_name]
    
    # Factor subsets
    x_capm = sample_df[["MF"]]
    x_ff3 = sample_df[["MF", "SMB", "HML"]]
    x_ff4 = sample_df[["MF", "SMB", "HML", "WML"]]
    
    capm_res = run_time_series_regression(y, x_capm, hac_lags=hac_lags)
    ff3_res = run_time_series_regression(y, x_ff3, hac_lags=hac_lags)
    ff4_res = run_time_series_regression(y, x_ff4, hac_lags=hac_lags)
    
    return {
        "portfolio": portfolio_name,
        "CAPM": capm_res,
        "FF3": ff3_res,
        "FF4": ff4_res,
    }


def build_regression_table(
    model_name: str,
    df: pd.DataFrame = None,
    sample_start: str = config.FULL_SAMPLE_START,
    sample_end: str = config.FULL_SAMPLE_END,
    portfolios: list = config.ALL_TEST_PORTFOLIOS,
    hac_lags: int = config.HAC_LAGS_MONTHLY
) -> pd.DataFrame:
    """
    Construct a clean tabular representation of regression results across all test portfolios.
    Model options: 'CAPM', 'FF3', 'FF4'
    """
    if df is None:
        df = load_master_dataset()
        
    sample_df = df.loc[sample_start:sample_end]
    excess_ports = get_test_portfolio_excess_returns(sample_df)
    
    if model_name == "CAPM":
        factors = ["MF"]
    elif model_name == "FF3":
        factors = ["MF", "SMB", "HML"]
    elif model_name == "FF4":
        factors = ["MF", "SMB", "HML", "WML"]
    else:
        raise ValueError(f"Unknown model {model_name}")
        
    X = sample_df[factors]
    
    rows = []
    for p in portfolios:
        y = excess_ports[p]
        try:
            res = run_time_series_regression(y, X, hac_lags=hac_lags)
            
            row = {
                "Portfolio": p,
                "N": res["n_obs"],
                "Alpha (%/mo)": round(res["params"]["const"], 2),
                "t(Alpha)": round(res["tvalues"]["const"], 2),
                "p(Alpha)": round(res["pvalues"]["const"], 3),
            }
            
            for f in factors:
                row[f"Beta_{f}"] = round(res["params"][f], 2)
                row[f"t({f})"] = round(res["tvalues"][f], 2)
                row[f"p({f})"] = round(res["pvalues"][f], 3)
                
            row["R2 (%)"] = round(res["r2"] * 100, 1)
            row["Adj-R2 (%)"] = round(res["adj_r2"] * 100, 1)
            row["DW Stat"] = round(res["dw_stat"], 2)
            row["LB p-val"] = round(res["lb_pvalue"], 3)
            row["Resid Vol (%)"] = round(res["resid_std"], 2)
            
            rows.append(row)
        except Exception as e:
            print(f"Error fitting {model_name} on {p}: {e}")
            
    table_df = pd.DataFrame(rows)
    return table_df


def generate_and_save_all_regression_tables():
    """
    Generate and persist regression tables for Full Sample, In-Sample, and Out-of-Sample.
    """
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    periods = {
        "full_sample": (config.FULL_SAMPLE_START, config.FULL_SAMPLE_END),
        "in_sample": (config.IN_SAMPLE_START, config.IN_SAMPLE_END),
        "oos": (config.OUT_OF_SAMPLE_START, config.OUT_OF_SAMPLE_END),
    }
    
    models = ["CAPM", "FF3", "FF4"]
    all_results = {}
    
    for p_name, (start, end) in periods.items():
        all_results[p_name] = {}
        for m_name in models:
            table = build_regression_table(
                model_name=m_name,
                sample_start=start,
                sample_end=end,
            )
            all_results[p_name][m_name] = table
            
            # Save CSV
            csv_path = config.TABLES_DIR / f"regression_{m_name.lower()}_{p_name}.csv"
            table.to_csv(csv_path, index=False)
            
            # Save Markdown
            md_path = config.TABLES_DIR / f"regression_{m_name.lower()}_{p_name}.md"
            with open(md_path, "w") as f:
                f.write(f"# {m_name} Time-Series Regressions ({p_name.replace('_', ' ').title()}: {start} to {end})\n\n")
                f.write(table.to_markdown(index=False))
                f.write("\n\n*Note: t-statistics and p-values are computed using Newey-West HAC robust standard errors (Bartlett kernel, 3 lags).*\n")
                
    print("All regression tables successfully generated and saved.")
    return all_results


if __name__ == "__main__":
    res = generate_and_save_all_regression_tables()
    print("\nSample CAPM Full Sample Table:")
    print(res["full_sample"]["CAPM"].to_string(index=False))
