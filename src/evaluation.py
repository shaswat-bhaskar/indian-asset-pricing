"""
Pricing Error Analysis and Model Comparison Engine.
Computes Mean Absolute Alpha (MAA), RMS Alpha, Max Absolute Alpha,
and average explanatory power across CAPM, FF3, and Carhart FF4.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.regressions import build_regression_table


def compute_model_pricing_error_metrics(
    model_name: str,
    df: pd.DataFrame = None,
    sample_start: str = config.FULL_SAMPLE_START,
    sample_end: str = config.FULL_SAMPLE_END,
    portfolios: list = config.BALANCED_TEST_PORTFOLIOS,  # 9 balanced portfolios
    hac_lags: int = config.HAC_LAGS_MONTHLY
) -> dict:
    """
    Compute pricing error summary statistics for a given factor model.
    Uses balanced 9-portfolio set (or full 10) to ensure consistent cross-model comparison.
    """
    table = build_regression_table(
        model_name=model_name,
        df=df,
        sample_start=sample_start,
        sample_end=sample_end,
        portfolios=portfolios,
        hac_lags=hac_lags,
    )
    
    alphas = table["Alpha (%/mo)"]
    t_alphas = table["t(Alpha)"]
    p_alphas = table["p(Alpha)"]
    r2s = table["R2 (%)"]
    adj_r2s = table["Adj-R2 (%)"]
    resid_vols = table["Resid Vol (%)"]
    
    n_ports = len(alphas)
    maa = alphas.abs().mean()
    rms_alpha = np.sqrt((alphas ** 2).mean())
    max_abs_alpha = alphas.abs().max()
    
    # Significant alphas at 5% level
    sig_alphas_5pct = (p_alphas < 0.05).sum()
    sig_alphas_10pct = (p_alphas < 0.10).sum()
    
    avg_r2 = r2s.mean()
    avg_adj_r2 = adj_r2s.mean()
    avg_resid_vol = resid_vols.mean()
    
    return {
        "Model": model_name,
        "N_Portfolios": n_ports,
        "MAA (%/mo)": round(maa, 3),
        "MAA Annualized (%)": round(maa * 12, 2),
        "RMS Alpha (%/mo)": round(rms_alpha, 3),
        "RMS Alpha Annualized (%)": round(rms_alpha * 12, 2),
        "Max |Alpha| (%/mo)": round(max_abs_alpha, 3),
        "Max |Alpha| Annualized (%)": round(max_abs_alpha * 12, 2),
        "Sig Alphas (p<0.05)": f"{sig_alphas_5pct}/{n_ports}",
        "Sig Alphas (p<0.10)": f"{sig_alphas_10pct}/{n_ports}",
        "Mean R2 (%)": round(avg_r2, 1),
        "Mean Adj-R2 (%)": round(avg_adj_r2, 1),
        "Mean Resid Vol (%/mo)": round(avg_resid_vol, 2),
    }


def compare_all_models_pricing_errors(
    portfolios: list = config.BALANCED_TEST_PORTFOLIOS,
    sample_name: str = "full_sample"
) -> pd.DataFrame:
    """
    Generate comprehensive cross-model pricing error comparison table: CAPM vs FF3 vs FF4.
    """
    if sample_name == "full_sample":
        start, end = config.FULL_SAMPLE_START, config.FULL_SAMPLE_END
    elif sample_name == "in_sample":
        start, end = config.IN_SAMPLE_START, config.IN_SAMPLE_END
    elif sample_name == "oos":
        start, end = config.OUT_OF_SAMPLE_START, config.OUT_OF_SAMPLE_END
    else:
        raise ValueError(f"Unknown sample name {sample_name}")
        
    models = ["CAPM", "FF3", "FF4"]
    rows = []
    
    for m in models:
        metrics = compute_model_pricing_error_metrics(
            model_name=m,
            sample_start=start,
            sample_end=end,
            portfolios=portfolios,
        )
        rows.append(metrics)
        
    comp_df = pd.DataFrame(rows)
    
    # Save table
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(config.TABLES_DIR / f"pricing_error_comparison_{sample_name}.csv", index=False)
    
    with open(config.TABLES_DIR / f"pricing_error_comparison_{sample_name}.md", "w") as f:
        f.write(f"# Pricing Error Comparison: CAPM vs FF3 vs FF4 ({sample_name.replace('_', ' ').title()})\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n\n*Evaluated across 9 balanced test portfolios (SV, SN, SG, BN, BG, WS, WB, LS, LB).*\n")
        
    return comp_df


def generate_all_pricing_error_reports():
    """
    Generate comparison reports for full sample, in-sample, and out-of-sample periods.
    """
    res = {}
    for s in ["full_sample", "in_sample", "oos"]:
        res[s] = compare_all_models_pricing_errors(sample_name=s)
    return res


if __name__ == "__main__":
    r = generate_all_pricing_error_reports()
    print("Full Sample Pricing Error Comparison:")
    print(r["full_sample"].to_string(index=False))
    print("\nIn-Sample Pricing Error Comparison:")
    print(r["in_sample"].to_string(index=False))
    print("\nOut-of-Sample Pricing Error Comparison:")
    print(r["oos"].to_string(index=False))
