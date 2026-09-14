"""
Gibbons-Ross-Shanken (1989) Test Engine.
Computes exact finite-sample F-test for the joint hypothesis of zero pricing errors (alpha_1 = ... = alpha_N = 0).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.data_loader import load_master_dataset
from src.portfolios import get_test_portfolio_excess_returns


def compute_grs_test(
    portfolios: list,
    factor_names: list,
    df: pd.DataFrame = None,
    sample_start: str = config.FULL_SAMPLE_START,
    sample_end: str = config.FULL_SAMPLE_END,
) -> dict:
    """
    Compute exact Gibbons-Ross-Shanken (1989) F-statistic, degrees of freedom, and p-value.
    
    Formula:
    GRS = [(T - N - K) / N] * [1 + mu_f' * Sigma_f^(-1) * mu_f]^(-1) * [alpha' * Sigma_eps^(-1) * alpha]
    Distributed as F(N, T - N - K) under H0: alpha = 0.
    """
    if df is None:
        df = load_master_dataset()
        
    sample_df = df.loc[sample_start:sample_end]
    excess_ports = get_test_portfolio_excess_returns(sample_df)
    
    # Align data on common non-null observations
    combined = pd.concat([excess_ports[portfolios], sample_df[factor_names]], axis=1).dropna()
    
    Y = combined[portfolios].values  # T x N
    F = combined[factor_names].values  # T x K
    
    T, N = Y.shape
    K = F.shape[1]
    
    if T <= (N + K):
        raise ValueError(f"Insufficient degrees of freedom: T={T}, N={N}, K={K}")
        
    # 1. Estimate multivariate OLS: Y = X * B + E
    X = np.column_stack([np.ones(T), F])  # T x (K+1)
    
    # B_hat = (X'X)^(-1) X' Y -> shape: (K+1) x N
    XtX_inv = np.linalg.inv(X.T @ X)
    B_hat = XtX_inv @ (X.T @ Y)
    
    alpha_hat = B_hat[0, :]  # 1 x N
    
    # Residuals: E = Y - X * B_hat -> T x N
    E_hat = Y - X @ B_hat
    
    # Unbiased Residual Covariance Matrix: Sigma_eps = (E' E) / (T - K - 1)
    Sigma_eps = (E_hat.T @ E_hat) / (T - K - 1)
    Sigma_eps_inv = np.linalg.inv(Sigma_eps)
    
    # 2. Factor sample moments
    mu_f = np.mean(F, axis=0)  # K x 1
    Sigma_f = np.cov(F, rowvar=False, ddof=1)
    if K == 1:
        Sigma_f_inv = 1.0 / Sigma_f
        sh2_f = (mu_f[0] ** 2) * Sigma_f_inv
    else:
        Sigma_f_inv = np.linalg.inv(Sigma_f)
        sh2_f = mu_f.T @ Sigma_f_inv @ mu_f
        
    # 3. GRS Quadratic Form
    alpha_quad = alpha_hat.T @ Sigma_eps_inv @ alpha_hat
    
    # 4. GRS Statistic
    scaling = (T - N - K) / N
    factor_adjustment = 1.0 / (1.0 + sh2_f)
    grs_stat = scaling * factor_adjustment * alpha_quad
    
    # 5. F-distribution p-value
    df1 = N
    df2 = T - N - K
    p_value = 1.0 - stats.f.cdf(grs_stat, df1, df2)
    
    # Economic metrics
    maa = np.mean(np.abs(alpha_hat))
    rms_alpha = np.sqrt(np.mean(alpha_hat ** 2))
    
    return {
        "T": T,
        "N": N,
        "K": K,
        "df1": df1,
        "df2": df2,
        "GRS_stat": grs_stat,
        "p_value": p_value,
        "reject_H0_5pct": bool(p_value < 0.05),
        "reject_H0_1pct": bool(p_value < 0.01),
        "Sh2_factors": sh2_f,
        "MAA (%/mo)": maa,
        "RMS_alpha (%/mo)": rms_alpha,
    }


def run_grs_suite(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Run GRS test across CAPM, FF3, and FF4 for multiple asset sets and sample splits.
    """
    if df is None:
        df = load_master_dataset()
        
    models = {
        "CAPM": ["MF"],
        "FF3": ["MF", "SMB", "HML"],
        "FF4": ["MF", "SMB", "HML", "WML"],
    }
    
    portfolio_sets = {
        "9 Balanced Portfolios (SV,SN,SG,BN,BG,WS,WB,LS,LB)": config.BALANCED_TEST_PORTFOLIOS,
        "5 Size-Value Portfolios (SV,SN,SG,BN,BG)": [p for p in config.SIZE_VALUE_PORTFOLIOS if p != "BV"],
        "4 Size-Momentum Portfolios (WS,WB,LS,LB)": config.SIZE_MOMENTUM_PORTFOLIOS,
    }
    
    sample_periods = {
        "Full Sample (1993-2025)": (config.FULL_SAMPLE_START, config.FULL_SAMPLE_END),
        "In-Sample (1993-2015)": (config.IN_SAMPLE_START, config.IN_SAMPLE_END),
        "Out-of-Sample (2016-2025)": (config.OUT_OF_SAMPLE_START, config.OUT_OF_SAMPLE_END),
    }
    
    results = []
    
    for s_name, (start, end) in sample_periods.items():
        for p_name, port_list in portfolio_sets.items():
            for m_name, factors in models.items():
                try:
                    res = compute_grs_test(
                        portfolios=port_list,
                        factor_names=factors,
                        df=df,
                        sample_start=start,
                        sample_end=end,
                    )
                    
                    results.append({
                        "Sample Period": s_name,
                        "Test Asset Set": p_name.split(" (")[0],
                        "Model": m_name,
                        "N": res["N"],
                        "K": res["K"],
                        "T": res["T"],
                        "GRS Stat": round(res["GRS_stat"], 3),
                        "df1, df2": f"({res['df1']}, {res['df2']})",
                        "p-value": round(res["p_value"], 4),
                        "Reject H0 (5%)": "Yes" if res["reject_H0_5pct"] else "No",
                        "MAA (%/mo)": round(res["MAA (%/mo)"], 3),
                        "RMS Alpha (%/mo)": round(res["RMS_alpha (%/mo)"], 3),
                    })
                except Exception as e:
                    print(f"Error running GRS for {m_name} on {p_name} ({s_name}): {e}")
                    
    grs_df = pd.DataFrame(results)
    
    # Save tables
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    grs_df.to_csv(config.TABLES_DIR / "grs_test_results.csv", index=False)
    
    with open(config.TABLES_DIR / "grs_test_results.md", "w") as f:
        f.write("# Gibbons-Ross-Shanken (GRS) Joint Hypothesis Test Results\n\n")
        f.write(grs_df.to_markdown(index=False))
        f.write("\n\n**Hypothesis:** $H_0: \\alpha_1 = \\alpha_2 = \\dots = \\alpha_N = 0$. Rejection indicates model failure to completely span the mean-variance efficient frontier of test assets.\n")
        
    return grs_df


if __name__ == "__main__":
    table = run_grs_suite()
    print("GRS Test Results Summary:")
    print(table.to_string(index=False))
