"""
Generate and execute research-quality Jupyter notebooks with live code, LaTeX math, and rich outputs.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_master_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Intro
    cells.append(nbf.v4.new_markdown_cell(r"""# Empirical Asset Pricing in Indian Equities: Factor Premia, Pricing Errors and Regime Dependence
### Testing CAPM, Fama–French Three-Factor and Carhart Four-Factor Models (1993–2025)

**Author:** Shaswat Bhaskar  
**Dataset:** IIM Ahmedabad Fama-French & Momentum Factors Library (CMIE Prowess DX + RBI)  
**Sample Period:** November 1993 to December 2025 ($T = 386$ monthly observations / 8,004 daily observations)

---
## Theoretical Overview & Asset Pricing Foundations

### 1. Stochastic Discount Factor (SDF) Representation
Under no-arbitrage, the price of any asset satisfies the Euler equation:
$$P_t = \mathbb{E}_t [m_{t+1} X_{t+1}]$$
For gross returns $R_{i,t+1} = X_{i,t+1}/P_{i,t}$, we have $\mathbb{E}_t [m_{t+1} R_{i,t+1}] = 1$. Since $R_f = 1/\mathbb{E}_t[m_{t+1}]$, expected excess returns satisfy the fundamental covariance pricing equation:
$$\mathbb{E}_t[R_{i,t+1}] - R_f = - R_f \text{Cov}_t(m_{t+1}, R_{i,t+1}) = \beta_{i,m} \lambda_m$$

### 2. Factor Model Specifications
1. **Capital Asset Pricing Model (CAPM):**
   $$R_{it} - R_{ft} = \alpha_i + \beta_{iM} (R_{mt} - R_{ft}) + \epsilon_{it}$$
2. **Fama–French Three-Factor Model (FF3):**
   $$R_{it} - R_{ft} = \alpha_i + \beta_{iM} MF_t + \beta_{iS} SMB_t + \beta_{iH} HML_t + \epsilon_{it}$$
3. **Carhart Four-Factor Model (FF4):**
   $$R_{it} - R_{ft} = \alpha_i + \beta_{iM} MF_t + \beta_{iS} SMB_t + \beta_{iH} HML_t + \beta_{iW} WML_t + \epsilon_{it}$$
"""))

    # Cell: Imports & Setup
    cells.append(nbf.v4.new_code_cell("""# Setup environment and imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "notebooks" else notebook_dir
sys.path.append(str(project_root))

from config import config
from src.data_loader import load_master_dataset
from src.cleaning import generate_data_quality_report, generate_subsample_breakdown
from src.factors import compute_factor_descriptive_stats, compute_factor_correlations, compute_rolling_factor_stats
from src.portfolios import get_test_portfolio_excess_returns, compute_portfolio_summary_stats
from src.regressions import build_regression_table, generate_and_save_all_regression_tables
from src.evaluation import compare_all_models_pricing_errors
from src.grs import compute_grs_test, run_grs_suite
from src.regimes import fit_market_regime_hmm, analyze_factors_across_regimes
from src.backtest import run_factor_backtests, run_transaction_cost_sensitivity
from src.robustness import run_subsample_robustness_analysis, run_hac_lag_sensitivity

# Set plotting aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 120
print("Environment configured successfully. Ready for empirical analysis.")
"""))

    # Section 1: Data Ingestion
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 1: Data Ingestion & Quality Audit (Steps 0.1–0.12)
We load the synchronized monthly master panel covering **November 1993 to December 2025** ($T=386$ months).
"""))

    cells.append(nbf.v4.new_code_cell("""master_df = load_master_dataset()
print(f"Master Dataset Dimensions: {master_df.shape} (Months x Variables)")
print(f"Sample Date Range: {master_df.index[0]} to {master_df.index[-1]}")
master_df.head()
"""))

    cells.append(nbf.v4.new_code_cell("""# Data quality and missingness audit
quality_rep = generate_data_quality_report()
quality_rep
"""))

    # Section 2: Factor Descriptive Statistics
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 2: Factor Descriptive Statistics & Correlation Structure
We compute annualized means, standard deviations, Sharpe ratios, skewness, kurtosis, autocorrelations, and $t$-tests for each factor.
"""))

    cells.append(nbf.v4.new_code_cell("""factor_stats = compute_factor_descriptive_stats(master_df)
factor_stats
"""))

    cells.append(nbf.v4.new_code_cell("""# Factor Correlation Matrices
corrs = compute_factor_correlations(master_df)
print("=== Full Sample Factor Correlation Matrix (1993-2025) ===")
display(corrs['full'].round(3))
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualize Cumulative Factor Returns
dates = master_df.index.to_timestamp()
plt.figure(figsize=(10, 5.5))
for f, col in [('MF', '#1f77b4'), ('SMB', '#d62728'), ('HML', '#2ca02c'), ('WML', '#ff7f0e')]:
    w = (1.0 + master_df[f]/100.0).cumprod()
    plt.plot(dates, w, label=f"{f} Factor", color=col, lw=1.8)

mf_3f = (master_df['MF'] + master_df['HML'] + master_df['WML']) / (3.0 * 100.0)
plt.plot(dates, (1.0 + mf_3f).cumprod(), label="Multi-Factor (MF+HML+WML)", color='#9467bd', lw=2.4, ls='--')

plt.yscale('log')
plt.title("Cumulative Factor Growth (Log Scale, 1993–2025)", fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Wealth ($1 Initial Investment)")
plt.legend(frameon=True)
plt.show()
"""))

    # Section 3: Time-Series Regressions
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 3: Time-Series Asset Pricing Regressions (CAPM, FF3, Carhart FF4)
All regressions are estimated via OLS with **Newey-West (1987) HAC robust standard errors** ($L=3$ lags).
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. CAPM Regressions
capm_table = build_regression_table(model_name="CAPM", df=master_df)
print("=== CAPM Time-Series Regression Results (Full Sample) ===")
display(capm_table)
"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Fama-French 3-Factor Regressions
ff3_table = build_regression_table(model_name="FF3", df=master_df)
print("=== Fama-French 3-Factor Regression Results (Full Sample) ===")
display(ff3_table)
"""))

    cells.append(nbf.v4.new_code_cell("""# 3. Carhart 4-Factor Regressions (with Momentum)
ff4_table = build_regression_table(model_name="FF4", df=master_df)
print("=== Carhart 4-Factor Regression Results (Full Sample) ===")
display(ff4_table)
"""))

    # Section 4: Pricing Errors
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 4: Pricing Error Analysis & Cross-Model Comparison
We compare Mean Absolute Alpha ($MAA$), RMS Alpha, Max $|\alpha|$, and Mean $R^2$ across models.
"""))

    cells.append(nbf.v4.new_code_cell("""pricing_errors = compare_all_models_pricing_errors(sample_name="full_sample")
print("=== Full Sample Pricing Error Comparison (CAPM vs FF3 vs FF4) ===")
display(pricing_errors)
"""))

    cells.append(nbf.v4.new_code_cell("""# Plot Pricing Error Comparison
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(pricing_errors))
width = 0.35
ax.bar(x - width/2, pricing_errors['MAA (%/mo)'], width, label='Mean Absolute Alpha (MAA)', color='#1f77b4', edgecolor='black')
ax.bar(x + width/2, pricing_errors['RMS Alpha (%/mo)'], width, label='RMS Alpha', color='#ff7f0e', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(pricing_errors['Model'], fontweight='bold')
ax.set_ylabel("Monthly Pricing Error (%/mo)")
ax.set_title("Cross-Model Pricing Error Comparison (Full Sample: 1993–2025)", fontweight='bold')
ax.legend(frameon=True)
plt.show()
"""))

    # Section 5: GRS Test
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 5: Gibbons–Ross–Shanken (GRS) Exact Joint Hypothesis Test
$$H_0: \boldsymbol{\alpha} = [\alpha_1, \alpha_2, \dots, \alpha_N]' = \mathbf{0}$$
$$GRS = \frac{T - N - K}{N} \left( 1 + \hat{\boldsymbol{\mu}}_F' \hat{\boldsymbol{\Sigma}}_F^{-1} \hat{\boldsymbol{\mu}}_F \right)^{-1} \hat{\boldsymbol{\alpha}}' \hat{\boldsymbol{\Sigma}}_\epsilon^{-1} \hat{\boldsymbol{\alpha}} \sim F_{N, T - N - K}$$
"""))

    cells.append(nbf.v4.new_code_cell("""grs_results = run_grs_suite(master_df)
display(grs_results)
"""))

    # Section 6: Regime Analysis (HMM)
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 6: Market Regime Analysis (2-State Gaussian Hidden Markov Model)
We classify market conditions into **State 0 (Normal / Low-Volatility)** and **State 1 (Crisis / High-Volatility)**.
"""))

    cells.append(nbf.v4.new_code_cell("""regime_dict = fit_market_regime_hmm(master_df)
print("=== HMM Regime Parameters & Transition Persistence ===")
display(regime_dict['char_table'])
"""))

    cells.append(nbf.v4.new_code_cell("""factor_regimes = analyze_factors_across_regimes(regime_dict)
print("=== Factor Risk Premia Conditional on Market Regimes ===")
display(factor_regimes)
"""))

    cells.append(nbf.v4.new_code_cell("""# Plot Regime Probabilities
reg_df = regime_dict['regime_df']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
ax1.plot(dates, reg_df['MF'], color='#1f77b4', lw=1.1, label='Market Excess Return (MF)')
ax1.axhline(0, color='black', ls='--', lw=0.7)
ax1.set_ylabel("Monthly Return (%)")
ax1.set_title("Market Returns and HMM Crisis Regime Probabilities (1993–2025)", fontweight='bold')
ax1.legend(loc='upper left')

ax2.fill_between(dates, 0, reg_df['Prob_HighVol'], color='#d62728', alpha=0.65, label='P(Crisis / High-Vol Regime)')
ax2.set_ylabel("Crisis Probability")
ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Year")
ax2.legend(loc='upper left')
plt.show()
"""))

    # Section 7: Out-of-Sample Testing
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 7: Out-of-Sample Testing (2016–2025)
We validate factor models on a strictly chronological out-of-sample period ($N=120$ months).
"""))

    cells.append(nbf.v4.new_code_cell("""pricing_errors_oos = compare_all_models_pricing_errors(sample_name="oos")
print("=== Out-of-Sample Pricing Error Comparison (2016–2025) ===")
display(pricing_errors_oos)
"""))

    # Section 8: Portfolio Backtest & Transaction Costs
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 8: Portfolio Backtest & Transaction Cost Sensitivity
We evaluate factor strategies and equal-weighted multi-factor combinations under **0, 25, 50, and 100 bps** transaction cost friction.
"""))

    cells.append(nbf.v4.new_code_cell("""backtest_summary = run_transaction_cost_sensitivity(master_df)
display(backtest_summary)
"""))

    cells.append(nbf.v4.new_code_cell("""# Plot Transaction Cost Sensitivity
strategies = ["MF", "HML", "WML", "MultiFactor_3F (MF+HML+WML)"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for strat in strategies:
    sub = backtest_summary[backtest_summary["Strategy"] == strat]
    ax1.plot(sub["Cost (bps)"], sub["Sharpe Ratio"], marker="o", lw=1.8, label=strat)
    ax2.plot(sub["Cost (bps)"], sub["CAGR (%)"], marker="s", lw=1.8, label=strat)

ax1.set_title("Net Sharpe Ratio vs Transaction Costs", fontweight="bold")
ax1.set_xlabel("Transaction Cost (bps)")
ax1.set_ylabel("Annualized Sharpe Ratio")
ax1.set_xticks(config.TRANSACTION_COST_BPS)
ax1.legend(frameon=True)

ax2.set_title("Net CAGR vs Transaction Costs", fontweight="bold")
ax2.set_xlabel("Transaction Cost (bps)")
ax2.set_ylabel("Net CAGR (%)")
ax2.set_xticks(config.TRANSACTION_COST_BPS)
ax2.legend(frameon=True)

plt.suptitle("Strategy Performance Resilience Under Cost Friction", fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
"""))

    # Section 9: Robustness
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 9: Subsample Robustness & HAC Lag Sensitivity
We test model stability across historical eras (Pre-GFC vs Post-GFC) and Newey-West lag choices ($L=1, 3, 6, 12$).
"""))

    cells.append(nbf.v4.new_code_cell("""subsample_rob = run_subsample_robustness_analysis(master_df)
print("=== Subsample Robustness Analysis ===")
display(subsample_rob)
"""))

    cells.append(nbf.v4.new_code_cell("""lag_sens = run_hac_lag_sensitivity(master_df)
print("=== Newey-West HAC Lag Sensitivity Analysis ===")
display(lag_sens)
"""))

    # Section 10: Conclusion
    cells.append(nbf.v4.new_markdown_cell(r"""---
## Section 10: Empirical Conclusions
1. **CAPM is Inadequate:** Pricing errors are large ($MAA = 4.12\%$ p.a.) and GRS strongly rejects $H_0$ ($p=0.0016$).
2. **Value ($HML$) and Momentum ($WML$) are Dominant:** $HML$ earns $+8.50\%$ p.a. ($t=2.37$), and $WML$ earns $+13.35\%$ p.a. ($t=3.08$).
3. **Size ($SMB$) Fails in India:** Small caps earn $-2.92\%$ p.a. ($t=-0.99$) and suffer severe losses during crisis regimes ($-7.76\%$).
4. **Carhart 4-Factor Outperforms:** Reduces Mean Absolute Pricing Errors by $31\%$ and survives 10-year out-of-sample testing.
5. **Multi-Factor Strategy ($MF + HML + WML$):** Delivers a Sharpe ratio of **$0.81$ net of $50\text{ bps}$ friction** with a maximum drawdown of $-34.8\%$ vs $-75.1\%$ for the Market.
"""))

    nb.cells = cells
    nb_path = NOTEBOOKS_DIR / "empirical_asset_pricing_master.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Master notebook generated at {nb_path}")
    return nb_path


def execute_notebook(nb_path: Path):
    """Execute the notebook and save it with all live outputs."""
    print(f"Executing notebook: {nb_path.name} ...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook executed and saved successfully with all inline outputs: {nb_path.name}")


if __name__ == "__main__":
    master_path = build_master_notebook()
    execute_notebook(master_path)
