# Empirical Asset Pricing in Indian Equities: Factor Premia, Pricing Errors and Regime Dependence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Econometrics: Statsmodels](https://img.shields.io/badge/Econometrics-Statsmodels-orange.svg)](https://www.statsmodels.org/)

A research-grade empirical asset pricing study testing the **Capital Asset Pricing Model (CAPM)**, **Fama–French Three-Factor Model (FF3)**, and **Carhart Four-Factor Model (FF4)** across the cross-section of Indian equities over a 32-year sample period (**November 1993 to December 2025**, $T = 386$ monthly observations / 8,004 daily observations).

---

## 📌 Executive Findings Summary

1. **Failure of the CAPM:** CAPM produces large, economically significant pricing errors ($MAA = 4.12\%$ p.a., $RMS\alpha = 4.61\%$ p.a.) and is decisively rejected by the Gibbons–Ross–Shanken (GRS) test ($F = 3.05, p = 0.0016$).
2. **Dominance of Value ($HML$) and Momentum ($WML$):**
   - **Value ($HML$):** Generates $+8.50\%$ annualized premium ($t = 2.37, p = 0.0185$, Sharpe $0.42$).
   - **Momentum ($WML$):** Delivers $+13.35\%$ annualized premium ($t = 3.08, p = 0.0022$, Sharpe $0.54$).
   - **Negative Factor Correlation:** $r(HML, WML) = -0.368$, creating a natural diversification hedge.
3. **Ineffectiveness of Size ($SMB$):** Small caps generate a statistically insignificant premium ($-2.92\%$ p.a., $t = -0.99$). In crisis regimes, $SMB$ plunges to $-7.76\%$ p.a.
4. **Carhart Four-Factor Outperformance:** Adding Momentum reduces Mean Absolute Alpha by **$31\%$** (to $2.85\%$ p.a.) and increases explanatory power to $R^2 = 93.4\%$.
5. **Out-of-Sample Stability (2016–2025):** In the 10-year OOS test, FF4 is the only specification that fails to reject the GRS zero-alpha null on Size-Value ($p = 0.0501$) and Size-Momentum ($p = 0.0524$) portfolios.
6. **Multi-Factor Portfolio ($MF + HML + WML$):** Achieves an annualized Sharpe ratio of **$0.85$ gross / $0.81$ net of $50\text{ bps}$ transaction costs**, cutting maximum drawdown to $-33.3\%$ (vs. $-75.1\%$ for the Market).

---

## 🗂 Project Repository Architecture

```
indian_asset_pricing/
├── README.md                           # Main repository documentation
├── requirements.txt                    # Python package dependencies
├── run_pipeline.py                     # Master one-click reproducible runner
├── class_presentation.md               # 10-mark course presentation summary
├── resume_bullets.md                   # Concise resume description
├── interview_prep.md                   # Complete Theory & Econometrics Q&A
│
├── config/
│   └── config.py                       # Central parameters & sample periods
│
├── data/
│   ├── raw/                            # Downloaded IIM-A CSV factor files
│   ├── processed/                      # Synchronized master panel & excess returns
│   ├── data_dictionary.csv             # Formal variable dictionary
│   └── README.md                       # Data provenance & quality audit
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Ingestion & date indexing engine
│   ├── cleaning.py                     # Missingness & sample continuity checks
│   ├── factors.py                      # Factor moments, correlations, rolling stats
│   ├── portfolios.py                   # Test asset excess returns
│   ├── regressions.py                  # OLS time-series with Newey-West HAC inference
│   ├── evaluation.py                   # Pricing error metrics (MAA, RMS Alpha)
│   ├── grs.py                          # Exact Gibbons-Ross-Shanken (1989) F-test
│   ├── regimes.py                      # 2-State Gaussian Hidden Markov Model (HMM)
│   ├── backtest.py                     # Strategy backtesting & transaction costs
│   ├── robustness.py                   # Subsample eras & HAC lag sensitivity
│   └── plotting.py                     # Publication-quality figure renderer (14 plots)
│
├── results/
│   ├── figures/                        # 14 High-resolution PNG figures (300 DPI)
│   └── tables/                         # CSV and Markdown econometric tables
│
├── notebooks/
│   └── empirical_asset_pricing_master.ipynb # Executed interactive research notebook
│
└── report/
    └── final_report.md                 # Complete academic research paper
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/shaswat-bhaskar/indian-asset-pricing.git
cd indian-asset-pricing
pip install -r requirements.txt
```

### 2. Execute Entire Pipeline
Run the master script to download data, run econometric regressions, compute GRS tests, fit HMM regimes, evaluate backtests, and generate all 14 figures:
```bash
python run_pipeline.py
```

---

## 📊 Summary of Research Figures (Saved in `results/figures/`)

| File | Description |
|---|---|
| `fig01_cumulative_factor_returns.png` | Cumulative wealth index of factors and multi-factor strategies (log scale) |
| `fig02_factor_distributions.png` | Empirical factor return histograms, KDE, and Gaussian fits |
| `fig03_factor_correlation_heatmap.png` | Pairwise factor correlation heatmaps across Full, In-Sample, and OOS |
| `fig04_rolling_factor_sharpe.png` | 36-month rolling annualized Sharpe ratios through time |
| `fig05_factor_drawdowns.png` | Historical underwater drawdown profiles |
| `fig06_capm_alpha_comparison.png` | CAPM pricing errors ($\alpha_i$) with Newey-West 95% confidence intervals |
| `fig07_ff3_alpha_comparison.png` | Fama-French 3-Factor pricing errors across test portfolios |
| `fig08_ff4_alpha_comparison.png` | Carhart 4-Factor pricing errors across test portfolios |
| `fig09_model_pricing_error_comparison.png` | Cross-model MAA and RMS Alpha comparison (Full Sample vs OOS) |
| `fig10_factor_loading_heatmap.png` | Factor loading matrix ($\beta_M, \beta_S, \beta_H, \beta_W$) across cross-section |
| `fig11_regime_probabilities.png` | HMM posterior probability of Crisis regime with historical shock annotations |
| `fig12_factor_performance_by_regime.png` | Factor returns and Sharpe ratios in Normal vs Crisis regimes |
| `fig13_out_of_sample_cumulative_returns.png` | Out-of-sample wealth trajectories (2016–2025) |
| `fig14_transaction_cost_sensitivity.png` | Strategy Sharpe ratio and CAGR decay across 0 to 100 bps friction |

---

## 📚 Academic Citation & Data Provenance

Factor data and test portfolios are sourced from the **IIM Ahmedabad Fama-French and Momentum Data Library for the Indian Market**:
```bibtex
@techreport{agarwalla2013four,
  author      = {Agarwalla, Sobhesh K. and Jacob, Joshy and Varma, Jayanth R.},
  title       = {Four factor model in Indian equities market},
  institution = {Indian Institute of Management, Ahmedabad},
  number      = {W.P. No. 2013-09-05},
  year        = {2013},
  url         = {https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/}
}
```

---

## 📄 License
This repository is licensed under the MIT License.
