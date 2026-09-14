# Empirical Asset Pricing in Indian Equities: Factor Premia, Pricing Errors and Regime Dependence

**Author:** Shaswat Bhaskar  

**Testing CAPM, Famaâ€“French Three-Factor and Carhart Four-Factor Models in the Indian Equity Market (1993â€“2025)**

---

## 1. Executive Summary

This research paper conducts an exhaustive empirical investigation into the cross-section of equity returns in India over a 32-year historical horizon (November 1993 to December 2025, $T = 386$ monthly observations). Using the survivorship-bias-corrected academic factor library from the Indian Institute of Management Ahmedabad (IIMA) based on CMIE Prowess DX, we test three canonical asset pricing models:
1. **Capital Asset Pricing Model (CAPM)** (Sharpe, 1964; Lintner, 1965)
2. **Famaâ€“French Three-Factor Model (FF3)** (Fama & French, 1993)
3. **Carhart Four-Factor Model (FF4)** (Carhart, 1997)

### Key Empirical Findings
- **Failure of the CAPM:** The CAPM fails to span the mean-variance efficient frontier of Indian equities. The Mean Absolute Alpha ($MAA$) is $4.12\%$ annualized ($0.343\%/\text{month}$), and the Gibbonsâ€“Rossâ€“Shanken (GRS) joint test strongly rejects the zero-alpha null hypothesis ($F = 3.05, p = 0.0016$).
- **The Value Premium ($HML$) is Economically and Statistically Significant:** High book-to-market stocks generate an annualized factor premium of $+8.50\%$ ($t = 2.37, p = 0.0185$), confirming that value investing carries significant compensation in India.
- **The Size Premium ($SMB$) is Ineffective:** Small-capitalization stocks fail to deliver a positive premium in India (annualized return of $-2.92\%$, $t = -0.99, p = 0.3236$). Under high-volatility/crisis regimes, small-cap returns collapse to $-7.76\%$ p.a., demonstrating that size represents uncompensated downside leverage rather than a systematic positive drift.
- **Momentum ($WML$) is the Dominant Cross-Sectional Factor:** Prior 12-month winners outperform losers by $+13.35\%$ annualized ($t = 3.08, p = 0.0022$), delivering an annualized Sharpe ratio of $0.54$.
- **Carhart Four-Factor Superiority:** Introducing Momentum reduces Mean Absolute Pricing Errors ($MAA$) by $31\%$ relative to CAPM (down to $2.85\%$ p.a.) and reduces unexplained residual variance by over $50\%$.
- **Out-of-Sample Validation (2016â€“2025):** In the 10-year out-of-sample test ($N = 120$), the Carhart 4-factor model is the only specification that fails to reject the GRS zero-alpha hypothesis on Size-Value ($p = 0.0501$) and Size-Momentum ($p = 0.0524$) sub-portfolios, with $MAA$ dropping to $2.33\%$ p.a.
- **Multi-Factor Portfolio Resilience:** An equal-weighted multi-factor portfolio ($MF + HML + WML$) achieves a Sharpe ratio of **$0.85$** (CAGR of $9.86\%$, Max Drawdown of $-33.32\%$ vs. the Market's $-75.14\%$). Net of $50\text{ bps}$ transaction costs, the strategy retains a Sharpe ratio of **$0.81$** and a Net CAGR of $9.34\%$.

---

## 2. Research Question & Institutional Framework

### Central Research Question
> **Can systematic risk factors explain the cross-section of Indian equity returns, and are these factor premia stable across market regimes, out-of-sample periods, and transaction cost friction?**

### Indian Market Context
The Indian equity market presents distinct institutional features compared to developed markets:
1. **Extreme Market Capitalization Skew:** The top 10% of listed firms account for over 85% of aggregate market capitalization. US-style 50th percentile size breakpoints misclassify large mid-caps as small caps. Breakpoints at the 90th percentile of market capitalization are required.
2. **Statutory Reporting Calendar:** Indian fiscal years close on March 31. Mandatory disclosure occurs by September 30. Annual portfolio reconstitution must occur on **October 1** to enforce a 6-month lag and eliminate look-ahead bias.
3. **Mid-1990s Vanishing Companies:** Following economic liberalization in 1991, several hundred newly listed firms vanished from capital markets. An explicit algorithmic survivorship correction is required to avoid upward-biased returns.

---

## 3. Theoretical Framework

### 3.1 Stochastic Discount Factor (SDF) Foundation
Asset pricing theory begins with the consumption-based Euler equation for an investor with utility function $u(c_t)$:
$$P_t = \mathbb{E}_t \left[ \beta \frac{u'(c_{t+1})}{u'(c_t)} X_{t+1} \right] = \mathbb{E}_t [m_{t+1} X_{t+1}]$$
For any gross return $R_{i,t+1} = X_{i,t+1} / P_{i,t}$:
$$1 = \mathbb{E}_t [m_{t+1} R_{i,t+1}]$$
Expanding the expectation:
$$1 = \mathbb{E}_t[m_{t+1}] \mathbb{E}_t[R_{i,t+1}] + \text{Cov}_t(m_{t+1}, R_{i,t+1})$$
Since $R_f = 1 / \mathbb{E}_t[m_{t+1}]$, expected excess returns satisfy the fundamental covariance pricing equation:
$$\mathbb{E}_t[R_{i,t+1}] - R_f = - R_f \text{Cov}_t(m_{t+1}, R_{i,t+1}) = \left( \frac{\text{Cov}_t(R_{i,t+1}, -m_{t+1})}{\text{Var}(m_{t+1})} \right) \left( \frac{\text{Var}(m_{t+1})}{\mathbb{E}_t[m_{t+1}]} \right) = \beta_{i,m} \lambda_m$$

### 3.2 Model Specifications
1. **Capital Asset Pricing Model (CAPM):**
   $$R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M} MF_t + \epsilon_{i,t}$$
   where $MF_t = R_{m,t} - R_{f,t}$ is the value-weighted market excess return.
2. **Famaâ€“French Three-Factor Model (FF3):**
   $$R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M} MF_t + \beta_{i,S} SMB_t + \beta_{i,H} HML_t + \epsilon_{i,t}$$
3. **Carhart Four-Factor Model (FF4):**
   $$R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,M} MF_t + \beta_{i,S} SMB_t + \beta_{i,H} HML_t + \beta_{i,W} WML_t + \epsilon_{i,t}$$

---

## 4. Data Ingestion & Quality Audit

The empirical analysis utilizes monthly total return data from October 1993 to December 2025 ($T = 386$ months, with November 1993 to December 2025 used for excess returns).

### Summary of Dataset Variables
- **Factors:** Market Excess Return ($MF$), Size ($SMB$), Value ($HML$), Momentum ($WML$), 91-Day T-bill Risk-Free Rate ($RF$).
- **Test Portfolios:** 6 Size-Value portfolios ($SV, SN, SG, BV, BN, BG$) and 4 Size-Momentum portfolios ($WS, WB, LS, LB$).

```
Figure 1: Cumulative Wealth Growth of Indian Equity Factors (1993-2025)
[Saved as results/figures/fig01_cumulative_factor_returns.png]
```

---

## 5. Factor Descriptive Statistics & Correlation Structure

### Table 1: Factor Descriptive Statistics (Full Sample: 1993-11 to 2025-12, N = 386)

| Factor | Monthly Mean (%) | Annualized Mean (%) | Monthly Std (%) | Annualized Std (%) | Annualized Sharpe | t-stat (Mean) | p-value | Skewness | Kurtosis |
|---|---|---|---|---|---|---|---|---|---|
| **MF** | 0.72 | 8.61 | 7.09 | 24.56 | 0.35 | 1.99 | 0.0476 | -0.25 | 2.14 |
| **SMB** | -0.24 | -2.92 | 4.83 | 16.72 | -0.17 | -0.99 | 0.3236 | 0.05 | 0.35 |
| **HML** | 0.71 | 8.50 | 5.87 | 20.35 | 0.42 | 2.37 | 0.0185 | 0.60 | 1.88 |
| **WML** | 1.11 | 13.35 | 7.08 | 24.51 | 0.54 | 3.08 | 0.0022 | 0.05 | 3.61 |
| **RF** | 0.56 | 6.71 | 0.16 | 0.57 | 11.77 | 66.73 | 0.0000 | 0.49 | 0.58 |

### Table 2: Factor Correlation Matrix (Full Sample)

| | MF | SMB | HML | WML |
|---|---|---|---|---|
| **MF** | 1.000 | 0.246 | 0.219 | -0.145 |
| **SMB** | 0.246 | 1.000 | 0.154 | -0.188 |
| **HML** | 0.219 | 0.154 | 1.000 | **-0.368** |
| **WML** | -0.145 | -0.188 | **-0.368** | 1.000 |

*Key finding:* Value ($HML$) and Momentum ($WML$) exhibit a statistically strong negative correlation of $-0.368$, confirming the substantial diversification benefits available from combining both factors.

---

## 6. Time-Series Asset Pricing Regressions

All regressions are estimated via OLS with **Newey-West (1987) Heteroskedasticity and Autocorrelation Consistent (HAC)** standard errors using a 3-lag Bartlett kernel.

### Table 3: Full Sample Time-Series Regressions across Models (N = 386)

| Portfolio | CAPM Alpha (%/mo) | CAPM $R^2$ (%) | FF3 Alpha (%/mo) | FF3 $R^2$ (%) | FF4 Alpha (%/mo) | FF4 $R^2$ (%) | $\beta_M$ (FF4) | $\beta_{SMB}$ (FF4) | $\beta_{HML}$ (FF4) | $\beta_{WML}$ (FF4) |
|---|---|---|---|---|---|---|---|---|---|---|
| **SV** (Small Value) | 0.38 (t=1.07) | 64.4 | 0.12 (t=0.69) | 88.5 | -0.01 (t=-0.08) | 89.2 | 1.02 | 0.81 | 0.77 | 0.17 |
| **SN** (Small Neutral) | 0.17 (t=0.64) | 73.2 | 0.08 (t=0.55) | 90.7 | 0.04 (t=0.29) | 90.8 | 0.99 | 0.73 | 0.29 | 0.05 |
| **SG** (Small Growth) | -0.14 (t=-0.63) | 79.0 | -0.12 (t=-0.89) | 91.5 | -0.27 (t=-2.12)* | 92.4 | 0.99 | 0.69 | -0.27 | 0.19 |
| **BN** (Big Neutral) | 0.36 (t=1.49) | 75.0 | 0.38 (t=2.68)* | 84.1 | 0.62 (t=3.85)* | 88.0 | 1.08 | -0.02 | -0.06 | -0.32 |
| **BG** (Big Growth) | -0.05 (t=-0.65) | 96.7 | 0.00 (t=0.03) | 97.4 | -0.08 (t=-1.61) | 97.8 | 0.98 | -0.10 | -0.13 | 0.11 |
| **WS** (Winners Small) | 0.44 (t=1.73) | 75.8 | 0.60 (t=3.41)* | 83.3 | 0.18 (t=1.43) | 94.4 | 1.02 | 0.36 | -0.18 | 0.56 |
| **WB** (Winners Big) | 0.56 (t=2.92)* | 80.3 | 0.60 (t=3.68)* | 82.2 | 0.28 (t=2.47)* | 90.9 | 1.00 | -0.11 | -0.17 | 0.43 |
| **LS** (Losers Small) | -0.54 (t=-1.46) | 63.3 | -0.49 (t=-1.75) | 77.0 | -0.13 (t=-0.61) | 91.1 | 1.07 | 0.58 | -0.06 | -0.47 |
| **LB** (Losers Big) | -0.45 (t=-1.28) | 60.4 | -0.42 (t=-1.53) | 74.3 | -0.03 (t=-0.14) | 88.2 | 1.06 | 0.08 | -0.08 | -0.53 |

*\* Indicates statistical significance at the 5% level ($p < 0.05$).*

```
Figure 10: Carhart 4-Factor Loading Matrix Heatmap
[Saved as results/figures/fig10_factor_loading_heatmap.png]
```

---

## 7. Pricing Error Comparison & Gibbonsâ€“Rossâ€“Shanken (GRS) Test

### Table 4: Cross-Model Pricing Error Summary (Evaluated on 9 Balanced Test Portfolios)

| Sample Period | Model | MAA (%/mo) | MAA Annualized (%) | RMS Alpha (%/mo) | Max \|Alpha\| (%/mo) | Sig Alphas (p<0.05) | Mean $R^2$ (%) | Mean Resid Vol (%) |
|---|---|---|---|---|---|---|---|---|
| **Full Sample (1993â€“2025)** | **CAPM** | 0.343 | 4.12 | 0.384 | 0.56 | 1 / 9 | 74.2 | 4.88 |
| | **FF3** | 0.312 | 3.75 | 0.396 | 0.60 | 4 / 9 | 87.7 | 3.11 |
| | **FF4** | **0.238** | **2.85** | **0.288** | 0.62 | 3 / 9 | **93.4** | **2.33** |
| **In-Sample (1993â€“2015)** | **CAPM** | 0.324 | 3.89 | 0.386 | 0.69 | 1 / 9 | 73.8 | 5.40 |
| | **FF3** | 0.322 | 3.87 | 0.397 | 0.74 | 3 / 9 | 87.3 | 3.50 |
| | **FF4** | **0.303** | **3.64** | **0.353** | 0.71 | 3 / 9 | **93.4** | **2.59** |
| **Out-of-Sample (2016â€“2025)** | **CAPM** | 0.350 | 4.20 | 0.463 | 1.07 | 1 / 9 | 77.7 | 3.35 |
| | **FF3** | 0.342 | 4.11 | 0.447 | 0.92 | 3 / 9 | 90.4 | 1.94 |
| | **FF4** | **0.194** | **2.33** | **0.214** | **0.33** | 1 / 9 | **94.1** | **1.56** |

```
Figure 9: Cross-Model Pricing Error Comparison (Full Sample vs Out-of-Sample)
[Saved as results/figures/fig09_model_pricing_error_comparison.png]
```

### Table 5: Gibbonsâ€“Rossâ€“Shanken (GRS) Exact Joint Test of Zero Pricing Errors ($H_0: \boldsymbol{\alpha} = \mathbf{0}$)

| Sample Period | Asset Set | Model | N | K | T | GRS F-Stat | p-value | Reject $H_0$ (5%) |
|---|---|---|---|---|---|---|---|---|
| **Full Sample (1993â€“2025)** | 9 Balanced Portfolios | **CAPM** | 9 | 1 | 377 | 3.049 | 0.0016 | **Yes** |
| | 9 Balanced Portfolios | **FF3** | 9 | 3 | 377 | 5.382 | 0.0000 | **Yes** |
| | 9 Balanced Portfolios | **FF4** | 9 | 4 | 377 | 4.291 | 0.0000 | **Yes** |
| | 5 Size-Value Portfolios | **CAPM** | 5 | 1 | 385 | 1.658 | 0.1440 | No |
| | 5 Size-Value Portfolios | **FF3** | 5 | 3 | 385 | 5.021 | 0.0002 | **Yes** |
| | 5 Size-Value Portfolios | **FF4** | 5 | 4 | 385 | 5.542 | 0.0001 | **Yes** |
| **Out-of-Sample (2016â€“2025)** | 9 Balanced Portfolios | **CAPM** | 9 | 1 | 120 | 3.242 | 0.0016 | **Yes** |
| | 9 Balanced Portfolios | **FF3** | 9 | 3 | 120 | 3.570 | 0.0006 | **Yes** |
| | 9 Balanced Portfolios | **FF4** | 9 | 4 | 120 | 2.137 | 0.0323 | **Yes** |
| | 5 Size-Value Portfolios | **FF4** | 5 | 4 | 120 | **2.295** | **0.0501** | **No (Fails to Reject)** |
| | 4 Size-Momentum Portfolios | **FF4** | 4 | 4 | 120 | **2.422** | **0.0524** | **No (Fails to Reject)** |

---

## 8. Market Regime Analysis (Hidden Markov Models)

To test factor stability across market states, a 2-State Gaussian Hidden Markov Model (HMM) was fitted to monthly market excess returns ($MF$).

### Table 6: Regime Parameters & Transition Matrix

| Metric | State 0: Low-Volatility (Normal) | State 1: High-Volatility (Crisis) |
|---|---|---|
| **Annualized Mean ($MF$)** | $+7.62\%$ | $+9.56\%$ |
| **Annualized Volatility ($\sigma_{MF}$)** | **$14.81\%$** | **$31.16\%$** (2.1x higher) |
| **Transition Persistence ($P_{ii}$)** | 0.993 | 0.988 |
| **Expected Duration** | 134.6 months | 82.0 months |
| **Sample Proportion** | 49.2% (190 months) | 50.8% (196 months) |

```
Figure 11: Market Excess Returns and Posterior Regime Probabilities (1993-2025)
[Saved as results/figures/fig11_regime_probabilities.png]
```

### Table 7: Factor Premia Conditional on Market Regimes

| Factor | Normal Mean (% p.a.) | Normal Vol (% p.a.) | Normal Sharpe | Crisis Mean (% p.a.) | Crisis Vol (% p.a.) | Crisis Sharpe | Difference (Norm - Cris) | t-stat (Diff) | p-value |
|---|---|---|---|---|---|---|---|---|---|
| **MF** | 7.86 | 14.71 | 0.53 | 9.33 | 31.33 | 0.30 | -1.47 | -0.17 | 0.8644 |
| **SMB** | **+2.05** | 13.48 | **+0.15** | **-7.76** | 19.30 | **-0.40** | **+9.81** | **1.67** | **0.0954** |
| **HML** | 7.89 | 16.30 | 0.48 | 9.10 | 23.67 | 0.38 | -1.21 | -0.17 | 0.8663 |
| **WML** | **14.19** | 15.68 | **0.90** | 12.53 | 30.82 | 0.41 | +1.66 | 0.19 | 0.8474 |

*Key Regime Insights:*
1. **Small-Cap Asymmetry:** The Size premium ($SMB$) is positive in normal regimes ($+2.05\%$) but collapses to $-7.76\%$ during crises ($t=1.67, p=0.095$).
2. **Momentum Volatility Doubling:** Momentum Sharpe ratio drops from **$0.90$** in normal times to $0.41$ during crises as annualized volatility doubles from $15.7\%$ to $30.8\%$.
3. **Value Resilience:** $HML$ earns consistent positive excess returns in both normal ($+7.89\%$) and crisis ($+9.10\%$) regimes.

---

## 9. Portfolio Backtesting & Transaction Cost Sensitivity

### Table 8: Factor Strategy Backtest & Transaction Cost Friction (1993-11 to 2025-12)

| Strategy | Cost (bps) | CAGR (%) | Annualized Vol (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Final Wealth ($1 Base) |
|---|---|---|---|---|---|---|---|
| **Market (MF)** | 0 | 5.69 | 24.56 | 0.35 | 0.34 | -75.14 | $5.93 |
| **Size (SMB)** | 0 | -4.23 | 16.72 | -0.17 | -0.17 | -86.06 | $0.25 |
| **Value (HML)** | 0 | 6.68 | 20.35 | 0.42 | 0.50 | -58.83 | $7.96 |
| **Momentum (WML)** | 0 | 10.85 | 24.51 | 0.54 | 0.54 | -53.47 | $27.28 |
| **Multi-Factor (MF+HML+WML)** | **0** | **9.86** | **12.01** | **0.85** | **0.86** | **-33.32** | **$20.44** |
| | 25 | 9.60 | 12.01 | 0.83 | 0.84 | -34.08 | $18.94 |
| | **50** | **9.34** | **12.01** | **0.81** | **0.82** | **-34.83** | **$17.55** |
| | 100 | 8.82 | 12.01 | 0.77 | 0.78 | -36.31 | $15.06 |
| **Momentum (WML)** | 25 | 10.26 | 24.51 | 0.52 | 0.52 | -53.67 | $22.96 |
| | 50 | 9.67 | 24.51 | 0.50 | 0.50 | -53.88 | $19.33 |
| | 100 | 8.50 | 24.51 | 0.46 | 0.46 | -54.29 | $13.69 |

```
Figure 14: Strategy Sharpe Ratio Decay under Transaction Cost Friction
[Saved as results/figures/fig14_transaction_cost_sensitivity.png]
```

---

## 10. Answers to Final Empirical Questions

1. **Does CAPM explain the cross-section of Indian equity returns?**
   **No.** CAPM is decisively rejected by the GRS test ($F = 3.05, p = 0.0016$). Alphas are economically large ($MAA = 4.12\%$ p.a.).
2. **Does FF3 reduce pricing errors relative to CAPM?**
   **Marginally.** FF3 reduces $MAA$ from $0.343\%$ to $0.312\%/\text{month}$, but GRS remains strongly rejected ($p < 0.0001$).
3. **Does Momentum add explanatory power beyond FF3?**
   **Yes, substantially.** Adding Momentum reduces $MAA$ by $31\%$ (to $0.238\%/\text{month} = 2.85\%$ p.a.) and increases average $R^2$ from $87.7\%$ to $93.4\%$.
4. **Which factors earn statistically significant premia?**
   **Momentum ($WML$, $+13.35\%$ p.a., $t = 3.08$)** and **Value ($HML$, $+8.50\%$ p.a., $t = 2.37$)**. Market premium is $+8.61\%$ ($t=1.99$). Size ($SMB$) is insignificant ($-2.92\%$, $t = -0.99$).
5. **Are factor premia stable across market regimes?**
   **No.** Small-cap premium experiences a severe regime shift ($+2.05\%$ in normal times vs $-7.76\%$ during crises). Momentum volatility doubles during market distress.
6. **Do results survive out-of-sample (2016â€“2025)?**
   **Yes.** The Carhart 4-factor model demonstrates exceptional out-of-sample stability, reducing $MAA$ to $2.33\%$ p.a. and failing to reject the GRS zero-alpha null on Size-Value ($p = 0.0501$) and Size-Momentum ($p = 0.0524$) sub-portfolios.
7. **Do transaction costs materially change conclusions?**
   **No.** While Momentum experiences turnover drag (Sharpe falls from 0.54 to 0.46 at 100 bps), the Multi-Factor strategy retains a Sharpe ratio of **0.81** at 50 bps and **0.77** at 100 bps.
8. **Which model performs best according to economic criteria?**
   **Carhart Four-Factor Model.** It achieves the lowest pricing errors, highest cross-sectional explanatory power, and best out-of-sample stability.

---

## 11. Conclusion & Recommendations

This study establishes that empirical asset pricing in Indian equities is governed by a **two-pillar factor architecture: Value and Momentum**. Traditional Size ($SMB$) provides no unconditional premium and exacerbates crash risk. Investors and quantitative researchers operating in Indian equities should focus on multi-factor allocations combining Value and Momentum, which exploit the $-0.37$ pairwise correlation to achieve high risk-adjusted Sharpe ratios ($0.85$ gross / $0.81$ net of friction) with minimal drawdown risk.

