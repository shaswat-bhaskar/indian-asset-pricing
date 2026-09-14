# Résumé & Portfolio Deliverables: Empirical Asset Pricing in Indian Equities

---

## 1. Final 3-Bullet Résumé Version (78 words total)

**Empirical Asset Pricing in Indian Equities** \hfill *Python, Pandas, NumPy, Statsmodels*  
*Focus: CAPM, Fama–French Factor Models & Portfolio Analysis*
* Constructed a 32-year monthly panel (1993–2025) of Indian equity risk factors and 10 characteristic-sorted portfolios, enforcing survivorship corrections and 6-month accounting lags to eliminate look-ahead bias.
* Evaluated CAPM, Fama–French 3-Factor, and Carhart 4-Factor models via HAC-robust regressions; Carhart reduced mean absolute pricing errors by 31% to 2.85% p.a. ($R^2=93.4\%$).
* Found that Value (+8.50% p.a., $t=2.37$) and Momentum (+13.35% p.a., $t=3.08$) drive cross-sectional returns while Size is ineffective (-2.92% p.a.); combining them yields a net Sharpe ratio of 0.81.

---

## 2. One-Sentence LinkedIn / GitHub Tagline

> Built an econometric asset pricing pipeline analyzing 32 years of Indian equity data (1993–2025) to test CAPM, Fama–French, and Carhart factor models across 10 characteristic portfolios under transaction cost friction.

---

## 3. 30-Second Interview Explanation

> "I investigated whether systematic risk factors explain the cross-section of Indian equity returns using a 32-year survivorship-bias-corrected panel from 1993 to 2025.
> 
> I found that the traditional CAPM and Fama-French 3-factor models fail because the Size premium is completely absent in India ($-2.92\%$ p.a.). However, Value ($+8.50\%$ p.a.) and Momentum ($+13.35\%$ p.a.) are both economically and statistically significant.
> 
> Adding Momentum into the Carhart four-factor model reduced pricing errors by 31% and survived 10-year out-of-sample testing. Combining Value and Momentum in an equal-weighted multi-factor portfolio yielded an annualized Sharpe ratio of 0.81 net of 50 basis points transaction costs, cutting maximum drawdown in half relative to the market index."

---

## 4. Detailed Project Description for GitHub README

### Project Overview
An empirical asset pricing investigation testing whether systematic multi-factor models can explain the cross-section of Indian equity returns over a 32-year period (November 1993 to December 2025, $T=386$ months, 8,004 daily observations). Sourced from the IIM Ahmedabad factor database (CMIE Prowess DX + RBI), the study implements rigorous econometrics including Newey-West HAC inference, Gibbons-Ross-Shanken (GRS) exact $F$-tests, Gaussian Hidden Markov Models (HMM) for regime dependence, and turnover-adjusted backtesting across 0–100 bps transaction cost tiers.

### Key Quantitative Results
- **Factor Risk Premia:**
  - **Market ($MF$):** $+8.61\%$ p.a. ($t=1.99, p=0.048$, Sharpe $0.35$).
  - **Value ($HML$):** $+8.50\%$ p.a. ($t=2.37, p=0.018$, Sharpe $0.42$).
  - **Momentum ($WML$):** $+13.35\%$ p.a. ($t=3.08, p=0.002$, Sharpe $0.54$).
  - **Size ($SMB$):** $-2.92\%$ p.a. ($t=-0.99, p=0.324$, insignificant).
- **Pricing Errors & Explanatory Power:**
  - CAPM produces a Mean Absolute Alpha ($MAA$) of $4.12\%$ p.a. and is rejected by GRS ($p=0.0016$).
  - Carhart 4-Factor reduces $MAA$ to $2.85\%$ p.a. ($-31\%$), increases average $R^2$ to $93.4\%$, and cuts residual volatility to $2.33\%$/month.
- **Out-of-Sample & Regime Resilience:**
  - Out-of-Sample (2016–2025, $N=120$): Carhart 4-factor reduces $MAA$ to $2.33\%$ p.a. and is the only model failing to reject the GRS zero-alpha hypothesis on Size-Value ($p=0.0501$) and Size-Momentum ($p=0.0524$) portfolios.
  - Multi-Factor Combination ($MF + HML + WML$): Delivers an annualized Sharpe ratio of **$0.81$ net of $50\text{ bps}$ transaction costs** with a maximum drawdown of $-34.8\%$ (vs. $-75.1\%$ for the Market).
