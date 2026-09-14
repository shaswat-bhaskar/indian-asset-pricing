# Interview Preparation Guide: Empirical Asset Pricing in Indian Equities

This document provides rigorous, graduate-level interview responses spanning theoretical asset pricing, econometrics, empirical finance, and portfolio implementation, grounded in the actual results obtained from our 32-year study of Indian equities (1993–2025).

---

## 1. Theoretical Asset Pricing

### Q1: What is a Stochastic Discount Factor (SDF) and how does it relate to expected returns?
**Answer:**
The SDF, denoted $m_{t+1}$, is a random variable representing the marginal rate of substitution between consumption today and consumption in state $s$ tomorrow:
$$m_{t+1} = \beta \frac{u'(c_{t+1})}{u'(c_t)}$$
Under the law of one price and absence of arbitrage, the fundamental asset pricing equation states:
$$P_t = \mathbb{E}_t [m_{t+1} X_{t+1}]$$
For gross returns $R_{i,t+1} = X_{i,t+1}/P_{i,t}$, this simplifies to $\mathbb{E}_t [m_{t+1} R_{i,t+1}] = 1$. Expanding the expectation:
$$1 = \mathbb{E}_t[m_{t+1}] \mathbb{E}_t[R_{i,t+1}] + \text{Cov}_t(m_{t+1}, R_{i,t+1})$$
Since for a risk-free asset $R_f = 1/\mathbb{E}_t[m_{t+1}]$, substituting yields the general risk-return relation:
$$\mathbb{E}_t[R_{i,t+1}] - R_f = - R_f \text{Cov}_t(m_{t+1}, R_{i,t+1}) = \left( \frac{\text{Cov}_t(R_{i,t+1}, -m_{t+1})}{\text{Var}(m_{t+1})} \right) \left( \frac{\text{Var}(m_{t+1})}{\mathbb{E}_t[m_{t+1}]} \right) = \beta_{i,m} \lambda_m$$
**Intuition:** An asset commands an expected return premium if and only if its payoff co-varies negatively with marginal utility ($m_{t+1}$). Assets that pay off poorly when marginal utility is high (e.g. during market crashes or recessions) are risky and require higher expected returns to induce risk-averse investors to hold them.

---

### Q2: Why does CAPM price only covariance with the market portfolio rather than total variance?
**Answer:**
Under mean-variance optimization and homogeneous expectations (Markowitz-Sharpe-Lintner), all investors allocate between the risk-free rate and the unique tangency portfolio (Two-Fund Separation). In equilibrium, the market portfolio is the tangency portfolio.
An asset's total risk decomposes into:
$$\text{Var}(R_i) = \beta_i^2 \text{Var}(R_m) + \text{Var}(\epsilon_i)$$
Because idiosyncratic risk $\epsilon_i$ is uncorrelated across assets, an investor holding the diversified market portfolio achieves $\lim_{N \to \infty} \text{Var}(\sum w_i \epsilon_i) = 0$. Since idiosyncratic risk can be eliminated at zero cost via diversification, the market does not compensate investors for bearing it. Only systematic covariance with the aggregate wealth portfolio ($\beta_i = \text{Cov}(R_i, R_m)/\text{Var}(R_m)$) cannot be diversified away and must earn an equilibrium risk premium.

---

### Q3: What is the economic intuition behind SMB (Size) and HML (Value)? Are they risk factors or behavioral anomalies?
**Answer:**
- **SMB (Small Minus Big):** Captures the return spread of small-capitalization firms over large-capitalization firms.
  - *Risk interpretation (Fama-French 1993):* Small firms suffer higher distress risk, less access to external capital markets, and lower liquidity during credit contractions.
  - *Empirical Indian Finding:* In India, SMB earns a **statistically insignificant -2.92% p.a.** ($t=-0.99$). In normal times, small caps earn +2.05% p.a., but during crisis regimes, they plunge to **-7.76% p.a.** Small-cap exposure acts as severe downside leverage rather than a structural positive drift.
- **HML (High Minus Low):** Captures the spread between high book-to-market (value) and low book-to-market (growth) stocks.
  - *Risk interpretation (Zhang 2005 / Fama-French):* Value firms possess inflexible, high-capital-intensity assets and suffer high adjustment costs when downsizing during macroeconomic downturns.
  - *Behavioral interpretation (Lakonishok, Shleifer & Vishny 1994):* Investors systematically overextrapolate recent growth trends, overpricing glamour/growth stocks and underpricing unglamorous distressed value stocks.
  - *Empirical Indian Finding:* HML earns a robust **+8.50% p.a.** ($t=2.37, p=0.018$), demonstrating that the value premium is economically large in Indian equities.

---

### Q4: Why does Momentum (WML) exist, and why is it distinct from traditional risk factors?
**Answer:**
Momentum (Jegadeesh & Titman, 1993; Carhart, 1997) sorts on intermediate prior returns ($t-12$ to $t-2$).
- *Behavioral drivers:* Underreaction to corporate fundamental news (analyst conservatism, disposition effect) causes prices to drift toward intrinsic value over 3–12 months. Once a trend establishes, self-reinforcing herd behavior and overreaction amplify momentum until it eventually reverses over 3–5 years.
- *Risk drivers:* Momentum is not easily explained by static macroeconomic state variables. It delivers high returns (+13.35% p.a. in India, $t=3.08$) but suffers from rare, catastrophic "momentum crashes" (Barroso & Santa-Clara, 2015; Daniel & Moskowitz, 2016) during market turning points.

---

## 2. Econometrics and Empirical Asset Pricing

### Q5: What is an alpha ($\alpha_i$), and why is a high $R^2$ insufficient to validate a factor model?
**Answer:**
In a time-series regression $R_{i,t} - R_{f,t} = \alpha_i + \boldsymbol{\beta}_i' \mathbf{F}_t + \epsilon_{i,t}$, the intercept $\alpha_i$ represents the **pricing error** — the average excess return unexplained by the asset's exposure to systematic risk factors.
- Under an exact asset pricing model, the factors span the mean-variance efficient frontier of test assets, implying $\alpha_i = 0$ for all $i$.
- **Why high $R^2$ is misleading:** $R^2 = 1 - \frac{\text{Var}(\epsilon_i)}{\text{Var}(R_i)}$ measures the fraction of *time-series variance* explained by the factors. Adding spurious correlated factors increases $R^2$ mechanically without reducing the pricing error $\alpha_i$. A model can have $R^2 = 95\%$ but still exhibit large, statistically significant alphas (e.g. $\alpha = 0.6\%/\text{month} = 7.2\%\text{ p.a.}$, rejecting the pricing model).

---

### Q6: What is the Gibbons-Ross-Shanken (GRS) test, and how is it derived?
**Answer:**
The GRS (1989) statistic is the exact finite-sample $F$-test of the joint null hypothesis:
$$H_0: \boldsymbol{\alpha} = [\alpha_1, \alpha_2, \dots, \alpha_N]' = \mathbf{0}$$
The test statistic is:
$$GRS = \frac{T - N - K}{N} \left( 1 + \hat{\boldsymbol{\mu}}_F' \hat{\boldsymbol{\Sigma}}_F^{-1} \hat{\boldsymbol{\mu}}_F \right)^{-1} \hat{\boldsymbol{\alpha}}' \hat{\boldsymbol{\Sigma}}_\epsilon^{-1} \hat{\boldsymbol{\alpha}} \sim F_{N, T - N - K}$$
**Economic Intuition:**
The term $\hat{\boldsymbol{\mu}}_F' \hat{\boldsymbol{\Sigma}}_F^{-1} \hat{\boldsymbol{\mu}}_F = \text{Sh}^2(F)$ is the squared Sharpe ratio of the optimal factor tangency portfolio.
The term $\hat{\boldsymbol{\alpha}}' \hat{\boldsymbol{\Sigma}}_\epsilon^{-1} \hat{\boldsymbol{\alpha}}$ measures the improvement in the squared Sharpe ratio attained by adding test assets to the factor frontier:
$$\text{Sh}^2(F + \text{Assets}) - \text{Sh}^2(F) = \hat{\boldsymbol{\alpha}}' \hat{\boldsymbol{\Sigma}}_\epsilon^{-1} \hat{\boldsymbol{\alpha}}$$
Therefore, GRS tests whether the factors alone are mean-variance efficient relative to the universe of test assets. If $GRS$ rejects $H_0$, an investor can construct an active portfolio with a significantly higher Sharpe ratio than the factor benchmark.

---

### Q7: Why must we use Newey-West HAC standard errors for financial time series?
**Answer:**
Standard OLS assumes independent and identically distributed errors: $\mathbb{E}[\boldsymbol{\epsilon} \boldsymbol{\epsilon}' | \mathbf{X}] = \sigma^2 \mathbf{I}$.
Financial asset return residuals violate both assumptions:
1. **Heteroskedasticity:** Volatility clustering (periods of high turbulence followed by calm).
2. **Autocorrelation:** Microstructural friction, stale prices, or lag dynamics generate serial correlation in residuals.
If OLS standard errors are used, the covariance matrix $\text{Var}(\hat{\boldsymbol{\beta}}) = (X'X)^{-1} X' \Omega X (X'X)^{-1}$ is misestimated, leading to severely inflated $t$-statistics and false discoveries.
The Newey-West (1987) estimator guarantees positive semi-definiteness using a Bartlett triangular weighting kernel:
$$\hat{\boldsymbol{\Omega}} = \hat{\boldsymbol{\Gamma}}_0 + \sum_{j=1}^L \left( 1 - \frac{j}{L + 1} \right) (\hat{\boldsymbol{\Gamma}}_j + \hat{\boldsymbol{\Gamma}}_j')$$
where $\hat{\boldsymbol{\Gamma}}_j = \frac{1}{T} \sum_{t=j+1}^T \mathbf{x}_t \mathbf{x}_{t-j}' e_t e_{t-j}$.

---

### Q8: What are Survivorship Bias and Look-Ahead Bias, and how did your empirical design control for them?
**Answer:**
- **Survivorship Bias:** Occurs when a dataset contains only companies that currently exist, omitting bankrupt, liquidated, or delisted firms. This inflates historical average returns and suppresses risk.
  - *Our Control:* Used the IIMA academic library, which tracks all historic listed firms on BSE/NSE from 1993 to 2025 and applies an explicit algorithm correcting for the mid-1990s "vanishing companies" era.
- **Look-Ahead Bias:** Occurs when conditioning portfolio formation on information that was not yet publicly known at formation time.
  - *Our Control:* Indian corporate fiscal years end on March 31. Annual financial statements are finalized by September 30. All size and value characteristic portfolios are reconstituted on **October 1**, guaranteeing a 6-month information publication lag.

---

## 3. Empirical Findings in Indian Equities

### Q9: Summarize your key empirical findings: Does CAPM or Fama-French work in India?
**Answer:**
1. **CAPM Fails:** CAPM exhibits large pricing errors (Mean Absolute Alpha = 4.12% p.a., Max Alpha = 6.72% p.a.). GRS rejects CAPM across the full sample ($p=0.0016$).
2. **Value (HML) is Robust:** High B/M stocks earn a statistically significant premium of +8.50% p.a. ($t=2.37$).
3. **Size (SMB) is Ineffective:** Small-cap premium is negative (-2.92% p.a., $t=-0.99$) and suffers severely during market stress (-7.76% p.a.).
4. **Momentum (WML) is Dominant:** Momentum generates +13.35% p.a. ($t=3.08$, Sharpe 0.54), providing the strongest explanatory and risk-adjusted performance.
5. **Carhart 4-Factor Outperforms:** Adding Momentum to FF3 reduces Mean Absolute Alpha by 31% (down to 2.85% p.a.) and cuts residual variance by over 50%. In the Out-of-Sample period (2016–2025), FF4 is the only model that fails to reject the GRS zero-alpha null on Size-Value ($p=0.0501$) and Size-Momentum ($p=0.0524$) sub-portfolios.

---

### Q10: How do transaction costs impact factor strategies in practice?
**Answer:**
Because factor models assume friction-free rebalancing, academic premia can be severely degraded by transaction costs:
- **Momentum (WML):** Requires high monthly turnover (~18%/month). While gross CAGR is 10.85% (Sharpe 0.54), at 50 bps round-trip transaction costs, net CAGR is 9.67% (Sharpe 0.50), and at 100 bps it drops to 8.50% (Sharpe 0.46). Momentum remains economically profitable even after high turnover costs.
- **Value (HML):** Annual rebalancing yields low turnover (~5%/month). Net CAGR degrades modestly from 6.68% (0 bps) to 6.38% (50 bps).
- **Multi-Factor Combination ($MF + HML + WML$):** Generates gross CAGR of 9.86% with a Sharpe of **0.85** (Max Drawdown -33.3% vs Market's -75.1%). Under 50 bps friction, net CAGR is **9.34%** and net Sharpe is **0.81**, retaining strong risk-adjusted alpha due to negative factor correlation ($r(HML, WML) = -0.37$).
