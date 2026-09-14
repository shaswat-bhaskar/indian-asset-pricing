# Class Project Presentation (10-Mark Executive Version)

## Project Title
**Empirical Asset Pricing in Indian Equities: Testing CAPM, Fama–French Three-Factor and Carhart Four-Factor Models (1993–2025)**

---

### Slide 1: Research Question & Motivation
- **Core Question:** Can systematic risk factors explain the cross-section of Indian equity returns, or are excess returns driven by unexplained anomalies?
- **Dataset:** 32-year monthly panel (Nov 1993 – Dec 2025, $T=386$ months) sourced from IIM Ahmedabad (CMIE Prowess DX + RBI).
- **Quality Controls:** Corrected for survivorship bias (vanishing companies), 6-month accounting lag (look-ahead bias elimination), and investability filters (price > ₹1, mcap > 10% median).

---

### Slide 2: Theoretical & Empirical Framework
- **Models Tested:**
  1. **CAPM:** $R_{it} - R_{ft} = \alpha_i + \beta_{iM} (R_{mt} - R_{ft}) + \epsilon_{it}$
  2. **Fama–French 3-Factor (FF3):** Adds Size ($SMB$) and Value ($HML$).
  3. **Carhart 4-Factor (FF4):** Adds Momentum ($WML$).
- **Test Assets:** 10 characteristic-sorted portfolios (6 Size-Value + 4 Size-Momentum).
- **Econometric Inference:** OLS with Newey-West (1987) HAC robust standard errors + Gibbons-Ross-Shanken (1989) joint $F$-test.

---

### Slide 3: Main Empirical Results (Factor Premia)
- **Market Premium ($MF$):** $+8.61\%$ p.a. ($t=1.99, p=0.048$, Sharpe $0.35$).
- **Value Premium ($HML$):** $+8.50\%$ p.a. ($t=2.37, p=0.018$, Sharpe $0.42$) — **Statistically significant**.
- **Size Premium ($SMB$):** $-2.92\%$ p.a. ($t=-0.99, p=0.324$) — **Statistically insignificant; fails in India**.
- **Momentum Premium ($WML$):** $+13.35\%$ p.a. ($t=3.08, p=0.002$, Sharpe $0.54$) — **Strongest factor in the cross-section**.
- **Diversification Property:** $HML$ and $WML$ exhibit a **$-0.37$ pairwise correlation**.

---

### Slide 4: Model Comparison & Pricing Errors
| Metric | CAPM | Fama–French 3-Factor | Carhart 4-Factor |
|---|---|---|---|
| **Mean Absolute Alpha ($MAA$)** | $4.12\%$ p.a. | $3.75\%$ p.a. | **$2.85\%$ p.a. ($-31\%$)** |
| **RMS Alpha** | $4.61\%$ p.a. | $4.75\%$ p.a. | **$3.46\%$ p.a. ($-25\%$)** |
| **Average Explanatory Power ($R^2$)** | $74.2\%$ | $87.7\%$ | **$93.4\%$** |
| **Unexplained Residual Volatility** | $4.88\%$/mo | $3.11\%$/mo | **$2.33\%$/mo** |
| **GRS Test (Full Sample)** | $F=3.05 (p=0.0016)$ | $F=5.38 (p=0.0000)$ | $F=4.29 (p=0.0000)$ |
| **GRS Test (Out-of-Sample Size-Value)** | $F=2.67 (p=0.025)$ | $F=2.99 (p=0.014)$ | **$F=2.30 (p=0.0501)$ [Fails to Reject]** |

---

### Slide 5: Market Regime & Out-of-Sample Robustness
- **Regime Shifts (Gaussian HMM):**
  - Small caps ($SMB$) earn $+2.05\%$ in normal times but collapse to **$-7.76\%$ during crisis regimes**.
  - Momentum ($WML$) volatility doubles from $15.7\%$ to $30.8\%$ during market distress.
- **Out-of-Sample Validation (2016–2025, $N=120$):**
  - Carhart 4-Factor cuts $MAA$ to **$2.33\%$ p.a.**
- **Transaction Costs (50 bps round-trip friction):**
  - Standalone Momentum Net Sharpe: **$0.50$** (Gross $0.54$).
  - Equal-Weighted Multi-Factor ($MF + HML + WML$) Net Sharpe: **$0.81$** (Max DD $-34.8\%$ vs Market's $-75.1\%$).

---

### Slide 6: Key Takeaway & Conclusion
1. **CAPM and FF3 are inadequate** in Indian equities due to large pricing errors and an ineffective Size factor.
2. **Carhart 4-Factor is the dominant empirical specification**, reducing pricing errors by $31\%$ and surviving out-of-sample testing.
3. Combining Value and Momentum creates a highly resilient multi-factor portfolio that withstands real-world transaction cost friction.
