# Data Provenance and Quality Documentation

## 1. Primary Source Identification

All factor series, test portfolio returns, and breakpoint time series used in this study are sourced from the **Indian Institute of Management, Ahmedabad (IIMA) Fama-French & Momentum Factors Data Library for the Indian Market**.

- **URL:** [https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/](https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/)
- **Maintainers:** Prof. Sobhesh K. Agarwalla, Prof. Joshy Jacob, and Prof. Jayanth R. Varma
- **Foundational Paper:** Agarwalla, S. K., Jacob, J., & Varma, J. R. (2013), *"Four factor model in Indian equities market"*, IIMA Working Paper No. 2013-09-05.
- **Underlying Database:** CMIE Prowess DX (Centre for Monitoring Indian Economy)
- **Sample Coverage:** October 1993 to December 2025 (387 monthly observations / 8,004 daily observations)

---

## 2. Institutional Framework & Construction Methodology

### A. Universe & Coverage
The database covers all companies listed on the **Bombay Stock Exchange (BSE)** and the **National Stock Exchange (NSE)**. For securities dually listed on both exchanges, price and volume series are extracted from the exchange exhibiting the higher aggregate trading liquidity.

### B. Investability & Liquidity Screening
To ensure that empirical factor portfolios represent actionable, investable cross-sections rather than microstructural artifacts, two rigorous filters are imposed:
1. **Market Capitalization Filter:** Stocks whose market capitalization during the preceding year falls below 10% of the median market capitalization across the universe are excluded.
2. **Penny Stock / Price Filter:** Stocks whose median price during the preceding year is strictly less than INR 1.00 are excluded.

Together, these filters eliminate approximately 26.8% of micro-cap/illiquid securities, which contribute negligible capitalization but generate substantial microstructure noise.

### C. Breakpoint Methodology
- **Size Breakpoint:** 90th percentile of market capitalization. In emerging markets such as India, market capitalization is heavily skewed; using a 50th percentile (median) as in the US CRSP universe would classify large institutional firms into the small-cap bucket. The 90th percentile accurately separates the top liquid large caps from the broader mid/small-cap universe.
- **Value (Book-to-Market) Breakpoints:** 30th and 70th percentiles of Book Equity to Market Equity ($BE/ME$).
- **Momentum Breakpoints:** 30th and 70th percentiles of cumulative past return from month $t-12$ to $t-2$.

### D. Look-Ahead Bias Mitigation
In India, the financial year concludes on March 31. Financial statements are audited and submitted under statutory disclosure timelines by September 30. Therefore, portfolios formed on size and value are reconstituted annually on **October 1** of year $t$ using accounting data for the fiscal year ending March 31 of year $t$. This guarantees a minimum information lag of 6 months, eliminating look-ahead bias.

### E. Survivorship Bias Mitigation
The IIMA series explicitly applies an algorithmic correction for the mid-1990s "vanishing companies" phenomenon in India, where hundreds of newly listed firms ceased reporting. Delisted firms are included up to their delisting date with proceeds reinvested across the remaining portfolio.

---

## 3. Variable Summary and Storage

| File Name | Description | Rows | Frequency |
|---|---|---|---|
| `2025-12_FourFactors_and_Market_Returns_Monthly_SurvivorshipBiasAdjusted.csv` | Monthly Factor returns (MF, SMB, HML, WML) and Risk-Free Rate (RF) | 387 | Monthly |
| `2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv` | Daily Factor returns (MF, SMB, HML, WML) and Risk-Free Rate (RF) | 8,004 | Daily |
| `2025-12_Size_Value_Portfolio_Returns_Monthly_SurvivorshipBiasAdjusted.csv` | Monthly returns for 6 Size-Value portfolios (BV, BN, BG, SV, SN, SG) | 387 | Monthly |
| `2025-12_Size_Momentum_Portfolio_Returns_Monthly_SurvivorshipBiasAdjusted.csv` | Monthly returns for 4 Size-Momentum portfolios (WB, WS, LB, LS) | 387 | Monthly |
| `2025-12_Size_and_Value_Break_Points_for_Size_Value_Portfolios.csv` | Historical annual breakpoints for Size and B/M | 35 | Annual |
| `2025-12_Size_and_Momentum_Break_Points_for_Size_Momentum_Portfolios.csv` | Historical monthly breakpoints for Size and Momentum | 423 | Monthly |

---

## 4. Known Structural Characteristics & Limitations

1. **Big-Value Portfolio Sparsity:**
   In 154 months out of 387 (39.8%), the Big-Value ($BV$) portfolio contains fewer than 5 stocks and is set to missing ($NaN$) by the IIMA library to prevent un-diversified portfolio noise. As noted by Agarwalla et al. (2013), the value premium in India is heavily concentrated in small-cap stocks. For GRS joint hypothesis testing and multivariate regressions requiring a balanced panel, the standard 9-portfolio universe (excluding $BV$) or pairwise-complete subsets are analyzed and reported.

2. **Absence of Public FF5 Factors (RMW / CMA):**
   The public IIMA library provides the 4-factor Carhart model. Operating profitability and total asset growth for constructing Fama-French (2015) 5-factor series require paid subscription access to CMIE Prowess DX.
