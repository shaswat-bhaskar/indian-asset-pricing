"""
Central configuration for Empirical Asset Pricing in Indian Equities project.
Defines paths, dates, factor names, test portfolios, and econometric parameters.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

# Sample Dates
FULL_SAMPLE_START = "1993-11"
FULL_SAMPLE_END = "2025-12"

# In-Sample & Out-of-Sample Split
IN_SAMPLE_START = "1993-11"
IN_SAMPLE_END = "2015-12"
OUT_OF_SAMPLE_START = "2016-01"
OUT_OF_SAMPLE_END = "2025-12"

# Sub-samples for Robustness
SUBSAMPLE_EARLY_START = "1993-11"
SUBSAMPLE_EARLY_END = "2008-12"
SUBSAMPLE_MODERN_START = "2009-01"
SUBSAMPLE_MODERN_END = "2025-12"

# Crisis Periods
CRISIS_GFC_START = "2007-10"
CRISIS_GFC_END = "2009-03"
CRISIS_COVID_START = "2020-01"
CRISIS_COVID_END = "2020-12"

# Factor Definitions
FACTOR_COLS = ["MF", "SMB", "HML", "WML"]  # Market Factor (Mkt-Rf), Size, Value, Momentum
RISK_FREE_COL = "RF"

# Test Portfolios
# 6 Size-Value portfolios
SIZE_VALUE_PORTFOLIOS = [
    "SV",  # Small-Value
    "SN",  # Small-Neutral
    "SG",  # Small-Growth
    "BV",  # Big-Value (note: contains missing data due to <5 stocks filter)
    "BN",  # Big-Neutral
    "BG",  # Big-Growth
]

# 4 Size-Momentum portfolios
SIZE_MOMENTUM_PORTFOLIOS = [
    "WS",  # Winners-Small
    "WB",  # Winners-Big
    "LS",  # Losers-Small
    "LB",  # Losers-Big
]

ALL_TEST_PORTFOLIOS = SIZE_VALUE_PORTFOLIOS + SIZE_MOMENTUM_PORTFOLIOS

# Complete Portfolios (excluding BV if needed for balanced panel)
BALANCED_TEST_PORTFOLIOS = [p for p in ALL_TEST_PORTFOLIOS if p != "BV"]

# Econometric Parameters
HAC_LAGS_MONTHLY = 3  # Newey-West lag truncation for monthly data: int(4*(T/100)^(2/9)) ~ 3-4
SIGNIFICANCE_LEVEL = 0.05
RANDOM_SEED = 42

# Backtest Parameters
ANNUALIZATION_FACTOR = 12  # Monthly returns to annual
TRANSACTION_COST_BPS = [0, 25, 50, 100]  # Basis points per round-trip trade
