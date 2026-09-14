"""
Master End-to-End Pipeline Execution Script.
Executes data ingestion, econometric regressions, GRS testing, regime modeling,
portfolio backtesting, robustness audits, and visualization rendering.
"""

import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import config
from src.data_loader import download_raw_data, load_master_dataset
from src.cleaning import generate_data_quality_report, generate_subsample_breakdown
from src.factors import compute_factor_descriptive_stats, compute_factor_correlations
from src.portfolios import compute_portfolio_summary_stats
from src.regressions import generate_and_save_all_regression_tables
from src.evaluation import generate_all_pricing_error_reports
from src.grs import run_grs_suite
from src.regimes import fit_market_regime_hmm, analyze_factors_across_regimes
from src.backtest import run_transaction_cost_sensitivity
from src.robustness import run_subsample_robustness_analysis, run_hac_lag_sensitivity
from src.plotting import generate_all_figures


def run_full_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("EMPIRICAL ASSET PRICING IN INDIAN EQUITIES: PIPELINE EXECUTION")
    print("Testing CAPM, Fama-French 3-Factor, and Carhart 4-Factor Models")
    print("=" * 80)
    
    # 1. Ingestion
    print("\n[Step 1/9] Ingesting and syncing raw IIM-A data...")
    download_raw_data()
    master_df = load_master_dataset()
    print(f"  Master dataset synchronized: N={len(master_df)} months ({master_df.index[0]} to {master_df.index[-1]})")
    
    # 2. Data Quality
    print("\n[Step 2/9] Running data quality and missingness audit...")
    generate_data_quality_report()
    generate_subsample_breakdown()
    print("  Data quality reports saved to results/tables/")
    
    # 3. Factor & Portfolio Moments
    print("\n[Step 3/9] Computing factor and portfolio moments...")
    compute_factor_descriptive_stats(master_df)
    compute_factor_correlations(master_df)
    compute_portfolio_summary_stats(master_df)
    print("  Factor moments and correlation structures saved.")
    
    # 4. Time-Series Regressions (CAPM, FF3, FF4)
    print("\n[Step 4/9] Estimating time-series regressions with Newey-West HAC inference...")
    generate_and_save_all_regression_tables()
    print("  Regression tables saved for Full Sample, In-Sample, and Out-of-Sample.")
    
    # 5. Pricing Error Metrics
    print("\n[Step 5/9] Evaluating cross-model pricing errors (MAA, RMS Alpha, Max |Alpha|)...")
    generate_all_pricing_error_reports()
    print("  Pricing error comparison tables saved.")
    
    # 6. GRS Joint Hypothesis Tests
    print("\n[Step 6/9] Running Gibbons-Ross-Shanken (GRS) exact F-tests...")
    run_grs_suite(master_df)
    print("  GRS test suite completed.")
    
    # 7. Regime Analysis (HMM)
    print("\n[Step 7/9] Fitting 2-State Gaussian Hidden Markov Model on Market Factors...")
    r_dict = fit_market_regime_hmm(master_df)
    analyze_factors_across_regimes(r_dict)
    print("  Regime parameters, transition persistence, and conditional moments saved.")
    
    # 8. Portfolio Backtest & Transaction Costs
    print("\n[Step 8/9] Running factor strategy backtests across transaction cost tiers...")
    run_transaction_cost_sensitivity(master_df)
    run_subsample_robustness_analysis(master_df)
    run_hac_lag_sensitivity(master_df)
    print("  Backtest metrics and robustness evaluations saved.")
    
    # 9. Visualizations
    print("\n[Step 9/9] Generating all 14 research figures...")
    generate_all_figures()
    print("  All 14 figures rendered and saved to results/figures/")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
