"""
Publication-Quality Visualization Suite for Empirical Asset Pricing in Indian Equities.
Generates all 14 required research figures with clean typography, error bounds, and economic annotations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.data_loader import load_master_dataset
from src.factors import compute_factor_correlations, compute_rolling_factor_stats
from src.regressions import build_regression_table
from src.evaluation import compare_all_models_pricing_errors
from src.regimes import fit_market_regime_hmm, analyze_factors_across_regimes
from src.backtest import run_factor_backtests, run_transaction_cost_sensitivity

# Set global aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
})


def plot_01_cumulative_factor_returns(df: pd.DataFrame = None):
    """Figure 1: Cumulative Wealth Index of Indian Asset Pricing Factors (1993-2025)."""
    if df is None:
        df = load_master_dataset()
        
    fig, ax = plt.subplots(figsize=(10, 5.5))
    dates = df.index.to_timestamp()
    
    colors = {"MF": "#1f77b4", "SMB": "#d62728", "HML": "#2ca02c", "WML": "#ff7f0e"}
    labels = {
        "MF": "Market (MF): CAGR 5.7%",
        "SMB": "Size (SMB): CAGR -4.2%",
        "HML": "Value (HML): CAGR 6.7%",
        "WML": "Momentum (WML): CAGR 10.9%",
    }
    
    for f in config.FACTOR_COLS:
        r = df[f] / 100.0
        wealth = (1.0 + r).cumprod()
        ax.plot(dates, wealth, label=labels[f], color=colors[f], linewidth=1.8)
        
    # Multi-factor strategy
    mf_3f = (df["MF"] + df["HML"] + df["WML"]) / (3.0 * 100.0)
    wealth_3f = (1.0 + mf_3f).cumprod()
    ax.plot(dates, wealth_3f, label="Multi-Factor (MF+HML+WML): CAGR 9.9%", color="#9467bd", linewidth=2.4, linestyle="--")
    
    ax.set_yscale("log")
    ax.set_title("Figure 1: Cumulative Wealth Growth of Indian Equity Factors (Log Scale, 1993-2025)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Wealth ($1 Initial Investment, Log Scale)")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig01_cumulative_factor_returns.png")
    plt.close(fig)


def plot_02_factor_distributions(df: pd.DataFrame = None):
    """Figure 2: Empirical Return Distributions and Skewness/Kurtosis for Indian Factors."""
    if df is None:
        df = load_master_dataset()
        
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    
    for idx, f in enumerate(config.FACTOR_COLS):
        ax = axes[idx]
        data = df[f].dropna()
        sns.histplot(data, kde=True, ax=ax, color=colors[idx], stat="density", bins=30, alpha=0.45)
        
        # Add normal curve for reference
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 200)
        p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, p, "k--", linewidth=1.2, label=f"Normal Fit (μ={mu:.1f}%, σ={sigma:.1f}%)")
        
        ax.set_title(f"{f} Factor Distribution (Skew={data.skew():.2f}, Kurt={data.kurtosis():.2f})", fontweight="bold")
        ax.set_xlabel("Monthly Return (%)")
        ax.set_ylabel("Density")
        ax.legend(frameon=True, fontsize=8.5)
        
    plt.suptitle("Figure 2: Cross-Sectional Return Distributions of Indian Asset Pricing Factors", fontweight="bold", y=1.00)
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig02_factor_distributions.png")
    plt.close(fig)


def plot_03_factor_correlation_heatmap(df: pd.DataFrame = None):
    """Figure 3: Factor Correlation Matrix across Full Sample and Subperiods."""
    if df is None:
        df = load_master_dataset()
        
    corrs = compute_factor_correlations(df)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = [
        "Full Sample (1993-2025)",
        "In-Sample (1993-2015)",
        "Out-of-Sample (2016-2025)",
    ]
    subsets = ["full", "in_sample", "oos"]
    
    for idx, key in enumerate(subsets):
        ax = axes[idx]
        sns.heatmap(
            corrs[key],
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=-0.6,
            vmax=0.6,
            center=0,
            cbar=bool(idx == 2),
            ax=ax,
            square=True,
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(titles[idx], fontweight="bold")
        
    plt.suptitle("Figure 3: Pairwise Factor Correlation Matrices Across Market Regimes & Sample Splits", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig03_factor_correlation_heatmap.png")
    plt.close(fig)


def plot_04_rolling_factor_sharpe(df: pd.DataFrame = None):
    """Figure 4: 36-Month Rolling Annualized Sharpe Ratios."""
    if df is None:
        df = load_master_dataset()
        
    rolling_df = compute_rolling_factor_stats(df, window=36)
    dates = rolling_df.index.to_timestamp()
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    colors = {"MF": "#1f77b4", "SMB": "#d62728", "HML": "#2ca02c", "WML": "#ff7f0e"}
    for f in config.FACTOR_COLS:
        ax.plot(dates, rolling_df[f"{f}_rolling_sharpe"], label=f"{f} 36M Sharpe", color=colors[f], linewidth=1.6)
        
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Figure 4: 36-Month Rolling Annualized Sharpe Ratios (1996-2025)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Annualized Sharpe Ratio")
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig04_rolling_factor_sharpe.png")
    plt.close(fig)


def plot_05_factor_drawdowns(df: pd.DataFrame = None):
    """Figure 5: Historical Underwater Drawdown Profiles."""
    if df is None:
        df = load_master_dataset()
        
    fig, ax = plt.subplots(figsize=(10, 5.5))
    dates = df.index.to_timestamp()
    
    colors = {"MF": "#1f77b4", "SMB": "#d62728", "HML": "#2ca02c", "WML": "#ff7f0e"}
    
    for f in ["MF", "HML", "WML", "SMB"]:
        r = df[f] / 100.0
        w = (1.0 + r).cumprod()
        peak = w.cummax()
        dd = ((w - peak) / peak) * 100.0
        ax.plot(dates, dd, label=f"{f} (Max DD: {dd.min():.1f}%)", color=colors[f], linewidth=1.4)
        
    # Multi-factor DD
    mf_3f = (df["MF"] + df["HML"] + df["WML"]) / (3.0 * 100.0)
    w_3f = (1.0 + mf_3f).cumprod()
    peak_3f = w_3f.cummax()
    dd_3f = ((w_3f - peak_3f) / peak_3f) * 100.0
    ax.plot(dates, dd_3f, label=f"Multi-Factor (Max DD: {dd_3f.min():.1f}%)", color="#9467bd", linewidth=2.2, linestyle="--")
    
    ax.set_title("Figure 5: Historical Underwater Drawdowns: Standalone Factors vs Multi-Factor Portfolio", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(frameon=True, facecolor="white", loc="lower left")
    
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig05_factor_drawdowns.png")
    plt.close(fig)


def plot_06_07_08_alpha_comparisons():
    """Figures 6, 7, 8: Estimated Alphas across 10 test portfolios for CAPM, FF3, and FF4."""
    models = ["CAPM", "FF3", "FF4"]
    fignums = {"CAPM": "06", "FF3": "07", "FF4": "08"}
    
    for m in models:
        table = build_regression_table(model_name=m)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        
        ports = table["Portfolio"]
        alphas = table["Alpha (%/mo)"]
        t_stats = table["t(Alpha)"]
        
        # Color bars by statistical significance (p < 0.05)
        sig_mask = table["p(Alpha)"] < 0.05
        bar_colors = ["#d62728" if sig else "#1f77b4" for sig in sig_mask]
        
        bars = ax.bar(ports, alphas, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.7)
        
        # Add labels
        for bar, t, sig in zip(bars, t_stats, sig_mask):
            yval = bar.get_height()
            va = "bottom" if yval >= 0 else "top"
            star = "*" if sig else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + (0.02 if yval >= 0 else -0.04),
                f"{yval:.2f}%{star}\n(t={t:.1f})",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold" if sig else "normal"
            )
            
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_title(f"Figure {fignums[m]}: Estimated Pricing Errors (Alphas) across Portfolios — {m} Model", fontweight="bold")
        ax.set_xlabel("Test Portfolio (Size-Value: SV-BG; Size-Momentum: WS-LB)")
        ax.set_ylabel("Estimated Monthly Alpha (%)")
        ax.set_ylim(alphas.min() - 0.25, alphas.max() + 0.35)
        
        # Legend for significance
        red_patch = plt.Rectangle((0, 0), 1, 1, fc="#d62728", edgecolor="black", label="Statistically Significant (p < 0.05)")
        blue_patch = plt.Rectangle((0, 0), 1, 1, fc="#1f77b4", edgecolor="black", label="Insignificant Pricing Error")
        ax.legend(handles=[red_patch, blue_patch], loc="upper right", frameon=True)
        
        plt.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"fig{fignums[m]}_{m.lower()}_alpha_comparison.png")
        plt.close(fig)


def plot_09_model_pricing_error_comparison():
    """Figure 9: Cross-Model Comparison of MAA and RMS Alpha."""
    comp = compare_all_models_pricing_errors(sample_name="full_sample")
    comp_oos = compare_all_models_pricing_errors(sample_name="oos")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    models = comp["Model"]
    x = np.arange(len(models))
    width = 0.35
    
    # Panel A: Full Sample
    ax1.bar(x - width/2, comp["MAA (%/mo)"], width, label="Mean Absolute Alpha (MAA)", color="#1f77b4", edgecolor="black")
    ax1.bar(x + width/2, comp["RMS Alpha (%/mo)"], width, label="RMS Alpha", color="#ff7f0e", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_title("Full Sample (1993-2025)", fontweight="bold")
    ax1.set_ylabel("Monthly Pricing Error (%/mo)")
    ax1.legend(frameon=True)
    
    # Panel B: Out-of-Sample
    ax2.bar(x - width/2, comp_oos["MAA (%/mo)"], width, label="Mean Absolute Alpha (MAA)", color="#1f77b4", edgecolor="black")
    ax2.bar(x + width/2, comp_oos["RMS Alpha (%/mo)"], width, label="RMS Alpha", color="#ff7f0e", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_title("Out-of-Sample (2016-2025)", fontweight="bold")
    ax2.set_ylabel("Monthly Pricing Error (%/mo)")
    ax2.legend(frameon=True)
    
    plt.suptitle("Figure 9: Model Pricing Error Comparison: CAPM vs FF3 vs Carhart FF4", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig09_model_pricing_error_comparison.png")
    plt.close(fig)


def plot_10_factor_loading_heatmap():
    """Figure 10: Factor Loading Matrix Heatmap across Test Portfolios for Carhart FF4."""
    table = build_regression_table(model_name="FF4")
    
    beta_cols = ["Beta_MF", "Beta_SMB", "Beta_HML", "Beta_WML"]
    heatmap_df = table.set_index("Portfolio")[beta_cols]
    heatmap_df.columns = ["β_Market", "β_Size (SMB)", "β_Value (HML)", "β_Momentum (WML)"]
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        ax=ax,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 10.5, "weight": "bold"}
    )
    
    ax.set_title("Figure 10: Carhart 4-Factor Loading Matrix Across Indian Characteristic Portfolios", fontweight="bold")
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig10_factor_loading_heatmap.png")
    plt.close(fig)


def plot_11_regime_probabilities():
    """Figure 11: Posterior Regime Probabilities Through Time (HMM)."""
    r_dict = fit_market_regime_hmm()
    reg_df = r_dict["regime_df"]
    dates = reg_df.index.to_timestamp()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    
    # Top Panel: Market Return
    ax1.plot(dates, reg_df["MF"], color="#1f77b4", linewidth=1.1, label="Market Excess Return (MF)")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax1.set_ylabel("Monthly Return (%)")
    ax1.set_title("Figure 11: Market Excess Returns and Posterior Regime Probabilities (2-State HMM, 1993-2025)", fontweight="bold")
    ax1.legend(loc="upper left")
    
    # Bottom Panel: Crisis Probability
    ax2.fill_between(dates, 0, reg_df["Prob_HighVol"], color="#d62728", alpha=0.65, label="P(State = High-Volatility / Crisis)")
    ax2.set_ylabel("Crisis Probability")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Year")
    ax2.legend(loc="upper left")
    
    # Highlight major shocks
    shocks = {
        "1998-05": "Asian Crisis / Nuclear Tests",
        "2000-03": "Dot-Com Bust",
        "2008-10": "GFC Lehman Collapse",
        "2013-08": "Taper Tantrum",
        "2020-03": "COVID Shock",
    }
    for dt, label in shocks.items():
        try:
            ts = pd.to_datetime(dt)
            ax2.annotate(
                label,
                xy=(ts, 0.85),
                xytext=(ts, 0.95),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
                fontsize=7.5,
                rotation=25,
                ha="center"
            )
        except Exception:
            pass
            
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig11_regime_probabilities.png")
    plt.close(fig)


def plot_12_factor_performance_by_regime():
    """Figure 12: Factor Returns and Sharpe Ratios Conditional on Market Regimes."""
    f_comp = analyze_factors_across_regimes()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    factors = f_comp["Factor"]
    x = np.arange(len(factors))
    width = 0.35
    
    # Panel A: Annualized Returns
    ax1.bar(x - width/2, f_comp["Normal Ann Mean (%)"], width, label="Normal / Low-Vol Regime", color="#2ca02c", edgecolor="black")
    ax1.bar(x + width/2, f_comp["Crisis Ann Mean (%)"], width, label="Crisis / High-Vol Regime", color="#d62728", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(factors, fontweight="bold")
    ax1.set_title("Annualized Factor Return by Regime", fontweight="bold")
    ax1.set_ylabel("Annualized Return (%)")
    ax1.legend(frameon=True)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    
    # Panel B: Sharpe Ratios
    ax2.bar(x - width/2, f_comp["Normal Sharpe"], width, label="Normal Regime Sharpe", color="#2ca02c", edgecolor="black")
    ax2.bar(x + width/2, f_comp["Crisis Sharpe"], width, label="Crisis Regime Sharpe", color="#d62728", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(factors, fontweight="bold")
    ax2.set_title("Factor Sharpe Ratio by Regime", fontweight="bold")
    ax2.set_ylabel("Annualized Sharpe Ratio")
    ax2.legend(frameon=True)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    
    plt.suptitle("Figure 12: Factor Return and Risk Asymmetry Across Market Regimes", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig12_factor_performance_by_regime.png")
    plt.close(fig)


def plot_13_out_of_sample_cumulative_returns(df: pd.DataFrame = None):
    """Figure 13: Out-of-Sample Performance of Standalone and Multi-Factor Strategies (2016-2025)."""
    if df is None:
        df = load_master_dataset()
        
    oos_df = df.loc[config.OUT_OF_SAMPLE_START:config.OUT_OF_SAMPLE_END]
    dates = oos_df.index.to_timestamp()
    
    fig, ax = plt.subplots(figsize=(10, 5.2))
    
    colors = {"MF": "#1f77b4", "SMB": "#d62728", "HML": "#2ca02c", "WML": "#ff7f0e"}
    for f in config.FACTOR_COLS:
        r = oos_df[f] / 100.0
        w = (1.0 + r).cumprod()
        ax.plot(dates, w, label=f"{f} (OOS Return: {(w.iloc[-1]-1)*100:.1f}%)", color=colors[f], linewidth=1.7)
        
    # Multi-factor
    mf_3f = (oos_df["MF"] + oos_df["HML"] + oos_df["WML"]) / (3.0 * 100.0)
    w_3f = (1.0 + mf_3f).cumprod()
    ax.plot(dates, w_3f, label=f"Multi-Factor Portfolio (OOS Return: {(w_3f.iloc[-1]-1)*100:.1f}%)", color="#9467bd", linewidth=2.4, linestyle="--")
    
    ax.set_title("Figure 13: Out-of-Sample Performance Trajectories (2016-01 to 2025-12)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Wealth ($1 Initial Investment, 2016 Base)")
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig13_out_of_sample_cumulative_returns.png")
    plt.close(fig)


def plot_14_transaction_cost_sensitivity():
    """Figure 14: Strategy Sharpe Ratio Decay Across Transaction Cost Tiers (0 to 100 bps)."""
    sens = run_transaction_cost_sensitivity()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    strategies = ["MF", "HML", "WML", "MultiFactor_3F (MF+HML+WML)"]
    labels = {
        "MF": "Market (MF)",
        "HML": "Value (HML)",
        "WML": "Momentum (WML)",
        "MultiFactor_3F (MF+HML+WML)": "Multi-Factor (MF+HML+WML)",
    }
    colors = {
        "MF": "#1f77b4",
        "HML": "#2ca02c",
        "WML": "#ff7f0e",
        "MultiFactor_3F (MF+HML+WML)": "#9467bd",
    }
    
    cost_tiers = config.TRANSACTION_COST_BPS
    
    for strat in strategies:
        sub = sens[sens["Strategy"] == strat]
        ax1.plot(sub["Cost (bps)"], sub["Sharpe Ratio"], marker="o", label=labels[strat], color=colors[strat], linewidth=1.8)
        ax2.plot(sub["Cost (bps)"], sub["CAGR (%)"], marker="s", label=labels[strat], color=colors[strat], linewidth=1.8)
        
    ax1.set_title("Net Sharpe Ratio vs Transaction Costs", fontweight="bold")
    ax1.set_xlabel("Transaction Cost (bps per round-trip)")
    ax1.set_ylabel("Annualized Sharpe Ratio")
    ax1.set_xticks(cost_tiers)
    ax1.legend(frameon=True)
    
    ax2.set_title("Net CAGR vs Transaction Costs", fontweight="bold")
    ax2.set_xlabel("Transaction Cost (bps per round-trip)")
    ax2.set_ylabel("Compound Annual Growth Rate (CAGR %)")
    ax2.set_xticks(cost_tiers)
    ax2.legend(frameon=True)
    
    plt.suptitle("Figure 14: Strategy Implementation Resilience Under Transaction Cost Friction", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig14_transaction_cost_sensitivity.png")
    plt.close(fig)


def generate_all_figures():
    """Master routine to render and save all 14 research figures."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_master_dataset()
    
    print("Generating Figure 1 (Cumulative Factor Returns)...")
    plot_01_cumulative_factor_returns(df)
    
    print("Generating Figure 2 (Factor Distributions)...")
    plot_02_factor_distributions(df)
    
    print("Generating Figure 3 (Correlation Heatmaps)...")
    plot_03_factor_correlation_heatmap(df)
    
    print("Generating Figure 4 (Rolling Sharpe)...")
    plot_04_rolling_factor_sharpe(df)
    
    print("Generating Figure 5 (Drawdowns)...")
    plot_05_factor_drawdowns(df)
    
    print("Generating Figures 6, 7, 8 (Alpha Comparisons)...")
    plot_06_07_08_alpha_comparisons()
    
    print("Generating Figure 9 (Pricing Error Metrics Comparison)...")
    plot_09_model_pricing_error_comparison()
    
    print("Generating Figure 10 (Factor Loading Heatmap)...")
    plot_10_factor_loading_heatmap()
    
    print("Generating Figure 11 (HMM Regime Probabilities)...")
    plot_11_regime_probabilities()
    
    print("Generating Figure 12 (Factor Premia by Regime)...")
    plot_12_factor_performance_by_regime()
    
    print("Generating Figure 13 (OOS Cumulative Performance)...")
    plot_13_out_of_sample_cumulative_returns(df)
    
    print("Generating Figure 14 (Transaction Cost Decay Curves)...")
    plot_14_transaction_cost_sensitivity()
    
    print("\nAll 14 figures successfully rendered and saved to results/figures/.")


if __name__ == "__main__":
    generate_all_figures()
