"""
Market Regime Analysis Engine using Gaussian Hidden Markov Models (HMM).
Identifies Low-Volatility (Normal) vs High-Volatility (Crisis) regimes,
and assesses the stability of factor premia and correlation structures across regimes.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy import stats
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from src.data_loader import load_master_dataset


def fit_market_regime_hmm(
    df: pd.DataFrame = None,
    n_components: int = 2,
    random_state: int = config.RANDOM_SEED
) -> dict:
    """
    Fit a 2-state Gaussian Hidden Markov Model on monthly Market Factor (MF) excess returns.
    Sorts states so that State 0 = Low-Volatility / Normal, State 1 = High-Volatility / Crisis.
    """
    if df is None:
        df = load_master_dataset()
        
    mf_series = df["MF"].dropna()
    X = mf_series.values.reshape(-1, 1)
    dates = mf_series.index
    
    # Fit HMM
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=1000,
        random_state=random_state,
    )
    model.fit(X)
    
    # Posterior state probabilities
    post_probs = model.predict_proba(X)
    hidden_states = model.predict(X)
    
    means = model.means_.flatten()
    vars_ = model.covars_.flatten()
    stds = np.sqrt(vars_)
    
    # Identify which state is Low-Vol vs High-Vol
    if stds[0] > stds[1]:
        # Swap states so state 0 is Low-Vol, state 1 is High-Vol
        state_map = {0: 1, 1: 0}
        hidden_states = np.array([state_map[s] for s in hidden_states])
        post_probs = post_probs[:, [1, 0]]
        means = means[[1, 0]]
        stds = stds[[1, 0]]
        trans_mat = model.transmat_[[1, 0], :][:, [1, 0]]
    else:
        trans_mat = model.transmat_
        
    # Transition persistence and expected durations
    p00 = trans_mat[0, 0]
    p11 = trans_mat[1, 1]
    dur0 = 1.0 / (1.0 - p00) if p00 < 1.0 else np.inf
    dur1 = 1.0 / (1.0 - p11) if p11 < 1.0 else np.inf
    
    regime_df = pd.DataFrame(
        {
            "MF": mf_series.values,
            "Regime": hidden_states,
            "Prob_LowVol": post_probs[:, 0],
            "Prob_HighVol": post_probs[:, 1],
        },
        index=dates,
    )
    
    # Merge factor returns
    for f in config.FACTOR_COLS:
        regime_df[f] = df.loc[dates, f]
        
    # Regime characteristics table
    char_table = pd.DataFrame({
        "Regime": ["Low-Volatility (Normal)", "High-Volatility (Crisis)"],
        "Monthly Mean MF (%)": [round(means[0], 2), round(means[1], 2)],
        "Annualized Mean MF (%)": [round(means[0] * 12, 2), round(means[1] * 12, 2)],
        "Monthly Std MF (%)": [round(stds[0], 2), round(stds[1], 2)],
        "Annualized Std MF (%)": [round(stds[0] * np.sqrt(12), 2), round(stds[1] * np.sqrt(12), 2)],
        "Stay Probability (P_ii)": [round(p00, 3), round(p11, 3)],
        "Expected Duration (Months)": [round(dur0, 1), round(dur1, 1)],
        "Months in Regime": [(hidden_states == 0).sum(), (hidden_states == 1).sum()],
        "Proportion of Sample (%)": [
            round((hidden_states == 0).mean() * 100, 1),
            round((hidden_states == 1).mean() * 100, 1),
        ],
    })
    
    # Save table
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    char_table.to_csv(config.TABLES_DIR / "regime_characteristics.csv", index=False)
    
    with open(config.TABLES_DIR / "regime_characteristics.md", "w") as f:
        f.write("# HMM Market Regime Characteristics (1993-11 to 2025-12)\n\n")
        f.write(char_table.to_markdown(index=False))
        f.write("\n\n*Estimated via 2-state Gaussian Hidden Markov Model on Monthly Market Excess Returns (MF).*\n")
        
    return {
        "model": model,
        "regime_df": regime_df,
        "char_table": char_table,
        "trans_mat": trans_mat,
        "means": means,
        "stds": stds,
    }


def analyze_factors_across_regimes(regime_dict: dict = None) -> pd.DataFrame:
    """
    Compare empirical factor premia and volatility across Low-Vol vs High-Vol regimes.
    Conducts two-sample t-tests for equality of means.
    """
    if regime_dict is None:
        regime_dict = fit_market_regime_hmm()
        
    reg_df = regime_dict["regime_df"]
    
    rows = []
    for f in config.FACTOR_COLS:
        norm = reg_df[reg_df["Regime"] == 0][f].dropna()
        cris = reg_df[reg_df["Regime"] == 1][f].dropna()
        
        # Test mean difference: H0: mu_norm = mu_cris
        t_stat, p_val = stats.ttest_ind(norm, cris, equal_var=False)
        
        rows.append({
            "Factor": f,
            "Normal Mean (%/mo)": round(norm.mean(), 2),
            "Normal Ann Mean (%)": round(norm.mean() * 12, 2),
            "Normal Ann Vol (%)": round(norm.std(ddof=1) * np.sqrt(12), 2),
            "Normal Sharpe": round((norm.mean() * 12) / (norm.std(ddof=1) * np.sqrt(12)), 2),
            "Crisis Mean (%/mo)": round(cris.mean(), 2),
            "Crisis Ann Mean (%)": round(cris.mean() * 12, 2),
            "Crisis Ann Vol (%)": round(cris.std(ddof=1) * np.sqrt(12), 2),
            "Crisis Sharpe": round((cris.mean() * 12) / (cris.std(ddof=1) * np.sqrt(12)), 2),
            "Diff (Norm - Cris)": round((norm.mean() - cris.mean()) * 12, 2),
            "t-stat (Diff)": round(t_stat, 2),
            "p-value (Diff)": round(p_val, 4),
        })
        
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(config.TABLES_DIR / "factor_premia_by_regime.csv", index=False)
    
    with open(config.TABLES_DIR / "factor_premia_by_regime.md", "w") as f:
        f.write("# Factor Premia and Risk Across Market Regimes\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n\n*Normal = Low-Volatility state; Crisis = High-Volatility state. Welch's t-test with unequal variances.*\n")
        
    # Correlations by regime
    corr_norm = reg_df[reg_df["Regime"] == 0][config.FACTOR_COLS].corr()
    corr_cris = reg_df[reg_df["Regime"] == 1][config.FACTOR_COLS].corr()
    
    corr_norm.to_csv(config.TABLES_DIR / "factor_correlations_normal_regime.csv")
    corr_cris.to_csv(config.TABLES_DIR / "factor_correlations_crisis_regime.csv")
    
    with open(config.TABLES_DIR / "factor_correlations_by_regime.md", "w") as f:
        f.write("# Factor Correlations by Market Regime\n\n")
        f.write("### Normal / Low-Volatility Regime\n\n")
        f.write(corr_norm.round(3).to_markdown())
        f.write("\n\n### Crisis / High-Volatility Regime\n\n")
        f.write(corr_cris.round(3).to_markdown())
        
    return comp_df


if __name__ == "__main__":
    r_dict = fit_market_regime_hmm()
    print("Regime Characteristics:")
    print(r_dict["char_table"].to_string(index=False))
    
    f_comp = analyze_factors_across_regimes(r_dict)
    print("\nFactor Premia Across Regimes:")
    print(f_comp.to_string(index=False))
