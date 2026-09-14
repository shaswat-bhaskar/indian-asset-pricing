"""
Data loading and preprocessing module for IIM-A Indian asset pricing dataset.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import urllib.request
import sys

# Ensure config is importable
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config

IIMA_BASE_URL = "https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/DATA/"

FILES_DICT = {
    "factors_monthly": "2025-12_FourFactors_and_Market_Returns_Monthly_SurvivorshipBiasAdjusted.csv",
    "factors_daily": "2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv",
    "portfolios_size_value": "2025-12_Size_Value_Portfolio_Returns_Monthly_SurvivorshipBiasAdjusted.csv",
    "portfolios_size_mom": "2025-12_Size_Momentum_Portfolio_Returns_Monthly_SurvivorshipBiasAdjusted.csv",
    "breakpoints_size_value": "2025-12_Size_and_Value_Break_Points_for_Size_Value_Portfolios.csv",
    "breakpoints_size_mom": "2025-12_Size_and_Momentum_Break_Points_for_Size_Momentum_Portfolios.csv",
}


def download_raw_data(force=False):
    """Download raw datasets from IIM Ahmedabad data library if not already present."""
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in FILES_DICT.items():
        out_path = config.RAW_DATA_DIR / filename
        if not out_path.exists() or force:
            url = IIMA_BASE_URL + filename
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, out_path)
            print(f"  Saved to {out_path} ({out_path.stat().st_size:,} bytes)")
        else:
            print(f"File exists: {filename}")


def load_raw_factors_monthly() -> pd.DataFrame:
    """Load and clean monthly factor returns."""
    file_path = config.RAW_DATA_DIR / FILES_DICT["factors_monthly"]
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Parse Date
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    df.set_index("Date", inplace=True)
    df.index = df.index.to_period("M")
    
    # Drop first month where MF and RF are NaN (1993-10)
    df = df.dropna(subset=["MF", "RF"])
    return df


def load_raw_factors_daily() -> pd.DataFrame:
    """Load and clean daily factor returns."""
    file_path = config.RAW_DATA_DIR / FILES_DICT["factors_daily"]
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.dropna(subset=["MF", "RF"])
    return df


def load_raw_portfolios_size_value() -> pd.DataFrame:
    """
    Load monthly returns for 6 Size-Value portfolios.
    Columns: SV (Small-Value), SN (Small-Neutral), SG (Small-Growth),
             BV (Big-Value), BN (Big-Neutral), BG (Big-Growth)
    """
    file_path = config.RAW_DATA_DIR / FILES_DICT["portfolios_size_value"]
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Rename columns to standard abbreviations
    col_mapping = {
        "Date": "Date",
        "Portfolios BV returns (%)": "BV",
        "Portfolios BN returns (%)": "BN",
        "Portfolios BG returns (%)": "BG",
        "Portfolios SV returns (%)": "SV",
        "Portfolios SN returns (%)": "SN",
        "Portfolios SG returns (%)": "SG",
    }
    df.rename(columns=col_mapping, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    df.set_index("Date", inplace=True)
    df.index = df.index.to_period("M")
    
    # Filter to start from 1993-11 to align with factors
    df = df.loc[config.FULL_SAMPLE_START:config.FULL_SAMPLE_END]
    return df


def load_raw_portfolios_size_mom() -> pd.DataFrame:
    """
    Load monthly returns for 4 Size-Momentum portfolios.
    Columns: WB (Winners-Big), WS (Winners-Small), LB (Losers-Big), LS (Losers-Small)
    """
    file_path = config.RAW_DATA_DIR / FILES_DICT["portfolios_size_mom"]
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    
    col_mapping = {
        "Date": "Date",
        "Portfolio WB returns (%)": "WB",
        "Portfolio WS returns (%)": "WS",
        "Portfolio LB returns (%)": "LB",
        "Portfolio LS returns (%)": "LS",
    }
    df.rename(columns=col_mapping, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    df.set_index("Date", inplace=True)
    df.index = df.index.to_period("M")
    
    df = df.loc[config.FULL_SAMPLE_START:config.FULL_SAMPLE_END]
    return df


def load_master_dataset() -> pd.DataFrame:
    """
    Combine Factors, Risk-Free Rate, Size-Value Portfolios, and Size-Momentum Portfolios
    into a single synchronized monthly DataFrame.
    """
    factors = load_raw_factors_monthly()
    sv_ports = load_raw_portfolios_size_value()
    sm_ports = load_raw_portfolios_size_mom()
    
    master = factors.join(sv_ports, how="inner").join(sm_ports, how="inner")
    
    # Save processed copy
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(config.PROCESSED_DATA_DIR / "master_monthly_panel.csv")
    return master


if __name__ == "__main__":
    download_raw_data()
    df = load_master_dataset()
    print("\nMaster dataset loaded successfully:")
    print(f"Shape: {df.shape}")
    print(f"Index: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
