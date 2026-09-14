"""Test yfinance data availability for Indian stocks."""
import yfinance as yf
import pandas as pd

test_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'WIPRO.NS']

print("=" * 80)
print("YFINANCE TEST: Indian Stock Data Quality Audit")
print("=" * 80)

for ticker_str in test_tickers:
    print(f"\n--- {ticker_str} ---")
    t = yf.Ticker(ticker_str)

    # 1. Price history coverage
    hist = t.history(period="max")
    if len(hist) > 0:
        print(f"  Price history: {hist.index[0].date()} to {hist.index[-1].date()} ({len(hist)} obs)")
    else:
        print("  Price history: NO DATA")

    # 2. Balance sheet
    bs = t.balance_sheet
    if bs is not None and len(bs) > 0:
        print(f"  Balance sheet years: {[str(c.date()) for c in bs.columns]}")
        key_bs = ["Total Assets", "Stockholders Equity", "Total Liabilities Net Minority Interest"]
        for k in key_bs:
            if k in bs.index:
                avail = bs.loc[k].notna().sum()
                print(f"    {k}: {avail}/{len(bs.columns)} years")
            else:
                print(f"    {k}: NOT FOUND")
    else:
        print("  Balance sheet: NO DATA")

    # 3. Income statement
    inc = t.financials
    if inc is not None and len(inc) > 0:
        print(f"  Income stmt years: {[str(c.date()) for c in inc.columns]}")
        key_inc = ["Total Revenue", "Operating Income", "EBIT", "Net Income", "Gross Profit"]
        for k in key_inc:
            if k in inc.index:
                avail = inc.loc[k].notna().sum()
                print(f"    {k}: {avail}/{len(inc.columns)} years")
            else:
                print(f"    {k}: NOT FOUND")
    else:
        print("  Income statement: NO DATA")

    # 4. Market cap info
    info = t.info
    mcap = info.get("marketCap", "N/A")
    bv = info.get("bookValue", "N/A")
    print(f"  Market cap (current): {mcap}")
    print(f"  Book value per share: {bv}")

print("\n\nDone.")
