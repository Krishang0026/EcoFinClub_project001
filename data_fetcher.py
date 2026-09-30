"""
Data Fetcher Module using yfinance
Fetches financial metrics, balance sheet items, valuation ratios, and industry categories.
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf

# Representative list of Indian Nifty 50, Nifty Next 50, and mid-cap companies across various sectors
NIFTY_TICKERS = [
    # Information Technology & Telecom
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "LTIM.NS", "TECHM.NS", "BHARTIARTL.NS", "IDEA.NS", "PERSISTENT.NS", "COFORGE.NS",
    # Financial Services & Banks
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "PFC.NS", "REC.NS",
    # Energy, Oil & Gas
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS",
    # Consumer Goods & Retail
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "TITAN.NS", "ASIANPAINT.NS", "TRENT.NS", "DMART.NS", "VBL.NS",
    # Automobiles & Transportation
    "TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "BHARATFORG.NS",
    # Pharmaceuticals & Healthcare
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "MANKIND.NS", "LUPIN.NS", "TORNTPHARM.NS",
    # Metals, Mining & Materials
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS", "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "APLAPOLLO.NS",
    # Industrials, Defense & Capital Goods
    "LT.NS", "BEL.NS", "HAL.NS", "SIEMENS.NS", "ABB.NS", "ADANIENT.NS", "ADANIPORTS.NS", "BHEL.NS", "SOLARINDS.NS"
]

REQUIRED_FIELDS = {
    'symbol': 'symbol',
    'shortName': 'shortName',
    'sector': 'sector',
    'industry': 'industry',
    'marketCap': 'marketCap',
    'trailingPE': 'trailingPE',
    'forwardPE': 'forwardPE',
    'priceToBook': 'priceToBook',
    'priceToSalesTrailing12Months': 'priceToSales',
    'enterpriseToEbitda': 'enterpriseToEbitda',
    'debtToEquity': 'debtToEquity',
    'currentRatio': 'currentRatio',
    'quickRatio': 'quickRatio',
    'totalRevenue': 'totalRevenue',
    'ebitda': 'ebitda',
    'ebitdaMargins': 'ebitdaMargins',
    'profitMargins': 'profitMargins',
    'grossMargins': 'grossMargins',
    'operatingMargins': 'operatingMargins',
    'returnOnEquity': 'returnOnEquity',
    'returnOnAssets': 'returnOnAssets',
    'revenueGrowth': 'revenueGrowth',
    'earningsGrowth': 'earningsGrowth',
    'totalDebt': 'totalDebt',
    'totalCash': 'totalCash',
    'freeCashflow': 'freeCashflow',
    'operatingCashflow': 'operatingCashflow'
}

def fetch_financial_data(tickers=NIFTY_TICKERS, save_filename="nifty50_company_financial_data.csv", delay=0.2):
    """
    Fetch company financial data from yfinance and save to CSV.
    """
    print(f"Starting data collection for {len(tickers)} tickers via yfinance...")
    records = []
    
    for idx, ticker_symbol in enumerate(tickers, start=1):
        try:
            print(f"[{idx}/{len(tickers)}] Fetching data for {ticker_symbol}...", end=" ", flush=True)
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            if not info or info.get('marketCap') is None:
                print("SKIPPED (No market cap available)")
                continue
                
            record = {'ticker': ticker_symbol}
            for yf_key, col_name in REQUIRED_FIELDS.items():
                val = info.get(yf_key, np.nan)
                record[col_name] = val
                
            records.append(record)
            print(f"SUCCESS (Market Cap: {record['marketCap']:,.0f}, Sector: {record['sector']})")
        except Exception as e:
            print(f"FAILED ({str(e)})")
            
        time.sleep(delay)
        
    df = pd.DataFrame(records)
    print(f"\nSuccessfully collected financial data for {len(df)} companies.")
    
    # Clean data types
    numeric_cols = [c for c in df.columns if c not in ['ticker', 'symbol', 'shortName', 'sector', 'industry']]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df.to_csv(save_filename, index=False)
    print(f"Data saved to {save_filename}")
    return df

if __name__ == "__main__":
    df = fetch_financial_data()
    print(df.head())
