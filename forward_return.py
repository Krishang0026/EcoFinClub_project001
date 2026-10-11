"""
from trading multiple
Now we have the company's current market cap and an estimated market cap, this delta will give us a sense of direction about the stock price movements in the future. But future is not definite. we need an N-day period. 
We further use machine learning models to get this N.
consider c=(market cap- our claim)/(avg of its classification)
now we can model N with c and market price.
select a historical date and feed historical data. based on c calculations, we can vary n to observe delta price. 
my claim: delta price/proce bough is directly proportional to c
so we can infact get linear regression to work here
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score

DATA_FILE = "valuation_results.csv"

def calculate_c_mispricing(df):
    """
    Calculate c = (our claim - current market cap) / (avg market cap of its classification/sector)
    c > 0 indicates undervalued stock, c < 0 indicates overvalued stock.
    """
    df = df.copy()
    
    sector_avg_mcap = df.groupby('sector')['marketCap'].transform('mean')
    df['sector_avg_marketCap'] = sector_avg_mcap
    
    df['c_mispricing'] = (df['estimated_marketCap'] - df['marketCap']) / (df['sector_avg_marketCap'] + 1.0)
    return df

def fetch_historical_returns(tickers, lookback_days=504):
    """
    Fetch historical daily prices for tickers and compute price dynamics over N-day horizons.
    """
    print(f"Fetching historical price series for {len(tickers)} tickers (Lookback: {lookback_days} days)...")
    price_histories = {}
    
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="2y")
            if len(hist) > 0 and 'Close' in hist.columns:
                price_histories[t] = hist['Close']
        except Exception:
            continue
            
    print(f"Successfully retrieved price histories for {len(price_histories)} tickers.")
    return price_histories

def train_forward_return_linear_model(df_with_c, price_histories):
    """
    Train regularized regression model with enhanced factors to model forward return % and horizon N based on c-factor.
    """
    training_data = []
    horizons_N = [30, 60, 90, 180, 252, 365, 504]
    
    print("\nCalibrating Forward Return & Horizon Model using historical price data & enhanced stats...")
    
    for _, row in df_with_c.iterrows():
        ticker = row['ticker']
        c_val = row['c_mispricing']
        pe = row.get('trailingPE', 20.0)
        de = row.get('debtToEquity', 10.0)
        rev_growth = row.get('revenueGrowth', 0.1)
        margin = row.get('ebitdaMargins', 0.15)
        
        if ticker not in price_histories:
            continue
            
        prices = price_histories[ticker].values
        if len(prices) < 252:
            continue
            
        p_base = prices[-252]
        
        for n_days in horizons_N:
            if len(prices) >= 252 + n_days:
                p_future = prices[-252 + n_days]
            else:
                p_future = prices[-1]
                
            delta_price_ratio = (p_future - p_base) / p_base
            
            training_data.append({
                'ticker': ticker,
                'c_mispricing': c_val,
                'trailingPE': pe,
                'debtToEquity': de,
                'revenueGrowth': rev_growth,
                'ebitdaMargins': margin,
                'horizon_N_days': n_days,
                'target_forward_return': delta_price_ratio
            })
            
    train_df = pd.DataFrame(training_data)
    
    if len(train_df) == 0:
        scaler = RobustScaler()
        model = Ridge(alpha=10.0)
        return model, scaler, ['c_mispricing', 'trailingPE', 'debtToEquity', 'revenueGrowth', 'ebitdaMargins']
        
    feature_cols = ['c_mispricing', 'trailingPE', 'debtToEquity', 'revenueGrowth', 'ebitdaMargins']
    X = train_df[feature_cols].fillna(0)
    y = train_df['target_forward_return'].fillna(0)
    
    # Feature Scaling & Regularization (ElasticNet / Ridge for lower variance)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    reg_model = Ridge(alpha=15.0)
    reg_model.fit(X_scaled, y)
    
    train_r2 = r2_score(y, reg_model.predict(X_scaled))
    print(f"Forward Return Model Fitted | Regularized Ridge R²: {train_r2:.4f}")
    
    return reg_model, scaler, feature_cols

def predict_forward_returns(df_with_c, reg_model, scaler, feature_cols=None):
    """
    Predict forward expected return % and estimated horizon N days for current stock universe.
    """
    if feature_cols is None:
        feature_cols = ['c_mispricing', 'trailingPE', 'debtToEquity', 'revenueGrowth', 'ebitdaMargins']
        
    for c in feature_cols:
        if c not in df_with_c.columns:
            df_with_c[c] = 0.0
            
    X_curr = df_with_c[feature_cols].fillna(0)
    X_curr_scaled = scaler.transform(X_curr)
    
    pred_returns = reg_model.predict(X_curr_scaled)
    
    res_df = df_with_c.copy()
    res_df['expected_forward_return_pct'] = (pred_returns * 100).round(2)
    
    # Estimate convergence time horizon N (in trading days)
    base_horizon_days = 252
    res_df['estimated_horizon_days_N'] = (base_horizon_days * (1.0 + 0.4 * np.abs(res_df['c_mispricing']))).astype(int)
    res_df['estimated_horizon_years'] = (res_df['estimated_horizon_days_N'] / 252.0).round(1)
    
    return res_df

def main():
    print("="*75)
    print(" STEP 2: FORWARD RETURN & HORIZON N MODELING (forward_return.py)")
    print("="*75)
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found. Please run 'trading_multiple.py' first.")
        return None
        
    df_val = pd.read_csv(DATA_FILE)
    df_with_c = calculate_c_mispricing(df_val)
    
    tickers = df_with_c['ticker'].tolist()
    price_histories = fetch_historical_returns(tickers)
    
    reg_model, scaler, fcols = train_forward_return_linear_model(df_with_c, price_histories)
    res_df = predict_forward_returns(df_with_c, reg_model, scaler, fcols)
    
    res_df.to_csv("forward_return_results.csv", index=False)
    print("\nForward Return & Horizon predictions saved to 'forward_return_results.csv'.")
    return res_df

if __name__ == "__main__":
    main()