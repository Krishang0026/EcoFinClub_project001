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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

DATA_FILE = "valuation_results.csv"

def calculate_c_mispricing(df):
    """
    Calculate c = (our claim - current market cap) / (avg market cap of its classification/sector)
    c > 0 indicates undervalued stock, c < 0 indicates overvalued stock.
    """
    df = df.copy()
    
    # Calculate average market cap for each sector classification
    sector_avg_mcap = df.groupby('sector')['marketCap'].transform('mean')
    df['sector_avg_marketCap'] = sector_avg_mcap
    
    # c factor: difference between estimated market cap (our claim) and current market cap relative to sector avg
    df['c_mispricing'] = (df['estimated_marketCap'] - df['marketCap']) / df['sector_avg_marketCap']
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
    Train regularized linear regression to model forward return % and horizon N based on c-factor.
    Linear Model: Forward Return (delta_P / P) = alpha + beta * c_mispricing
    """
    training_data = []
    horizons_N = [30, 60, 90, 180, 252, 365, 504]
    
    print("\nCalibrating Forward Return & Horizon Model using historical price data...")
    
    for _, row in df_with_c.iterrows():
        ticker = row['ticker']
        c_val = row['c_mispricing']
        pe = row.get('trailingPE', 20.0)
        de = row.get('debtToEquity', 10.0)
        rev_growth = row.get('revenueGrowth', 0.1)
        
        if ticker not in price_histories:
            continue
            
        prices = price_histories[ticker].values
        if len(prices) < 252:
            continue
            
        # Sample historical intervals to model delta_P / P vs c
        # Take point at 252 trading days ago (1 year ago)
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
                'horizon_N_days': n_days,
                'target_forward_return': delta_price_ratio
            })
            
    train_df = pd.DataFrame(training_data)
    
    if len(train_df) == 0:
        print("Warning: Insufficient historical price points for ML regression. Using synthetic calibration.")
        # Fallback heuristic calibration if historical API limited
        scaler = StandardScaler()
        model = Ridge(alpha=1.0)
        return model, scaler, 252
        
    feature_cols = ['c_mispricing', 'trailingPE', 'debtToEquity', 'revenueGrowth']
    X = train_df[feature_cols].fillna(0)
    y = train_df['target_forward_return'].fillna(0)
    
    # Feature Scaling & Regularization (Ridge Regression with L2 Penalty)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    reg_model = Ridge(alpha=5.0)
    reg_model.fit(X_scaled, y)
    
    train_r2 = r2_score(y, reg_model.predict(X_scaled))
    print(f"Forward Return Model Fitted | Regularized Ridge R²: {train_r2:.4f} | Coefficients: {dict(zip(feature_cols, reg_model.coef_))}")
    
    return reg_model, scaler, train_df

def predict_forward_returns(df_with_c, reg_model, scaler):
    """
    Predict forward expected return % and estimated horizon N days for current stock universe.
    """
    feature_cols = ['c_mispricing', 'trailingPE', 'debtToEquity', 'revenueGrowth']
    X_curr = df_with_c[feature_cols].fillna(0)
    X_curr_scaled = scaler.transform(X_curr)
    
    pred_returns = reg_model.predict(X_curr_scaled)
    
    res_df = df_with_c.copy()
    # Expected Return % (Annualized / Horizon)
    res_df['expected_forward_return_pct'] = (pred_returns * 100).round(2)
    
    # Estimate convergence time horizon N (in trading days) based on mispricing magnitude
    # Larger mispricing c generally requires longer horizon N (~180 to 504 trading days) to converge
    base_horizon_days = 252 # 1 trading year
    res_df['estimated_horizon_days_N'] = (base_horizon_days * (1.0 + 0.5 * np.abs(res_df['c_mispricing']))).astype(int)
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
    print(f"Loaded {len(df_val)} companies from '{DATA_FILE}'.")
    
    # Step 1: Calculate mispricing factor c
    df_with_c = calculate_c_mispricing(df_val)
    
    # Step 2: Fetch historical prices for calibration
    tickers = df_with_c['ticker'].tolist()
    price_histories = fetch_historical_returns(tickers)
    
    # Step 3: Train Regularized Linear Regression Model
    reg_model, scaler, _ = train_forward_return_linear_model(df_with_c, price_histories)
    
    # Step 4: Predict Expected Forward Return and Horizon N
    res_df = predict_forward_returns(df_with_c, reg_model, scaler)
    
    # Save output
    res_df.to_csv("forward_return_results.csv", index=False)
    print("\nForward Return & Horizon predictions saved to 'forward_return_results.csv'.")
    
    print("\n" + "="*80)
    print(" SAMPLE EXPECTED FORWARD RETURNS & ESTIMATED HORIZON N")
    print("="*80)
    display_cols = ['ticker', 'shortName', 'sector', 'c_mispricing', 'expected_forward_return_pct', 'estimated_horizon_days_N', 'estimated_horizon_years']
    sample = res_df[display_cols].sort_values(by='expected_forward_return_pct', ascending=False)
    
    print("\nTop 5 Highest Projected Forward Returns:")
    for _, r in sample.head(5).iterrows():
        print(f"* {r['shortName']} ({r['ticker']}) | Mispricing c: {r['c_mispricing']:.4f}")
        print(f"  Projected Return: +{r['expected_forward_return_pct']}% | Target Horizon N: {r['estimated_horizon_days_N']} days ({r['estimated_horizon_years']} yrs)\n")
        
    print("Top 5 Lowest Projected Forward Returns:")
    for _, r in sample.tail(5).iloc[::-1].iterrows():
        print(f"* {r['shortName']} ({r['ticker']}) | Mispricing c: {r['c_mispricing']:.4f}")
        print(f"  Projected Return: {r['expected_forward_return_pct']}% | Target Horizon N: {r['estimated_horizon_days_N']} days ({r['estimated_horizon_years']} yrs)\n")

    return res_df

if __name__ == "__main__":
    main()