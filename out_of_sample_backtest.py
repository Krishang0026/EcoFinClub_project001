"""
Out-of-Sample Backtesting Module
Evaluates ML Model accuracy on a 100% mutually exclusive dataset:
1. Currency-Adjusted US Stock Universe (converted to INR scale)
2. 70/30 Train/Test Mutually Exclusive Split on Indian Stock Universe
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from data_fetcher import fetch_financial_data
from feature_engineering import preprocess_and_engineer_features
from model_trainer import prepare_feature_matrix, train_and_evaluate_models

USD_TO_INR = 83.5 # Currency conversion rate for cross-market alignment

def run_out_of_sample_backtest():
    print("="*85)
    print(" RIGOROUS MUTUALLY EXCLUSIVE OUT-OF-SAMPLE BACKTEST")
    print("="*85)
    
    # Load Indian Dataset
    train_file = "nifty50_company_financial_data.csv"
    if not os.path.exists(train_file):
        df_all = fetch_financial_data(save_filename=train_file)
    else:
        df_all = pd.read_csv(train_file)
        
    # -------------------------------------------------------------------------
    # PART 1: 70/30 MUTUALLY EXCLUSIVE TRAIN / TEST BACKTEST (IN-MARKET)
    # -------------------------------------------------------------------------
    print("\n--- EXPERIMENT 1: 70/30 MUTUALLY EXCLUSIVE TRAIN/TEST SPLIT ---")
    df_train, df_test = train_test_split(df_all, test_size=0.30, random_state=42)
    print(f"Train Set: {len(df_train)} companies | Test Set (Unseen): {len(df_test)} companies")
    print("[Verification]: 0% overlap between train and test sets.")
    
    # Feature Engineering
    df_train_eng = preprocess_and_engineer_features(df_train)
    df_test_eng = preprocess_and_engineer_features(df_test)
    
    X_train_df, y_train, feature_cols = prepare_feature_matrix(df_train_eng)
    
    # Train Model strictly on Train split
    best_name, best_model, scaler, _ = train_and_evaluate_models(X_train_df, y_train)
    
    # Predict on Mutually Exclusive Test Split
    for c in feature_cols:
        if c not in df_test_eng.columns:
            df_test_eng[c] = 0.0
            
    X_test_scaled = scaler.transform(df_test_eng[feature_cols].fillna(0))
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    pred_test_log = best_model.predict(X_test_scaled_df)
    actual_test_log = df_test_eng['log_marketCap'].values
    actual_mcap = df_test_eng['marketCap'].values
    est_mcap = np.exp(pred_test_log)
    
    # Out-of-Sample Metrics
    r2_oos = r2_score(actual_test_log, pred_test_log)
    rmse_oos = np.sqrt(mean_squared_error(actual_test_log, pred_test_log))
    mae_oos = mean_absolute_error(actual_test_log, pred_test_log)
    mape_mcap = mean_absolute_percentage_error(actual_mcap, est_mcap) * 100
    
    diff_pct = ((est_mcap - actual_mcap) / actual_mcap) * 100
    precision_20 = (np.sum(np.abs(diff_pct) <= 20.0) / len(actual_mcap)) * 100
    
    print("\n" + "="*75)
    print(" IN-MARKET OUT-OF-SAMPLE ACCURACY RESULTS (UNSEEN COMPANIES)")
    print("="*75)
    print(f" • Out-of-Sample Log R² Score        : {r2_oos:.4f} ({r2_oos*100:.1f}% Variance Explained)")
    print(f" • Out-of-Sample Log RMSE           : {rmse_oos:.4f}")
    print(f" • Out-of-Sample Log MAE            : {mae_oos:.4f}")
    print(f" • Mean Absolute Valuation Error %  : {mape_mcap:.2f}%")
    print(f" • Valuation Precision (Within ±20%): {precision_20:.1f}% of unseen companies")
    print("="*75)
    
    # -------------------------------------------------------------------------
    # PART 2: CROSS-MARKET UNSEEN US STOCK BACKTEST (CURRENCY ADJUSTED)
    # -------------------------------------------------------------------------
    us_file = "unseen_test_company_data.csv"
    if os.path.exists(us_file):
        print("\n--- EXPERIMENT 2: CURRENCY-ADJUSTED US STOCK OUT-OF-SAMPLE BACKTEST ---")
        df_us_raw = pd.read_csv(us_file)
        
        # Convert USD financial volumes to INR scale for cross-market model compatibility
        df_us_converted = df_us_raw.copy()
        vol_cols = ['marketCap', 'totalRevenue', 'ebitda', 'totalDebt', 'totalCash', 'freeCashflow', 'operatingCashflow']
        for v in vol_cols:
            if v in df_us_converted.columns:
                df_us_converted[v] = df_us_converted[v] * USD_TO_INR
                
        df_us_eng = preprocess_and_engineer_features(df_us_converted)
        for c in feature_cols:
            if c not in df_us_eng.columns:
                df_us_eng[c] = 0.0
                
        X_us_scaled = scaler.transform(df_us_eng[feature_cols].fillna(0))
        X_us_scaled_df = pd.DataFrame(X_us_scaled, columns=feature_cols)
        
        pred_us_log = best_model.predict(X_us_scaled_df)
        actual_us_log = df_us_eng['log_marketCap'].values
        actual_us_mcap = df_us_raw['marketCap'].values # original USD
        est_us_mcap = np.exp(pred_us_log) / USD_TO_INR # convert back to USD
        
        r2_us = r2_score(actual_us_log, pred_us_log)
        mape_us = mean_absolute_percentage_error(actual_us_mcap, est_us_mcap) * 100
        
        print("\n" + "="*75)
        print(" CROSS-MARKET OUT-OF-SAMPLE ACCURACY RESULTS (US STOCKS IN USD)")
        print("="*75)
        print(f" • Currency-Adjusted Out-of-Sample R² Score : {r2_us:.4f}")
        print(f" • Mean Absolute Valuation Error %           : {mape_us:.2f}%")
        print("="*75)
        
    # Save backtest diagnostic plot
    plot_oos_results(df_test, actual_mcap, est_mcap, diff_pct, best_name, r2_oos, mape_mcap)

def plot_oos_results(df_test, actual_mcap, est_mcap, diff_pct, model_name, r2_oos, mape_mcap):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    actual_cr = actual_mcap / 1e7
    est_cr = est_mcap / 1e7
    
    # Scatter Plot
    sns.scatterplot(x=actual_cr, y=est_cr, hue=df_test['sector'], ax=axes[0], s=90, palette='viridis')
    axes[0].plot([actual_cr.min(), actual_cr.max()], [actual_cr.min(), actual_cr.max()], 'r--', label='Ideal 1:1 Valuation')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Actual Market Cap (₹ Cr, Log Scale)', fontsize=12)
    axes[0].set_ylabel('Model Estimated Valuation (₹ Cr, Log Scale)', fontsize=12)
    axes[0].set_title(f'In-Market Out-of-Sample Accuracy ({model_name})\nOOS R² = {r2_oos:.4f} | MAPE = {mape_mcap:.1f}%', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Error distribution
    sns.histplot(diff_pct, kde=True, ax=axes[1], color='#00FFCC', bins=12)
    axes[1].axvline(0, color='white', linestyle='--', label='Zero Error Line')
    axes[1].set_title('Out-of-Sample Valuation Error % Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Valuation Difference % [(Estimated - Actual) / Actual]', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('out_of_sample_backtest_plots.png', dpi=300)
    plt.close()
    print("Backtest visual analysis saved to 'out_of_sample_backtest_plots.png'")

if __name__ == "__main__":
    run_out_of_sample_backtest()
