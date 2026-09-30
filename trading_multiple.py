"""
1. We need to figure out the current valuation of the company and what we claim to be our estimate of their valulation based on market forces. (we will consider market cap)
2. I think we need to make an ML model to estimate the market cap based on its financials
3. For this we can model P/E,D/E,P/B,etc ratios (to get market sentiment), revenue growth, ebidta as percentage of revenue, and critical factors from balance sheet such as current ratio, total liabilities, etc (to get financial strength)
4. We need to figure out mathematical relations between these factors to consider higher power terms, cross terms and possibly exponential or logarthimically related terms using correlations and some financial intuition.
5. One key factor here: We need to classify companies by industry,growth and size. (primarily because industries have different standards, Eg: a saturated industry like Telecom will not have acceptably high debt ratios as compared to upcoming tech and AI companies. Within the telecom industry, service providers and telecom backend (like ericsson, nokia, huawei) will have different financial standards. within a same sub-category, we need to classify them by size because it is very natural for a startup to have high debt ratio, and it not implying that something is broken.

"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from data_fetcher import fetch_financial_data
from feature_engineering import preprocess_and_engineer_features
from model_trainer import (
    prepare_feature_matrix, 
    train_and_evaluate_models, 
    calculate_valuation_estimates, 
    plot_model_insights
)

DATA_FILE = "nifty50_company_financial_data.csv"

def main(force_refresh=False):
    print("="*75)
    print(" FINANCIAL VALUATION ML MODEL - MULTIPLE VALUATION PIPELINE")
    print("="*75)
    
    # Step 1: Data Collection using yfinance
    if force_refresh or not os.path.exists(DATA_FILE):
        print(f"\n[1/4] Fetching latest market and financial data using yfinance...")
        df_raw = fetch_financial_data(save_filename=DATA_FILE)
    else:
        print(f"\n[1/4] Loading existing financial dataset from '{DATA_FILE}'...")
        df_raw = pd.read_csv(DATA_FILE)
        
    print(f"Dataset contains {len(df_raw)} companies and {df_raw.shape[1]} raw financial attributes.")
    
    # Step 2: Classification & Feature Engineering
    print("\n[2/4] Executing Feature Engineering & Industry/Size/Growth Classification...")
    df_engineered = preprocess_and_engineer_features(df_raw)
    
    # Step 3: Model Training & Selection
    print("\n[3/4] Preparing Feature Matrix & Training Machine Learning Models...")
    X, y, feature_cols = prepare_feature_matrix(df_engineered)
    best_model_name, best_model, scaler, all_results = train_and_evaluate_models(X, y)
    
    # Step 4: Valuation Estimation & Mispricing Analysis
    print(f"\n[4/4] Calculating Valuation Estimates using Best Model ({best_model_name})...")
    res_df = calculate_valuation_estimates(df_engineered, best_model, scaler, feature_cols)
    
    # Save results to CSV
    output_cols = [
        'ticker', 'shortName', 'sector', 'size_category', 'growth_category',
        'marketCap', 'estimated_marketCap', 'valuation_diff_pct', 'valuation_status',
        'trailingPE', 'debtToEquity', 'ebitdaMargins', 'revenueGrowth'
    ]
    
    available_cols = [c for c in output_cols if c in res_df.columns]
    summary_df = res_df[available_cols].copy()
    
    # Format Market Cap to Crore INR (1 Cr = 10,000,000 INR)
    summary_df['Actual MarketCap (INR Cr)'] = summary_df['marketCap'] / 1e7
    summary_df['Estimated Valuation (INR Cr)'] = summary_df['estimated_marketCap'] / 1e7
    summary_df['Valuation Diff (%)'] = summary_df['valuation_diff_pct'].round(2)
    
    summary_df = summary_df.sort_values(by='Valuation Diff (%)', ascending=False)
    summary_df.to_csv("valuation_results.csv", index=False)
    print("Full valuation analysis saved to 'valuation_results.csv'.")
    
    # Plot analysis charts
    plot_model_insights(res_df, best_model, feature_cols, best_model_name)
    
    # Display Top Undervalued and Top Overvalued stocks
    print("\n" + "="*80)
    print(" TOP 5 RELATIVELY UNDERVALUED STOCKS (Model Estimated Valuation > Actual Market Cap)")
    print("="*80)
    undervalued = summary_df.head(5)
    for idx, row in undervalued.iterrows():
        print(f"* {row['shortName']} ({row['ticker']}) | Sector: {row['sector']} | Size: {row['size_category']}")
        print(f"  Actual MarketCap: INR {row['Actual MarketCap (INR Cr)']:,.0f} Cr | Estimated Valuation: INR {row['Estimated Valuation (INR Cr)']:,.0f} Cr")
        print(f"  Discount: +{row['Valuation Diff (%)']}% | P/E: {row['trailingPE']:.1f} | D/E: {row['debtToEquity']:.2f} | EBITDA Margin: {row['ebitdaMargins']*100:.1f}%\n")
        
    print("="*80)
    print(" TOP 5 RELATIVELY OVERVALUED STOCKS (Model Estimated Valuation < Actual Market Cap)")
    print("="*80)
    overvalued = summary_df.tail(5).iloc[::-1]
    for idx, row in overvalued.iterrows():
        print(f"* {row['shortName']} ({row['ticker']}) | Sector: {row['sector']} | Size: {row['size_category']}")
        print(f"  Actual MarketCap: INR {row['Actual MarketCap (INR Cr)']:,.0f} Cr | Estimated Valuation: INR {row['Estimated Valuation (INR Cr)']:,.0f} Cr")
        print(f"  Premium: {row['Valuation Diff (%)']}% | P/E: {row['trailingPE']:.1f} | D/E: {row['debtToEquity']:.2f} | EBITDA Margin: {row['ebitdaMargins']*100:.1f}%\n")

    return summary_df, best_model_name, all_results

if __name__ == "__main__":
    main(force_refresh=False)
