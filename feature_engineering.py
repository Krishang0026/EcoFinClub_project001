"""
Feature Engineering Module
Implements mathematical transformations, non-linear & interaction features,
logarithmic scaling, and industry/size/growth classifications as specified in docstring.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

def classify_companies(df):
    """
    Classify companies by Industry/Sector, Size, and Growth profiles.
    (Requirement 5 from docstring)
    """
    df = df.copy()
    
    # 1. Size Classification based on Market Cap (in INR)
    def get_size_cat(mcap):
        if pd.isna(mcap):
            return 'Mid-Cap'
        if mcap >= 1e12: # >= 1 Lakh Crore
            return 'Mega-Cap'
        elif mcap >= 2.5e11: # >= 25k Crore
            return 'Large-Cap'
        elif mcap >= 5e10: # >= 5k Crore
            return 'Mid-Cap'
        else:
            return 'Small-Cap'

    df['size_category'] = df['marketCap'].apply(get_size_cat)

    # 2. Growth Classification based on revenue growth
    def get_growth_cat(rev_growth):
        if pd.isna(rev_growth):
            return 'Moderate-Growth'
        if rev_growth >= 0.15:
            return 'High-Growth'
        elif rev_growth >= 0.05:
            return 'Moderate-Growth'
        else:
            return 'Value/Mature'

    df['growth_category'] = df['revenueGrowth'].apply(get_growth_cat)
    
    # 3. Clean sector & industry fallback
    df['sector'] = df['sector'].fillna('Unknown')
    df['industry'] = df['industry'].fillna('Unknown')
    
    return df

def preprocess_and_engineer_features(df):
    """
    Construct logarithmic terms, cross-terms, higher power terms, and sector-relative features.
    (Requirements 3, 4, 5 from docstring)
    """
    df = classify_companies(df)
    
    num_cols = [
        'trailingPE', 'forwardPE', 'priceToBook', 'priceToSales', 'enterpriseToEbitda',
        'debtToEquity', 'currentRatio', 'quickRatio', 'totalRevenue', 'ebitda',
        'ebitdaMargins', 'profitMargins', 'grossMargins', 'operatingMargins',
        'returnOnEquity', 'returnOnAssets', 'revenueGrowth', 'earningsGrowth',
        'totalDebt', 'totalCash', 'freeCashflow', 'operatingCashflow'
    ]
    
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
            
    # Sector median imputation for missing values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for c in num_cols:
            df[c] = df.groupby('sector')[c].transform(lambda grp: grp.fillna(grp.median() if not grp.isna().all() else 0))
            df[c] = df[c].fillna(df[c].median() if not df[c].isna().all() else 0).fillna(0)

    # Target variable transformation: Log(MarketCap)
    df['log_marketCap'] = np.log(np.maximum(df['marketCap'], 1.0))
    
    # 1. Logarithmic scale features
    df['log_totalRevenue'] = np.log1p(np.maximum(df['totalRevenue'], 0.0))
    df['log_ebitda'] = np.log1p(np.maximum(df['ebitda'], 0.0))
    df['log_totalDebt'] = np.log1p(np.maximum(df['totalDebt'], 0.0))
    df['log_totalCash'] = np.log1p(np.maximum(df['totalCash'], 0.0))
    df['log_freeCashflow'] = np.log1p(np.maximum(df['freeCashflow'], 0.0))
    
    # 2. Sector-Relative Metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sector_medians = df.groupby('sector')[['debtToEquity', 'trailingPE', 'ebitdaMargins', 'priceToBook']].transform('median').fillna(0)
    
    df['DE_relative_to_sector'] = df['debtToEquity'] - sector_medians['debtToEquity']
    df['PE_relative_to_sector'] = df['trailingPE'] - sector_medians['trailingPE']
    df['Margin_relative_to_sector'] = df['ebitdaMargins'] - sector_medians['ebitdaMargins']
    df['PB_relative_to_sector'] = df['priceToBook'] - sector_medians['priceToBook']
    
    # 3. Cross-Terms (Interaction terms based on financial intuition)
    df['cross_rev_x_ebitdaMargin'] = df['log_totalRevenue'] * df['ebitdaMargins']
    df['cross_DE_x_ebitdaMargin'] = df['debtToEquity'] * df['ebitdaMargins']
    df['cross_growth_x_margin'] = df['revenueGrowth'] * df['profitMargins']
    df['cross_PB_x_ROE'] = df['priceToBook'] * df['returnOnEquity']
    df['cross_liquidity_solvency'] = df['currentRatio'] / (np.abs(df['debtToEquity']) + 0.1)
    df['cross_PEG_proxy'] = df['trailingPE'] / (np.maximum(df['revenueGrowth'], 0.001) * 100 + 1.0)
    
    # 4. Higher Power Terms (Quadratic terms for non-linear relations)
    df['poly_ebitdaMargin_sq'] = df['ebitdaMargins'] ** 2
    df['poly_revenueGrowth_sq'] = df['revenueGrowth'] ** 2
    df['poly_debtToEquity_sq'] = df['debtToEquity'] ** 2
    df['poly_currentRatio_sq'] = df['currentRatio'] ** 2
    
    # One-hot encoding for categorical variables
    cat_df = pd.get_dummies(df[['sector', 'size_category', 'growth_category']], drop_first=True)
    df_engineered = pd.concat([df, cat_df], axis=1)
    
    return df_engineered

if __name__ == "__main__":
    import os
    if os.path.exists("nifty50_company_financial_data.csv"):
        df = pd.read_csv("nifty50_company_financial_data.csv")
        df_prep = preprocess_and_engineer_features(df)
        print("Engineered DataFrame shape:", df_prep.shape)
