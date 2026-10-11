"""
Feature Engineering Module (Enhanced)
Implements rich financial ratios, capital efficiency factors, non-linear & higher-power terms,
sector Z-score normalizations, and industry/size/growth classifications.
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
        if mcap >= 1e12: # >= 1 Lakh Crore (1 Trillion INR)
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
            
    # Sector median & global fallback imputation for missing values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for c in num_cols:
            df[c] = df.groupby('sector')[c].transform(lambda grp: grp.fillna(grp.median() if not grp.isna().all() else 0))
            df[c] = df[c].fillna(df[c].median() if not df[c].isna().all() else 0).fillna(0)

    # Target variable transformation: Log(MarketCap)
    df['log_marketCap'] = np.log(np.maximum(df['marketCap'], 1.0))
    
    # 1. Logarithmic scale features for financial volume (revenue, debt, cash, ebitda)
    df['log_totalRevenue'] = np.log1p(np.maximum(df['totalRevenue'], 0.0))
    df['log_ebitda'] = np.log1p(np.maximum(df['ebitda'], 0.0))
    df['log_totalDebt'] = np.log1p(np.maximum(df['totalDebt'], 0.0))
    df['log_totalCash'] = np.log1p(np.maximum(df['totalCash'], 0.0))
    df['log_freeCashflow'] = np.log1p(np.maximum(df['freeCashflow'], 0.0))
    
    # 2. Advanced Financial Health & Efficiency Factors
    # Cash-to-Debt buffer
    df['cash_to_debt'] = df['totalCash'] / (df['totalDebt'] + 1.0)
    # Net Debt to EBITDA (leverage stress)
    net_debt = df['totalDebt'] - df['totalCash']
    df['net_debt_to_ebitda'] = net_debt / (np.maximum(df['ebitda'], 1.0) + 1.0)
    # Free Cash Flow Yield
    df['fcf_yield'] = df['freeCashflow'] / np.maximum(df['marketCap'], 1.0)
    # FCF to Debt
    df['fcf_to_debt'] = df['freeCashflow'] / (df['totalDebt'] + 1.0)
    # Operating Cash Flow to Revenue (Cash conversion quality)
    df['ocf_to_revenue'] = df['operatingCashflow'] / (df['totalRevenue'] + 1.0)
    # Return on Invested Capital (ROIC) proxy
    invested_capital = df['totalDebt'] + (df['marketCap'] / (np.maximum(df['priceToBook'], 0.1)))
    df['roic_proxy'] = np.maximum(df['ebitda'], 0.0) / (invested_capital + 1.0)
    
    # 3. Sector Z-Score Normalization (Standardizing relative to industry baseline)
    for ratio_name in ['debtToEquity', 'trailingPE', 'ebitdaMargins', 'priceToBook', 'returnOnEquity']:
        grp_mean = df.groupby('sector')[ratio_name].transform('mean')
        grp_std = df.groupby('sector')[ratio_name].transform('std').fillna(1.0)
        df[f'{ratio_name}_sector_zscore'] = (df[ratio_name] - grp_mean) / (grp_std + 1e-4)

    # 4. Cross-Terms & Interaction Effects (Financial Intuition)
    # Scale x Efficiency = Profit engine
    df['cross_rev_x_ebitdaMargin'] = df['log_totalRevenue'] * df['ebitdaMargins']
    # Leverage x Operating Margin (risk vs return)
    df['cross_DE_x_ebitdaMargin'] = (df['debtToEquity'] / 100.0) * df['ebitdaMargins']
    # Earnings Yield x Return on Equity (Quality-at-value)
    df['cross_earningsYield_x_ROE'] = (1.0 / (np.maximum(df['trailingPE'], 1.0) + 0.1)) * df['returnOnEquity']
    # Growth-at-Reasonable-Price (GARP proxy)
    df['cross_GARP'] = (df['revenueGrowth'] * 100.0) / (np.maximum(df['trailingPE'], 1.0) + 0.1)
    # Dupont Valuation Cross Term: Price-to-Book x ROE
    df['cross_PB_x_ROE'] = df['priceToBook'] * df['returnOnEquity']
    # Solvency & Liquidity Buffer
    df['cross_liquidity_solvency'] = df['currentRatio'] / (np.abs(df['debtToEquity'] / 100.0) + 0.1)
    # Cashflow Quality Cross: OCF Margin x Net Profit Margin
    df['cross_ocfMargin_x_profitMargin'] = df['ocf_to_revenue'] * df['profitMargins']
    
    # 5. Higher Power & Non-Linear Polynomial Terms
    df['poly_ebitdaMargin_sq'] = df['ebitdaMargins'] ** 2
    df['poly_revenueGrowth_sq'] = df['revenueGrowth'] ** 2
    df['poly_roe_sq'] = df['returnOnEquity'] ** 2
    df['poly_debtToEquity_sq'] = (df['debtToEquity'] / 100.0) ** 2
    df['poly_log_PE'] = np.log1p(np.maximum(df['trailingPE'], 0.0))
    df['poly_log_PB'] = np.log1p(np.maximum(df['priceToBook'], 0.0))
    
    # One-hot encoding for categorical variables (Sector, Size, Growth)
    cat_df = pd.get_dummies(df[['sector', 'size_category', 'growth_category']], drop_first=True)
    df_engineered = pd.concat([df, cat_df], axis=1)
    
    return df_engineered

if __name__ == "__main__":
    import os
    if os.path.exists("nifty50_company_financial_data.csv"):
        df = pd.read_csv("nifty50_company_financial_data.csv")
        df_prep = preprocess_and_engineer_features(df)
        print("Engineered DataFrame shape:", df_prep.shape)
        print("Total Features:", df_prep.shape[1])
