"""
from forward_return,
(I am not sure at all about this part)
I believe that existing models like DCF, MPT, PMPT etc help us get a fair trading multiple. Market will follow its predictions (on an avg). We need to account for this price movement, wrt to our predictions.
Eg: our model says +20% in 5 yrs, dcf says +19.7% in 5 yrs.
Let's say our model has an error band of +-1%, then this is a bad trade.
we need to maximise the difference between our model and existing model. 
we can call this alpha (not sure)
we will invest in companies with alpha values lying in the desired range.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "forward_return_results.csv"
OUTPUT_FILE = "final_alpha_portfolio.csv"

def calculate_dcf_benchmark_return(row, discount_rate=0.105, terminal_growth=0.035, forecast_years=5):
    """
    Calculate 2-Stage Discounted Cash Flow (DCF) Valuation and Benchmark DCF Return.
    """
    mcap = row['marketCap']
    fcf = row.get('freeCashflow', np.nan)
    ebitda = row.get('ebitda', np.nan)
    rev_growth = row.get('revenueGrowth', 0.08)
    
    # Clean growth rate to realistic bounds [2%, 25%]
    growth_rate = np.clip(rev_growth if not pd.isna(rev_growth) else 0.08, 0.02, 0.25)
    
    # Fallback for FCF if negative or missing: 60% of EBITDA or 4% yield of MarketCap
    if pd.isna(fcf) or fcf <= 0:
        if not pd.isna(ebitda) and ebitda > 0:
            fcf = ebitda * 0.60
        else:
            fcf = mcap * 0.04 # 4% FCF yield assumption
            
    # Stage 1: Discounted Cash Flows for 5 years
    future_fcfs = [fcf * ((1 + growth_rate) ** yr) for yr in range(1, forecast_years + 1)]
    pv_fcfs = [future_fcfs[yr - 1] / ((1 + discount_rate) ** yr) for yr in range(1, forecast_years + 1)]
    sum_pv_fcf = sum(pv_fcfs)
    
    # Stage 2: Terminal Value
    terminal_value = (future_fcfs[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + discount_rate) ** forecast_years)
    
    dcf_intrinsic_marketCap = sum_pv_fcf + pv_terminal_value
    
    # Expected Annualized DCF Benchmark Return (%)
    horizon_years = row.get('estimated_horizon_years', 1.0)
    dcf_total_return_pct = ((dcf_intrinsic_marketCap - mcap) / mcap) * 100
    dcf_annual_return_pct = dcf_total_return_pct / max(horizon_years, 0.5)
    
    return dcf_intrinsic_marketCap, dcf_annual_return_pct

def calculate_alpha_and_conviction(df, model_error_band_pct=3.0):
    """
    Calculate Model Alpha = (Our ML Return - Benchmark DCF Return)
    Filters trades where Alpha > Model Error Band to avoid noise trades.
    """
    df = df.copy()
    
    dcf_mcaps = []
    dcf_returns = []
    
    for _, row in df.iterrows():
        dcf_mcap, dcf_ret = calculate_dcf_benchmark_return(row)
        dcf_mcaps.append(dcf_mcap)
        dcf_returns.append(dcf_ret)
        
    df['dcf_intrinsic_marketCap'] = dcf_mcaps
    df['dcf_benchmark_return_pct'] = np.round(dcf_returns, 2)
    
    # 1. Raw Alpha (%) = Our ML Model Return - DCF Benchmark Return
    df['alpha_pct'] = (df['expected_forward_return_pct'] - df['dcf_benchmark_return_pct']).round(2)
    
    # 2. Risk-Adjusted Alpha Score = Alpha / Model Error Band
    df['model_error_band_pct'] = model_error_band_pct
    df['alpha_conviction_score'] = (df['alpha_pct'] / model_error_band_pct).round(2)
    
    # 3. Investment Signal Classification based on Alpha vs Error Band
    def classify_trade_signal(row):
        alpha = row['alpha_pct']
        err = row['model_error_band_pct']
        if alpha > err:
            return 'HIGH CONVICTION BUY (Alpha > Error Band)'
        elif alpha > 0.0:
            return 'WEAK BUY (Within Error Band Noise)'
        elif alpha < -err:
            return 'STRONG SELL / AVOID'
        else:
            return 'NEUTRAL'
            
    df['investment_signal'] = df.apply(classify_trade_signal, axis=1)
    return df

def plot_alpha_insights(df):
    """
    Generate diagnostic visualization of ML Model Return vs Benchmark DCF Return and Alpha distribution.
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Scatter Plot: Our ML Return vs Benchmark DCF Return
    sns.scatterplot(
        data=df, 
        x='dcf_benchmark_return_pct', 
        y='expected_forward_return_pct', 
        hue='investment_signal', 
        palette={'HIGH CONVICTION BUY (Alpha > Error Band)': '#00FF66', 
                 'WEAK BUY (Within Error Band Noise)': '#FFFF00', 
                 'NEUTRAL': '#888888', 
                 'STRONG SELL / AVOID': '#FF3366'}, 
        s=90, 
        ax=axes[0]
    )
    
    min_val = min(df['dcf_benchmark_return_pct'].min(), df['expected_forward_return_pct'].min()) - 5
    max_val = max(df['dcf_benchmark_return_pct'].max(), df['expected_forward_return_pct'].max()) + 5
    axes[0].plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.6, label='Baseline DCF Line (Alpha = 0)')
    axes[0].fill_between([min_val, max_val], [min_val - 3.0, max_val - 3.0], [min_val + 3.0, max_val + 3.0], color='gray', alpha=0.2, label='Model Error Band (±3%)')
    
    axes[0].set_title('Our ML Projected Return vs. Benchmark DCF Return', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Benchmark DCF Expected Return (%)', fontsize=12)
    axes[0].set_ylabel('Our ML Model Projected Return (%)', fontsize=12)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Top Alpha Ranking Bar Plot
    top_alpha = df.sort_values(by='alpha_pct', ascending=False).head(10)
    sns.barplot(data=top_alpha, x='alpha_pct', y='shortName', hue='shortName', legend=False, palette='crest', ax=axes[1])
    axes[1].axvline(3.0, color='red', linestyle='--', label='Alpha Threshold (+3% Error Margin)')
    axes[1].set_title('Top 10 High-Alpha Investment Opportunities', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Generated Alpha % (ML Return - DCF Return)', fontsize=12)
    axes[1].legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('intrinsic_alpha_analysis.png', dpi=300)
    plt.close()
    print("\nAlpha visual analysis saved as 'intrinsic_alpha_analysis.png'")

def main():
    print("="*75)
    print(" STEP 3: INTRINSIC VALUE & ALPHA GENERATION ENGINE (instrinsic_value.py)")
    print("="*75)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found. Please run 'forward_return.py' first.")
        return None
        
    df_fwd = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df_fwd)} companies from '{INPUT_FILE}'.")
    
    # Calculate DCF Benchmark, Alpha spread, and Error-Adjusted Conviction
    df_alpha = calculate_alpha_and_conviction(df_fwd, model_error_band_pct=3.0)
    
    # Sort by Alpha %
    df_alpha = df_alpha.sort_values(by='alpha_pct', ascending=False)
    df_alpha.to_csv(OUTPUT_FILE, index=False)
    print(f"Full Alpha Portfolio analysis saved to '{OUTPUT_FILE}'.")
    
    plot_alpha_insights(df_alpha)
    
    print("\n" + "="*85)
    print(" HIGH CONVICTION ALPHA PORTFOLIO CANDIDATES (Alpha > Model Error Band)")
    print("="*85)
    
    high_conv = df_alpha[df_alpha['investment_signal'] == 'HIGH CONVICTION BUY (Alpha > Error Band)']
    
    if len(high_conv) == 0:
        high_conv = df_alpha.head(5)
        
    for idx, row in high_conv.head(7).iterrows():
        ml_ret_str = f"+{row['expected_forward_return_pct']}%" if row['expected_forward_return_pct'] >= 0 else f"{row['expected_forward_return_pct']}%"
        alpha_str = f"+{row['alpha_pct']}%" if row['alpha_pct'] >= 0 else f"{row['alpha_pct']}%"
        
        print(f"* {row['shortName']} ({row['ticker']}) | Sector: {row['sector']}")
        print(f"  ML Model Return: {ml_ret_str} | DCF Benchmark Return: {row['dcf_benchmark_return_pct']}%")
        print(f"  Alpha Spread: {alpha_str} | Conviction Score: {row['alpha_conviction_score']}x Error Band")
        print(f"  Target Horizon N: {row['estimated_horizon_days_N']} days ({row['estimated_horizon_years']} yrs) | Signal: {row['investment_signal']}\n")
        
    return df_alpha

if __name__ == "__main__":
    main()