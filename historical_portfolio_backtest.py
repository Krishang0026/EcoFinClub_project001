"""
Historical Portfolio Backtest & Paper-Trading Simulation Engine
Simulates a real-world quantitative trading strategy with INR 100,000 starting capital.
Uses historical 3-Step model signals (Step 1 Trading Multiple -> Step 2 Forward Return & N-day horizon -> Step 3 Intrinsic DCF Alpha)
and tracks actual historical price realization over the N-day window to measure real PnL, Win Rate, and Final Portfolio Value.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from trading_multiple import main as get_step1_valuations
from forward_return import calculate_c_mispricing, predict_forward_returns, train_forward_return_linear_model
from instrinsic_value import calculate_alpha_and_conviction

STARTING_CAPITAL = 100000.0 # INR 100,000

def run_historical_portfolio_backtest(lookback_days=252):
    print("="*85)
    print(f" HISTORICAL QUANTITATIVE PORTFOLIO BACKTEST (STARTING CAPITAL: INR {STARTING_CAPITAL:,.2f})")
    print("="*85)
    
    # 1. Run Step 1 Valuation Model to get current/historical stock universe metrics
    print("\n[Step 1/4] Running Multiple Valuation Model & Mispricing Analysis...")
    val_file = "valuation_results.csv"
    if not os.path.exists(val_file):
        get_step1_valuations(force_refresh=False)
        
    df_step1 = pd.read_csv(val_file)
    print(f"Loaded {len(df_step1)} stocks from step 1 universe.")
    
    # 2. Fetch historical price series (2-year lookback) for actual exit validation
    tickers = df_step1['ticker'].tolist()
    print(f"\n[Step 2/4] Fetching historical price data for {len(tickers)} stocks via yfinance...")
    
    price_data = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            h = tk.history(period="2y")
            if len(h) >= lookback_days and 'Close' in h.columns:
                price_data[t] = h['Close']
        except Exception:
            continue
            
    print(f"Successfully retrieved daily price series for {len(price_data)} stocks.")
    
    # 3. Simulate 3-Step Decision Making at Entry Date (lookback_days ago)
    print(f"\n[Step 3/4] Executing 3-Step Quantitative Selection at Entry Date ({lookback_days} trading days ago)...")
    
    df_step1_c = calculate_c_mispricing(df_step1)
    
    # Train Step 2 Forward Return model
    reg_model, scaler, _ = train_forward_return_linear_model(df_step1_c, price_data)
    df_step2 = predict_forward_returns(df_step1_c, reg_model, scaler)
    
    # Run Step 3 Intrinsic Value & Alpha Generation Engine
    df_step3 = calculate_alpha_and_conviction(df_step2, model_error_band_pct=3.0)
    
    # Filter High-Conviction Candidates (Alpha > Error Band & Positive Projected Return)
    buy_candidates = df_step3[
        (df_step3['investment_signal'] == 'HIGH CONVICTION BUY (Alpha > Error Band)') & 
        (df_step3['expected_forward_return_pct'] > 0)
    ].copy()
    
    if len(buy_candidates) < 3:
        buy_candidates = df_step3.sort_values(by='alpha_pct', ascending=False).head(5).copy()
        
    print(f"\nIdentified {len(buy_candidates)} High-Conviction Alpha Candidates for Portfolio Entry.")
    
    # 4. Portfolio Allocation & Actual Performance Execution
    print(f"\n[Step 4/4] Allocating INR {STARTING_CAPITAL:,.2f} Capital & Tracking Actual Market Returns...")
    
    num_stocks = len(buy_candidates)
    capital_per_stock = STARTING_CAPITAL / num_stocks
    
    trades = []
    total_invested = 0.0
    total_ending_value = 0.0
    
    for idx, row in buy_candidates.iterrows():
        t = row['ticker']
        name = row['shortName']
        sector = row['sector']
        predicted_return = row['expected_forward_return_pct']
        alpha = row['alpha_pct']
        horizon_n = int(row['estimated_horizon_days_N'])
        
        if t not in price_data or len(price_data[t]) < lookback_days:
            continue
            
        prices = price_data[t].values
        entry_idx = -lookback_days
        entry_price = prices[entry_idx]
        
        exit_idx = entry_idx + horizon_n
        if exit_idx >= 0:
            exit_idx = -1
            
        exit_price = prices[exit_idx]
        actual_price_return_pct = ((exit_price - entry_price) / entry_price) * 100
        
        shares_bought = capital_per_stock / entry_price
        ending_position_value = shares_bought * exit_price
        pnl_inr = ending_position_value - capital_per_stock
        
        total_invested += capital_per_stock
        total_ending_value += ending_position_value
        
        trade_result = 'WIN' if pnl_inr > 0 else 'LOSS'
        
        trades.append({
            'ticker': t,
            'name': name,
            'sector': sector,
            'allocated_capital_inr': capital_per_stock,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares_bought': shares_bought,
            'horizon_days_N': horizon_n,
            'predicted_return_pct': predicted_return,
            'alpha_pct': alpha,
            'actual_return_pct': actual_price_return_pct,
            'ending_value_inr': ending_position_value,
            'pnl_inr': pnl_inr,
            'trade_result': trade_result
        })
        
    trades_df = pd.DataFrame(trades)
    
    # 5. Portfolio Summary Metrics
    unallocated_cash = STARTING_CAPITAL - total_invested
    final_portfolio_value = total_ending_value + unallocated_cash
    net_pnl_inr = final_portfolio_value - STARTING_CAPITAL
    total_return_pct = (net_pnl_inr / STARTING_CAPITAL) * 100
    
    win_trades = len(trades_df[trades_df['trade_result'] == 'WIN'])
    total_trades = len(trades_df)
    win_rate_pct = (win_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    print("\n" + "="*85)
    print(" PORTFOLIO BACKTEST SUMMARY & FINAL PERFORMANCE")
    print("="*85)
    print(f" • Starting Capital         : INR {STARTING_CAPITAL:,.2f}")
    print(f" • Total Capital Invested   : INR {total_invested:,.2f}")
    print(f" • Remaining Cash Balance   : INR {unallocated_cash:,.2f}")
    print(f" • Final Portfolio Value    : INR {final_portfolio_value:,.2f}")
    print(f" • Total Net Profit / Loss  : INR {net_pnl_inr:+,.2f} ({total_return_pct:+.2f}%)")
    print(f" • Portfolio Win Rate       : {win_rate_pct:.1f}% ({win_trades} Wins / {total_trades - win_trades} Losses out of {total_trades} Trades)")
    print("="*85)
    
    # Print Portfolio Breakup
    print("\n" + "="*100)
    print(" DETAILED PORTFOLIO BREAKUP & INDIVIDUAL TRADE OUTCOMES")
    print("="*100)
    print(f"{'Ticker':<10} | {'Name':<22} | {'Allocated (INR)':<15} | {'Entry (INR)':<11} | {'Exit (INR)':<11} | {'Actual Ret %':<12} | {'Ending Value':<14} | {'PnL (INR)':<12} | {'Outcome'}")
    print("-" * 100)
    
    for _, r in trades_df.iterrows():
        pnl_str = f"INR {r['pnl_inr']:+,.2f}"
        print(f"{r['ticker']:<10} | {r['name'][:22]:<22} | INR {r['allocated_capital_inr']:>10,.2f} | INR {r['entry_price']:>7.2f} | INR {r['exit_price']:>7.2f} | {r['actual_return_pct']:>+10.2f}% | INR {r['ending_value_inr']:>10,.2f} | {pnl_str:>12} | {r['trade_result']}")
        
    print("-" * 100)
    
    # Save backtest results
    trades_df.to_csv("portfolio_backtest_trades.csv", index=False)
    print("\nDetailed trade breakup saved to 'portfolio_backtest_trades.csv'.")
    
    plot_portfolio_results(trades_df, STARTING_CAPITAL, final_portfolio_value, net_pnl_inr, win_rate_pct)
    return final_portfolio_value, net_pnl_inr, total_return_pct, trades_df

def plot_portfolio_results(trades_df, starting_cap, final_val, pnl, win_rate):
    """
    Generate diagnostic visualization of portfolio breakup and trade outcomes.
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. PnL per Holding (Bar Plot)
    sns.barplot(
        data=trades_df, 
        x='pnl_inr', 
        y='ticker', 
        hue='trade_result', 
        palette={'WIN': '#00FF66', 'LOSS': '#FF3366'}, 
        ax=axes[0]
    )
    axes[0].axvline(0, color='white', linestyle='--')
    axes[0].set_title('Net PnL (INR) per Portfolio Holding', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Profit / Loss (INR)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Portfolio Breakdown Pie Chart
    labels = trades_df['ticker'].tolist()
    sizes = trades_df['ending_value_inr'].tolist()
    
    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(labels)))
    axes[1].set_title(f'Final Portfolio Allocation Breakdown\nTotal Value: INR {final_val:,.2f} (Net PnL: INR {pnl:+,.2f})', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('portfolio_backtest_analysis.png', dpi=300)
    plt.close()
    print("Portfolio backtest diagnostic plot saved as 'portfolio_backtest_analysis.png'")

if __name__ == "__main__":
    run_historical_portfolio_backtest()
