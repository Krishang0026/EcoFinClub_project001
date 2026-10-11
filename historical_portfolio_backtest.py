"""
Historical Portfolio Backtest & Risk-Adjusted Allocation Engine
Starting Capital: INR 100,000.
Implements Alpha-to-Beta Risk-Balanced Portfolio Allocation Strategy:
- Stocks with higher Alpha & higher expected forward return get larger capital weights.
- Stocks with higher Beta are penalized for risk balancing (Treynor-style risk budgeting).
- Position caps prevent excessive single-stock concentration.
- Calculates comprehensive performance: Final Value, Net PnL, Win Rate, Portfolio Beta, Sharpe & Sortino Ratios.
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
RISK_FREE_RATE = 6.8        # 6.8% Indian 10-Year Govt Bond Yield
BENCHMARK_TICKER = "^NSEI"  # Nifty 50 Index

def calculate_stock_beta(stock_closes, benchmark_closes):
    """
    Calculate systematic risk (Beta) relative to Nifty 50 index.
    Beta = Cov(R_stock, R_bench) / Var(R_bench)
    """
    try:
        r_stock = stock_closes.pct_change().dropna()
        r_bench = benchmark_closes.pct_change().dropna()
        common_dates = r_stock.index.intersection(r_bench.index)
        
        if len(common_dates) < 50:
            return 1.0
            
        cov_matrix = np.cov(r_stock.loc[common_dates], r_bench.loc[common_dates])
        var_bench = cov_matrix[1, 1]
        
        if var_bench == 0 or np.isnan(var_bench):
            return 1.0
            
        beta = cov_matrix[0, 1] / var_bench
        return float(np.clip(beta, 0.35, 2.5))
    except Exception:
        return 1.0

def calculate_risk_adjusted_weights(candidates_df, max_cap=0.22, min_floor=0.04):
    """
    Alpha-to-Beta Risk-Adjusted Weighting Strategy:
    Score = (Expected_Forward_Return * Alpha_Spread) / (Beta ^ 0.75)
    Constrains maximum position size to max_cap (22%) to prevent single-stock blowout.
    """
    scores = []
    for _, row in candidates_df.iterrows():
        fwd_ret = max(row.get('expected_forward_return_pct', 5.0), 1.0)
        alpha = max(row.get('alpha_pct', 5.0), 1.0)
        beta = max(row.get('beta', 1.0), 0.4)
        
        # Risk-Balanced Treynor Score: rewards return & alpha, penalizes systematic beta risk
        score = (fwd_ret * alpha) / (beta ** 0.75)
        scores.append(score)
        
    scores = np.array(scores)
    raw_weights = scores / np.sum(scores)
    
    # Position size capping & re-normalization
    weights = np.clip(raw_weights, min_floor, max_cap)
    weights = weights / np.sum(weights)
    
    for _ in range(5):
        weights = np.clip(weights, min_floor, max_cap)
        weights = weights / np.sum(weights)
        
    return weights

def run_historical_portfolio_backtest(lookback_days=252):
    print("="*90)
    print(f" RISK-BALANCED QUANTITATIVE PORTFOLIO BACKTEST (STARTING CAPITAL: INR {STARTING_CAPITAL:,.2f})")
    print("="*90)
    
    # 1. Run Enhanced Step 1 Valuation Model
    print("\n[Step 1/5] Executing Enhanced Valuation Model (trading_multiple.py)...")
    val_file = "valuation_results.csv"
    get_step1_valuations(force_refresh=False)
    df_step1 = pd.read_csv(val_file)
    print(f"Evaluated {len(df_step1)} stocks in fundamental universe.")
    
    # 2. Fetch Stock Price Histories & Nifty 50 Benchmark
    tickers = df_step1['ticker'].tolist()
    print(f"\n[Step 2/5] Fetching historical price series for {len(tickers)} stocks and Nifty 50 index...")
    
    price_data = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            h = tk.history(period="2y")
            if len(h) >= lookback_days and 'Close' in h.columns:
                price_data[t] = h['Close']
        except Exception:
            continue
            
    bench_hist = yf.Ticker(BENCHMARK_TICKER).history(period="2y")['Close']
    print(f"Retrieved price histories for {len(price_data)} stocks + Nifty 50 benchmark.")
    
    # 3. Simulate 3-Step Selection
    print(f"\n[Step 3/5] Executing 3-Step Quantitative Selection at Entry Date ({lookback_days} days ago)...")
    df_step1_c = calculate_c_mispricing(df_step1)
    
    reg_model, scaler, fcols = train_forward_return_linear_model(df_step1_c, price_data)
    df_step2 = predict_forward_returns(df_step1_c, reg_model, scaler, fcols)
    df_step3 = calculate_alpha_and_conviction(df_step2, model_error_band_pct=3.0)
    
    # Filter High-Conviction Alpha Candidates with Positive Expected Return
    buy_candidates = df_step3[
        (df_step3['investment_signal'] == 'HIGH CONVICTION BUY (Alpha > Error Band)') & 
        (df_step3['expected_forward_return_pct'] > 0)
    ].copy()
    
    if len(buy_candidates) < 4:
        buy_candidates = df_step3[df_step3['expected_forward_return_pct'] > 0].sort_values(by='alpha_pct', ascending=False).head(8).copy()
        
    print(f"Identified {len(buy_candidates)} High-Conviction Alpha Candidates for Risk-Adjusted Allocation.")
    
    # 4. Calculate Beta for Each Candidate
    betas = []
    for _, row in buy_candidates.iterrows():
        t = row['ticker']
        if t in price_data:
            b = calculate_stock_beta(price_data[t], bench_hist)
        else:
            b = 1.0
        betas.append(b)
        
    buy_candidates['beta'] = betas
    
    # 5. Calculate Dynamic Alpha-to-Beta Risk-Adjusted Weights
    print("\n[Step 4/5] Computing Dynamic Alpha-to-Beta Portfolio Weights...")
    weights = calculate_risk_adjusted_weights(buy_candidates, max_cap=0.22, min_floor=0.04)
    buy_candidates['portfolio_weight'] = weights
    buy_candidates['allocated_capital_inr'] = buy_candidates['portfolio_weight'] * STARTING_CAPITAL
    
    # 6. Execute Trades & Track Actual Market Returns over N-Day Horizon
    print("\n[Step 5/5] Executing Trades & Measuring Actual Realized Market PnL...")
    trades = []
    total_invested = 0.0
    total_ending_value = 0.0
    
    for idx, row in buy_candidates.iterrows():
        t = row['ticker']
        name = row['shortName']
        sector = row['sector']
        alloc_cap = row['allocated_capital_inr']
        weight_pct = row['portfolio_weight'] * 100
        predicted_ret = row['expected_forward_return_pct']
        alpha = row['alpha_pct']
        beta = row['beta']
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
        
        shares_bought = alloc_cap / entry_price
        ending_pos_value = shares_bought * exit_price
        pnl_inr = ending_pos_value - alloc_cap
        
        total_invested += alloc_cap
        total_ending_value += ending_pos_value
        
        trade_result = 'WIN' if pnl_inr > 0 else 'LOSS'
        
        trades.append({
            'ticker': t,
            'name': name,
            'sector': sector,
            'weight_pct': weight_pct,
            'allocated_capital_inr': alloc_cap,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares_bought': shares_bought,
            'horizon_days_N': horizon_n,
            'beta': beta,
            'alpha_pct': alpha,
            'predicted_return_pct': predicted_ret,
            'actual_return_pct': actual_price_return_pct,
            'ending_value_inr': ending_pos_value,
            'pnl_inr': pnl_inr,
            'trade_result': trade_result
        })
        
    trades_df = pd.DataFrame(trades)
    
    # 7. Portfolio Performance & Sharpe Ratio Calculation
    unallocated_cash = STARTING_CAPITAL - total_invested
    final_portfolio_value = total_ending_value + unallocated_cash
    net_pnl_inr = final_portfolio_value - STARTING_CAPITAL
    total_portfolio_return_pct = (net_pnl_inr / STARTING_CAPITAL) * 100
    
    trades_df['weighted_return'] = (trades_df['weight_pct'] / 100.0) * trades_df['actual_return_pct']
    portfolio_beta = np.sum((trades_df['weight_pct'] / 100.0) * trades_df['beta'])
    
    ret_series = trades_df['actual_return_pct'].values
    weights_series = trades_df['weight_pct'].values / 100.0
    
    # Position-weighted standard deviation
    portfolio_volatility = np.sqrt(np.sum(weights_series * ((ret_series - total_portfolio_return_pct) ** 2)))
    
    # Sharpe Ratio: (Portfolio Return - Risk Free Rate) / Volatility
    sharpe_ratio = (total_portfolio_return_pct - RISK_FREE_RATE) / (portfolio_volatility + 1e-4)
    gross_sharpe = total_portfolio_return_pct / (portfolio_volatility + 1e-4)
    
    # Sortino Ratio
    downside_diffs = np.minimum(ret_series - RISK_FREE_RATE, 0.0)
    downside_std = np.sqrt(np.sum(weights_series * (downside_diffs ** 2)))
    sortino_ratio = (total_portfolio_return_pct - RISK_FREE_RATE) / (downside_std + 1e-4) if downside_std > 0 else sharpe_ratio
    
    # Treynor Ratio
    treynor_ratio = (total_portfolio_return_pct - RISK_FREE_RATE) / portfolio_beta
    
    win_trades = len(trades_df[trades_df['trade_result'] == 'WIN'])
    total_trades = len(trades_df)
    win_rate_pct = (win_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    print("\n" + "="*85)
    print(" ENHANCED PORTFOLIO PERFORMANCE & SHARPE RATIO AUDIT")
    print("="*85)
    print(f" * Starting Capital              : INR {STARTING_CAPITAL:,.2f}")
    print(f" * Final Portfolio Value         : INR {final_portfolio_value:,.2f}")
    print(f" * Total Net Profit / Loss       : INR {net_pnl_inr:+,.2f} ({total_portfolio_return_pct:+.2f}%)")
    print(f" * Portfolio Win Rate            : {win_rate_pct:.1f}% ({win_trades} Wins / {total_trades - win_trades} Losses)")
    print(f" * Portfolio Systematic Beta     : {portfolio_beta:.2f} (Market Sensitivity)")
    print(f" * Portfolio Volatility (Risk)   : {portfolio_volatility:.2f}%")
    print(f" * Net Sharpe Ratio (Rf = {RISK_FREE_RATE}%): {sharpe_ratio:.3f}")
    print(f" * Gross Sharpe Ratio (Rf = 0%)  : {gross_sharpe:.3f}")
    print(f" * Sortino Ratio (Downside Risk) : {sortino_ratio:.3f}")
    print(f" * Treynor Ratio (Excess Ret / b): {treynor_ratio:.3f}")
    print("="*85)
    
    trades_df = trades_df.sort_values(by='allocated_capital_inr', ascending=False)
    
    print("\n" + "="*110)
    print(" RISK-BALANCED PORTFOLIO HOLDINGS BREAKUP")
    print("="*110)
    print(f"{'Ticker':<12} | {'Name':<20} | {'Weight':<7} | {'Allocated':<12} | {'Beta':<5} | {'Alpha':<7} | {'Ret %':<8} | {'Ending Val':<12} | {'Net PnL':<12} | {'Outcome'}")
    print("-" * 110)
    for _, r in trades_df.iterrows():
        pnl_str = f"INR {r['pnl_inr']:+,.2f}"
        print(f"{r['ticker']:<12} | {r['name'][:20]:<20} | {r['weight_pct']:>5.1f}% | INR {r['allocated_capital_inr']:>7,.0f} | {r['beta']:>4.2f} | {r['alpha_pct']:>+5.1f}% | {r['actual_return_pct']:>+6.1f}% | INR {r['ending_value_inr']:>8,.0f} | {pnl_str:>12} | {r['trade_result']}")
    print("-" * 110)
    
    trades_df.to_csv("portfolio_backtest_trades.csv", index=False)
    print("\nSaved enhanced trade breakup to 'portfolio_backtest_trades.csv'.")
    
    plot_enhanced_portfolio_results(trades_df, STARTING_CAPITAL, final_portfolio_value, net_pnl_inr, sharpe_ratio, win_rate_pct)
    return final_portfolio_value, net_pnl_inr, total_portfolio_return_pct, sharpe_ratio, trades_df

def plot_enhanced_portfolio_results(trades_df, start_val, final_val, pnl, sharpe, win_rate):
    """
    Generate professional diagnostic plots:
    1. Trade PnL bar chart with exact value annotations
    2. Capital Allocation Donut Chart
    3. Alpha vs Beta Risk Balancing Map with labeled tickers
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. PnL by Position Size with Value Annotations
    bars = sns.barplot(
        data=trades_df, 
        x='pnl_inr', 
        y='ticker', 
        hue='trade_result', 
        palette={'WIN': '#00FF66', 'LOSS': '#FF3366'}, 
        ax=axes[0]
    )
    axes[0].axvline(0, color='white', linestyle='--', alpha=0.7)
    axes[0].set_title(f'Trade Net PnL (Total: INR {pnl:+,.2f})\nSharpe Ratio: {sharpe:.3f}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Profit / Loss (INR)', fontsize=11)
    axes[0].set_ylabel('Holding Ticker', fontsize=11)
    axes[0].grid(True, alpha=0.25)
    
    # Annotate bar values
    for p in axes[0].patches:
        val = p.get_width()
        if abs(val) > 0.01:
            x_pos = val + (200 if val >= 0 else -1400)
            axes[0].text(x_pos, p.get_y() + p.get_height() / 2, f'{val:+,.0f}', va='center', fontsize=8, color='white')
            
    # 2. Portfolio Allocation Donut Chart
    labels = trades_df['ticker'].tolist()
    weights = trades_df['weight_pct'].tolist()
    colors = sns.color_palette('tab10', len(labels))
    
    wedges, texts, autotexts = axes[1].pie(
        weights, 
        labels=labels, 
        autopct='%1.1f%%', 
        pctdistance=0.75, 
        colors=colors, 
        wedgeprops=dict(width=0.45, edgecolor='black')
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color('white')
    axes[1].set_title(f'Capital Allocation Weights\nTotal Capital: INR {start_val:,.0f} -> {final_val:,.0f}', fontsize=12, fontweight='bold')
    
    # 3. Alpha vs. Beta Risk Balancing Map
    scatter = sns.scatterplot(
        data=trades_df, 
        x='beta', 
        y='alpha_pct', 
        size='weight_pct', 
        sizes=(80, 350), 
        hue='trade_result', 
        palette={'WIN': '#00FF66', 'LOSS': '#FF3366'}, 
        ax=axes[2]
    )
    axes[2].set_title(f'Alpha vs. Beta Risk Balancing Map\nWin Rate: {win_rate:.1f}%', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Stock Beta (Systematic Market Risk)', fontsize=11)
    axes[2].set_ylabel('Model Generated Alpha %', fontsize=11)
    axes[2].grid(True, alpha=0.25)
    
    # Annotate ticker labels on scatter plot
    for _, r in trades_df.iterrows():
        axes[2].annotate(
            r['ticker'].replace('.NS', ''), 
            (r['beta'], r['alpha_pct']), 
            textcoords="offset points", 
            xytext=(0, 7), 
            ha='center', 
            fontsize=8, 
            fontweight='bold', 
            color='yellow'
        )
        
    plt.tight_layout()
    plt.savefig('portfolio_backtest_analysis.png', dpi=300)
    plt.close()
    print("Enhanced backtest visualization saved as 'portfolio_backtest_analysis.png'")

if __name__ == "__main__":
    run_historical_portfolio_backtest()
