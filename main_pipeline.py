"""
Master Pipeline Orchestrator for 3-Step Quantitative Investment Engine
1. Step 1: Multiple Valuation Model (trading_multiple.py)
2. Step 2: Forward Return & N-Day Horizon Modeling (forward_return.py)
3. Step 3: Intrinsic Value & Alpha Generation Engine (instrinsic_value.py)
"""

import sys
import time
import os

from trading_multiple import main as run_step_1
from forward_return import main as run_step_2
from instrinsic_value import main as run_step_3

def run_full_pipeline(force_refresh_data=False):
    start_time = time.time()
    print("#"*80)
    print(" ECOFIN QUANTITATIVE INVESTMENT PRODUCT - 3-STEP PIPELINE EXECUTION")
    print("#"*80)
    
    # Step 1: Valuation Multiple & Mispricing Model
    print("\n" + "="*80)
    print(" RUNNING STEP 1: MARKET CAP VALUATION MODEL (trading_multiple.py)")
    print("="*80)
    step1_df, best_model_name, _ = run_step_1(force_refresh=force_refresh_data)
    
    # Step 2: Forward Return & Horizon N Modeling
    print("\n" + "="*80)
    print(" RUNNING STEP 2: FORWARD RETURN & HORIZON N MODELING (forward_return.py)")
    print("="*80)
    step2_df = run_step_2()
    
    # Step 3: Intrinsic Value & Alpha Generation Engine
    print("\n" + "="*80)
    print(" RUNNING STEP 3: INTRINSIC VALUE & ALPHA GENERATION ENGINE (instrinsic_value.py)")
    print("="*80)
    step3_df = run_step_3()
    
    elapsed = time.time() - start_time
    print("#"*80)
    print(f" PIPELINE COMPLETE! Total Execution Time: {elapsed:.2f} seconds")
    print(" Output Artifacts Generated:")
    print("  - Step 1 Results: valuation_results.csv & valuation_analysis_plots.png")
    print("  - Step 2 Results: forward_return_results.csv")
    print("  - Step 3 Results: final_alpha_portfolio.csv & intrinsic_alpha_analysis.png")
    print("#"*80)

if __name__ == "__main__":
    force_refresh = "--refresh" in sys.argv
    run_full_pipeline(force_refresh_data=force_refresh)
