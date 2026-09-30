"""
Model Trainer Module
Trains machine learning regression models (Ridge, Random Forest, XGBoost, LightGBM)
to estimate Market Cap based on financial ratios, log terms, cross-terms, and classifications.
Generates performance metrics and valuation mispricing analysis.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from feature_engineering import preprocess_and_engineer_features

def prepare_feature_matrix(df_engineered):
    """
    Separate features (X) and target (y), drop non-feature metadata.
    """
    drop_cols = ['ticker', 'symbol', 'shortName', 'marketCap', 'log_marketCap', 'sector', 'industry', 'size_category', 'growth_category']
    feature_cols = [c for c in df_engineered.columns if c not in drop_cols]
    
    X = df_engineered[feature_cols].copy()
    y = df_engineered['log_marketCap'].values
    
    # Replace any remaining Inf or NaN with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return X, y, feature_cols

def train_and_evaluate_models(X, y):
    """
    Train multiple models using 5-Fold Cross Validation and train/test splits.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    
    models = {
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model_obj = None
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*60)
    print(" MACHINE LEARNING MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        # Cross validation scores on scaled data
        cv_scores = cross_val_score(model, X_scaled_df, y, cv=kf, scoring='r2')
        mean_cv_r2 = cv_scores.mean()
        
        # Fit on train split and test on test split
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mean_cv_r2': mean_cv_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        
        print(f"Model: {name:<20} | 5-Fold CV R²: {mean_cv_r2:.4f} | Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")
        
        if mean_cv_r2 > best_r2:
            best_r2 = mean_cv_r2
            best_model_name = name
            best_model_obj = model
            
    print("-" * 60)
    print(f"Selected Best Model: {best_model_name} (CV R² = {best_r2:.4f})")
    
    # Fit best model on full dataset for maximum prediction accuracy
    best_model_obj.fit(X_scaled_df, y)
    
    return best_model_name, best_model_obj, scaler, results

def calculate_valuation_estimates(df_engineered, model, scaler, feature_cols):
    """
    Predict market cap, compare with actual market cap, and determine valuation metrics.
    """
    X_full = df_engineered[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(X_full)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    pred_log_mcap = model.predict(X_scaled_df)
    estimated_mcap = np.exp(pred_log_mcap)
    
    res_df = df_engineered.copy()
    res_df['estimated_marketCap'] = estimated_mcap
    
    # Calculate difference percentage: (Estimated - Actual) / Actual * 100
    res_df['valuation_diff_pct'] = ((res_df['estimated_marketCap'] - res_df['marketCap']) / res_df['marketCap']) * 100
    
    # Relative thresholding for valuation status
    def label_valuation(diff):
        if diff > 3.0:
            return 'Undervalued'
        elif diff < -3.0:
            return 'Overvalued'
        else:
            return 'Fairly Valued'
            
    res_df['valuation_status'] = res_df['valuation_diff_pct'].apply(label_valuation)
    return res_df

def plot_model_insights(res_df, model, feature_cols, model_name):
    """
    Generate and save visualization plots: Scatter plot, Feature Importance.
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Actual vs Estimated Market Cap (in Cr INR)
    actual_cr = res_df['marketCap'] / 1e7
    estimated_cr = res_df['estimated_marketCap'] / 1e7
    
    sns.scatterplot(x=actual_cr, y=estimated_cr, hue=res_df['sector'], ax=axes[0], s=80, palette='viridis')
    axes[0].plot([actual_cr.min(), actual_cr.max()], [actual_cr.min(), actual_cr.max()], 'r--', label='Ideal 1:1 Valuation')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Actual Market Cap (₹ Cr, Log Scale)', fontsize=12)
    axes[0].set_ylabel('ML Model Estimated Valuation (₹ Cr, Log Scale)', fontsize=12)
    axes[0].set_title(f'Actual vs Model Valuation ({model_name})', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # 2. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        imp_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(12)
        sns.barplot(x=imp_series.values, y=imp_series.index, ax=axes[1], hue=imp_series.index, legend=False, palette='rocket')
        axes[1].set_title('Top 12 Features Driving Valuation Estimate', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Feature Importance', fontsize=12)
    elif hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_)
        coef_series = pd.Series(coefs, index=feature_cols).sort_values(ascending=False).head(12)
        sns.barplot(x=coef_series.values, y=coef_series.index, ax=axes[1], hue=coef_series.index, legend=False, palette='rocket')
        axes[1].set_title('Top 12 Feature Coefficient Magnitudes (Ridge)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Absolute Coefficient', fontsize=12)
        
    plt.tight_layout()
    plt.savefig('valuation_analysis_plots.png', dpi=300)
    plt.close()
    print("\nAnalysis plot saved as 'valuation_analysis_plots.png'")

if __name__ == "__main__":
    if os.path.exists("nifty50_company_financial_data.csv"):
        raw_df = pd.read_csv("nifty50_company_financial_data.csv")
        eng_df = preprocess_and_engineer_features(raw_df)
        X, y, fcols = prepare_feature_matrix(eng_df)
        best_name, best_model, scaler, _ = train_and_evaluate_models(X, y)
        res = calculate_valuation_estimates(eng_df, best_model, scaler, fcols)
        plot_model_insights(res, best_model, fcols, best_name)
