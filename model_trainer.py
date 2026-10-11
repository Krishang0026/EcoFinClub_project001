"""
Model Trainer Module (Enhanced & Overfitting-Regularized)
Trains machine learning regression models with strong L1/L2 regularization,
depth constraints, cross-validation tuning, and an Ensemble Blended meta-regressor.
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
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
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
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y, feature_cols

def train_and_evaluate_models(X, y):
    """
    Train multiple models using 5-Fold Cross Validation with StandardScaler and regularization.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    
    # Regularized models
    m_ridge = Ridge(alpha=50.0, random_state=42)
    m_elastic = ElasticNet(alpha=0.3, l1_ratio=0.2, random_state=42)
    m_rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_split=4, min_samples_leaf=2, max_features='sqrt', random_state=42)
    m_xgb = XGBRegressor(
        n_estimators=80, 
        learning_rate=0.04, 
        max_depth=3, 
        subsample=0.8, 
        colsample_bytree=0.7, 
        reg_alpha=2.0, 
        reg_lambda=10.0, 
        random_state=42, 
        verbosity=0
    )
    m_gbr = GradientBoostingRegressor(n_estimators=80, learning_rate=0.04, max_depth=3, subsample=0.8, max_features='sqrt', random_state=42)
    
    models = {
        'Regularized Ridge': m_ridge,
        'ElasticNet': m_elastic,
        'Regularized Random Forest': m_rf,
        'Regularized XGBoost': m_xgb,
        'Regularized Gradient Boosting': m_gbr
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model_obj = None
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*70)
    print(" ENHANCED REGULARIZED MODEL PERFORMANCE EVALUATION")
    print("="*70)
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_scaled_df, y, cv=kf, scoring='r2')
        mean_cv_r2 = cv_scores.mean()
        
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
        
        print(f"Model: {name:<30} | 5-Fold CV R²: {mean_cv_r2:.4f} | Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")
        
        if mean_cv_r2 > best_r2:
            best_r2 = mean_cv_r2
            best_model_name = name
            best_model_obj = model
            
    print("-" * 70)
    print(f"Selected Best Regularized Architecture: {best_model_name} (CV R² = {best_r2:.4f})")
    
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
    res_df['valuation_diff_pct'] = ((res_df['estimated_marketCap'] - res_df['marketCap']) / res_df['marketCap']) * 100
    
    def label_valuation(diff):
        if diff > 5.0:
            return 'Undervalued'
        elif diff < -5.0:
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
    
    actual_cr = res_df['marketCap'] / 1e7
    estimated_cr = res_df['estimated_marketCap'] / 1e7
    
    sns.scatterplot(x=actual_cr, y=estimated_cr, hue=res_df['sector'], ax=axes[0], s=80, palette='viridis')
    axes[0].plot([actual_cr.min(), actual_cr.max()], [actual_cr.min(), actual_cr.max()], 'r--', label='Ideal 1:1 Valuation')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Actual Market Cap (INR Cr, Log Scale)', fontsize=12)
    axes[0].set_ylabel('Model Estimated Valuation (INR Cr, Log Scale)', fontsize=12)
    axes[0].set_title(f'Actual vs Model Valuation ({model_name})', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    axes[1].set_title('Model Valuation Spread Distribution', fontsize=14, fontweight='bold')
    sns.histplot(res_df['valuation_diff_pct'], kde=True, ax=axes[1], color='#00FFCC')
    axes[1].set_xlabel('Valuation Spread % (Estimated vs Actual)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
        
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
