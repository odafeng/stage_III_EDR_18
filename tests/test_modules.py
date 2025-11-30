import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import clean_data, create_preprocessor
from feature_selection import select_features_lasso, select_features_xgboost
from modeling import train_xgboost_nested_cv, calibrate_model, train_logistic_regression
from evaluation import calculate_metrics, bootstrap_ci, get_optimal_cutoff
from visualization import plot_roc_curve

def test_data_processing():
    print("Testing data_processing...")
    # Create dummy data
    df = pd.DataFrame({
        'LNR': ['10%', '20%', '5%'],
        'Sex': [1, 2, 1],
        'Tumor_Location_Group': [1, 2, 1],
        'LVI': [0, 1, 0],
        'PNI': [0, 0, 1],
        'ECOG': [0, 1, 0],
        'edr_18m': [0, 1, 0],
        'edr_24m': [0, 1, 0],
        'Differentiation': [1, 2, 9],
        'MSI_Status': ['MSS', 'MSI-H', 'MSS'],
        'pT_Stage': ['3', '4A', '2'],
        'pN_Stage': ['1A', '2B', '1B'],
        'AJCC_Substage': ['3A', '3C', '3B']
    })
    
    cleaned_df = clean_data(df)
    
    assert cleaned_df['LNR'].dtype == 'float64'
    assert cleaned_df['Sex'].dtype == 'int64'
    assert cleaned_df['Tumor_Location_Group'].dtype == 'int64'
    assert cleaned_df['Differentiation'].isna().sum() == 1
    assert 'MSI_High' in cleaned_df.columns
    assert 'pT_Stage_Num' in cleaned_df.columns
    
    print("data_processing passed!")
    return cleaned_df

def test_feature_selection(df):
    print("Testing feature_selection...")
    
    # Prepare data for selection
    # We need to define feature lists
    num_features = ['LNR']
    bin_features = ['Sex', 'Tumor_Location_Group', 'LVI', 'PNI', 'ECOG', 'MSI_High']
    nom_features = [] # Let's assume no nominals for this simple test or add one
    
    # Add a dummy nominal feature for testing pipeline
    df['DummyNom'] = ['A', 'B', 'A']
    nom_features = ['DummyNom']
    
    # Target
    y = df['edr_18m'].astype(int)
    X = df[num_features + bin_features + nom_features]
    
    # Preprocessor
    preprocessor = create_preprocessor(num_features, bin_features, nom_features)
    
    # Test LASSO
    # We need more samples for CV to work without warning/error, but let's try
    # Duplicate data to have enough samples
    X_large = pd.concat([X]*10, ignore_index=True)
    y_large = pd.concat([y]*10, ignore_index=True)
    
    selected_features = select_features_lasso(X_large, y_large, preprocessor)
    print(f"LASSO selected: {selected_features}")
    
    # Test XGBoost
    # For XGBoost we might need numeric input if we don't use the preprocessor inside
    # select_features_xgboost takes X directly. 
    # Let's pass X_large but we need to handle 'DummyNom' if it's string.
    # The clean_data handles specific columns. 'DummyNom' is not handled.
    # Let's drop it for XGBoost test or encode it.
    X_xgb = X_large.drop(columns=['DummyNom'])
    selected_features_xgb = select_features_xgboost(X_xgb, y_large)
    print(f"XGBoost selected: {selected_features_xgb}")
    
    print("feature_selection passed!")
    return X_large, y_large, preprocessor

def test_modeling(X, y):
    print("Testing modeling...")
    
    param_grid = {
        'n_estimators': [10, 20],
        'max_depth': [2, 3],
        'learning_rate': [0.1]
    }
    
    # We need to drop non-numeric for XGBoost if not using pipeline inside (which we aren't in the function)
    if 'DummyNom' in X.columns:
        X = X.drop(columns=['DummyNom'])
        
    search = train_xgboost_nested_cv(X, y, param_grid, outer_splits=2, inner_splits=2)
    print(f"Best params: {search.best_params_}")
    
    # Test calibration
    calibrated = calibrate_model(search.best_estimator_, X, y, cv=2)
    
    print("modeling passed!")
    return calibrated

def test_evaluation():
    print("Testing evaluation...")
    y_true = [0, 1, 0, 1, 0]
    y_prob = [0.1, 0.9, 0.2, 0.8, 0.4]
    
    metrics = calculate_metrics(y_true, np.array(y_prob), 0.5)
    print(f"Metrics: {metrics}")
    
    cutoff = get_optimal_cutoff(y_true, y_prob)
    print(f"Optimal cutoff: {cutoff}")
    
    ci = bootstrap_ci(y_true, y_prob, lambda y, p: calculate_metrics(y, p, 0.5)['Sensitivity'], n_boots=10)
    print(f"CI: {ci}")
    
    print("evaluation passed!")

if __name__ == "__main__":
    cleaned_df = test_data_processing()
    X, y, preprocessor = test_feature_selection(cleaned_df)
    model = test_modeling(X, y)
    test_evaluation()
    print("All tests passed!")
