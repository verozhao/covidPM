# Data preparation for modeling
"""
Data preparation for machine learning modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import warnings

from ..config import RANDOM_SEED, TEST_SIZE
from ..data.imputation import advanced_imputation_pipeline
from ..utils.diagnostics import check_nan_status


def prepare_data_for_modeling_publication(
    df: pd.DataFrame, 
    target_col: str = 'COVID_severity_binary'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], Dict[str, Any]]:
    """
    Publication-quality data preparation pipeline for modeling:
    1. Apply advanced imputation with multiple strategies
    2. Process categorical features with appropriate encoding
    3. Scale numerical features using robust techniques
    4. Apply feature selection to improve model stability
    5. Split data with stratification to maintain class balance
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to prepare
    target_col : str
        The target column for prediction
        
    Returns:
    --------
    Tuple containing:
    - X_train: Training features
    - X_test: Test features
    - y_train: Training target
    - y_test: Test target
    - feature_cols: Selected feature names
    - imputation_metadata: Dictionary with imputation documentation for methods section
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("\n=== PREPARING DATA FOR MODELING (Publication Quality) ===")
    
    # 1. Apply advanced imputation with documentation
    print("\nStep 1: Advanced Multi-Strategy Imputation")
    df_imputed, imputation_metadata = advanced_imputation_pipeline(
        df, 
        target_col=['COVID_severity_3group', 'COVID_severity_binary']
    )
    
    # 2. Process categorical features
    print("\nStep 2: Encoding Categorical Features")
    # Identify categorical columns (excluding target variables)
    categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols
                      if col not in ['COVID_severity_3group', 'COVID_severity_binary']]
    
    # Use pandas get_dummies for one-hot encoding of categorical variables
    # This is a cleaner approach for scientific research than label encoding
    if categorical_cols:
        print(f"One-hot encoding {len(categorical_cols)} categorical features")
        df_encoded = pd.get_dummies(
            df_imputed, 
            columns=categorical_cols,
            drop_first=True,  # Drop first to avoid multicollinearity
            dummy_na=False    # We've already handled NAs
        )
    else:
        df_encoded = df_imputed.copy()
    
    # Add explicit encoding for Gender if present (binary feature)
    if 'Gender' in df_encoded.columns:
        df_encoded['Gender_binary'] = df_encoded['Gender'].map({'M': 1, 'F': 0})
        # Drop original to avoid duplication
        df_encoded = df_encoded.drop(columns=['Gender'])
    
    # 3. Scale numerical features with robust scaling
    print("\nStep 3: Scaling Numerical Features")
    # Identify numerical columns (excluding target variables and IDs)
    numerical_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols
                     if col not in ['COVID_severity_3group', 'COVID_severity_binary',
                                   'MRN', 'CSN', 'HAR']]
    
    # Use RobustScaler to handle outliers better (important for clinical data)
    try:
        print(f"Robust scaling {len(numerical_cols)} numerical features")
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        
        df_scaled = df_encoded.copy()
        df_scaled[numerical_cols] = pd.DataFrame(
            scaler.fit_transform(df_encoded[numerical_cols]),
            columns=numerical_cols,
            index=df_encoded.index
        )
    except Exception as e:
        print(f"Error in robust scaling: {e}")
        print("Falling back to standard scaling")
        
        # Fallback to standard scaling
        scaler = StandardScaler()
        df_scaled = df_encoded.copy()
        df_scaled[numerical_cols] = pd.DataFrame(
            scaler.fit_transform(df_encoded[numerical_cols]),
            columns=numerical_cols,
            index=df_encoded.index
        )
    
    # 4. Define features and target
    print("\nStep 4: Feature Selection")
    # Identify potential features (exclude targets and identifiers)
    potential_features = [col for col in df_scaled.columns 
                         if col != target_col
                         and 'COVID_severity' not in col
                         and col not in ['MRN', 'CSN', 'HAR', 'Date.Tested', 'Admission.Arrival.Date']]
    
    X = df_scaled[potential_features].copy()
    y = df_scaled[target_col].copy()
    
    # Optional: Feature selection with Random Forest
    # This is an accepted approach in scientific literature
    try:
        print("Performing feature selection...")
        # Initial feature selector
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight='balanced'),
            threshold='median'  # Use median importance as threshold
        )
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        # If we'd lose too many features, adjust selection criteria
        if len(selected_features) < max(10, len(potential_features) // 4):
            print("Too few features selected, adjusting threshold...")
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight='balanced'),
                threshold='mean'  # Use mean importance (selects more features)
            )
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"Selected {len(selected_features)} features from {len(potential_features)} candidates")
        
        # If we still have too few features, skip feature selection
        if len(selected_features) < max(5, len(potential_features) // 10):
            print("Feature selection would eliminate too many features, skipping...")
            feature_cols = potential_features
        else:
            feature_cols = selected_features
            X = X[feature_cols]
            
    except Exception as e:
        print(f"Feature selection error: {e}")
        print("Using all features instead")
        feature_cols = potential_features
    
    # 5. Final verification before splitting
    print("\nStep 5: Final Data Verification")
    # Verify no missing values
    if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
        raise ValueError(f"ERROR: Dataset still contains NaN values after preprocessing! X: {X.isna().sum().sum()}, y: {y.isna().sum()}")
    else:
        print("✓ No missing values detected")
    
    # Verify no infinite values
    if np.isinf(X).sum().sum() > 0:
        print("WARNING: Infinite values detected in features, replacing with large values")
        X = X.replace([np.inf, -np.inf], [1e9, -1e9])
    else:
        print("✓ No infinite values detected")
    
    # 6. Split data into train and test sets with stratification
    print("\nStep 6: Train-Test Splitting with Stratification")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
    print(f"Class distribution - Test: {np.bincount(y_test.astype(int))}")
    
    # Final NaN check
    check_nan_status(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test, feature_cols, imputation_metadata