# Missing value imputation
"""
Advanced imputation strategies for missing data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
import warnings

from ..config import MISSING_VALUE_THRESHOLDS, IMPUTATION_STRATEGIES


def advanced_imputation_pipeline(
    df: pd.DataFrame, 
    target_col: Optional[Union[str, List[str]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Advanced imputation pipeline suitable for scientific research publications.
    Uses multiple imputation techniques tailored to feature characteristics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset with missing values
    target_col : str or list, optional
        Target column(s) to exclude from imputation
        
    Returns:
    --------
    pd.DataFrame
        The imputed dataset with no missing values
    dict
        Imputation metadata for documentation in methods section
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Initialize metadata for documentation
    imputation_metadata = {
        'initial_missingness': {},
        'strategies_used': {},
        'feature_groups': {},
        'final_check': {}
    }
    
    # Document initial missingness
    missing_percent = (data.isna().sum() / len(data) * 100).round(2)
    imputation_metadata['initial_missingness'] = {
        'total_missing_values': data.isna().sum().sum(),
        'percent_missing_by_feature': missing_percent.to_dict(),
        'features_with_high_missingness': missing_percent[missing_percent > 30].index.tolist()
    }
    
    print(f"Initial data shape: {data.shape}")
    print(f"Initial missing values: {data.isna().sum().sum()}")
    
    # Exclude target column(s) from imputation if specified
    if target_col:
        if isinstance(target_col, str):
            target_cols = [target_col]
        else:
            target_cols = target_col
            
        target_data = data[target_cols].copy()
        impute_data = data.drop(columns=target_cols)
    else:
        target_cols = []
        target_data = None
        impute_data = data
    
    # Split features by type and missingness level
    numeric_cols = impute_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = impute_data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Further classify numeric features by missingness
    high_missing_numeric = [col for col in numeric_cols 
                           if impute_data[col].isna().mean() > MISSING_VALUE_THRESHOLDS['moderate']]
    moderate_missing_numeric = [col for col in numeric_cols 
                               if MISSING_VALUE_THRESHOLDS['low'] < impute_data[col].isna().mean() <= MISSING_VALUE_THRESHOLDS['moderate']]
    low_missing_numeric = [col for col in numeric_cols 
                          if impute_data[col].isna().mean() <= MISSING_VALUE_THRESHOLDS['low']]
    
    # Classify categorical features by missingness
    high_missing_cat = [col for col in categorical_cols 
                       if impute_data[col].isna().mean() > MISSING_VALUE_THRESHOLDS['moderate']]
    low_missing_cat = [col for col in categorical_cols 
                      if impute_data[col].isna().mean() <= MISSING_VALUE_THRESHOLDS['moderate']]
    
    # Document feature grouping
    imputation_metadata['feature_groups'] = {
        'numeric_features': len(numeric_cols),
        'categorical_features': len(categorical_cols),
        'high_missing_numeric': high_missing_numeric,
        'moderate_missing_numeric': moderate_missing_numeric,
        'low_missing_numeric': low_missing_numeric,
        'high_missing_categorical': high_missing_cat,
        'low_missing_categorical': low_missing_cat
    }
    
    print(f"\nFeature grouping:")
    print(f"- {len(low_missing_numeric)} numeric features with low missingness (<10%)")
    print(f"- {len(moderate_missing_numeric)} numeric features with moderate missingness (10-30%)")
    print(f"- {len(high_missing_numeric)} numeric features with high missingness (>30%)")
    print(f"- {len(low_missing_cat)} categorical features with low missingness (<30%)")
    print(f"- {len(high_missing_cat)} categorical features with high missingness (>30%)")
    
    # 1. Handle categorical features first
    if categorical_cols:
        print("\nImputing categorical features...")
        # For low missingness categorical features: most frequent value
        if low_missing_cat:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            impute_data[low_missing_cat] = pd.DataFrame(
                cat_imputer.fit_transform(impute_data[low_missing_cat]),
                columns=low_missing_cat,
                index=impute_data.index
            )
            imputation_metadata['strategies_used']['low_missing_categorical'] = 'most_frequent'
        
        # For high missingness categorical features: add missing indicator + most frequent
        if high_missing_cat:
            # Add missing indicators
            for col in high_missing_cat:
                impute_data[f"{col}_was_missing"] = impute_data[col].isna().astype(int)
            
            # Then impute with most frequent
            high_cat_imputer = SimpleImputer(strategy='most_frequent')
            impute_data[high_missing_cat] = pd.DataFrame(
                high_cat_imputer.fit_transform(impute_data[high_missing_cat]),
                columns=high_missing_cat,
                index=impute_data.index
            )
            imputation_metadata['strategies_used']['high_missing_categorical'] = 'missing_indicator + most_frequent'
    
    # 2. Handle numeric features with different strategies based on missingness
    if numeric_cols:
        print("\nImputing numeric features...")
        # For low missingness: median imputation (more robust than mean)
        if low_missing_numeric:
            num_imputer = SimpleImputer(strategy='median')
            impute_data[low_missing_numeric] = pd.DataFrame(
                num_imputer.fit_transform(impute_data[low_missing_numeric]),
                columns=low_missing_numeric,
                index=impute_data.index
            )
            imputation_metadata['strategies_used']['low_missing_numeric'] = IMPUTATION_STRATEGIES['low_missing_numeric']
        
        # For moderate missingness: KNN imputation
        if moderate_missing_numeric:
            # Start with median imputation for any features needed as predictors by KNN
            init_imputer = SimpleImputer(strategy='median')
            impute_data[numeric_cols] = pd.DataFrame(
                init_imputer.fit_transform(impute_data[numeric_cols]),
                columns=numeric_cols,
                index=impute_data.index
            )
            
            # Then apply KNN to the moderate missing features
            # Use a modest n_neighbors to avoid overfitting
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            features_for_knn = moderate_missing_numeric + low_missing_numeric
            if features_for_knn:
                tmp_knn_data = impute_data[features_for_knn].copy()
                imputed_knn = knn_imputer.fit_transform(tmp_knn_data)
                # Only update the moderate missing columns
                moderate_indices = [features_for_knn.index(col) for col in moderate_missing_numeric]
                for i, col_idx in enumerate(moderate_indices):
                    impute_data[moderate_missing_numeric[i]] = imputed_knn[:, col_idx]
                
                imputation_metadata['strategies_used']['moderate_missing_numeric'] = IMPUTATION_STRATEGIES['moderate_missing_numeric']
        
        # For high missingness: MICE (Multiple Imputation by Chained Equations) with safeguards
        if high_missing_numeric:
            # First ensure we have initial values for all features
            # This prevents errors in the iterative imputer
            for col in high_missing_numeric:
                if impute_data[col].isna().any():
                    impute_data[col] = impute_data[col].fillna(impute_data[col].median())
            
            # Add missing indicators for high missingness features
            for col in high_missing_numeric:
                impute_data[f"{col}_was_missing"] = df[col].isna().astype(int)
            
            # Setup the iterative imputer with robust estimator
            try:
                # Use Extra Trees which handle mixed data types better
                estimator = ExtraTreesRegressor(n_estimators=50, random_state=42)
                iterative_imputer = IterativeImputer(
                    estimator=estimator,
                    max_iter=10,
                    random_state=42,
                    verbose=0,
                    imputation_order='random'
                )
                
                # Prepare features for imputation - use all numeric plus encoded categoricals
                numeric_features = numeric_cols + [c for c in impute_data.columns if '_was_missing' in c]
                
                # Apply iterative imputation
                impute_data[numeric_features] = pd.DataFrame(
                    iterative_imputer.fit_transform(impute_data[numeric_features]),
                    columns=numeric_features,
                    index=impute_data.index
                )
                
                imputation_metadata['strategies_used']['high_missing_numeric'] = IMPUTATION_STRATEGIES['high_missing_numeric']
                
            except Exception as e:
                print(f"Iterative imputation failed: {e}")
                print("Falling back to median imputation for high missingness features")
                
                # Fallback to median imputation
                for col in high_missing_numeric:
                    if impute_data[col].isna().any():
                        impute_data[col] = impute_data[col].fillna(impute_data[col].median())
                
                imputation_metadata['strategies_used']['high_missing_numeric'] = 'median (fallback)'
    
    # 3. Reconstruct the full dataframe if target column was excluded
    if target_cols:
        imputed_df = pd.concat([impute_data, target_data], axis=1)
    else:
        imputed_df = impute_data
    
    # 4. Final check for any remaining missing values
    final_missing = imputed_df.isna().sum().sum()
    
    # If we still have missing values, apply final median/mode imputation
    if final_missing > 0:
        print(f"\nWARNING: {final_missing} missing values remain after primary imputation")
        cols_with_missing = imputed_df.columns[imputed_df.isna().any()].tolist()
        print(f"Columns with missing values: {cols_with_missing}")
        
        # Apply final imputation
        for col in cols_with_missing:
            if imputed_df[col].dtype in ['int64', 'float64']:
                imputed_df[col] = imputed_df[col].fillna(imputed_df[col].median() if not pd.isna(imputed_df[col].median()) else 0)
            else:
                imputed_df[col] = imputed_df[col].fillna(imputed_df[col].mode()[0] if not imputed_df[col].mode().empty else "Unknown")
    
    # Final verification
    post_imputation_missing = imputed_df.isna().sum().sum()
    imputation_metadata['final_check'] = {
        'remaining_missing_values': post_imputation_missing,
        'imputation_successful': post_imputation_missing == 0
    }
    
    print(f"\nImputation complete: {post_imputation_missing} missing values remain")
    if post_imputation_missing > 0:
        raise ValueError("Imputation failed to address all missing values.")
    
    return imputed_df, imputation_metadata


def handle_missing_values(df: pd.DataFrame, 
                         numerical_strategy: str = 'iterative', 
                         categorical_strategy: str = 'mode') -> pd.DataFrame:
    """
    Basic missing value handling with various strategies.
    For more sophisticated imputation, use advanced_imputation_pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe with missing values
    numerical_strategy : str
        Strategy for numerical columns: 'iterative', 'knn', or 'median'
    categorical_strategy : str
        Strategy for categorical columns: 'mode' or 'new_category'
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with imputed values
    """
    # Make a copy of the dataframe
    df_imputed = df.copy()
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

    # Handle numerical missing values
    if numerical_strategy == 'iterative':
        # Multivariate imputation by chained equations
        imp = IterativeImputer(max_iter=10, random_state=42)
        df_imputed[numerical_cols] = pd.DataFrame(
            imp.fit_transform(df[numerical_cols]),
            columns=numerical_cols,
            index=df.index
        )
    elif numerical_strategy == 'knn':
        # KNN imputation
        imp = KNNImputer(n_neighbors=5)
        df_imputed[numerical_cols] = pd.DataFrame(
            imp.fit_transform(df[numerical_cols]),
            columns=numerical_cols,
            index=df.index
        )
    elif numerical_strategy == 'median':
        for col in numerical_cols:
            df_imputed[col].fillna(df[col].median(), inplace=True)

    # Handle categorical missing values
    if categorical_strategy == 'mode':
        for col in categorical_cols:
            df_imputed[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    elif categorical_strategy == 'new_category':
        for col in categorical_cols:
            df_imputed[col].fillna('Unknown', inplace=True)

    # Final check for any remaining missing values
    remaining_missing = df_imputed.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"WARNING: {remaining_missing} missing values remain after imputation!")
        
        # Last resort: fill remaining missing values
        for col in df_imputed.columns[df_imputed.isnull().any()]:
            if df_imputed[col].dtype in ['int64', 'float64']:
                df_imputed[col].fillna(0, inplace=True)
            else:
                df_imputed[col].fillna('Unknown', inplace=True)
    
    print(f"Imputation complete. Remaining missing values: {df_imputed.isnull().sum().sum()}")
    return df_imputed