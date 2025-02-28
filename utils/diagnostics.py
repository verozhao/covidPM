# Diagnostic tools
"""
Diagnostic tools for data analysis and quality assurance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any

def diagnose_problematic_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Diagnose problematic columns in a dataframe:
    1. Check for columns with high percentage of missing values
    2. Check for columns with abnormal distributions
    3. Check for columns with unexpected data types
    4. Check for columns with potential invalid data (inf, very large numbers)
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to diagnose
        
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary with lists of problematic columns by category
    """
    results = {}
    
    print("\n=== DATA DIAGNOSIS REPORT ===")
    
    # 1. Missing values analysis
    print("\n-- MISSING VALUES ANALYSIS --")
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percent': missing_percent.round(2)
    }).sort_values('Missing Percent', ascending=False)
    
    # Show columns with any missing values
    missing_cols = missing_df[missing_df['Missing Count'] > 0]
    print(f"Found {len(missing_cols)} columns with missing values")
    print(missing_cols.head(20))
    
    # High-missing columns (>30%)
    high_missing = missing_cols[missing_cols['Missing Percent'] > 30]
    if not high_missing.empty:
        print(f"\nWARNING: {len(high_missing)} columns have >30% missing values")
        print(high_missing)
        results['high_missing_cols'] = high_missing.index.tolist()
    
    # 2. Check for data type issues
    print("\n-- DATA TYPE ANALYSIS --")
    dtypes = df.dtypes.astype(str)
    dtype_counts = dtypes.value_counts()
    print("Data type distribution:")
    print(dtype_counts)
    
    # Check for mixed types (numerical columns with strings)
    mixed_type_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        # Try to convert to numeric
        numeric_conversion = pd.to_numeric(df[col], errors='coerce')
        # If some (but not all) values converted successfully
        na_count = numeric_conversion.isna().sum()
        if na_count > 0 and na_count < len(df):
            non_na_count = len(df) - na_count
            if non_na_count > 5:  # Only flag if we have a significant number of numeric values
                mixed_type_cols.append(col)
    
    if mixed_type_cols:
        print(f"\nWARNING: {len(mixed_type_cols)} columns appear to have mixed types (numeric + non-numeric)")
        for col in mixed_type_cols:
            print(f"- {col}: Sample values: {df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist()}")
        results['mixed_type_cols'] = mixed_type_cols
    
    # 3. Check for extreme values and outliers
    print("\n-- EXTREME VALUES ANALYSIS --")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    extreme_value_cols = []
    
    for col in numeric_cols:
        # Check for infinities
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"Column '{col}' has {inf_count} infinite values")
            extreme_value_cols.append(col)
            continue
            
        # Check for extreme values
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
        lower_bound = q1 - 3 * iqr
        
        extreme_count = ((data > upper_bound) | (data < lower_bound)).sum()
        extreme_percent = 100 * extreme_count / len(data)
        
        if extreme_percent > 5:
            print(f"Column '{col}' has {extreme_percent:.1f}% extreme outliers")
            print(f"  Range: [{data.min()}, {data.max()}], IQR: {iqr}")
            print(f"  Outliers: < {lower_bound} or > {upper_bound}")
            extreme_value_cols.append(col)
    
    if extreme_value_cols:
        results['extreme_value_cols'] = extreme_value_cols
    
    # 4. Check for near-constant columns
    print("\n-- VARIANCE ANALYSIS --")
    low_variance_cols = []
    
    for col in numeric_cols:
        # Skip columns with too many NaNs
        if df[col].isna().sum() > len(df) * 0.5:
            continue
            
        # Check variance
        variance = df[col].var()
        if pd.isna(variance) or variance == 0:
            print(f"Column '{col}' has zero variance (constant value)")
            low_variance_cols.append(col)
        elif variance < 0.01:
            unique_count = df[col].nunique()
            print(f"Column '{col}' has very low variance ({variance:.6f}), {unique_count} unique values")
            low_variance_cols.append(col)
    
    if low_variance_cols:
        results['low_variance_cols'] = low_variance_cols
    
    # 5. Check for problematic PM columns (for this specific dataset)
    print("\n-- PM COLUMNS ANALYSIS --")
    pm_cols = [col for col in df.columns if 'PM2.5' in col]
    if pm_cols:
        pm_missing = df[pm_cols].isna().sum()
        pm_missing_percent = 100 * pm_missing / len(df)
        print(f"Found {len(pm_cols)} PM2.5 columns with average {pm_missing_percent.mean():.1f}% missing values")
        
        # Check for PM columns with all NaN
        all_nan_pm = [col for col in pm_cols if df[col].isna().all()]
        if all_nan_pm:
            print(f"WARNING: {len(all_nan_pm)} PM columns have ALL missing values: {all_nan_pm}")
    
    # Summary of findings
    print("\n=== DIAGNOSIS SUMMARY ===")
    for issue, cols in results.items():
        print(f"- {issue}: {len(cols)} columns identified")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    print("Based on the diagnosis, consider:")
    
    if 'high_missing_cols' in results:
        print("1. Drop columns with >80% missing values")
        print("2. Use advanced imputation for columns with 30-80% missing values")
    
    if 'mixed_type_cols' in results:
        print("3. Clean mixed-type columns by standardizing numeric formats")
    
    if 'extreme_value_cols' in results:
        print("4. Cap extreme values or transform columns with outliers")
    
    if 'low_variance_cols' in results:
        print("5. Consider dropping near-constant columns that provide little information")
    
    print("6. Ensure consistent data types are used throughout your pipeline")
    
    return results


def check_nan_status(X_train: Any, X_test: Any, y_train: Any, y_test: Any) -> None:
    """
    Check for NaN values in training and testing datasets.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Training target
    y_test : array-like
        Testing target
    """
    # Convert to appropriate format for checking
    def count_nans(arr):
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            return arr.isna().sum().sum()
        elif hasattr(arr, 'shape'):  # numpy array or similar
            return np.isnan(arr).sum()
        else:
            try:
                return np.isnan(np.asarray(arr)).sum()
            except:
                return "Unknown - cannot check NaN status"
    
    # Print NaN status
    print("\n=== NaN STATUS CHECK ===")
    print(f"X_train NaNs: {count_nans(X_train)}")
    print(f"X_test NaNs: {count_nans(X_test)}")
    print(f"y_train NaNs: {count_nans(y_train)}")
    print(f"y_test NaNs: {count_nans(y_test)}")
    
    # Return datasets after checking
    return
    
    
def plot_missing_data_heatmap(df: pd.DataFrame, max_cols: int = 30) -> None:
    """
    Create a heatmap visualization of missing data patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    max_cols : int, default=30
        Maximum number of columns to include in heatmap
    """
    # Calculate missing percentages
    missing = df.isnull().sum() / len(df)
    
    # Select top columns with missing data
    missing = missing.sort_values(ascending=False)
    cols_to_plot = missing.index[:max_cols]
    
    # Create the heatmap matrix (white=present, red=missing)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols_to_plot].isnull(), 
                cmap=['white', 'red'],
                cbar=False, 
                yticklabels=False)
    
    plt.title('Missing Value Patterns (Red = Missing)', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Observations', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Additionally, show a bar chart of missing percentages
    plt.figure(figsize=(10, 6))
    missing[cols_to_plot].plot(kind='bar')
    plt.title('Percentage of Missing Values by Feature', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Percentage Missing', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()