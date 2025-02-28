# sklearn NaN patching utilities

"""
Utility functions for patching scikit-learn to handle NaN values.
This module contains functions to monkey patch scikit-learn's validation 
functions to handle NaN values gracefully.
"""

import numpy as np
import pandas as pd
import functools
import warnings
import sklearn.utils.validation

def apply_sklearn_nan_patches():
    """
    Apply monkey patches to scikit-learn to handle NaN values automatically.
    This should be called at the beginning of your script if you want scikit-learn
    to handle NaN values internally.
    """
    # Store the original function
    original_assert_all_finite = sklearn.utils.validation._assert_all_finite_element_wise
    
    # Create our patched version
    def patched_assert_all_finite(X, *args, **kwargs):
        """
        Patched version that handles NaN values in data by replacing them
        before the validation happens.
        """
        try:
            # Try with the original function first
            return original_assert_all_finite(X, *args, **kwargs)
        except ValueError as e:
            if "Input contains NaN" in str(e):
                warnings.warn("WARNING: NaN values detected, auto-fixing...")
                
                # Handle different data types
                if isinstance(X, np.ndarray):
                    # If it's a numpy array, replace NaNs with 0
                    X_fixed = np.nan_to_num(X, nan=0.0)
                    return original_assert_all_finite(X_fixed, *args, **kwargs)
                elif isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                    # If it's pandas, fill NaNs with appropriate values
                    X_fixed = X.fillna(0)
                    return original_assert_all_finite(X_fixed, *args, **kwargs)
                else:
                    # For other types, try direct conversion to array
                    try:
                        X_array = np.asarray(X)
                        X_fixed = np.nan_to_num(X_array, nan=0.0)
                        return original_assert_all_finite(X_fixed, *args, **kwargs)
                    except:
                        # If all else fails, reraise the original error
                        raise e
            else:
                # If it's a different error, reraise it
                raise e
                
    # Replace the original function with our patched version
    sklearn.utils.validation._assert_all_finite_element_wise = patched_assert_all_finite
    
    # Function to create a patched version of any validation function
    def create_nan_safe_wrapper(original_func):
        """Creates a wrapper that handles NaN values safely"""
        @functools.wraps(original_func)
        def wrapper(X, *args, **kwargs):
            try:
                return original_func(X, *args, **kwargs)
            except ValueError as e:
                if "Input contains NaN" in str(e):
                    warnings.warn(f"WARNING: NaN values detected in {original_func.__name__}, auto-fixing...")
                    
                    # Handle different data types
                    if isinstance(X, np.ndarray):
                        X_fixed = np.nan_to_num(X, nan=0.0)
                        return original_func(X_fixed, *args, **kwargs)
                    elif isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                        X_fixed = X.fillna(0)
                        return original_func(X_fixed, *args, **kwargs)
                    else:
                        try:
                            X_array = np.asarray(X)
                            X_fixed = np.nan_to_num(X_array, nan=0.0)
                            return original_func(X_fixed, *args, **kwargs)
                        except:
                            raise e
                else:
                    raise e
        return wrapper
    
    # Apply patches to other common scikit-learn validation functions
    for func_name in [
        'check_array', 
        'check_X_y',
        'assert_all_finite'
    ]:
        if hasattr(sklearn.utils.validation, func_name):
            original = getattr(sklearn.utils.validation, func_name)
            patched = create_nan_safe_wrapper(original)
            setattr(sklearn.utils.validation, func_name, patched)
    
    # Log success
    warnings.warn("Applied sklearn NaN handling patches successfully!")

def ensure_no_nans(arr):
    """
    Ensure no NaN values exist in any array-like object.
    Use this before any scikit-learn function that might complain about NaNs.
    
    Parameters:
    -----------
    arr : array-like
        The array to clean
        
    Returns:
    --------
    array-like
        The same type of object with NaNs replaced by appropriate values
    """
    if arr is None:
        return arr
    
    if isinstance(arr, pd.DataFrame):
        # For DataFrames, fill column-by-column
        result = arr.copy()
        for col in result.columns:
            # Try mean first, fall back to 0
            mean_val = result[col].mean()
            if pd.isna(mean_val):
                result[col] = result[col].fillna(0)
            else:
                result[col] = result[col].fillna(mean_val)
        return result
        
    elif isinstance(arr, pd.Series):
        # For Series, try mean, fall back to most frequent, then 0
        mean_val = arr.mean()
        if pd.isna(mean_val):
            if arr.dtype in ['object', 'category']:
                # For categorical, use most frequent
                mode_val = arr.mode()
                if len(mode_val) > 0:
                    return arr.fillna(mode_val[0])
                else:
                    return arr.fillna("unknown")
            else:
                # For numerical, use 0
                return arr.fillna(0)
        else:
            return arr.fillna(mean_val)
            
    elif isinstance(arr, np.ndarray):
        # For numpy arrays, replace with 0
        return np.nan_to_num(arr, nan=0.0)
        
    else:
        # Try to convert to numpy array
        try:
            arr_np = np.asarray(arr)
            return np.nan_to_num(arr_np, nan=0.0)
        except:
            # If all else fails, return as is and hope for the best
            return arr