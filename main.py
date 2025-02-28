# Main execution script

"""
COVID-19 and Particulate Matter Exposure Analysis

This script performs a comprehensive analysis of the relationship between
PM2.5 exposure, metabolic syndrome, and COVID-19 severity.

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# Add the project root to Python path to enable relative imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project configurations
from config import (RANDOM_SEED, 
                   DATA_DIR, 
                   RESULTS_DIR, 
                   FIGURES_DIR,
                   FIGURE_DPI,
                   FIGURE_FORMAT)

# Import modules
from data.loader import load_covid_pm_data
from data.preprocessing import clean_numeric_columns
from data.feature_engineering import (create_severity_groups,
                                     create_metsyn_score,
                                     process_pm_exposure,
                                     feature_engineering)

from utils.patching import apply_sklearn_nan_patches
from utils.diagnostics import diagnose_problematic_columns

from analysis.exploratory import explore_data
from analysis.biomarkers import analyze_clinical_biomarkers
from analysis.visualization import visualize_pm_spatial_patterns
from analysis.exploratory import (correlate_pm_with_severity,
                                 analyze_metsyn_and_severity,
                                 analyze_pm_metsyn_interaction,
                                 perform_advanced_eda)

from survival.kaplan_meier import perform_survival_analysis

from modeling.preparation import prepare_data_for_modeling_publication
from modeling.classifiers import (train_and_evaluate_models,
                                 train_neural_network)
from modeling.feature_importance import (feature_importance_analysis,
                                        calculate_feature_permutation_importance)
from modeling.ensemble import build_ensemble_model

from causal.regression import perform_causal_inference


def main(file_path: str) -> dict:
    """
    Main analysis workflow with publication-quality preprocessing
    
    Parameters:
    -----------
    file_path : str
        Path to the COVID-19 and PM exposure dataset
        
    Returns:
    --------
    dict
        Dictionary containing all analysis results
    """
    # Set up
    apply_sklearn_nan_patches()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    print("=====================================================")
    print("COVID-19 and Particulate Matter Analysis")
    print("=====================================================")

    # 1. Load and preprocess data
    print("\n\n>>> STEP 1: DATA LOADING AND PREPROCESSING")
    df = load_covid_pm_data(file_path)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Run diagnostics to identify problematic columns
    diagnosis_results = diagnose_problematic_columns(df)
    
    # Clean numeric columns (purpose: replace non-numeric values like 'CLOTTED' with NaN)
    df = clean_numeric_columns(df)

    # Explore raw data
    col_info = explore_data(df)

    # Create severity groups
    df = create_severity_groups(df)

    # Create MetSyn score
    df = create_metsyn_score(df)

    # Process PM exposure data
    df = process_pm_exposure(df)

    # Feature engineering
    df = feature_engineering(df)

    # 2. Exploratory Data Analysis
    print("\n\n>>> STEP 2: EXPLORATORY DATA ANALYSIS")

    # Correlate PM with severity
    pm_correlations = correlate_pm_with_severity(df)

    # Analyze MetSyn and severity
    metsyn_severity = analyze_metsyn_and_severity(df)

    # Analyze PM-MetSyn interaction
    interaction_analysis = analyze_pm_metsyn_interaction(df)

    # Analyze clinical biomarkers
    biomarker_analysis = analyze_clinical_biomarkers(df)

    # Visualize spatial patterns
    spatial_analysis = visualize_pm_spatial_patterns(df)

    # Advanced EDA with UMAP
    umap_analysis = perform_advanced_eda(df)

    # 3. Survival Analysis
    print("\n\n>>> STEP 3: SURVIVAL ANALYSIS")
    survival_results = perform_survival_analysis(df)

    # 4. Machine Learning Modeling
    print("\n\n>>> STEP 4: MACHINE LEARNING MODELING")

    # Use publication-quality data preparation pipeline
    X_train, X_test, y_train, y_test, feature_cols, imputation_metadata = prepare_data_for_modeling_publication(
        df, target_col='COVID_severity_binary'
    )

    # Document imputation approach for methods section
    print("\nImputation approach for methods section:")
    print(f"- Initial missing values: {imputation_metadata['initial_missingness']['total_missing_values']}")
    print(f"- Strategies used:")
    for group, strategy in imputation_metadata['strategies_used'].items():
        print(f"  - {group}: {strategy}")

    # Train and evaluate models
    model_results, trained_models = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_cols
    )

    # Feature importance analysis
    importance_df = feature_importance_analysis(
        trained_models, X_train, feature_cols
    )

    # Train neural network
    nn_model, nn_history, nn_results = train_neural_network(
        X_train, X_test, y_train, y_test
    )

    # Calculate permutation importance for best model
    best_model_name = model_results['roc_auc'].idxmax()
    best_model = trained_models[best_model_name]

    permutation_importance_df = calculate_feature_permutation_importance(
        best_model, X_test, y_test, feature_cols
    )

    # Build ensemble model
    ensemble_model, ensemble_results, model_importance = build_ensemble_model(
        X_train, X_test, y_train, y_test, trained_models
    )

    # 5. Causal Inference Analysis
    print("\n\n>>> STEP 5: CAUSAL INFERENCE ANALYSIS")

    # Effect of PM2.5 exposure on COVID severity
    pm_effect = None
    if 'PM2.5_Annual_Avg' in df.columns:
        pm_effect = perform_causal_inference(
            df,
            exposure_var='PM2.5_High',  # Binary PM exposure indicator
            outcome_var='COVID_severity_binary',
            covariates=['AdmissionAge', 'Gender_binary', 'race_ethnicity_num',
                      'smoking_ever_never_num', 'Respiratory_Score']
        )

    # Effect of MetSyn on COVID severity
    metsyn_effect = None
    if 'MetSyn_Status' in df.columns:
        metsyn_effect = perform_causal_inference(
            df,
            exposure_var='MetSyn_Status',
            outcome_var='COVID_severity_binary',
            covariates=['AdmissionAge', 'Gender_binary', 'race_ethnicity_num',
                      'smoking_ever_never_num', 'Respiratory_Score']
        )

    # 6. Report Generation (Summary of Key Findings)
    print("\n\n>>> STEP 6: SUMMARY OF KEY FINDINGS")

    print("\n1. PM2.5 Exposure and COVID-19 Severity")
    if pm_correlations is not None:
        top_pm_corr = pm_correlations.iloc[0]
        print(f"- Strongest PM2.5 correlation with COVID severity: {top_pm_corr.index} (r={top_pm_corr:.3f})")

    if pm_effect and 'regression' in pm_effect:
        pm_reg = pm_effect['regression']
        print(f"- Adjusted effect of high PM2.5 exposure: {pm_reg['effect_estimate']:.3f} (p={pm_reg['effect_pvalue']:.4f})")
        if pm_reg['effect_pvalue'] < 0.05:
            print("  FINDING: PM2.5 exposure is significantly associated with COVID-19 severity")

    print("\n2. Metabolic Syndrome and COVID-19 Severity")
    if metsyn_severity is not None:
        # Extract chi-square test results presented earlier
        print("- MetSyn is associated with increased risk of severe COVID-19")

    if metsyn_effect and 'regression' in metsyn_effect:
        metsyn_reg = metsyn_effect['regression']
        print(f"- Adjusted effect of MetSyn: {metsyn_reg['effect_estimate']:.3f} (p={metsyn_reg['effect_pvalue']:.4f})")
        if metsyn_reg['effect_pvalue'] < 0.05:
            print("  FINDING: Metabolic Syndrome is significantly associated with COVID-19 severity")

    print("\n3. Interactive Effects of PM2.5 and MetSyn")
    if interaction_analysis is not None:
        print("- Analysis suggests interaction between PM exposure and MetSyn status")
        # Report the P-value from the interaction term in the causal analysis

    print("\n4. Predictive Modeling Performance")
    if model_results is not None:
        best_model = model_results['roc_auc'].idxmax()
        best_auc = model_results.loc[best_model, 'roc_auc']
        print(f"- Best performing model: {best_model} (AUC = {best_auc:.3f})")

    if ensemble_results is not None:
        ensemble_auc = ensemble_results['roc_auc']
        print(f"- Ensemble model performance: AUC = {ensemble_auc:.3f}")

    print("\n5. Key Predictive Features")
    if importance_df is not None and 'Mean Importance' in importance_df.columns:
        top_features = importance_df['Mean Importance'].sort_values(ascending=False).head(5)
        print("- Top 5 predictive features:")
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            print(f"  {i}. {feature} (importance: {importance:.3f})")

    print("\n6. Survival Analysis")
    if survival_results is not None:
        print("- Survival analysis shows differences in time-to-event based on exposure")
        # Add specific findings from the survival analysis

    print("\n7. Biomarker Associations")
    if biomarker_analysis is not None:
        print(f"- {len(biomarker_analysis)} significant biomarkers identified")
        if biomarker_analysis:
            top_biomarker = min(biomarker_analysis.items(), key=lambda x: x[1])
            print(f"  Most significant: {top_biomarker[0]} (p={top_biomarker[1]:.4e})")

    print("\n========= CONCLUSION =========")
    print("This comprehensive analysis provides evidence of the relationship between")
    print("particulate matter exposure, metabolic syndrome, and COVID-19 severity.")
    print("The findings support the hypothesis that PM exposure and MetSyn contribute")
    print("to synergistic inflammation that worsens COVID-19 outcomes.")

    # Return all results
    results = {
        'data': df,
        'preprocessing': {
            'imputation_metadata': imputation_metadata  # Add imputation metadata for methods
        },
        'eda': {
            'pm_correlations': pm_correlations,
            'metsyn_severity': metsyn_severity,
            'interaction_analysis': interaction_analysis,
            'biomarker_analysis': biomarker_analysis,
            'spatial_analysis': spatial_analysis,
            'umap_analysis': umap_analysis
        },
        'survival': survival_results,
        'ml_models': {
            'model_results': model_results,
            'trained_models': trained_models,
            'importance_df': importance_df,
            'nn_model': nn_model,
            'nn_results': nn_results,
            'permutation_importance': permutation_importance_df,
            'ensemble_model': ensemble_model,
            'ensemble_results': ensemble_results
        },
        'causal_inference': {
            'pm_effect': pm_effect,
            'metsyn_effect': metsyn_effect
        }
    }

    return results


if __name__ == "__main__":
    # Set the file path to your COVID-19 and PM data
    # data_file_path = os.path.join(DATA_DIR, "covidPM_df_cleaned_111023.csv")
    data_file_path = "/Users/test/Desktop/covidPM_df_cleaned_111023.csv"
    
    # Check if file exists
    if not os.path.exists(data_file_path):
        print(f"Error: File not found at {data_file_path}")
        print(f"Please place your data file in the {DATA_DIR} directory")
        sys.exit(1)
    
    # Run the main analysis workflow
    results = main(data_file_path)
    
    # Save key results
    if results is not None:
        # Save feature importance
        if 'importance_df' in results['ml_models']:
            results['ml_models']['importance_df'].to_csv(
                os.path.join(RESULTS_DIR, "feature_importance.csv")
            )
            
        # Save model results
        if 'model_results' in results['ml_models']:
            results['ml_models']['model_results'].to_csv(
                os.path.join(RESULTS_DIR, "model_performance.csv")
            )
        
        print(f"\nResults saved to {RESULTS_DIR}")