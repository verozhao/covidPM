# Configuration parameters

"""
Configuration parameters for COVID-19 and PM2.5 analysis
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data_files"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Analysis parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Figure parameters
FIGURE_DPI = 300
FIGURE_SIZE_LARGE = (12, 8)
FIGURE_SIZE_MEDIUM = (10, 6)
FIGURE_SIZE_SMALL = (8, 5)
FIGURE_FORMAT = 'png'

# Data parameters
PM_THRESHOLD_LOW = 12   # EPA standard for low PM2.5 (μg/m³)
PM_THRESHOLD_HIGH = 35  # EPA standard for high PM2.5 (μg/m³)
METSYN_THRESHOLD = 3    # Number of components to diagnose MetSyn

# Feature engineering parameters
RESPIRATORY_CONDITIONS = [
    'COPD', 'Asthma', 'PULMONARY_FIBROSIS', 'EMPHYSEMA',
    'INTERSTITIAL_LUNG_DISEASE', 'BRONCHIECTASIS', 'CYSTIC_FIBROSIS', 'SLEEP_APNEA'
]

# Biomarker list
BIOMARKERS = [
    'FIRST_CRP', 'FIRST_FERRITIN', 'FIRST_DDIMER', 'LACTATE',
    'TROPONIN', 'PROCALCITONIN', 'WBC', 'PLATELET',
    'BNP', 'LDH', 'Lymphocytes_abs', 'Neutrophils_abs'
]

# Machine learning parameters
ML_MODELS = {
    'logistic_regression': {
        'name': 'Logistic Regression', 
        'params': {'max_iter': 1000, 'class_weight': 'balanced'}
    },
    'random_forest': {
        'name': 'Random Forest', 
        'params': {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': RANDOM_SEED}
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting', 
        'params': {'random_state': RANDOM_SEED}
    },
    'xgboost': {
        'name': 'XGBoost', 
        'params': {'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': RANDOM_SEED}
    },
    'svm': {
        'name': 'SVM', 
        'params': {'probability': True, 'class_weight': 'balanced', 'random_state': RANDOM_SEED}
    },
    'knn': {
        'name': 'KNN', 
        'params': {}
    },
    'naive_bayes': {
        'name': 'Naive Bayes', 
        'params': {}
    }
}

# Neural network parameters
NN_PARAMS = {
    'hidden_layers': [64, 32],  
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 10
}

# Causal inference parameters
PROPENSITY_SCORE_BINS = 5

# Imputation strategies
IMPUTATION_STRATEGIES = {
    'low_missing_numeric': 'median',
    'moderate_missing_numeric': 'knn',
    'high_missing_numeric': 'iterative',
    'low_missing_categorical': 'mode',
    'high_missing_categorical': 'missing_indicator_and_mode'
}

# Missing value thresholds
MISSING_VALUE_THRESHOLDS = {
    'low': 0.1,      # 10%
    'moderate': 0.3, # 30%
    'high': 0.5      # 50%
}