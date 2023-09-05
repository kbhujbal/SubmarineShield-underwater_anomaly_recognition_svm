"""
Configuration file for Submarine AI Sonar Classification Project.

This module centralizes all hyperparameters, data sources, and reproducibility settings
for the sonar rock vs mine classification system.
"""

# Data Source Configuration
# Path to the manually downloaded sonar dataset
DATA_FILE_PATH = "data/raw/sonar.all-data"

# Reproducibility
RANDOM_SEED = 42

# Train-Test Split Configuration
TEST_SIZE = 0.2
STRATIFY = True  # Ensures balanced class distribution in train/test splits

# Model Persistence
MODEL_OUTPUT_PATH = "models/sonar_svm_model.joblib"

# SVM Hyperparameter Grid for GridSearchCV
# These parameters control the complexity and decision boundary of the SVM
PARAM_GRID = {
    'svc__C': [0.1, 1, 10, 100],  # Regularization parameter (lower = more regularization)
    'svc__kernel': ['rbf', 'linear', 'poly'],  # Kernel type for non-linear decision boundaries
    'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]  # Kernel coefficient (rbf, poly, sigmoid)
}

# Cross-Validation Configuration
CV_FOLDS = 5  # Number of folds for cross-validation in GridSearchCV
CV_SCORING = 'accuracy'  # Metric to optimize during hyperparameter search

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
