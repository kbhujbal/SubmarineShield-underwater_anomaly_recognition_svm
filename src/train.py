"""
Main training script for Submarine AI Sonar Classification.

This module orchestrates the complete ML pipeline:
1. Data loading from local file
2. Preprocessing and label encoding
3. Train-test split with stratification
4. SVM model training with hyperparameter tuning via GridSearchCV
5. Model evaluation and persistence
"""

import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Import project modules
import config
from src.data_loader import load_sonar_data, split_features_labels
from src.preprocessing import SonarLabelEncoder, validate_labels
from src.evaluation import (
    evaluate_model,
    print_confusion_matrix,
    calculate_threat_detection_metrics
)

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def build_svm_pipeline():
    """
    Build the SVM classification pipeline.

    Pipeline steps:
    1. StandardScaler: Normalizes feature values to have mean=0 and variance=1
       WHY: SVM is sensitive to feature scales. Sonar frequency bands have
       different energy ranges, and StandardScaler ensures all features
       contribute equally to the decision boundary.

    2. SVC: Support Vector Classifier with RBF kernel
       WHY: SVM with RBF kernel can capture non-linear patterns in sonar
       signatures that distinguish mines from rocks.

    Returns:
        sklearn Pipeline object
    """
    logger.info("Building SVM pipeline with StandardScaler")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Feature normalization (critical for SVM)
        ('svc', SVC(random_state=config.RANDOM_SEED))  # SVM classifier
    ])

    logger.info("Pipeline structure:")
    for name, step in pipeline.steps:
        logger.info(f"  - {name}: {step.__class__.__name__}")

    return pipeline


def perform_hyperparameter_tuning(pipeline, X_train, y_train):
    """
    Perform grid search for optimal SVM hyperparameters.

    Tuned parameters:
    - C: Regularization strength (controls margin vs misclassification trade-off)
    - kernel: Decision boundary shape (linear, rbf, polynomial)
    - gamma: Kernel coefficient (controls influence of single training examples)

    Args:
        pipeline: sklearn Pipeline
        X_train: Training features
        y_train: Training labels

    Returns:
        GridSearchCV object fitted to training data
    """
    logger.info("Starting GridSearchCV for hyperparameter optimization")
    logger.info(f"Parameter grid: {config.PARAM_GRID}")
    logger.info(f"Cross-validation folds: {config.CV_FOLDS}")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config.PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring=config.CV_SCORING,
        n_jobs=-1,  # Use all CPU cores
        verbose=2,
        return_train_score=True
    )

    logger.info(f"Fitting {len(config.PARAM_GRID['svc__C']) * len(config.PARAM_GRID['svc__kernel']) * len(config.PARAM_GRID['svc__gamma'])} candidate models...")

    grid_search.fit(X_train, y_train)

    logger.info("GridSearchCV complete!")
    logger.info(f"Best score (CV accuracy): {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")

    return grid_search


def main():
    """
    Main execution function for training pipeline.
    """
    logger.info("=" * 70)
    logger.info("SUBMARINE AI - SONAR ROCK VS MINE CLASSIFICATION")
    logger.info("=" * 70)

    # Step 1: Load data
    logger.info("\n[STEP 1] Loading dataset from local file")
    df = load_sonar_data(config.DATA_FILE_PATH)

    # Step 2: Split features and labels
    logger.info("\n[STEP 2] Splitting features and labels")
    X, y = split_features_labels(df)

    # Step 3: Validate and encode labels
    logger.info("\n[STEP 3] Encoding labels (R -> 0, M -> 1)")
    validate_labels(y)
    label_encoder = SonarLabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Step 4: Train-test split with stratification
    logger.info("\n[STEP 4] Creating train-test split")
    logger.info(f"Test size: {config.TEST_SIZE * 100}%")
    logger.info(f"Stratified: {config.STRATIFY}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y_encoded if config.STRATIFY else None
    )

    logger.info(f"Training set size: {len(X_train)} samples")
    logger.info(f"Test set size: {len(X_test)} samples")

    # Step 5: Build pipeline
    logger.info("\n[STEP 5] Building SVM pipeline")
    pipeline = build_svm_pipeline()

    # Step 6: Hyperparameter tuning
    logger.info("\n[STEP 6] Hyperparameter tuning with GridSearchCV")
    grid_search = perform_hyperparameter_tuning(pipeline, X_train, y_train)

    # Step 7: Evaluate on test set
    logger.info("\n[STEP 7] Evaluating best model on test set")
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    # Compute evaluation metrics
    metrics = evaluate_model(
        y_test,
        y_pred,
        label_names=['Rock', 'Mine']
    )

    # Print confusion matrix
    print_confusion_matrix(metrics['confusion_matrix'], label_names=['Rock', 'Mine'])

    # Calculate submarine-specific metrics
    threat_metrics = calculate_threat_detection_metrics(metrics['confusion_matrix'])

    # Step 8: Save the model
    logger.info("\n[STEP 8] Saving trained model")
    os.makedirs(os.path.dirname(config.MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_model, config.MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to: {config.MODEL_OUTPUT_PATH}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - MODEL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Best Hyperparameters: {grid_search.best_params_}")
    logger.info(f"Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    logger.info(f"Test Set Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Mine Detection Rate: {threat_metrics['mine_detection_rate']:.2%}")
    logger.info(f"False Alarm Rate: {threat_metrics['false_alarm_rate']:.2%}")
    logger.info("=" * 70)

    return best_model, metrics


if __name__ == "__main__":
    try:
        model, metrics = main()
        logger.info("\nTraining pipeline executed successfully!")
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)
