"""
Model evaluation module for sonar classification.

This module provides functions to assess model performance using
confusion matrices and classification reports.
"""

import logging
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, label_names=None):
    """
    Comprehensive model evaluation with multiple metrics.

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        label_names: Optional list of label names for display (e.g., ['Rock', 'Mine'])

    Returns:
        dict containing evaluation metrics
    """
    logger.info("Evaluating model performance")

    if label_names is None:
        label_names = ['Rock (0)', 'Mine (1)']

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report (detailed per-class metrics)
    class_report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0
    )

    # Log results
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{class_report}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': class_report
    }


def print_confusion_matrix(cm, label_names=None):
    """
    Pretty print confusion matrix.

    Args:
        cm: Confusion matrix (2D numpy array)
        label_names: Optional list of label names
    """
    if label_names is None:
        label_names = ['Rock', 'Mine']

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print(f"\n{'':10} {'Predicted ' + label_names[0]:>15} {'Predicted ' + label_names[1]:>15}")
    print(f"{'Actual ' + label_names[0]:10} {cm[0, 0]:15} {cm[0, 1]:15}")
    print(f"{'Actual ' + label_names[1]:10} {cm[1, 0]:15} {cm[1, 1]:15}")
    print("=" * 50 + "\n")


def calculate_threat_detection_metrics(cm):
    """
    Calculate submarine-specific threat detection metrics.

    For submarine operations, we care deeply about:
    - Mine Detection Rate (Recall for Mine class): How many actual mines we detect
    - False Alarm Rate: How often we mistake rocks for mines

    Args:
        cm: Confusion matrix [[TN, FP], [FN, TP]]

    Returns:
        dict with submarine-relevant metrics
    """
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    mine_detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    precision_mines = TP / (TP + FP) if (TP + FP) > 0 else 0

    logger.info("Submarine Threat Detection Metrics:")
    logger.info(f"  Mine Detection Rate (Recall): {mine_detection_rate:.2%}")
    logger.info(f"  False Alarm Rate: {false_alarm_rate:.2%}")
    logger.info(f"  Mine Precision: {precision_mines:.2%}")

    return {
        'mine_detection_rate': mine_detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'mine_precision': precision_mines
    }
