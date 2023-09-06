"""
Data loader module for UCI Sonar dataset.

This module handles loading the sonar dataset from a local file.
The dataset contains sonar chirp frequency data across 60 bands.
"""

import logging
import os
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)


def load_sonar_data(file_path: str) -> pd.DataFrame:
    """
    Load the Sonar dataset from a local file.

    The dataset contains 208 samples of sonar returns bounced off either a metal
    cylinder (mine) or rocks. Each sample has 60 features representing energy
    within a particular frequency band, integrated over time.

    Args:
        file_path: Path to the sonar.all-data file (manually downloaded from UCI)

    Returns:
        DataFrame containing 60 feature columns (0-59) and 1 label column ('label')

    Raises:
        FileNotFoundError: If the dataset file does not exist
        Exception: If data cannot be loaded from the file
    """
    logger.info(f"Loading sonar data from local file: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        error_msg = (
            f"Dataset file not found at: {file_path}\n"
            f"Please download the dataset from:\n"
            f"https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data\n"
            f"And save it as: {file_path}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # The UCI sonar dataset has no header row
        # 60 numeric features + 1 label column ('R' or 'M')
        column_names = [f"feature_{i}" for i in range(60)] + ["label"]

        df = pd.read_csv(
            file_path,
            header=None,
            names=column_names
        )

        logger.info(f"Successfully loaded {len(df)} samples with {len(df.columns)} columns")
        logger.info(f"Class distribution:\n{df['label'].value_counts()}")

        # Validate data integrity
        if df.isnull().sum().sum() > 0:
            logger.warning("Dataset contains missing values")

        return df

    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        raise


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features (X) and labels (y).

    Args:
        df: DataFrame containing both features and label column

    Returns:
        Tuple of (features_df, labels_series)
    """
    logger.info("Splitting features and labels")

    X = df.drop('label', axis=1)
    y = df['label']

    logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    return X, y
