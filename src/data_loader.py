"""
Data loader module for UCI Sonar dataset.

This module handles fetching and loading the sonar dataset from the UCI Machine Learning
Repository. The dataset contains sonar chirp frequency data across 60 bands.
"""

import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)


def load_sonar_data(url: str) -> pd.DataFrame:
    """
    Load the Sonar dataset from UCI Machine Learning Repository.

    The dataset contains 208 samples of sonar returns bounced off either a metal
    cylinder (mine) or rocks. Each sample has 60 features representing energy
    within a particular frequency band, integrated over time.

    Args:
        url: URL to the sonar.all-data file from UCI repository

    Returns:
        DataFrame containing 60 feature columns (0-59) and 1 label column ('label')

    Raises:
        Exception: If data cannot be loaded from the URL
    """
    logger.info(f"Loading sonar data from URL: {url}")

    try:
        # The UCI sonar dataset has no header row
        # 60 numeric features + 1 label column ('R' or 'M')
        column_names = [f"feature_{i}" for i in range(60)] + ["label"]

        df = pd.read_csv(
            url,
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
        logger.error(f"Failed to load data from {url}: {str(e)}")
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
