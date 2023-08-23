"""
Preprocessing module for sonar signal data.

This module handles label encoding and any data transformations required
before model training.
"""

import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class SonarLabelEncoder:
    """
    Custom label encoder for sonar classification labels.

    Encodes categorical labels:
    - 'R' (Rock) -> 0
    - 'M' (Mine) -> 1

    This encoding makes the problem a binary classification task suitable
    for SVM and other ML algorithms.
    """

    def __init__(self):
        self.encoder = LabelEncoder()
        self._is_fitted = False

    def fit(self, labels):
        """
        Fit the encoder to the label data.

        Args:
            labels: Array-like of string labels ('R' or 'M')

        Returns:
            self
        """
        logger.info("Fitting label encoder")
        self.encoder.fit(labels)
        self._is_fitted = True

        # Log the encoding mapping
        mapping = dict(zip(self.encoder.classes_, self.encoder.transform(self.encoder.classes_)))
        logger.info(f"Label encoding mapping: {mapping}")

        return self

    def transform(self, labels):
        """
        Transform labels to numeric encoding.

        Args:
            labels: Array-like of string labels to encode

        Returns:
            numpy array of encoded labels (0 or 1)
        """
        if not self._is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        encoded = self.encoder.transform(labels)
        logger.info(f"Transformed {len(labels)} labels")

        return encoded

    def fit_transform(self, labels):
        """
        Fit and transform labels in one step.

        Args:
            labels: Array-like of string labels

        Returns:
            numpy array of encoded labels
        """
        return self.fit(labels).transform(labels)

    def inverse_transform(self, encoded_labels):
        """
        Convert numeric labels back to original string format.

        Useful for interpreting model predictions.

        Args:
            encoded_labels: Array-like of numeric labels (0 or 1)

        Returns:
            numpy array of string labels ('R' or 'M')
        """
        if not self._is_fitted:
            raise ValueError("Encoder must be fitted before inverse_transform")

        return self.encoder.inverse_transform(encoded_labels)


def validate_labels(labels):
    """
    Validate that labels contain only expected values.

    Args:
        labels: Array-like of labels to validate

    Raises:
        ValueError: If labels contain unexpected values
    """
    unique_labels = set(labels)
    expected_labels = {'R', 'M'}

    if not unique_labels.issubset(expected_labels):
        invalid = unique_labels - expected_labels
        raise ValueError(f"Invalid labels found: {invalid}. Expected only 'R' or 'M'")

    logger.info(f"Label validation passed. Found {len(unique_labels)} unique classes")
