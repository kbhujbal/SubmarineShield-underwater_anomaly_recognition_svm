# SubmarineAI - Sonar Rock vs Mine Classification

A production-grade machine learning system for underwater threat detection using sonar signal processing.

## Overview

This project implements a binary classification system to distinguish between underwater rocks and mines based on sonar chirp return signals. The system processes frequency energy across 60 bands to make critical threat detection decisions for submarine operations.

**Business Goal**: Enhance submarine safety by accurately identifying mines while minimizing false alarms that could disrupt operations.

## Dataset

- **Source**: UCI Machine Learning Repository - Connectionist Bench (Sonar, Mines vs. Rocks)
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data
- **Size**: 208 samples
- **Features**: 60 numeric features (frequency band energy readings)
- **Labels**:
  - `R` (Rock) → encoded as `0`
  - `M` (Mine) → encoded as `1`
- **Challenge**: Small dataset with high dimensionality

## Technical Architecture

### Project Structure

```
SubmarineAI-sonar_rock_vs_mine_classification/
├── config.py                    # Centralized configuration
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation (this file)
├── models/                      # Trained model artifacts
│   └── sonar_svm_model.joblib   # Saved SVM pipeline
└── src/                         # Source code modules
    ├── __init__.py
    ├── data_loader.py           # UCI dataset loading
    ├── preprocessing.py         # Label encoding
    ├── evaluation.py            # Model evaluation metrics
    └── train.py                 # Main training pipeline
```

### ML Pipeline

The system uses scikit-learn's `Pipeline` for a robust, reproducible workflow:

1. **StandardScaler**: Normalizes sonar frequency features
   - *Why*: SVM is sensitive to feature scales. Different frequency bands have varying energy ranges, and normalization ensures equal contribution to the decision boundary.

2. **Support Vector Classifier (SVC)**: Binary classification model
   - *Why*: SVMs excel at high-dimensional classification and can capture non-linear patterns with kernel tricks.

### Hyperparameter Tuning

GridSearchCV optimizes the following parameters:

- **C** (Regularization): `[0.1, 1, 10, 100]`
  - Controls trade-off between margin maximization and misclassification penalty

- **Kernel**: `['rbf', 'linear', 'poly']`
  - Determines decision boundary shape (non-linear vs linear)

- **Gamma**: `['scale', 'auto', 0.001, 0.01, 0.1, 1]`
  - Controls influence radius of support vectors

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone or navigate to project directory
cd SubmarineAI-sonar_rock_vs_mine_classification

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the complete training pipeline:

```bash
python src/train.py
```

This will:
1. Download the dataset from UCI repository
2. Preprocess and encode labels
3. Split data (80% train, 20% test) with stratification
4. Perform GridSearchCV for hyperparameter optimization (5-fold CV)
5. Evaluate the best model on the test set
6. Save the trained pipeline to `models/sonar_svm_model.joblib`

### Expected Output

The training script provides detailed logging:

- Data loading confirmation
- Label distribution
- Best hyperparameters found
- Cross-validation accuracy
- Test set performance metrics
- Confusion matrix
- Submarine-specific threat metrics:
  - **Mine Detection Rate** (Recall for Mine class)
  - **False Alarm Rate** (Rock misclassified as Mine)
  - **Mine Precision** (Accuracy of Mine predictions)

## Key Metrics

For submarine operations, we prioritize:

1. **Mine Detection Rate (Recall)**: Percentage of actual mines correctly identified
   - *Critical*: Missing a mine has severe consequences

2. **False Alarm Rate**: Percentage of rocks incorrectly flagged as mines
   - *Important*: Too many false alarms disrupt mission effectiveness

3. **Overall Accuracy**: General classification performance

## Configuration

All hyperparameters and settings are centralized in [`config.py`](config.py):

- Data source URL
- Random seed (42) for reproducibility
- Train-test split ratio (0.2)
- SVM hyperparameter grid
- Cross-validation settings
- Model save path

## Code Quality Standards

- **Logging**: All status updates use Python's `logging` module (no print statements)
- **Type Hints**: Function signatures include type annotations
- **Documentation**: Comprehensive docstrings explain *why* design choices were made
- **Reproducibility**: Fixed random seed (42) for consistent results
- **Modularity**: Separation of concerns across dedicated modules

## Model Persistence

The trained SVM pipeline is saved using `joblib`:

```python
import joblib

# Load the trained model
model = joblib.load('models/sonar_svm_model.joblib')

# Make predictions
predictions = model.predict(new_sonar_data)
```

The saved pipeline includes both the StandardScaler and SVC, so no separate preprocessing is needed.

## Future Enhancements

- Implement cross-validation visualization
- Add SHAP or LIME for model interpretability
- Experiment with ensemble methods (Random Forest, Gradient Boosting)
- Deploy as REST API for real-time sonar signal classification
- Collect more training data to improve generalization

## License

This project is for educational and research purposes.

## References

- Gorman, R. P., and Sejnowski, T. J. (1988). "Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets" in Neural Networks, Vol. 1, pp. 75-89.
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
