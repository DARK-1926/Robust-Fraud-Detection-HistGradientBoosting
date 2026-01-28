# Robust Fraud Detection with HistGradientBoosting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready fraud detection pipeline leveraging **HistGradientBoosting** with rigorous 10-fold stratified cross-validation, advanced feature engineering, and strategic domain-driven post-processing to achieve optimal robustness and generalization.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Feature Engineering](#feature-engineering--selection)
- [Modeling Architecture](#modeling-architecture)
- [Strategic Post-Processing](#strategic-post-processing)
- [Feature Importance](#feature-importance)
- [Results & Performance](#results--performance)
- [Future Roadmap](#future-roadmap)
- [Installation](#installation)
- [Usage](#usage)

---

## Executive Summary

This solution prioritizes **robustness over complexity**, utilizing a **HistGradientBoostingClassifier** within a rigorous **10-Fold Stratified Cross-Validation** framework. The approach combines domain-driven feature engineering to capture temporal transaction anomalies with strategic post-processing to minimize variance in Log Loss optimization.

**Key Achievements:**
- Stratified K-Fold validation for unbiased generalization estimates
- Domain-informed feature engineering with temporal decomposition
- Native missing value handling with HistGradientBoosting
- Deterministic post-processing constraints for production stability

---

## Feature Engineering & Selection

The feature engineering strategy focused on identifying **distributional anomalies** associated with fraudulent behavior, grounded in domain expertise and exploratory data analysis.

### Temporal Decomposition
- **Circadian Patterns:** The `step` variable was transformed into `hour` and `day` cycles to capture temporal transaction patterns
- Rationale: Fraudsters often operate at specific times; legitimate users follow consistent behavioral patterns

### Merchant Cohort Identification
- **Safe Cohort Detection:** A boolean feature `is_merchant` was derived from `nameDest` prefixes
- EDA Insight: Merchant accounts demonstrate negligible fraud probability, enabling deterministic filtering

### Interaction Features
- **Frequency Encoding:** 
  - `nameDest_count`: Flags high-velocity recipients (synthetic identity fraud indicator)
  - `nameOrig_count`: Identifies high-velocity senders
  
- **Relative Destination Value:** 
  - `amount_to_mean_dest`: Quantifies deviation from a recipient's typical intake, isolating statistical outliers

### Distributional Normalization
- **Log Transformation:** Applied `np.log1p(amount)` to mitigate heavy-tailed distributions
- Benefit: Improves gradient-based learner stability and convergence

---

## Modeling Architecture

### Algorithm Choice: HistGradientBoosting
Why HistGradientBoosting?
- **Native Missing Value Support:** Eliminates imputation overhead
- **Large-Scale Efficiency:** Histogram-based binning reduces memory footprint
- **Gradient-Based Optimization:** Fine-grained control over learning dynamics

### Validation Strategy
```
10-Fold Stratified Cross-Validation
├── Preserves class imbalance ratio across folds
├── Provides unbiased generalization estimates
└── Enables robust hyperparameter tuning
```

### Hyperparameter Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 0.01 | Granular convergence, reduced overfitting risk |
| **Max Iterations** | 2000 | Compensates for low learning rate |
| **L2 Regularization** | 1.0 | Prevents overfitting to training noise |
| **Validation Fraction** | 0.1 | Per-fold internal validation split |

---

## Strategic Post-Processing (Domain Adaptation)

To optimize the **Log Loss metric**, which heavily penalizes confident errors, deterministic constraints based on domain logic were implemented:

### The Merchant Filter
```python
# Hard constraint: merchant transactions → probability = 0.0
if is_merchant:
    prediction = 0.0
```
- **Impact:** Eliminates false positives in the dataset's safest segment
- **Loss Reduction:** Prevents penalties on inherently safe transactions

### Outlier Calibration
```python
# Soft ceiling: flagged suspicious transactions → probability ≤ 0.99
if isFlaggedFraud:
    prediction = min(prediction, 0.99)
```
- **Protection:** Shields against infinite Log Loss penalties from mislabeled outliers
- **Practical Insight:** Acknowledges inherent uncertainty in edge cases

---

## Feature Importance

The following infographic displays the relative importance ranking of engineered features as determined by the HistGradientBoosting model:

![Faculty Request: Feature Importance Ranking](https://ibb.co/Z1RG21pc)(https://postimg.cc/cgL7TzRz)

**Key Observations:**
- `is_merchant` emerges as the dominant predictive signal, validating the domain hypothesis that merchant accounts are inherently safe
- Temporal features (`hour`, `day`) capture circadian patterns of fraudulent activity
- Transaction `amount` provides strong discriminative power for outlier detection
- Frequency-based features (`destcount`, `orig_count`) demonstrate moderate importance in identifying high-velocity accounts

---

## Results & Performance

The solution achieves strong generalization through:

**Cross-Validation Robustness**
- Stratified folding ensures consistent performance across class distributions
- Multiple fold estimates reduce variance in reported metrics

**Feature Engineering Impact**
- Domain-informed features capture fraudulent behavior patterns
- Interaction features reveal high-velocity accounts and outlier transactions

**Post-Processing Gains**
- Domain constraints reduce Log Loss without sacrificing interpretability
- Deterministic rules ensure reproducibility in production

---

## Future Roadmap & Advanced Optimization

While the current solution prioritizes stability, future iterations can leverage advanced techniques:

### 1. Bayesian Hyperparameter Optimization (Optuna)
```
Manual Grid Search  →  Optuna Bayesian Search
                              ├── Dynamic tuning of tree depth
                              ├── Leaf node optimization
                              └── L2-regularization refinement
```
- **Benefit:** Discovers global minimum of loss function more efficiently

### 2. Stacking Generalization (Level-2 Ensemble)
```
Training Phase:
├── XGBoost on Fold 1-10 OOF predictions
├── LightGBM on Fold 1-10 OOF predictions
├── CatBoost on Fold 1-10 OOF predictions
└── Meta-Learner (Logistic Regression) synthesizes final prediction
```
- **Benefit:** Captures diverse decision boundaries, improves generalization

### 3. Adversarial Validation
- Train a classifier to detect distributional drift between train/test
- Remove drifting features to ensure safer generalization
- Identifies and mitigates "private leaderboard shakeup" risk

---

## Installation

### Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy

### Setup
```bash
# Clone repository
git clone https://github.com/DARK-1926/Robust-Fraud-Detection-HistGradientBoosting.git
cd Robust-Fraud-Detection-HistGradientBoosting

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# Load data
X = pd.read_csv('features.csv')
y = pd.read_csv('labels.csv')

# Initialize model
model = HistGradientBoostingClassifier(
    learning_rate=0.01,
    max_iter=2000,
    l2_regularization=1.0,
    random_state=42
)

# Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Train and predict
predictions = []
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    predictions.append(pred)
```

---

## Key Insights

1. **Temporal Patterns Matter:** Circadian decomposition captures fraud timing anomalies
2. **Cohort Safety:** Merchant accounts are inherently low-risk—domain expertise enables hard constraints
3. **Velocity as Signal:** High-frequency senders/recipients correlate with synthetic identity fraud
4. **Log Loss Sensitivity:** Strategic post-processing prevents catastrophic penalties on edge cases
5. **Stratified Validation:** Preserving class imbalance in folds ensures realistic generalization estimates

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

**DARK-1926** | Kaggle Knight Final Submission

---

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests for optimizations (Optuna, Stacking, etc.)

---

**Last Updated:** January 2026 | Status: Production-Ready
