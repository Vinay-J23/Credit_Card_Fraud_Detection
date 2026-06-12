# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using classification algorithms on a highly imbalanced dataset.

## Overview

This project applies supervised machine learning to classify credit card transactions as fraudulent or legitimate. The dataset is heavily imbalanced (far fewer fraud cases than normal ones), so SMOTE (Synthetic Minority Oversampling Technique) is used to balance the training data before model training.

## Dataset

- **File:** `creditcard.csv`
- **Features:** `Time`, `V1`–`V28` (PCA-transformed), `Amount`, `Class`
- **Target:** `Class` — `1` = Fraud, `0` = Legitimate
- **Key Insight from EDA:** Fraud transactions are predominantly below $2,500, with the most common fraudulent amount being around $1. A bimodal distribution in transaction time suggests fraud spikes during early morning hours.

## Project Structure

```
ML_project.ipynb       # Main notebook with EDA, preprocessing, and model training
creditcard.csv         # Input dataset (not included — upload to /content/ in Colab)
README.md
```

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked for duplicate columns (none found)
- Scatter plots and histograms to explore `Amount` vs `Class`
- Correlation heatmaps for the full dataset, fraud-only, and legitimate-only subsets
- Group-by analysis on `Class` and a custom `Amount_Group` feature (`0–2500` vs `2500+`)
- Pair plots to visually identify outliers associated with fraud

### 2. Feature Engineering
- Created `Amount_Group` to bin transaction amounts
- Filtered data to the `0–2500` amount range (where all fraud occurs)
- Selected 25 features for modeling: `Time`, `V1`–`V20`, `V27`, `V28`, `Amount`

### 3. Data Splitting & Balancing
- **Stratified Shuffle Split** (75/25 train-test) to preserve class distribution
- **SMOTE** (`k_neighbors=1`) applied to training data to address class imbalance

### 4. Models Trained

| Model | Key Hyperparameters | ROC-AUC |
|---|---|---|
| Logistic Regression | GridSearchCV over C, penalty; liblinear solver | ~98% |
| Decision Tree | `min_samples_leaf=30`, `criterion='gini'` | ~95% |
| Random Forest | `n_estimators=100`, `max_depth=30`, `min_samples_leaf=30` | ~99% |
| XGBoost | `eval_metric='logloss'` | Included in comparison |

Hyperparameter tuning was performed on Logistic Regression using `GridSearchCV` with F1 score as the metric and 5-fold `StratifiedKFold` cross-validation.

### 5. Evaluation Metrics
- Classification report (Precision, Recall, F1)
- Confusion matrix (heatmap)
- ROC-AUC score and curve

## Results

- **Random Forest** achieved the best performance at ~99% ROC-AUC.
- **Logistic Regression** reached ~98% ROC-AUC after hyperparameter tuning.
- **Decision Tree** achieved ~95% accuracy with `min_samples_leaf=30` to prevent overfitting.
- A model comparison table (Accuracy, Precision, Recall, F1) is generated at the end of the notebook.

## Key Findings

1. Fraud transactions are concentrated in very low amounts (mode ~$1, max ~$2,500–$3,000).
2. Features V1–V17 show notable correlations within both fraud and non-fraud groups.
3. Fraud activity peaks during early morning hours (bimodal time distribution).
4. SMOTE significantly improved model recall on the minority (fraud) class.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## Usage

1. Upload `creditcard.csv` to `/content/` in Google Colab (or update the file path locally).
2. Run all cells in `ML_project.ipynb` top to bottom.
3. Review EDA visualizations, then check the final model comparison table for results.