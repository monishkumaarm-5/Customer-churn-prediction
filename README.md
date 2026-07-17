# Customer Churn Prediction

Predicts telecom customer churn using the IBM Telco Customer Churn dataset.

## Files
- `Telco-Customer-Churn.csv` — dataset (487 customers, 21 features; a representative sample of the full IBM Telco set)
- `churn_prediction.py` — full pipeline: preprocessing, 3 models, evaluation, plots
- `model_comparison.csv` — metrics table
- `confusion_matrices.png` — confusion matrix per model
- `roc_curves.png` — ROC curves, all models
- `model_comparison.png` — bar chart of Accuracy/Precision/Recall/F1
- `feature_importance.png` — top drivers of churn (Random Forest)

## How to run
```
pip install pandas numpy scikit-learn matplotlib seaborn
python3 churn_prediction.py
```

## Pipeline
1. Drop `customerID`; coerce `TotalCharges` to numeric, fill missing with median.
2. One-hot encode 15 categorical features; scale numerics for Logistic Regression.
3. Stratified 75/25 train-test split.
4. Train Logistic Regression, Decision Tree (depth 5), Random Forest (300 trees, depth 8).
5. Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.754 | 0.515 | 0.548 | 0.531 | 0.782 |
| Decision Tree | 0.738 | 0.486 | 0.548 | 0.515 | 0.747 |
| Random Forest | 0.787 | 0.586 | 0.548 | 0.567 | 0.794 |

**Random Forest wins** on accuracy, precision, and ROC-AUC. All three models tie on recall (0.548) — with only 31 churners in the test set, catching more of them would need class-weighting or a lower decision threshold (a common next step in churn projects, since missing a churner is usually costlier than a false alarm).

Top churn drivers from the Random Forest (see `feature_importance.png`): **tenure**, **MonthlyCharges**/**TotalCharges**, and **Contract type** (month-to-month contracts churn far more than one/two-year contracts) — consistent with known patterns in this dataset.

## Note on data size
The environment fetched a 487-row sample of the dataset rather than the full 7,043-row version (network access is restricted to a fetch tool with an output-size cap). The script works unchanged on the full CSV — just swap the file and rerun for more robust, less variance-prone metrics.
