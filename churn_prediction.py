"""
Customer Churn Prediction
==========================
Dataset : IBM Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)
Models  : Logistic Regression, Decision Tree, Random Forest
Metrics : Confusion Matrix, Precision, Recall, F1, ROC-AUC

Run:  python3 churn_prediction.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, roc_curve, classification_report
)

RANDOM_STATE = 42
DATA_PATH = "Telco-Customer-Churn.csv"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------
# 2. Clean & preprocess
# ---------------------------------------------------------------
df.drop(columns=["customerID"], inplace=True)

# TotalCharges sometimes contains blanks in the full IBM dataset -> coerce
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

target = "Churn"
df[target] = df[target].map({"Yes": 1, "No": 0})

cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in df.columns if c not in cat_cols + [target]]

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numeric columns ({len(num_cols)}): {num_cols}")

# One-hot encode categoricals
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# ---------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE
    ),
}

results = {}
fig_cm, axes_cm = plt.subplots(1, 3, figsize=(15, 4))
fig_roc, ax_roc = plt.subplots(figsize=(6, 5))

for i, (name, model) in enumerate(models.items()):
    # Logistic Regression benefits from scaling; tree models don't need it
    # but scaling doesn't hurt them, so we keep the pipeline simple.
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1": f1, "ROC-AUC": auc, "ConfusionMatrix": cm.tolist()
    }

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_cm[i],
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    axes_cm[i].set_title(name)
    axes_cm[i].set_xlabel("Predicted")
    axes_cm[i].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

fig_cm.tight_layout()
fig_cm.savefig("confusion_matrices.png", dpi=150)

ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves")
ax_roc.legend()
fig_roc.tight_layout()
fig_roc.savefig("roc_curves.png", dpi=150)

# ---------------------------------------------------------------
# 5. Model comparison table
# ---------------------------------------------------------------
comparison = pd.DataFrame({
    name: {k: v for k, v in r.items() if k != "ConfusionMatrix"}
    for name, r in results.items()
}).T
comparison = comparison.round(3)
comparison.to_csv("model_comparison.csv")
print("\n=== Model comparison ===")
print(comparison)

fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
comparison[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=ax_bar)
ax_bar.set_title("Model Performance Comparison")
ax_bar.set_ylabel("Score")
ax_bar.set_ylim(0, 1)
ax_bar.legend(loc="lower right")
plt.xticks(rotation=15)
fig_bar.tight_layout()
fig_bar.savefig("model_comparison.png", dpi=150)

# ---------------------------------------------------------------
# 6. Feature importance (Random Forest)
# ---------------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
importances.sort_values().plot(kind="barh", ax=ax_fi, color="steelblue")
ax_fi.set_title("Top 15 Feature Importances (Random Forest)")
fig_fi.tight_layout()
fig_fi.savefig("feature_importance.png", dpi=150)

print("\nSaved: confusion_matrices.png, roc_curves.png, model_comparison.png, feature_importance.png, model_comparison.csv")
