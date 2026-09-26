import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# 1. Load the top-40 pruned dataset
data = np.load("/workspace/P/dataset/img_vit_top40.npz")
X = data["embeddings_top40"]
y = data["labels"].astype(int)

# Negative to positive class ratio for XGBoost
scale_pos_weight = (len(y) - np.sum(y)) / np.sum(y)

models = {
    "Logistic Regression (L2)": LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Linear SVM": SVC(
        C=0.1, kernel="linear", class_weight="balanced", probability=True, random_state=42
    ),
    "RBF Kernel SVM": SVC(
        C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42
    ),
    "XGBoost Classifier": XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# 2. Run 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model.fit(X_train_scaled, y_train)
        
        oof_probs[val_idx] = model.predict_proba(X_val_scaled)[:, 1]
        oof_preds[val_idx] = (oof_probs[val_idx] >= 0.5).astype(int)
        
    auc = roc_auc_score(y, oof_probs)
    acc = accuracy_score(y, oof_preds)
    prec = precision_score(y, oof_preds, zero_division=0)
    rec = recall_score(y, oof_preds)
    f1 = f1_score(y, oof_preds)
    
    results.append({
        "Model": name,
        "ROC-AUC": round(auc, 4),
        "F1-Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Accuracy": round(acc, 4)
    })

res_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

print("\n" + "=" * 65)
print("5-FOLD CROSS-VALIDATION BENCHMARK (TOP 40 FEATURES)")
print("=" * 65)
print(res_df.to_string(index=False))