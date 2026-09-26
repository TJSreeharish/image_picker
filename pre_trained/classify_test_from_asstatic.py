import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

# 1. Load Manifest CSV
csv_path = "/workspace/P/dataset/dataset_manifest.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} records.")
print("Class breakdown:\n", df["label"].value_counts())

# Group scores by class
scores_good = df[df["label"] == 1]["aesthetic_score"].values
scores_bad = df[df["label"] == 0]["aesthetic_score"].values

# 2. Statistical Analysis
mean_good, std_good = np.mean(scores_good), np.std(scores_good)
mean_bad, std_bad = np.mean(scores_bad), np.std(scores_bad)

# Two-sample t-test to check if distributions differ significantly
t_stat, p_val = stats.ttest_ind(scores_good, scores_bad, equal_var=False)

# Point-biserial correlation between continuous score and binary label
corr, corr_p = stats.pointbiserialr(df["aesthetic_score"], df["label"])

# Cohen's d (Effect Size)
n_good, n_bad = len(scores_good), len(scores_bad)
pooled_std = np.sqrt(((n_good - 1) * std_good**2 + (n_bad - 1) * std_bad**2) / (n_good + n_bad - 2))
cohens_d = (mean_good - mean_bad) / pooled_std if pooled_std > 0 else 0

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(f"Good Photos (Label 1)  -> Mean: {mean_good:.4f} | Std: {std_good:.4f}")
print(f"Bad Photos  (Label 0)  -> Mean: {mean_bad:.4f} | Std: {std_bad:.4f}")
print(f"Mean Difference        -> {mean_good - mean_bad:.4f}")
print(f"Cohen's d (Effect Size)-> {cohens_d:.4f} (|d| > 0.8 is considered strong)")
print(f"T-statistic            -> {t_stat:.4f} (p-value: {p_val:.4e})")
print(f"Correlation with Label -> {corr:.4f} (p-value: {corr_p:.4e})")

# 3. Class Separability via ROC-AUC & Optimal Threshold
fpr, tpr, thresholds = roc_curve(df["label"], df["aesthetic_score"])
auc_score = roc_auc_score(df["label"], df["aesthetic_score"])

# Find optimal threshold using Youden's J statistic (J = TPR - FPR)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
best_tpr = tpr[best_idx]
best_fpr = fpr[best_idx]

# Predict classes using the optimal cut-off
preds_at_best_thresh = (df["aesthetic_score"] >= best_threshold).astype(int)

print("\n" + "=" * 50)
print("SEPARABILITY & THRESHOLD ANALYSIS")
print("=" * 50)
print(f"ROC-AUC Score          -> {auc_score:.4f}")
print(f"Optimal Score Threshold-> {best_threshold:.4f}")
print(f"True Positive Rate     -> {best_tpr * 100:.2f}%")
print(f"False Positive Rate    -> {best_fpr * 100:.2f}%")

print("\nConfusion Matrix at Optimal Threshold:")
print(confusion_matrix(df["label"], preds_at_best_thresh))

print("\nClassification Report:")
print(classification_report(df["label"], preds_at_best_thresh, target_names=["Bad (0)", "Good (1)"]))

# 4. Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Histograms / KDE-like distributions
axes[0].hist(scores_bad, bins=25, alpha=0.6, color="crimson", label=f"Bad (Mean: {mean_bad:.2f})", density=True)
axes[0].hist(scores_good, bins=25, alpha=0.6, color="forestgreen", label=f"Good (Mean: {mean_good:.2f})", density=True)
axes[0].axvline(best_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Cut-off: {best_threshold:.2f}")
axes[0].set_title("Aesthetic Score Distribution by Class")
axes[0].set_xlabel("LAION Aesthetic Score")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot B: ROC Curve
axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
axes[1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
axes[1].scatter([best_fpr], [best_tpr], color="red", s=60, zorder=5, label=f"Best operating point")
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
axes[1].set_ylabel("True Positive Rate (Sensitivity)")
axes[1].set_title("ROC Curve for Aesthetic Score")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/workspace/P/dataset/aesthetic_separability.png", dpi=300)
print(f"\nPlot saved to -> /workspace/P/dataset/aesthetic_separability.png")