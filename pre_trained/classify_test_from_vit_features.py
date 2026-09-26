import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

# 1. Load the NPZ Dataset
npz_path = "/workspace/P/dataset/img_vit_dataset.npz"
data = np.load(npz_path)

X = data["embeddings"]  # Shape: (305, 512)
y = data["labels"].astype(int)  # Shape: (305,)

print(f"Dataset loaded: X = {X.shape}, y = {y.shape}")
print(f"Good samples (1): {np.sum(y == 1)}, Bad samples (0): {np.sum(y == 0)}")

good_mask = (y == 1)
bad_mask = (y == 0)

num_features = X.shape[1]
stats_records = []

# 2. Compute Statistical Metrics for Every Single Dimension
for dim_idx in range(num_features):
    vals_good = X[good_mask, dim_idx]
    vals_bad = X[bad_mask, dim_idx]
    
    mean_good, std_good = np.mean(vals_good), np.std(vals_good, ddof=1)
    mean_bad, std_bad = np.mean(vals_bad), np.std(vals_bad, ddof=1)
    
    # Welch's t-test (assumes unequal variances)
    t_stat, p_val = stats.ttest_ind(vals_good, vals_bad, equal_var=False)
    
    # Cohen's d effect size
    pooled_std = np.sqrt(((len(vals_good) - 1) * std_good**2 + (len(vals_bad) - 1) * std_bad**2) / (len(vals_good) + len(vals_bad) - 2))
    cohens_d = (mean_good - mean_bad) / pooled_std if pooled_std > 0 else 0.0
    
    # Univariate ROC-AUC score (handles positive or negative correlation)
    auc_raw = roc_auc_score(y, X[:, dim_idx])
    # Directed separability: how far from random (0.5) the dimension is
    auc_directed = max(auc_raw, 1.0 - auc_raw)
    
    stats_records.append({
        "feature_dim": dim_idx,
        "mean_diff": mean_good - mean_bad,
        "abs_t_stat": abs(t_stat),
        "t_stat": t_stat,
        "p_val": p_val,
        "cohens_d": cohens_d,
        "abs_cohens_d": abs(cohens_d),
        "roc_auc_directed": auc_directed,
        "correlation_direction": "Positive (+)" if mean_good > mean_bad else "Negative (-)"
    })

df_stats = pd.DataFrame(stats_records)

# 3. Select Top 40 Features Ranked by Welch's t-test significance
top_40_df = df_stats.sort_values(by="abs_t_stat", ascending=False).head(40).reset_index(drop=True)

print("\n" + "=" * 80)
print("TOP 40 DISCRIMINATIVE CLIP EMBEDDING FEATURES (RANKED BY SIGNIFICANCE)")
print("=" * 80)
print(top_40_df[["feature_dim", "abs_t_stat", "p_val", "abs_cohens_d", "roc_auc_directed", "correlation_direction"]].to_string(index=True))

# 4. Save Top 40 Feature Indices and Statistics
top_indices = top_40_df["feature_dim"].values
top_40_df.to_csv("/workspace/P/dataset/top_40_features_stats.csv", index=False)

# Save pruned matrix containing only the top 40 features
X_top40 = X[:, top_indices]
np.savez_compressed(
    "/workspace/P/dataset/img_vit_top40.npz",
    embeddings_top40=X_top40,
    top_indices=top_indices,
    labels=y
)
print("\nSaved files:")
print(" - Stats table : /workspace/P/dataset/top_40_features_stats.csv")
print(f" - Pruned NPZ  : /workspace/P/dataset/img_vit_top40.npz (Shape: {X_top40.shape})")

# 5. Visualization: Effect Size & Univariate AUC of the Top 40 Features
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].bar(range(40), top_40_df["abs_t_stat"], color="royalblue", edgecolor="black")
axes[0].set_ylabel("|t-statistic|")
axes[0].set_title("Top 40 Features: Welch's t-test Magnitude")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

axes[1].bar(range(40), top_40_df["roc_auc_directed"], color="darkorange", edgecolor="black")
axes[1].axhline(0.5, color="red", linestyle="--", label="Random Chance (0.50)")
axes[1].set_ylabel("Univariate AUC (|Directional|)")
axes[1].set_xlabel("Top Feature Rank (0 to 39)")
axes[1].set_xticks(range(40))
axes[1].set_xticklabels(top_40_df["feature_dim"], rotation=90, fontsize=8)
axes[1].set_title("Individual Dimension ROC-AUC Performance")
axes[1].legend(loc="lower right")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("/workspace/P/dataset/top_40_features_plot.png", dpi=300)
print(" - Plot saved  : /workspace/P/dataset/top_40_features_plot.png")