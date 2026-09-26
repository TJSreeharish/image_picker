"""Diagnoses why the classifiers are near chance level (AUC ~0.5).

Checks two things:
1. Do the explicit, human-interpretable features (blur, lighting, face,
   saliency) show ANY separation between good/bad groups on their own?
2. Cross-validated AUC (5-fold, more stable than one 61-sample test split)
   using engineered features only, CLS token only, and the full vector --
   isolates whether the 768-d embedding is helping, hurting, or irrelevant.

Usage:
    python3 diagnose_dataset.py --dataset-dir /workspace/P/dataset
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Matches the fixed layout from FeatureVectorPipeline.ordered_feature_names:
# [laplacian_var, brightness_mean, clip_shadow_pct, clip_highlight_pct,
#  histogram_std, has_face, blendshape(52), saliency_x, saliency_y,
#  thirds_dist, CLS_token(768)]
BLUR_LIGHTING_SLICE = slice(0, 5)
HAS_FACE_INDEX = 5
BLENDSHAPE_SLICE = slice(6, 58)
SALIENCY_SLICE = slice(58, 61)
ENGINEERED_SLICE = slice(0, 61)
CLS_SLICE = slice(61, 829)

ENGINEERED_NAMES = [
    "laplacian_var", "brightness_mean", "clip_shadow_pct",
    "clip_highlight_pct", "histogram_std", "has_face",
]


def print_group_separation(X: np.ndarray, y: np.ndarray) -> None:
    print("=== Per-feature group means (good vs bad) ===")
    good = X[y == 1]
    bad = X[y == 0]
    for i, name in enumerate(ENGINEERED_NAMES):
        good_mean, bad_mean = good[:, i].mean(), bad[:, i].mean()
        good_std, bad_std = good[:, i].std(), bad[:, i].std()
        diff_in_stds = abs(good_mean - bad_mean) / (bad_std + 1e-8)
        print(
            f"  {name:20s} good={good_mean:10.4f} (±{good_std:8.4f})  "
            f"bad={bad_mean:10.4f} (±{bad_std:8.4f})  "
            f"separation={diff_in_stds:.2f} std-devs"
        )
    print(
        "  (separation < ~0.2 std-devs means that feature alone barely "
        "differs between good/bad -- expected for a couple of these, "
        "concerning if ALL of them look like this)\n"
    )


def cv_auc(X: np.ndarray, y: np.ndarray, label: str) -> None:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg_scores = cross_val_score(logreg, X_scaled, y, cv=cv, scoring="roc_auc")

    svm = SVC(kernel="linear", probability=True, class_weight="balanced")
    svm_scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring="roc_auc")

    print(f"=== 5-fold CV AUC -- {label} ({X.shape[1]} dims) ===")
    print(f"  Logistic Regression: {logreg_scores.mean():.3f} (±{logreg_scores.std():.3f})")
    print(f"  SVM (linear):        {svm_scores.mean():.3f} (±{svm_scores.std():.3f})\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    data = np.load(dataset_dir / "features.npz")
    X, y = data["features"], data["labels"].astype(int)

    print(f"Total: {X.shape[0]} samples, {X.shape[1]} dims, "
          f"{y.sum()} good / {len(y) - y.sum()} bad\n")

    print_group_separation(X[:, ENGINEERED_SLICE], y)

    cv_auc(X[:, ENGINEERED_SLICE], y, "engineered features only (61-d)")
    cv_auc(X[:, CLS_SLICE], y, "CLS token only (768-d)")
    cv_auc(X, y, "full vector (829-d)")

    print(
        "If engineered-only clearly beats full-vector: the CLS token is "
        "adding noise at this sample size -- drop it or PCA it down hard.\n"
        "If ALL three hover near 0.5: the features (as currently computed) "
        "don't separate SELECTED from all_images for this dataset -- the "
        "selection criteria may be driven by something these features don't "
        "capture (e.g. peak action moment, subtle framing), or there's a "
        "pipeline bug worth double-checking (feature/label row alignment)."
    )


if __name__ == "__main__":
    main()