"""Analyzes which features separate good/bad across all three feature
sources you've built (fused engineered+DINOv3, NIMA/MUSIQ-style, VLM
hidden states), and trains XGBoost on each individually plus on their
concatenation (aligned by filename).

Auto-detects each npz's internal keys (label/filename/feature arrays)
rather than assuming a fixed schema, since img_vit_dataset.npz's exact
structure wasn't built as part of this conversation -- the inspection
printout at the top will show you exactly what was found, so you can
tell immediately if a guess was wrong.

Usage:
    python3 analyze_and_train.py
(edit the SOURCES dict below if your paths differ)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATASET_DIR = Path("/workspace/P/dataset")

# name -> (npz path, optional manifest csv for filename/label fallback)
SOURCES = {
    "engineered_dino": (DATASET_DIR / "features.npz", DATASET_DIR / "manifest.csv"),
    "img_vit":         (DATASET_DIR / "img_vit_dataset.npz", DATASET_DIR / "dataset_manifest.csv"),
    "vlm":             (DATASET_DIR / "dataset_vlm.npz", DATASET_DIR / "vlm_manifest.csv"),
}

TOP_N_FEATURES = 20
RANDOM_STATE = 42


def basename_stem(path_or_name) -> str:
    return Path(str(path_or_name)).stem


def load_manifest(csv_path: Path):
    df = pd.read_csv(csv_path)
    fname_col = next((c for c in df.columns if c.lower() in ("filepath", "filename", "path")), None)
    label_col = next((c for c in df.columns if c.lower() in ("label", "labels")), None)
    if fname_col is None or label_col is None:
        raise ValueError(f"Could not find filename/label columns in {csv_path}. Columns: {list(df.columns)}")
    stems = df[fname_col].apply(basename_stem).tolist()
    labels = df[label_col].astype(int).tolist()
    return stems, labels


def inspect_npz(npz_path: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)
    print(f"  {npz_path.name}:")
    for key in data.keys():
        arr = np.asarray(data[key])
        print(f"    {key:20s} shape={arr.shape} dtype={arr.dtype}")


def load_source(name: str, npz_path: Path, manifest_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    label_key = next((k for k in keys if k.lower() in ("label", "labels")), None)
    filename_key = next((k for k in keys if k.lower() in ("filename", "filenames", "filepath", "filepaths")), None)

    if filename_key is not None:
        stems = [basename_stem(f) for f in data[filename_key]]
    elif manifest_path.exists():
        stems, _ = load_manifest(manifest_path)
    else:
        raise ValueError(f"[{name}] no filename array in npz and no manifest found at {manifest_path}")

    if label_key is not None:
        y = np.asarray(data[label_key]).astype(int)
    else:
        _, y = load_manifest(manifest_path)
        y = np.array(y, dtype=int)

    feature_keys = [k for k in keys if k not in (label_key, filename_key)]
    blocks, feature_names = [], []
    for k in feature_keys:
        arr = np.asarray(data[k])
        if arr.dtype.kind in ("U", "S", "O"):  # skip string/object arrays (e.g. stray filename-like fields)
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
            feature_names.append(k)
        else:
            feature_names.extend([f"{k}[{i}]" for i in range(arr.shape[1])])
        blocks.append(arr.astype(np.float32))

    if not blocks:
        raise ValueError(f"[{name}] no usable numeric feature arrays found in {npz_path}")
    X = np.hstack(blocks)

    if not (len(stems) == X.shape[0] == len(y)):
        raise ValueError(
            f"[{name}] length mismatch -- stems={len(stems)} X={X.shape[0]} y={len(y)}. "
            "Check that the npz and its manifest correspond to the same run."
        )

    print(f"  [{name}] loaded: {X.shape[0]} samples, {X.shape[1]} dims "
          f"({y.sum()} good / {len(y) - y.sum()} bad)")
    return {"name": name, "stems": stems, "X": X, "y": y, "feature_names": feature_names}


def top_feature_separation(source: dict, top_n: int) -> pd.DataFrame:
    X, y = source["X"], source["y"]
    good, bad = X[y == 1], X[y == 0]
    good_mean, bad_mean = good.mean(axis=0), bad.mean(axis=0)
    pooled_std = np.sqrt((good.var(axis=0) + bad.var(axis=0)) / 2) + 1e-8
    cohens_d = np.abs(good_mean - bad_mean) / pooled_std

    order = np.argsort(-cohens_d)[:top_n]
    rows = [
        (source["feature_names"][i], cohens_d[i], good_mean[i], bad_mean[i])
        for i in order
    ]
    return pd.DataFrame(rows, columns=["feature", "cohens_d", "good_mean", "bad_mean"])


def cv_auc_xgb(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scale_pos_weight = (len(y) - y.sum()) / y.sum()
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    return scores


def align_sources_by_filename(sources: dict) -> tuple:
    """Inner-join every source on filename stem, so the concatenated
    matrix only includes images present in ALL sources, in matching row
    order across all of them."""
    stem_sets = [set(s["stems"]) for s in sources.values()]
    common_stems = sorted(set.intersection(*stem_sets))
    print(f"\nCommon filenames across all sources: {len(common_stems)}")

    aligned_blocks = []
    block_names = []
    y_ref = None
    for name, source in sources.items():
        stem_to_row = {stem: i for i, stem in enumerate(source["stems"])}
        indices = [stem_to_row[stem] for stem in common_stems]
        aligned_blocks.append(source["X"][indices])
        block_names.extend([f"{name}::{fn}" for fn in source["feature_names"]])

        y_this = source["y"][indices]
        if y_ref is None:
            y_ref = y_this
        elif not np.array_equal(y_ref, y_this):
            mismatches = int((y_ref != y_this).sum())
            print(f"  WARNING: {mismatches} label mismatches between sources on common filenames -- using engineered_dino's labels as ground truth.")

    X_combined = np.hstack(aligned_blocks)
    return X_combined, y_ref, block_names, common_stems


def main() -> None:
    print("=== Inspecting npz structure ===")
    for name, (npz_path, _) in SOURCES.items():
        if npz_path.exists():
            inspect_npz(npz_path)
        else:
            print(f"  [{name}] MISSING: {npz_path}")
    print()

    print("=== Loading sources ===")
    loaded = {name: load_source(name, npz_path, manifest_path) for name, (npz_path, manifest_path) in SOURCES.items()}
    print()

    print("=== Top separating features per source (Cohen's d) ===")
    for name, source in loaded.items():
        print(f"\n--- {name} ---")
        top_df = top_feature_separation(source, TOP_N_FEATURES)
        print(top_df.to_string(index=False))
        out_csv = DATASET_DIR / f"top_features_{name}.csv"
        top_df.to_csv(out_csv, index=False)
        print(f"  -> saved to {out_csv}")

    print("\n=== XGBoost, 5-fold CV AUC, per source ===")
    for name, source in loaded.items():
        scores = cv_auc_xgb(source["X"], source["y"])
        print(f"  {name:20s} ({source['X'].shape[1]:4d} dims): AUC = {scores.mean():.3f} (+/-{scores.std():.3f})")

    print("\n=== XGBoost, 5-fold CV AUC, concatenated (all sources, inner-joined by filename) ===")
    X_combined, y_combined, combined_names, common_stems = align_sources_by_filename(loaded)
    scores = cv_auc_xgb(X_combined, y_combined)
    print(f"  concatenated ({X_combined.shape[1]} dims, {X_combined.shape[0]} samples): "
          f"AUC = {scores.mean():.3f} (+/-{scores.std():.3f})")

    # Feature importance summed by source block, on a single fit over all
    # common-filename data (CV doesn't expose importances directly).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    scale_pos_weight = (len(y_combined) - y_combined.sum()) / y_combined.sum()
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_scaled, y_combined)

    importance_by_source = {}
    for name, importance in zip(combined_names, model.feature_importances_):
        source_name = name.split("::")[0]
        importance_by_source[source_name] = importance_by_source.get(source_name, 0.0) + importance

    print("\n=== Total XGBoost feature importance, summed by source ===")
    for source_name, total in sorted(importance_by_source.items(), key=lambda x: -x[1]):
        print(f"  {source_name:20s}: {total:.4f}")


if __name__ == "__main__":
    main()