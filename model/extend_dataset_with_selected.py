"""Extends an EXISTING dataset (already built from all_images, all labeled
0) by extracting features for SELECTED/ (label 1) and appending them --
without recomputing features for images already sitting in the dataset.

Usage:
    python3 extend_dataset_with_selected.py \
        --selected-dir /workspace/P/SELECTED \
        --face-model /workspace/model/face_landmarker.task \
        --dataset-dir /workspace/P/dataset
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from   blur import LaplacianBlurExtractor
from   embedding import DinoV3EmbeddingExtractor
from   face import MediaPipeFaceExtractor
from   lighting import LightingExtractor
from   saliency import SaliencyExtractor
from pipeline import FeatureVectorPipeline
from vector_builder import FeatureVectorAssembler

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}


def build_pipeline(face_model_path: str) -> Tuple[FeatureVectorPipeline, FeatureVectorAssembler]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    extractors = [
        LaplacianBlurExtractor(),
        LightingExtractor(),
        MediaPipeFaceExtractor(model_asset_path=face_model_path),
        SaliencyExtractor(),
        DinoV3EmbeddingExtractor(device=device),
    ]
    pipeline = FeatureVectorPipeline(extractors)
    assembler = FeatureVectorAssembler(pipeline.ordered_feature_names)
    return pipeline, assembler


def collect_image_paths(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)


def load_existing(dataset_dir: Path):
    data = np.load(dataset_dir / "features.npz")
    features = list(data["features"])
    labels = [int(label) for label in data["labels"]]  # avoid int8 overflow later

    paths: List[str] = []
    with open(dataset_dir / "manifest.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["filepath"])

    return features, labels, paths


def save_dataset(dataset_dir: Path, features, labels, paths) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_dir / "features.npz",
        features=np.stack(features).astype(np.float32),
        labels=np.array(labels, dtype=np.int8),
    )
    with open(dataset_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filepath", "label"])
        for idx, (path, label) in enumerate(zip(paths, labels)):
            writer.writerow([idx, path, label])


def main() -> None:
    parser = argparse.ArgumentParser(description="Append SELECTED (label=1) to an existing dataset.")
    parser.add_argument("--selected-dir", required=True)
    parser.add_argument("--face-model", default="face_landmarker.task")
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    features, labels, paths = load_existing(dataset_dir)
    existing_path_set = set(paths)
    print(
        f"Loaded existing dataset: {len(paths)} entries "
        f"({sum(labels)} good / {len(labels) - sum(labels)} bad)."
    )

    pipeline, assembler = build_pipeline(args.face_model)

    selected_paths = collect_image_paths(Path(args.selected_dir))
    print(f"Found {len(selected_paths)} images in SELECTED.")

    added = 0
    for i, path in enumerate(selected_paths, start=1):
        if str(path) in existing_path_set:
            print(f"  [skip] already in dataset: {path}")
            continue

        image = cv2.imread(str(path))
        if image is None:
            print(f"  [skip] could not read: {path}")
            continue

        try:
            feature_map = pipeline.run(image)
            vector = assembler.assemble(feature_map)
        except Exception as exc:
            print(f"  [skip] extraction failed for {path}: {exc}")
            continue

        features.append(vector)
        labels.append(1)
        paths.append(str(path))
        added += 1

        if i % 25 == 0 or i == len(selected_paths):
            print(f"  processed {i}/{len(selected_paths)}")

    save_dataset(dataset_dir, features, labels, paths)
    good_count = sum(labels)
    print(f"Added {added} new good-labeled entries.")
    print(
        f"Final dataset: {len(labels)} total "
        f"({good_count} good / {len(labels) - good_count} bad) -> {dataset_dir}/"
    )


if __name__ == "__main__":
    main()