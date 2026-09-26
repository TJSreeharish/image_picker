"""Builds a labeled feature-vector dataset from two folders:

    all_images/   -- the full pool of candidate photos
    SELECTED/     -- the subset you've hand-picked as "good" (by filename)

Every file in all_images gets label 1 if its filename also appears in
SELECTED, else label 0. Features are extracted once per image and written
to disk as:

    <output-dir>/features.npz   -- "features" (N, 829) float32, "labels" (N,) int8
    <output-dir>/manifest.csv   -- index, filepath, label   (for traceability)

Checkpoints every --checkpoint-every images so a crash partway through a
large folder doesn't lose everything already processed.

Usage:
    python3 build_dataset.py \
        --all-images-dir /workspace/P/all_images \
        --selected-dir /workspace/P/SELECTED \
        --face-model face_landmarker.task \
        --output-dir dataset
"""

import argparse
import csv
from pathlib import Path
from typing import List, Set, Tuple

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


def build_selected_filename_set(selected_dir: Path) -> Set[str]:
    """Matches by filename *stem*, lowercased -- not the exact filename or
    full path. This survives the common case where SELECTED was exported
    or copied through a tool that changed extension case or format
    (e.g. .JPG -> .jpeg) while keeping the base name the same."""
    return {p.stem.lower() for p in collect_image_paths(selected_dir)}


def save_checkpoint(
    output_dir: Path, features: List[np.ndarray], labels: List[int], paths: List[Path]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "features.npz",
        features=np.stack(features).astype(np.float32),
        labels=np.array(labels, dtype=np.int8),
    )
    with open(output_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filepath", "label"])
        for idx, (path, label) in enumerate(zip(paths, labels)):
            writer.writerow([idx, str(path), label])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a labeled feature dataset.")
    parser.add_argument("--all-images-dir", required=True)
    parser.add_argument("--selected-dir", required=True)
    parser.add_argument("--face-model", default="face_landmarker.task")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    args = parser.parse_args()

    all_images_dir = Path(args.all_images_dir)
    selected_dir = Path(args.selected_dir)
    output_dir = Path(args.output_dir)

    selected_filenames = build_selected_filename_set(selected_dir)
    print(f"Found {len(selected_filenames)} selected (good) filenames.")

    image_paths = collect_image_paths(all_images_dir)
    print(f"Found {len(image_paths)} total images in {all_images_dir}.")

    all_stems = {p.stem.lower() for p in image_paths}
    overlap = len(selected_filenames & all_stems)
    print(f"Overlap between SELECTED and all_images (by stem): {overlap}")
    if overlap == 0 and selected_filenames:
        print(
            "  WARNING: zero overlap -- filenames likely don't correspond "
            "between the two folders at all (not just extension/case). "
            "Sample SELECTED stems: "
            f"{sorted(selected_filenames)[:5]}"
        )
        print(f"  Sample all_images stems: {sorted(all_stems)[:5]}")

    pipeline, assembler = build_pipeline(args.face_model)

    features: List[np.ndarray] = []
    labels: List[int] = []
    processed_paths: List[Path] = []

    for i, path in enumerate(image_paths, start=1):
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

        label = 1 if path.stem.lower() in selected_filenames else 0
        features.append(vector)
        labels.append(label)
        processed_paths.append(path)

        if i % 25 == 0 or i == len(image_paths):
            print(f"  processed {i}/{len(image_paths)}")

        if i % args.checkpoint_every == 0:
            save_checkpoint(output_dir, features, labels, processed_paths)
            print(f"  checkpoint saved at {i} images -> {output_dir}")

    save_checkpoint(output_dir, features, labels, processed_paths)
    good_count = sum(labels)
    print(
        f"Done. {good_count} good / {len(labels) - good_count} bad "
        f"-> saved to {output_dir}/"
    )


if __name__ == "__main__":
    main()