"""Quick test of NIMA (trained on AVA, ~255k images) and MUSIQ aesthetic
scoring on a handful of your photos, via the `pyiqa` toolbox -- no need to
hunt for weights or convert Keras checkpoints by hand.

Loads one metric at a time (freeing GPU memory before loading the next)
to keep peak VRAM down on a small card. Set --max-side 0 (the default) to
score at full original resolution; pass a positive value to downscale
first if you hit out-of-memory errors again.

Usage:
    python3 try_nima_musiq.py --images-dir /workspace/P/SELECTED --limit 10
    python3 try_nima_musiq.py --images-dir /workspace/P/SELECTED --limit 10 --max-side 768
"""

import argparse
import gc
from pathlib import Path

import cv2
import torch
import pyiqa

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}


def collect_image_paths(folder: Path, limit: int):
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)
    return paths[:limit] if limit else paths


def resize_and_save_temp(path: Path, max_side: int, tmp_dir: Path) -> Path:
    """Downscales so the longer edge is at most max_side, preserving aspect
    ratio, and writes to a temp file. If max_side <= 0, returns the
    original path unchanged (full-resolution scoring)."""
    if max_side <= 0:
        return path

    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    tmp_path = tmp_dir / path.name
    cv2.imwrite(str(tmp_path), image)
    return tmp_path


def score_with_metric(metric_name: str, device: str, image_paths, max_side: int, tmp_dir: Path) -> dict:
    metric = pyiqa.create_metric(metric_name, device=device)
    scores = {}
    with torch.no_grad():
        for path in image_paths:
            scoring_path = resize_and_save_temp(path, max_side, tmp_dir)
            try:
                scores[path.name] = metric(str(scoring_path)).item()
            except torch.cuda.OutOfMemoryError:
                print(f"  [OOM] {metric_name} could not score {path.name} at this resolution -- skipping.")
                scores[path.name] = None
                if device == "cuda":
                    torch.cuda.empty_cache()

    # Free this metric's GPU memory before loading the next one.
    del metric
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--limit", type=int, default=10, help="0 = no limit")
    parser.add_argument(
        "--max-side", type=int, default=0,
        help="Resize longer edge to this many pixels before scoring. 0 = full resolution (default).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_paths = collect_image_paths(Path(args.images_dir), args.limit)
    resolution_note = "full resolution" if args.max_side <= 0 else f"resized to max side {args.max_side}px"
    print(f"Scoring {len(image_paths)} images from {args.images_dir} ({resolution_note})\n")

    tmp_dir = Path("/tmp/nima_musiq_resized")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # NIMA: AVA-trained aesthetic predictor (1-10 scale, higher = better).
    # MUSIQ: transformer-based, native-resolution IQA (also higher = better,
    # but on a different scale -- don't directly compare NIMA vs MUSIQ
    # numbers to each other, only within the same metric across images).
    # Scored one at a time, not both loaded simultaneously, to fit in a
    # small VRAM budget.
    nima_scores = score_with_metric("nima", device, image_paths, args.max_side, tmp_dir)
    musiq_scores = score_with_metric("musiq", device, image_paths, args.max_side, tmp_dir)

    for path in image_paths:
        nima_str = f"{nima_scores[path.name]:.3f}" if nima_scores[path.name] is not None else "OOM"
        musiq_str = f"{musiq_scores[path.name]:.3f}" if musiq_scores[path.name] is not None else "OOM"
        print(f"{path.name:30s}  NIMA={nima_str}   MUSIQ={musiq_str}")


if __name__ == "__main__":
    main()