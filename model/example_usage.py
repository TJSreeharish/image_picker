"""End-to-end example: build the pipeline and extract one feature vector.

    [laplacian_var, brightness_mean, clip_shadow_pct, clip_highlight_pct,
     histogram_std, has_face, blendshape_vector(52), saliency_x, saliency_y,
     thirds_dist, CLS_token(768)]                          -> 829-d total

Before running:
  pip install -r requirements.txt
  Download face_landmarker.task from:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/
    face_landmarker/float16/latest/face_landmarker.task
"""

from typing import Tuple

import cv2
import torch

from feature_extraction.extractors.blur import LaplacianBlurExtractor
from feature_extraction.extractors.embedding import DinoV3EmbeddingExtractor
from feature_extraction.extractors.face import MediaPipeFaceExtractor
from feature_extraction.extractors.lighting import LightingExtractor
from feature_extraction.extractors.saliency import SaliencyExtractor
from feature_extraction.pipeline import FeatureVectorPipeline
from feature_extraction.vector_builder import FeatureVectorAssembler


def build_pipeline(
    face_model_path: str, device: str = "cpu"
) -> Tuple[FeatureVectorPipeline, FeatureVectorAssembler]:
    """Composition root: the only place concrete extractor classes are
    instantiated. Everything else in the package only ever sees the
    FeatureExtractor abstraction.
    """
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


if __name__ == "__main__":
    # Same image runs on a GPU box or a CPU-only box unmodified -- no
    # rebuild needed when shipping to a machine without an NVIDIA GPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline, assembler = build_pipeline(
        face_model_path="face_landmarker.task", device=device
    )

    image = cv2.imread("sample.jpg")
    if image is None:
        raise FileNotFoundError("sample.jpg not found -- point this at a real image.")

    features = pipeline.run(image)
    vector = assembler.assemble(features)

    print(f"Feature vector length: {assembler.vector_length}")
    print(f"Vector shape: {vector.shape}")
