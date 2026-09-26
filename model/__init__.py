"""Feature extraction package for photo quality classification.

Exposes a composable pipeline of feature extractors that together produce
the feature vector:

    [laplacian_var, brightness_mean, clip_shadow_pct, clip_highlight_pct,
     histogram_std, has_face, blendshape_vector(52), saliency_x, saliency_y,
     thirds_dist, CLS_token(768)]                          -> 829-d total

Each block is implemented as its own FeatureExtractor (Single
Responsibility). The pipeline depends only on the FeatureExtractor
abstraction (Dependency Inversion), so new blocks can be added, or the
DINOv3 backbone swapped for another model, without modifying this package's
core classes (Open/Closed).
"""

from .pipeline import FeatureVectorPipeline
from .vector_builder import FeatureVectorAssembler

__all__ = ["FeatureVectorPipeline", "FeatureVectorAssembler"]
