"""Core abstractions.

Every concrete extractor depends on nothing but this module, and the
pipeline depends only on this abstraction, never on a concrete extractor
class. This is what lets you add, remove, or swap feature blocks (e.g.
DINOv3 -> SigLIP) without touching any other file.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

FeatureMap = Dict[str, float]


class FeatureExtractor(ABC):
    """Single-responsibility contract: extract one coherent group of
    features from a single BGR image (as returned by cv2.imread)."""

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Ordered list of feature keys this extractor guarantees to
        produce. Used by FeatureVectorAssembler to build a stable,
        reproducible vector layout."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, image: np.ndarray) -> FeatureMap:
        """Return a flat dict of {feature_name: value}. Must contain every
        key listed in `feature_names`, even on failure -- fill with a
        neutral default (0.0 / 0.5) rather than omitting the key, so the
        assembled vector's shape never changes between images."""
        raise NotImplementedError
