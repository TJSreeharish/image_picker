"""Flattens a feature dict into the fixed-order numpy vector consumed by
the downstream SVM/GBM classifier."""

from typing import List

import numpy as np

from .interfaces import FeatureMap


class FeatureVectorAssembler:
    """Single responsibility: turn {name: value} into a stable-ordered
    np.ndarray. Kept separate from extraction/pipeline so vector layout
    and dimensionality can be tested independently of how each value was
    computed.
    """

    def __init__(self, ordered_names: List[str]):
        self._ordered_names = ordered_names

    def assemble(self, features: FeatureMap) -> np.ndarray:
        missing = set(self._ordered_names) - set(features.keys())
        if missing:
            raise KeyError(f"Feature vector assembly missing keys: {missing}")
        return np.array(
            [features[name] for name in self._ordered_names], dtype=np.float32
        )

    @property
    def vector_length(self) -> int:
        return len(self._ordered_names)
