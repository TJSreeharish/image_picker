"""Runs a list of FeatureExtractor implementations over one image."""

from typing import List

import numpy as np

from interfaces import FeatureExtractor, FeatureMap


class FeatureVectorPipeline:
    """Depends only on the FeatureExtractor abstraction (Dependency
    Inversion). Extractors are injected at construction time, so adding a
    new feature block -- or removing/swapping one -- never requires
    modifying this class (Open/Closed). Any object satisfying the
    FeatureExtractor contract can be substituted for another (Liskov
    Substitution).
    """

    def __init__(self, extractors: List[FeatureExtractor]):
        self._extractors = extractors

    def run(self, image: np.ndarray) -> FeatureMap:
        features: FeatureMap = {}
        for extractor in self._extractors:
            extracted = extractor.extract(image)
            missing = set(extractor.feature_names) - set(extracted.keys())
            if missing:
                raise RuntimeError(
                    f"{type(extractor).__name__} did not produce its "
                    f"declared features: {missing}"
                )
            features.update(extracted)
        return features

    @property
    def ordered_feature_names(self) -> List[str]:
        """Concatenated feature-name order, matching the order extractors
        were passed in -- this defines the layout of the final vector."""
        names: List[str] = []
        for extractor in self._extractors:
            names.extend(extractor.feature_names)
        return names
