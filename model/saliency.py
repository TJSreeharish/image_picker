"""Saliency-based composition feature extraction."""

from typing import List, Tuple

import cv2
import numpy as np

from ..interfaces import FeatureExtractor, FeatureMap

# Normalized (x, y) rule-of-thirds intersection points.
THIRDS_POINTS = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]


class SaliencyExtractor(FeatureExtractor):
    """Reduces an OpenCV fine-grained saliency map to a normalized,
    saliency-weighted subject centroid plus distance to the nearest
    rule-of-thirds intersection.
    """

    SALIENCY_X = "saliency_x"
    SALIENCY_Y = "saliency_y"
    THIRDS_DIST = "thirds_dist"

    def __init__(self):
        # Requires opencv-contrib-python (cv2.saliency is not in the
        # base opencv-python package).
        self._saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    @property
    def feature_names(self) -> List[str]:
        return [self.SALIENCY_X, self.SALIENCY_Y, self.THIRDS_DIST]

    def extract(self, image: np.ndarray) -> FeatureMap:
        success, saliency_map = self._saliency.computeSaliency(image)
        if not success:
            return self._center_fallback()

        cx, cy = self._weighted_centroid(saliency_map)
        return {
            self.SALIENCY_X: cx,
            self.SALIENCY_Y: cy,
            self.THIRDS_DIST: self._thirds_distance(cx, cy),
        }

    def _weighted_centroid(self, saliency_map: np.ndarray) -> Tuple[float, float]:
        height, width = saliency_map.shape
        total = saliency_map.sum()
        if total <= 0:
            return 0.5, 0.5

        ys, xs = np.indices(saliency_map.shape)
        cx = float((xs * saliency_map).sum() / total) / width
        cy = float((ys * saliency_map).sum() / total) / height
        return cx, cy

    def _thirds_distance(self, cx: float, cy: float) -> float:
        return min(
            ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in THIRDS_POINTS
        )

    def _center_fallback(self) -> FeatureMap:
        return {
            self.SALIENCY_X: 0.5,
            self.SALIENCY_Y: 0.5,
            self.THIRDS_DIST: self._thirds_distance(0.5, 0.5),
        }
