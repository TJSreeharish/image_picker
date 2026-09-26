"""Blur / sharpness feature extraction."""

from typing import List

import cv2
import numpy as np

from ..interfaces import FeatureExtractor, FeatureMap


class LaplacianBlurExtractor(FeatureExtractor):
    """Variance of the Laplacian as a blur/sharpness proxy.

    Low variance -> low edge energy -> the image is likely blurry.
    """

    FEATURE_NAME = "laplacian_var"

    @property
    def feature_names(self) -> List[str]:
        return [self.FEATURE_NAME]

    def extract(self, image: np.ndarray) -> FeatureMap:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {self.FEATURE_NAME: float(variance)}
