"""Lighting / exposure feature extraction."""

from typing import List

import cv2
import numpy as np

from interfaces import FeatureExtractor, FeatureMap


class LightingExtractor(FeatureExtractor):
    """Exposure signals: mean brightness (Lab L-channel, perceptually
    uniform), shadow/highlight clipping percentages, and overall
    brightness-histogram spread.
    """

    BRIGHTNESS = "brightness_mean"
    CLIP_SHADOW = "clip_shadow_pct"
    CLIP_HIGHLIGHT = "clip_highlight_pct"
    HIST_STD = "histogram_std"

    def __init__(self, clip_tolerance: int = 0):
        """clip_tolerance: pixel values within this many levels of 0/255
        still count as clipped. 0 means only exact 0/255 values count."""
        self._clip_tolerance = clip_tolerance

    @property
    def feature_names(self) -> List[str]:
        return [self.BRIGHTNESS, self.CLIP_SHADOW, self.CLIP_HIGHLIGHT, self.HIST_STD]

    def extract(self, image: np.ndarray) -> FeatureMap:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l_channel = lab[:, :, 0]

        total_values = image.size  # height * width * 3 channels
        shadow_thresh = self._clip_tolerance
        highlight_thresh = 255 - self._clip_tolerance

        shadow_clipped = int(np.sum(image <= shadow_thresh))
        highlight_clipped = int(np.sum(image >= highlight_thresh))

        return {
            self.BRIGHTNESS: float(np.mean(l_channel)),
            self.CLIP_SHADOW: shadow_clipped / total_values,
            self.CLIP_HIGHLIGHT: highlight_clipped / total_values,
            self.HIST_STD: float(np.std(l_channel)),
        }
