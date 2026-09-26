"""Face presence + expression (blendshape) feature extraction via
MediaPipe's Face Landmarker task."""

from typing import List

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from interfaces import FeatureExtractor, FeatureMap

NUM_BLENDSHAPES = 52


class MediaPipeFaceExtractor(FeatureExtractor):
    """Detects the largest (most prominent) face and returns its 52-value
    blendshape vector.

    Zero-fills the whole block and sets has_face=0.0 when no face is
    found, so the assembled vector's shape and scale stay consistent
    whether or not a face is present.
    """

    HAS_FACE = "has_face"
    BLENDSHAPE_PREFIX = "blendshape_"

    def __init__(self, model_asset_path: str, num_faces: int = 3):
        base_options = mp_python.BaseOptions(model_asset_path=model_asset_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=num_faces,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    @property
    def feature_names(self) -> List[str]:
        return [self.HAS_FACE] + [
            f"{self.BLENDSHAPE_PREFIX}{i}" for i in range(NUM_BLENDSHAPES)
        ]

    def extract(self, image: np.ndarray) -> FeatureMap:
        rgb = np.ascontiguousarray(image[:, :, ::-1])  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_blendshapes:
            return self._empty_features()

        chosen = self._select_primary_face(result, image.shape)
        scores = [category.score for category in chosen]
        if len(scores) != NUM_BLENDSHAPES:
            # Defensive: keep the vector length fixed even if a future
            # MediaPipe version changes the blendshape count.
            scores = (scores + [0.0] * NUM_BLENDSHAPES)[:NUM_BLENDSHAPES]

        features: FeatureMap = {self.HAS_FACE: 1.0}
        features.update(
            {f"{self.BLENDSHAPE_PREFIX}{i}": float(s) for i, s in enumerate(scores)}
        )
        return features

    def _select_primary_face(self, result, image_shape) -> List:
        """Picks the largest face by landmark bounding-box area. Falls
        back to the first detected face when only one is present."""
        if len(result.face_blendshapes) == 1 or not result.face_landmarks:
            return result.face_blendshapes[0]

        height, width = image_shape[0], image_shape[1]
        best_idx, best_area = 0, -1.0
        for idx, landmarks in enumerate(result.face_landmarks):
            xs = [lm.x * width for lm in landmarks]
            ys = [lm.y * height for lm in landmarks]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > best_area:
                best_area, best_idx = area, idx
        return result.face_blendshapes[best_idx]

    def _empty_features(self) -> FeatureMap:
        features: FeatureMap = {self.HAS_FACE: 0.0}
        features.update(
            {f"{self.BLENDSHAPE_PREFIX}{i}": 0.0 for i in range(NUM_BLENDSHAPES)}
        )
        return features
