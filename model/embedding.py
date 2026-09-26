"""Semantic embedding extraction via a frozen DINOv3 backbone."""

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from interfaces import FeatureExtractor, FeatureMap

CLS_TOKEN_DIM = 768
DEFAULT_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DinoV3EmbeddingExtractor(FeatureExtractor):
    """Wraps a frozen DINOv3 ViT-B/16 backbone and returns its CLS token
    as the semantic-content feature block.

    Model name and device are constructor-injected, so the backbone can
    later be swapped (e.g. for SigLIP or DINOv2) without changing the
    pipeline or any other extractor (Open/Closed, Dependency Inversion).
    """

    PREFIX = "cls_"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        self._device = device
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device).eval()

    @property
    def feature_names(self) -> List[str]:
        return [f"{self.PREFIX}{i}" for i in range(CLS_TOKEN_DIM)]

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> FeatureMap:
        rgb_image = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        inputs = self._processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

        if cls_token.shape[0] != CLS_TOKEN_DIM:
            raise ValueError(
                f"Expected a {CLS_TOKEN_DIM}-d CLS token, got "
                f"{cls_token.shape[0]}. Check that the loaded model matches "
                "CLS_TOKEN_DIM, or update CLS_TOKEN_DIM to match the model."
            )
        return {f"{self.PREFIX}{i}": float(v) for i, v in enumerate(cls_token)}
