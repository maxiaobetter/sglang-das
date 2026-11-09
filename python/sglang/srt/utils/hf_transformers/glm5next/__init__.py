"""Bundled GLM-5-Next multimodal processor implementation.

This is used when a checkpoint declares the legacy nested ``processor_config.json``
format but does not provide ``auto_map`` entries for its processor classes.
"""

from .image_processing_glm5next import (
    Glm5nextImageProcessor,
    Glm5nextImageProcessorFast,
)
from .processing_glm5next import Glm5nextProcessor
from .video_processing_glm5next import Glm5nextVideoProcessor

__all__ = [
    "Glm5nextImageProcessor",
    "Glm5nextImageProcessorFast",
    "Glm5nextProcessor",
    "Glm5nextVideoProcessor",
]
