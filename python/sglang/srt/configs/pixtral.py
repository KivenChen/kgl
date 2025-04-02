import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig


@dataclass
class PixtralConfig(PixtralVisionConfig):
    def __init__(self, **kwargs):
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
