# A central config resolver for mistral models
from typing import Any
from pathlib import Path
from transformers import PretrainedConfig
import json

MISTRAL_AI_MODEL_NAMES = {
    "mistral",
    "pixtral"
}

def is_mistralai_model(model_name: str) -> bool:
    return any(mistral_model in model_name.lower() for mistral_model in MISTRAL_AI_MODEL_NAMES)

__all__ = [
    "is_mistralai_model"
]
    