from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
from transformers.models.pixtral.processing_pixtral import PixtralProcessor
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlamaTokenizerFast,
    LlamaTokenizer,
    PretrainedConfig,
    ProcessorMixin,
)
from dataclasses import dataclass


class PixtralConfig(PretrainedConfig):
    model_type = "pixtral"
    pass


class PixtralProcessor(PixtralProcessor):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    pass


AutoProcessor.register(PixtralConfig, PixtralProcessor)

# AutoTokenizer.register(PixtralConfig, LlamaTokenizer, LlamaTokenizerFast)
