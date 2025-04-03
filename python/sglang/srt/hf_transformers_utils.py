# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Huggingface Transformers."""

import contextlib
import os
import warnings
from pathlib import Path
import json
from typing import Dict, Optional, Type, Union

from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.srt.configs import (
    ChatGLMConfig,
    DbrxConfig,
    DeepseekVL2Config,
    ExaoneConfig,
    MultiModalityConfig,
    PixtralConfig,
)
from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url
from sglang.srt.configs.mistral import is_mistralai_model

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    ChatGLMConfig.model_type: ChatGLMConfig,
    DbrxConfig.model_type: DbrxConfig,
    ExaoneConfig.model_type: ExaoneConfig,
    DeepseekVL2Config.model_type: DeepseekVL2Config,
    MultiModalityConfig.model_type: MultiModalityConfig,
    # PixtralConfig.model_type: PixtralConfig,
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def download_from_hf(model_path: str) -> str:
    """
    Download model files from Huggingface.
    Args:
        model_path (str): The name or path of model.
    Returns:
        str: Folder path to the downloaded model.
    """
    if os.path.exists(model_path):
        return model_path

    return snapshot_download(model_path, allow_patterns=["*.json", "*.bin", "*.model"])


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = model
        model = Path(model).parent
    print("sglang.srt.hf_transformers_utils.get_config(): model =", model)
    
    config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )

    print("sglang.srt.hf_transformers_utils.get_config(): autoconfig() -> config =", config)

    # FIXME: Pour contents of janus-pro's langauge_config to first-level
    if isinstance(model, str) and model.lower().startswith("deepseek-ai/janus-pro"):
        assert hasattr(config, "language_config")
        for key, val in config.language_config.__dict__.items():
            setattr(config, key, val)
        setattr(config, "architectures", ["MultiModalityCausalLM"])
    
    if isinstance(model, str) and is_mistralai_model(model):
        config = load_mistral_config(model, revision)

    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model, revision=revision)
        # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
        setattr(config, "_name_or_path", model)
    if model_override_args:
        config.update(model_override_args)
    
    

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})
    print("sglang.srt.hf_transformers_utils.get_config(): config =", config)
    return config


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = Path(tokenizer_name).parent

    if is_remote_url(tokenizer_name):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(tokenizer_name)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        tokenizer_name = client.get_local_dir()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    attach_additional_stop_token_ids(tokenizer)
    return tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
):
    # pop 'revision' from kwargs if present.
    revision = kwargs.pop("revision", tokenizer_revision)

    config = AutoConfig.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )

    print("sglang.srt.hf_transformers_utils.get_processor(): config =", config)

    # fix: for Qwen2-VL model, inject default 'size' if not provided.
    if config.model_type in {"qwen2_vl"}:
        if "size" not in kwargs:
            kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 1003520}

    processor = AutoProcessor.from_pretrained(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )
    print("sglang.srt.hf_transformers_utils.get_processor(): processor =", processor)
    attach_additional_stop_token_ids(processor.tokenizer)
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None


def check_gguf_file(model: Union[str, os.PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"

def replace_key(dict_: dict, mapping: Dict[str, str]):
    new_dict = {}
    for key, value in dict_.items():
        new_key = mapping.get(key, key)
        if isinstance(value, dict):
            new_dict[new_key] = replace_key(value, mapping)
        else:
            new_dict[new_key] = value
    return new_dict


def fill_default(dict_: dict, mapping: Dict[str, str]):
    for key, value in mapping.items():
        if key not in dict_:
            dict_[key] = value
    return dict_

def config_ify(dict_: dict, cls=PretrainedConfig):
    for key, value in dict_.items():
        if isinstance(value, dict):
            dict_[key] = config_ify(value)
    return cls(**dict_)


def load_mistral_config(model, revision, config_file_name="params.json", cls=PretrainedConfig) -> PretrainedConfig:
    # This function loads a params.json config which
    # should be used when loading models in mistral format
    config_path = Path(model) / config_file_name
    if not config_path.is_file():
        config_path = Path(download_from_hf(model)) / config_file_name
    
    with open(config_path, "r") as file:
        config_dict = json.load(file)

    rename_mistral_to_hf = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "intermediate_size": "hidden_dim",
        "tie_embeddings": "tie_word_embeddings",
        "activation": "hidden_act",
        "vision_encoder": "vision_config",
    }

    hf_default_mapping = {
        "model_type": "transformer",
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "max_seq_len": 128_000,
    }

    config_dict = replace_key(config_dict, rename_mistral_to_hf)
    config_dict = fill_default(config_dict, hf_default_mapping)

    cls = PretrainedConfig
    if "moe" in config_dict:
        config_dict["architectures"] = ["MixtralForCausalLM"]
        from transformers.models.mixtral import MixtralConfig
        cls = MixtralConfig
    else:
        config_dict["architectures"] = ["MistralForCausalLM"]
        from transformers.models.mistral import MistralConfig
        cls = MistralConfig

    if "vision_config" in config_dict:
        vision_config = config_dict.pop("vision_config")
        text_config = config_dict
        from transformers.models.pixtral import PixtralVisionConfig
        from transformers.models.mistral import MistralConfig
        
        config_dict = {
            "text_config": MistralConfig(**text_config),
            "vision_config": PixtralVisionConfig(**vision_config)
        }
        config_dict["architectures"] = ["PixtralForConditionalGeneration"]
        config_dict["model_type"] = "pixtral"
        cls = PixtralConfig
    
    return config_ify(config_dict, cls)
