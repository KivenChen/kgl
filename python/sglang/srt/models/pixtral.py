# Copyright 2024 SGLang Team
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
"""Inference-only Pixtral model compatible with HuggingFace weights."""

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PixtralVisionConfig
from transformers.activations import ACT2FN
from transformers.models.pixtral.image_processing_pixtral import _num_image_tokens

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.mistral import MistralForCausalLM
from sglang.srt.models.llava import LlavaBaseForCausalLM
from sglang.srt.utils import add_prefix


cached_get_processor = lru_cache(get_processor)

PATCH_MERGE = "patch_merge"


# Types and Utility Functions
@dataclass
class VisionEncoderArgs:
    """Args for the vision encoder part of Pixtral."""
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float  # for rope-2D
    image_token_id: int
    adapter_bias: bool = True
    spatial_merge_size: int = 1
    add_pre_mm_projector_layer_norm: bool = False
    mm_projector_id: str = ""


def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    """
    Precomputes freqs_cis 2D tensor of shape (height, width, dim // 2) 
    to be indexed by (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2).float() / dim))
    h_idx = torch.arange(height).float()
    w_idx = torch.arange(width).float()
    
    # Compute positional embeddings for 2D grid
    freq_h = torch.outer(h_idx, freqs)
    freq_w = torch.outer(w_idx, freqs)
    
    # Combine height and width frequency components
    freqs_h = torch.cat([freq_h.cos().unsqueeze(-1), freq_h.sin().unsqueeze(-1)], dim=-1)
    freqs_w = torch.cat([freq_w.cos().unsqueeze(-1), freq_w.sin().unsqueeze(-1)], dim=-1)
    
    # Reshape to match requirements
    freqs_h = freqs_h.reshape(height, 1, 1, -1)
    freqs_w = freqs_w.reshape(1, width, 1, -1)
    
    # Broadcast and combine
    freqs_h = freqs_h.expand(-1, width, -1, -1)
    freqs_w = freqs_w.expand(height, -1, -1, -1)
    freqs_2d = torch.cat([freqs_h, freqs_w], dim=2)
    
    # Reshape to flat sequence
    freqs_2d = freqs_2d.reshape(-1, dim)
    
    # Convert to complex format
    freqs_cis = torch.complex(freqs_2d[..., 0::2], freqs_2d[..., 1::2])
    return freqs_cis


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting with input tensor.
    
    Args:
        freqs_cis: complex tensor of shape (seq_len, head_dim / 2)
        x: complex tensor of shape (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_vit(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    
    Args:
        xq: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        xk: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        freqs_cis: Frequency tensor of shape (seq_len, head_dim / 2)
        
    Returns:
        Tuple of tensors after applying rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    def __init__(self, args: VisionEncoderArgs, act_type: str = "silu"):
        super().__init__()
        self.act_type = act_type
        self.act_fn = ACT2FN(act_type)
        self.gate_proj = ColumnParallelLinear(
            args.hidden_size,
            args.intermediate_size,
            bias=True,
            quant_config=None,
            prefix="",
        )
        self.up_proj = ColumnParallelLinear(
            args.hidden_size,
            args.intermediate_size,
            bias=True,
            quant_config=None,
            prefix="",
        )
        self.down_proj = RowParallelLinear(
            args.intermediate_size,
            args.hidden_size,
            bias=True,
            quant_config=None,
            prefix="",
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_x, _ = self.gate_proj(x)
        gated_x = self.act_fn(gated_x)
        x_up, _ = self.up_proj(x)
        x_to_down = gated_x * x_up
        out, _ = self.down_proj(x_to_down)
        return out


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""
    
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.attention = VisionAttention(
            embed_dim=args.hidden_size,
            num_heads=args.num_attention_heads,
            projection_size=args.hidden_size,  # TODO: check again
            use_qkv_parallel=True,
            quant_config=None,
            dropout=0.0,
            use_context_forward=True,
            softmax_in_single_precision=False,
            flatten_batch=False,
            prefix="",
        )
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.hidden_size, eps=1e-5)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=1e-5)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis: torch.Tensor):
        # Self-attention block
        # TODO: figure out how this works
        h = x + self.attention(self.attention_norm(x), mask, freqs_cis)
        # Feed-forward block
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Stack of transformer blocks for vision processing."""
    
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(TransformerBlock(args))
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis: Optional[torch.Tensor]):
        for layer in self.layers:
            x = layer(x, mask, freqs_cis)
        return x


def position_meshgrid(patch_embeds_list: List[torch.Tensor]):
    """Create a mesh grid of positions for patches."""
    positions = []
    for patch in patch_embeds_list:
        h, w = patch.shape[-2:]
        h_pos = torch.arange(h, device=patch.device)
        w_pos = torch.arange(w, device=patch.device)
        pos = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).reshape(-1, 2)
        positions.append(pos)
    return torch.cat(positions)


class VisionTransformer(nn.Module):
    """Vision transformer for processing image inputs."""
    
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.args = args
        self.patch_conv = nn.Conv2d(
            in_channels=args.num_channels,
            out_channels=args.hidden_size,
            kernel_size=args.patch_size,
            stride=args.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(args.hidden_size, eps=1e-5)
        self.transformer = Transformer(args)
        
        head_dim = self.args.hidden_size // self.args.num_attention_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: Optional[torch.Tensor] = None
    
    @property
    def max_patches_per_side(self):
        return self.args.image_size // self.args.patch_size
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def freqs_cis(self):
        if self._freqs_cis is None:
            # Generate rotary embeddings for 2D positions
            max_side = self.max_patches_per_side
            head_dim = self.args.hidden_size // self.args.num_attention_heads
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=head_dim,
                height=max_side,
                width=max_side,
                theta=self.args.rope_theta,
            ).to(device=self.device, dtype=self.dtype)
        return self._freqs_cis
    
    def forward(self, images: List[torch.Tensor]):
        """
        Args:
            images: list of N_img images of variable sizes,
                each of shape (C, H, W)
        Returns:
            image_features: tensor of token features for
                all tokens of all images of shape (N_toks, D)
        """
        # Convert input images to patches
        patch_embeds_list = []
        for img in images:
            # Add batch dimension if not present
            if img.dim() == 3:
                img = img.unsqueeze(0)
            # Convert to patches
            patches = self.patch_conv(img)
            patch_embeds_list.append(patches)
        
        # Create attention mask for each image (block diagonal)
        batch_size = len(patch_embeds_list)
        seq_lengths = [p.shape[-2] * p.shape[-1] for p in patch_embeds_list]
        max_seq_len = max(seq_lengths)
        
        # Prepare mask (one per image)
        attention_mask = torch.zeros(
            (batch_size, 1, max_seq_len, max_seq_len),
            device=self.device,
            dtype=self.dtype,
        )
        for i, seq_len in enumerate(seq_lengths):
            # Create causal mask for each sequence
            attention_mask[i, 0, :seq_len, :seq_len] = torch.zeros(
                (seq_len, seq_len),
                device=self.device,
                dtype=self.dtype,
            )
        
        # Reshape patches to token sequences
        patch_embeds = []
        for p in patch_embeds_list:
            p = p.flatten(2).transpose(1, 2)
            patch_embeds.append(p)
        
        # Get positions for rotary embeddings
        pos = position_meshgrid(patch_embeds_list)
        pos_idx = pos[:, 0] * self.max_patches_per_side + pos[:, 1]
        pos_idx = pos_idx.to(device=self.device)
        freqs_cis = self.freqs_cis[pos_idx]
        
        # Combine patch embeddings from all images
        patch_embeds = torch.cat(patch_embeds, dim=1)  # [B, N, D]
        patch_embeds = self.ln_pre(patch_embeds)
        
        # Pass through transformer
        hidden_states = self.transformer(patch_embeds, attention_mask, freqs_cis)  # [B, N, D]
        
        # Split back by image
        image_features = []
        start_idx = 0
        for seq_len in seq_lengths:
            image_features.append(hidden_states[:, start_idx:start_idx+seq_len])
            start_idx += seq_len
        
        # Stack for efficient processing
        return image_features


class PatchMerger(nn.Module):
    """Learned merging of spatial_merge_size ** 2 patches."""
    
    def __init__(self, vision_encoder_dim: int, spatial_merge_size: int, use_mlp_bias: bool = False, quant_config=None, prefix="") -> None:
        super().__init__()
        
        mlp_input_dim = vision_encoder_dim * (spatial_merge_size**2)
        
        self.spatial_merge_size = spatial_merge_size
        self.mlp_input_dim = mlp_input_dim
        
        # Using RowParallelLinear since we're reducing dimensions (large input → smaller output)
        self.merging_layer = RowParallelLinear(
            mlp_input_dim,
            vision_encoder_dim,
            bias=use_mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("merging_layer", prefix)
        )
    
    def forward(self, x: torch.Tensor, image_sizes: list[tuple[int, int]]):
        """Merge patches spatially and project them."""
        if self.spatial_merge_size <= 1:
            return x
        
        # Get merged patches with shape [N / spatial_merge_size ** 2, D * spatial_merge_size ** 2]
        merged_x = self.permute(x, image_sizes)
        
        # Apply projection to get final shape [N / spatial_merge_size ** 2, D]
        out, _ = self.merging_layer(merged_x)
        return out
    
    def permute(self, x: torch.Tensor, image_sizes: list[tuple[int, int]]):
        """
        Args:
            x: (N, D) where N is flattened and concatenated patch tokens
                for all images
            image_sizes: list of tuple of (height, width) in tokens for
                each image
        Returns:
            image_features: reorders patch tokens so each grid of
                (spatial_merge_size, spatial_merge_size) is contiguous.
                now (N / spatial_merge_size ** 2, D * spatial_merge_size ** 2)
        """
        if self.spatial_merge_size <= 1:
            return x
            
        merging_size = self.spatial_merge_size
        sub_grids = []
        offset = 0
        
        for h, w in image_sizes:
            # Check if dimensions are compatible with merging size
            # Padding to ensure it works with any size
            pad_h = (merging_size - (h % merging_size)) % merging_size
            pad_w = (merging_size - (w % merging_size)) % merging_size
            padded_h, padded_w = h + pad_h, w + pad_w
            
            # Number of grid cells after merging
            grid_h, grid_w = padded_h // merging_size, padded_w // merging_size
            
            # Create padded tensor if needed
            valid_tokens = h * w
            if pad_h > 0 or pad_w > 0:
                padded_size = padded_h * padded_w
                padded_x = torch.zeros(
                    padded_size, x.shape[1],
                    device=x.device, dtype=x.dtype
                )
                padded_x[:valid_tokens] = x[offset:offset+valid_tokens]
                img_x = padded_x
            else:
                img_x = x[offset:offset+valid_tokens]
            
            # Reshape to 2D grid
            grid_tokens = img_x.reshape(padded_h, padded_w, -1)
            
            # Reshape to merge merging_size x merging_size patches
            grid_merged = grid_tokens.reshape(
                grid_h, merging_size, grid_w, merging_size, -1
            )
            
            # Transpose to get merging_size x merging_size patches together
            grid_merged = grid_merged.permute(0, 2, 1, 3, 4)
            
            # Reshape to get the final merged representation
            grid_merged = grid_merged.reshape(
                grid_h * grid_w, merging_size * merging_size * grid_tokens.shape[-1]
            )
            
            sub_grids.append(grid_merged)
            offset += valid_tokens
        
        # Combine all grids from all images
        return torch.cat(sub_grids, dim=0)


class VisionLanguageAdapter(nn.Module):
    """Adapter to connect vision and language models."""
    
    def __init__(self, args: VisionEncoderArgs, dim: int, quant_config=None, prefix=""):
        super().__init__()
        assert isinstance(args, VisionEncoderArgs)
        
        # Using ColumnParallelLinear for w_in as it might expand dimensions
        self.w_in = ColumnParallelLinear(
            args.hidden_size,
            dim,
            bias=args.adapter_bias,
            quant_config=quant_config,
            prefix=add_prefix("w_in", prefix)
        )
        self.gelu = nn.GELU()
        
        # Using ColumnParallelLinear for same dimension projection
        # Could also use RowParallelLinear if needed for tensor parallelism flow
        self.w_out = ColumnParallelLinear(
            dim, 
            dim, 
            bias=args.adapter_bias,
            quant_config=quant_config,
            prefix=add_prefix("w_out", prefix)
        )
    
    def forward(self, x: torch.Tensor):
        x_in, _ = self.w_in(x)
        x_activated = self.gelu(x_in)
        x_out, _ = self.w_out(x_activated)
        return x_out


class PixtralProcessorAdapter:
    """Adapter for Pixtral's image processor."""
    
    def __init__(self, vision_encoder_args: VisionEncoderArgs, processor=None):
        self.vision_encoder_args = vision_encoder_args
        self.processor = processor
        
        # These don't actually get used, but they're here for API compatibility
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
    
    @property
    def image_size(self):
        return self.vision_encoder_args.image_size
    
    def preprocess(self, images, return_tensors="pt"):
        if self.processor is not None:
            return self.processor(images=images, return_tensors=return_tensors)
        return None


class PixtralProcessingInfo:
    """Information required for processing multimodal inputs in Pixtral."""
    
    def __init__(self, config):
        self.padding_pattern = MultiModalityDataPaddingPatternTokenPairs(
            prefix_tokens=[],
            suffix_tokens=[],
            token_pair_prefix=[],
            token_pair_suffix=[],
        )
    
    @staticmethod
    def get_image_token_id(model_name: str, config, tokenizer) -> int:
        """Get the ID for the image token."""
        return tokenizer.convert_tokens_to_ids("<image>")


class PixtralMultiModalProcessor:
    """Processor for handling multimodal inputs in Pixtral."""
    
    def __init__(self, config, vision_encoder_args: VisionEncoderArgs):
        self.vision_encoder_args = vision_encoder_args
        self.processor_adapter = PixtralProcessorAdapter(vision_encoder_args)
        self.mm_processing_info = PixtralProcessingInfo(config)
        self.image_token_id = vision_encoder_args.image_token_id
    
    def _precompute_image_features(self, vision_tower, all_img_paths, model_device):
        """Process all images to extract features ahead of time."""
        image_sizes = []
        all_images = []
        
        for img_path in all_img_paths:
            # img could be a PIL Image, a tensor, or a list of those
            try:
                from PIL import Image
                img = Image.open(img_path)
                # Keep track of original image dimensions for PatchMerger
                w, h = img.size
                patch_size = self.vision_encoder_args.patch_size
                num_patches_w = w // patch_size
                num_patches_h = h // patch_size
                image_sizes.append((num_patches_h, num_patches_w))
                
                # Preprocess image
                proc_result = self.processor_adapter.preprocess(img)
                if proc_result:
                    img_tensor = proc_result.pixel_values[0]
                else:
                    # Fallback if processor is not available
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((self.vision_encoder_args.image_size, self.vision_encoder_args.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ])
                    img_tensor = transform(img)
                    
                all_images.append(img_tensor.to(model_device))
            except Exception as e:
                raise ValueError(f"Failed to process image {img_path}: {e}")
        
        # Process all images through vision tower
        with torch.no_grad():
            feats = vision_tower(all_images)
        
        # Validate output format
        if not isinstance(feats, list):
            feats = [feats]
        
        # Return both features and image sizes for patch merger
        return feats, image_sizes

    def process_all_mm_data(self, mm_batch_input, model):
        """Process all multimodal data in the given batch."""
        batch_size = len(mm_batch_input)
        if batch_size == 0:
            return {}
        
        all_img_paths = []
        batch_idx_to_img_idx_mapping = []
        
        # Collect all unique image paths
        for batch_idx, mm_input in enumerate(mm_batch_input):
            mm_data = mm_input.get_mm_data(None)
            img_paths = []
            for m in mm_data:
                if m.modality == Modality.IMAGE:
                    img_paths.append(m.data)
            all_img_paths.extend(img_paths)
            batch_idx_to_img_idx_mapping.append({
                f"image_{i}": len(all_img_paths) - len(img_paths) + i
                for i in range(len(img_paths))
            })
        
        # Extract image features using vision tower
        model_device = next(model.parameters()).device
        try:
            all_image_features, image_sizes = self._precompute_image_features(
                model.vision_encoder, all_img_paths, model_device
            )
        except Exception as e:
            raise ValueError(f"Error processing images: {e}")
        
        # Apply patch merger if configured
        if model.patch_merger and model.patch_merger.spatial_merge_size > 1:
            # Flatten and concatenate all features for efficient batch processing
            all_feats_flat = torch.cat([f.reshape(-1, f.shape[-1]) for f in all_image_features])
            merged_feats = model.patch_merger(all_feats_flat, image_sizes)
            
            # Adjust image sizes for merged patches
            merge_size = model.patch_merger.spatial_merge_size
            image_sizes = [
                ((h + merge_size - 1) // merge_size, (w + merge_size - 1) // merge_size)
                for h, w in image_sizes
            ]
        else:
            # No merging, just flatten the features
            merged_feats = torch.cat([f.reshape(-1, f.shape[-1]) for f in all_image_features])
            
        # Process through VL adapter if available
        if model.vision_proj_adapter is not None:
            merged_feats = model.vision_proj_adapter(merged_feats)
            
        # Create a dictionary of features indexed by image identifier
        return {
            "all_image_features": merged_feats,
            "image_sizes": image_sizes,
            "batch_idx_to_img_idx_mapping": batch_idx_to_img_idx_mapping,
        }

    def embed_mm_batch(self, batch, hidden_states, multimodal_features):
        """Embed multimodal data into the hidden states."""
        return general_mm_embed_routine(
            batch=batch,
            hidden_states=hidden_states,
            multimodal_features=multimodal_features,
            mm_processing_info=self.mm_processing_info,
            image_token_id=self.image_token_id,
        )


class PixtralForConditionalGeneration:
    """Main class for the Pixtral multimodal model."""
    
    def __init__(self, *, vllm_config, prefix: str = ""):
        self.prefix = prefix
        self.config = vllm_config.config
        self.weight_map = vllm_config.weight_map
        self.mapping = vllm_config.mapping
        self.quant_config = vllm_config.quant_config
        self.device = vllm_config.device
        
        # Load the base LLM
        self.llm = MistralForCausalLM(
            vllm_config=vllm_config,
            prefix=add_prefix(prefix, "language_model")
        )
        
        self._init_vision_encoder()
        
        # Initialize multimodal processor
        self.mm_process = PixtralMultiModalProcessor(
            config=self.config,
            vision_encoder_args=self.vision_encoder_args,
        )
    
    def _init_vision_encoder(self):
        """Initialize the vision encoder components."""
        # Try to load vision config or fallback to default parameters
        try:
            vis_config = AutoConfig.from_pretrained(
                self.config._name_or_path, subfolder="vision_config"
            )
            assert isinstance(vis_config, PixtralVisionConfig)
        except Exception:
            # Fallback to default values
            vision_width = 1536
            vision_layers = 6
            vision_heads = 16
            image_size = 336
            patch_size = 14
            mm_projector_id = ""
            adapter_bias = True
            spatial_merge_size = 1
            # Set theta for rotary embeddings
            rope_theta = 10000
        else:
            vision_width = vis_config.hidden_size
            vision_layers = vis_config.num_hidden_layers
            vision_heads = vis_config.num_attention_heads
            image_size = vis_config.image_size
            patch_size = vis_config.patch_size
            mm_projector_id = vis_config.mm_projector_id or ""
            adapter_bias = True
            spatial_merge_size = vis_config.patch_merger_params.get(
                "spatial_merge_size", 1
            ) if hasattr(vis_config, "patch_merger_params") else 1
            rope_theta = vis_config.get("rope_theta", 10000)
        
        # Set tokenizer
        image_token_id = LlavaBaseForCausalLM.get_image_token_id(
            self.config._name_or_path,
            self.config,
            None,  # tokenizer is not needed here
            subfolder="tokenizer",
        )
        
        # Configure vision encoder parameters
        self.vision_encoder_args = VisionEncoderArgs(
            hidden_size=vision_width,
            num_channels=3,
            image_size=image_size,
            patch_size=patch_size,
            intermediate_size=vision_width * 4,  # typical ratio
            num_hidden_layers=vision_layers,
            num_attention_heads=vision_heads,
            rope_theta=rope_theta,
            image_token_id=image_token_id,
            adapter_bias=adapter_bias,
            spatial_merge_size=spatial_merge_size,
            mm_projector_id=mm_projector_id,
        )
        
        # Initialize vision component
        self.vision_encoder = VisionTransformer(self.vision_encoder_args)
        
        # Patch merger for efficient processing
        spatial_merge_size = self.vision_encoder_args.spatial_merge_size
        if spatial_merge_size > 1:
            self.patch_merger = PatchMerger(
                vision_encoder_dim=self.vision_encoder_args.hidden_size,
                spatial_merge_size=spatial_merge_size,
                use_mlp_bias=adapter_bias,
            )
        else:
            self.patch_merger = None
        
        # Vision-language adapter
        hidden_size = self.llm.config.hidden_size
        self.vision_proj_adapter = VisionLanguageAdapter(
            self.vision_encoder_args, 
            hidden_size
        )
        
        # Load weights
        vision_dir = add_prefix(prefix, "vision_model")
        self._load_weights(
            self.vision_encoder,
            vision_dir,
            vllm=False,
        )
        
        # Load patch merger weights if needed
        if self.patch_merger is not None:
            patch_merger_dir = add_prefix(prefix, PATCH_MERGE)
            self._load_weights(
                self.patch_merger,
                patch_merger_dir,
                vllm=False,
            )
        
        # Load adapter weights
        mm_projector_dir = add_prefix(prefix, "mm_projector")
        self._load_weights(
            self.vision_proj_adapter,
            mm_projector_dir,
            vllm=False,
        )
    
    def _load_weights(self, model, prefix, vllm=True):
        """Load weights for a model component."""
        if vllm:
            # vLLM models load weights in a special way
            model.load_weights(self.weight_map, self.quant_config, prefix)
        else:
            # Standard PyTorch models
            default_weight_loader(
                model, self.weight_map, QuantizationConfig(), prefix
            )
    
    def embed_text_batch(self, batch: ForwardBatch):
        """Embed text inputs."""
        return self.llm.embed_text_batch(batch)

    def decode_hidden(self, hidden_states, multimodal_features):
        """Decode hidden states to logits."""
        return self.llm.decode_hidden(hidden_states, multimodal_features)

    def forward_batch(self, batch: ForwardBatch):
        """Forward pass for a batch of inputs."""
        if batch.multimodal_inputs:
            # Process multimodal data
            multimodal_features = self.mm_process.process_all_mm_data(
                batch.multimodal_inputs, self
            )
            # Get hidden states from text embeddings
            hidden_states = self.llm.embed_text_batch(batch)
            # Embed multimodal data into hidden states
            hidden_states = self.mm_process.embed_mm_batch(
                batch, hidden_states, multimodal_features
            )
            # Decode hidden states to get logits
            return self.llm.decode_hidden(hidden_states, multimodal_features)
        else:
            # Only text inputs, delegate to language model
            return self.llm.forward_batch(batch)

EntityClass = [PixtralForConditionalGeneration]
