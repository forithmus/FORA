"""
Visual encoder pipeline and LLM builder for SFT report generation.

Pipeline:
    MultiWindowEncoder  (frozen RADRATE + trainable per-window poolers)
        -> PerceiverResampler  (token compression)
        -> VisualProjection    (project to LLM dim + start/end tokens)

The LLM is Qwen3.5-9B with LoRA.
"""

import os
import sys
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from perceiver_resampler import PerceiverResampler

logger = logging.getLogger(__name__)


# =========================================================================
# WindowAttentionPooler
# =========================================================================

class WindowAttentionPooler(nn.Module):
    """Cross-attention pooling for a single window with learnable queries."""

    def __init__(
        self,
        input_dim: int = 1408,
        hidden_dim: int = 1408,
        num_queries: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """[B, N, input_dim] -> [B, num_queries, hidden_dim]"""
        B = visual_tokens.shape[0]
        layer_dtype = self.layers[0].linear1.weight.dtype
        visual_tokens = visual_tokens.to(layer_dtype)

        visual_tokens = self.input_proj(visual_tokens)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1).to(layer_dtype)

        for layer in self.layers:
            queries = layer(queries, visual_tokens)

        return self.final_norm(queries)


# =========================================================================
# MultiWindowEncoder
# =========================================================================

class MultiWindowEncoder(nn.Module):
    """Encodes 4 CT windows via frozen RADRATE + trainable per-window poolers.

    CT-RATE:  [full, mediastinal, lung, bone]  mask=[1,1,1,1]
    Merlin:   [full, soft_tissue, liver, bone]  mask=[1,1,1,1]
    X-rays:   [view, pad, pad, pad]             mask=[1,0,0,0]
    """

    def __init__(
        self,
        radrate_model: nn.Module,
        num_windows: int = 4,
        radrate_dim: int = 1408,
        per_window_num_queries: int = 256,
        per_window_num_layers: int = 2,
        per_window_num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.radrate = radrate_model
        self.num_windows = num_windows
        self.radrate_dim = radrate_dim

        # Freeze RADRATE (always)
        for param in self.radrate.parameters():
            param.requires_grad = False
        self.radrate.eval()

        self.window_poolers = nn.ModuleList([
            WindowAttentionPooler(
                input_dim=radrate_dim,
                hidden_dim=radrate_dim,
                num_queries=per_window_num_queries,
                num_layers=per_window_num_layers,
                num_heads=per_window_num_heads,
                dropout=dropout,
            )
            for _ in range(num_windows)
        ])

        self.window_type_embeddings = nn.Parameter(
            torch.randn(num_windows, 1, radrate_dim) * 0.02
        )

    def forward(
        self,
        volume: torch.Tensor,
        real_volume_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            volume: [B, num_windows, C, D, H, W]
            real_volume_mask: [B, num_windows] bool — True for real, False for padded

        Returns:
            [B, num_windows * per_window_queries, radrate_dim]
        """
        B = volume.shape[0]
        num_windows = volume.shape[1]

        if real_volume_mask is None:
            real_volume_mask = torch.ones(B, num_windows, dtype=torch.bool, device=volume.device)

        pooled_tokens: List[torch.Tensor] = []

        with torch.no_grad():
            for w in range(num_windows):
                single_window = volume[:, w]  # [B, C, D, H, W]
                window_tokens = self.radrate.visual_transformer(
                    single_window, return_encoded_tokens=True
                )
                if window_tokens.ndim == 5:
                    window_tokens = rearrange(window_tokens, 'b t h w d -> b (t h w) d')
                pooled = self.window_poolers[w](window_tokens)
                pooled = pooled + self.window_type_embeddings[w]

                # Zero out padded windows
                mask_w = real_volume_mask[:, w].view(B, 1, 1).to(pooled.dtype)
                pooled = pooled * mask_w

                pooled_tokens.append(pooled)

        return torch.cat(pooled_tokens, dim=1)


# =========================================================================
# VisualProjection
# =========================================================================

class VisualProjection(nn.Module):
    """Projects visual tokens to LLM dim with learned start/end separators."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.visual_type_embedding = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.img_start_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        self.img_end_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """[B, N, input_dim] -> [B, N+2, output_dim]"""
        B = visual_tokens.shape[0]
        projected = self.projection(visual_tokens) + self.visual_type_embedding
        start = self.img_start_token.expand(B, -1, -1)
        end = self.img_end_token.expand(B, -1, -1)
        return torch.cat([start, projected, end], dim=1)


# =========================================================================
# VisualEncoder (full pipeline)
# =========================================================================

class VisualEncoder(nn.Module):
    """MultiWindowEncoder -> PerceiverResampler -> VisualProjection

    Input:  [B, 4, 1, D, H, W] + real_volume_mask [B, 4]
    Output: [B, num_queries+2, llm_dim]
    """

    def __init__(
        self,
        multi_window_encoder: MultiWindowEncoder,
        perceiver: PerceiverResampler,
        visual_projection: VisualProjection,
    ):
        super().__init__()
        self.multi_window_encoder = multi_window_encoder
        self.perceiver = perceiver
        self.visual_projection = visual_projection

    def forward(
        self,
        volume: torch.Tensor,
        real_volume_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        concatenated = self.multi_window_encoder(volume, real_volume_mask)
        compressed = self.perceiver(concatenated)
        return self.visual_projection(compressed)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_trainable(self):
        """Unfreeze trainable parts (poolers, perceiver, projection) while
        keeping RADRATE frozen."""
        self.freeze()
        for param in self.multi_window_encoder.window_poolers.parameters():
            param.requires_grad = True
        self.multi_window_encoder.window_type_embeddings.requires_grad = True
        for param in self.perceiver.parameters():
            param.requires_grad = True
        for param in self.visual_projection.parameters():
            param.requires_grad = True


# =========================================================================
# Factory: build_visual_encoder
# =========================================================================

def build_visual_encoder(config: dict, device: torch.device) -> VisualEncoder:
    """Build the full visual encoder from config + RADRATE checkpoint."""

    # Add FORA source paths
    fora_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for subdir in ["rad_rate", "vision_encoder", "scripts"]:
        path = os.path.join(fora_root, subdir)
        if path not in sys.path:
            sys.path.insert(0, path)

    from vision_encoder import VJEPA2Encoder
    from transformers import BertModel
    from rad_rate import RADRATE

    model_cfg = config["model"]

    # 1. Build RADRATE
    logger.info("Building VJEPA2 encoder...")
    image_encoder = VJEPA2Encoder(
        input_channels=1,
        freeze_backbone=True,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
    )

    logger.info("Building text encoder...")
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    logger.info("Building RADRATE...")
    radrate = RADRATE(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_text=768,
        dim_image=image_encoder.output_dim,
        dim_latent=512,
        fusion_mode="late",
        use_gradient_checkpointing=True,
    )

    # 2. Load RADRATE checkpoint
    radrate_ckpt = model_cfg.get("radrate_checkpoint")
    if radrate_ckpt and os.path.exists(radrate_ckpt):
        logger.info("Loading RADRATE checkpoint: %s", radrate_ckpt)
        pt = torch.load(radrate_ckpt, map_location="cpu")
        clean_state = {}
        for k, v in pt.items():
            if k.startswith("module."):
                clean_state[k[len("module."):]] = v
            else:
                clean_state[k] = v
        radrate.load_state_dict(clean_state, strict=False)
        del pt, clean_state
    else:
        logger.warning("No RADRATE checkpoint — using random init for visual backbone.")

    # 3. Merge VJEPA LoRA into base (so we can freeze cleanly)
    if hasattr(radrate.visual_transformer, "merge_and_unload"):
        logger.info("Merging VJEPA LoRA into base weights...")
        radrate.visual_transformer.merge_and_unload()

    # 4. Build components
    radrate_dim = model_cfg.get("radrate_dim", 1408)
    num_windows = model_cfg.get("num_windows", 4)
    per_window_num_queries = model_cfg.get("per_window_num_queries", 256)
    per_window_num_layers = model_cfg.get("per_window_num_layers", 2)
    perceiver_num_heads = model_cfg.get("perceiver_num_heads", 16)
    perceiver_dropout = model_cfg.get("perceiver_dropout", 0.1)
    perceiver_hidden_dim = model_cfg.get("perceiver_hidden_dim", 2048)
    perceiver_num_queries = model_cfg.get("perceiver_num_queries", 512)
    perceiver_num_layers = model_cfg.get("perceiver_num_layers", 6)
    llm_hidden_dim = model_cfg.get("llm_hidden_dim", 4096)

    multi_window_encoder = MultiWindowEncoder(
        radrate_model=radrate,
        num_windows=num_windows,
        radrate_dim=radrate_dim,
        per_window_num_queries=per_window_num_queries,
        per_window_num_layers=per_window_num_layers,
        per_window_num_heads=perceiver_num_heads,
        dropout=perceiver_dropout,
    )

    perceiver = PerceiverResampler(
        input_dim=radrate_dim,
        hidden_dim=perceiver_hidden_dim,
        output_dim=llm_hidden_dim,
        num_queries=perceiver_num_queries,
        num_layers=perceiver_num_layers,
        num_heads=perceiver_num_heads,
        dropout=perceiver_dropout,
    )

    visual_projection = VisualProjection(
        input_dim=llm_hidden_dim,
        output_dim=llm_hidden_dim,
    )

    visual_encoder = VisualEncoder(multi_window_encoder, perceiver, visual_projection)
    visual_encoder = visual_encoder.to(device)
    return visual_encoder


# =========================================================================
# Factory: build_llm_with_lora
# =========================================================================

def build_llm_with_lora(
    config: dict,
    device: torch.device,
) -> Tuple[PeftModel, AutoTokenizer]:
    """Build Qwen3.5-9B with LoRA for SFT training.

    If a pretrained SFT checkpoint is provided, its LoRA weights are merged
    into the base model first, then a fresh LoRA adapter is added.
    """
    model_cfg = config["model"]
    llm_model_name = model_cfg.get("llm_model_name", "Qwen/Qwen3.5-9B")

    # 1. Load base LLM
    logger.info("Loading base LLM: %s", llm_model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        trust_remote_code=True,
        padding_side="right",  # right-padding for causal LM training
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Optionally merge pretrained LoRA
    pretrained_path = model_cfg.get("pretrained_checkpoint")
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info("Loading pretrained SFT checkpoint: %s", pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)

        llm_state = {}
        for key, value in state.items():
            if key.startswith("llm."):
                llm_state[key[len("llm."):]] = value

        if llm_state:
            lora_keys = [k for k in llm_state if "lora" in k.lower()]
            if lora_keys:
                lora_target = model_cfg.get("lora_target_modules") or [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]
                tmp_config = LoraConfig(
                    r=model_cfg.get("lora_r", 64),
                    lora_alpha=model_cfg.get("lora_alpha", 128),
                    target_modules=lora_target,
                    lora_dropout=model_cfg.get("lora_dropout", 0.1),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                llm = get_peft_model(llm, tmp_config, adapter_name="policy")
                llm.load_state_dict(llm_state, strict=False)
                logger.info("Merging pretrained LoRA into base...")
                llm = llm.merge_and_unload()
            else:
                llm.load_state_dict(llm_state, strict=False)
        del checkpoint, state

    # 4. Freeze base
    for param in llm.parameters():
        param.requires_grad = False

    # 5. Add fresh LoRA
    lora_r = model_cfg.get("lora_r", 64)
    lora_alpha = model_cfg.get("lora_alpha", 128)
    lora_dropout = model_cfg.get("lora_dropout", 0.1)
    lora_target = model_cfg.get("lora_target_modules") or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(llm, lora_config)
    peft_model = peft_model.to(device)

    total = sum(p.numel() for p in peft_model.parameters())
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    logger.info("LLM: %s total, %s trainable (%.2f%%)",
                f"{total:,}", f"{trainable:,}", 100 * trainable / total)

    return peft_model, tokenizer
