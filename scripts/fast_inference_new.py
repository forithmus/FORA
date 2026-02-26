# fast_inference_multireconstruction.py
import os
import torch
import gc
from pathlib import Path
import numpy as np
import tqdm
import pandas as pd
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

# Import your custom modules
from transformers import BertTokenizer, BertModel
from eval import evaluate_internal
from data_inference_nii_fixed import CTReportDatasetinfer
from rad_rate import RADRATE
from vision_encoder import VJEPA2Encoder

# ==============================================================================
# OPTIMIZATION SETTINGS
# ==============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def exists(val):
    return val is not None


def noop(*args, **kwargs):
    pass


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


class RadRateInferenceMultiReconstruction(nn.Module):
    def __init__(
            self,
            model: RADRATE,
            *,
            data_folder,
            reports_file,
            meta_file,
            results_folder='./results',
            labels="labels.csv",
            fusion_mode="mid_cnn",
            pooling_strategy="simple_attn",
            accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        # Initialize Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)

        self.model = model
        self.fusion_mode = fusion_mode
        self.pooling_strategy = pooling_strategy
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.result_folder_txt = str(self.results_folder) + "/"

        self.register_buffer('steps', torch.Tensor([0]))

        # Initialize Dataset
        self.ds = CTReportDatasetinfer(
            data_folder=data_folder,
            reports_file=reports_file,
            meta_file=meta_file,
            labels=labels
        )

        self.device = self.accelerator.device
        self.model.to(self.device)
        self.model.eval()

        # OPTIMIZATION: Compile the visual transformer if PyTorch 2.0+
        if hasattr(torch, "compile"):
            print("Compiling visual transformer for faster inference...")
            try:
                self.model.visual_transformer = torch.compile(self.model.visual_transformer)
            except Exception as e:
                print(f"Compilation failed (safe to ignore): {e}")

    def _encode_text_latents(self, prompts, device, use_extra_proj=False):
        """
        Encode text prompts to latent representations.
        Returns: [N, dim_latent] normalized tensor
        """
        # Tokenize
        tokenized = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.model.text_seq_len
        ).to(device)

        # Get text embeddings
        text_output = self.model.text_transformer(
            tokenized.input_ids,
            attention_mask=tokenized.attention_mask
        )
        enc_text = text_output.last_hidden_state  # [N, seq_len, dim_text]

        # Extract CLS token
        text_cls = enc_text[:, 0, :]  # [N, dim_text]

        # Project to latent space
        text_proj_layer = self.model.to_text_latent_extra if use_extra_proj else self.model.to_text_latent
        text_latents = text_proj_layer(text_cls)  # [N, dim_latent]

        return l2norm(text_latents)

    def _encode_visual_tokens_with_attention(self, images, real_volume_mask, text_latents=None, use_extra_proj=False):
        """
        Encode multi-reconstruction images to visual tokens with attention-based pooling.

        Args:
            images: [B, R, 1, D, H, W] where R is number of reconstructions
            real_volume_mask: [B, R] boolean mask for valid reconstructions
            text_latents: [B, D] optional text latents for cross-attention pooling
            use_extra_proj: whether to use extra projection layer

        Returns:
            visual_tokens: [B, num_tokens, dim_latent] normalized tokens
        """
        b, r, c, d, h, w = images.shape

        # Determine projection layer
        vis_proj_layer = self.model.to_visual_latent_extra if use_extra_proj else self.model.to_visual_latent

        if self.fusion_mode == "early":
            # Stack volumes into channels [B, R, D, H, W]
            img_in = images.squeeze(2)
            enc = self.model.visual_transformer(img_in)
            visual_tokens = vis_proj_layer(enc)

        elif self.fusion_mode == "mid_cnn":
            # CNN separate -> Merge -> Transformer
            flat_img = rearrange(images, 'b r c d h w -> (b r) c d h w')
            cnn_features = self.model.visual_transformer.forward_cnn(flat_img)
            cnn_features = rearrange(cnn_features, '(b r) t h w d -> b r t h w d', r=r)

            # Masked merge
            m = real_volume_mask.view(b, r, 1, 1, 1, 1).to(cnn_features.dtype)
            merged_features = (cnn_features * m).sum(1) / m.sum(1).clamp(min=1.0)

            # Transformer pass
            enc = self.model.visual_transformer.forward_transformer(merged_features)
            visual_tokens = vis_proj_layer(enc)

        elif self.fusion_mode == "late":
            # Siamese pass - process volumes separately and merge tokens
            all_tokens_list = []

            for i in range(r):
                single_vol = images[:, i]
                enc_single = self.model.visual_transformer(single_vol)
                vis_tokens = vis_proj_layer(enc_single)
                all_tokens_list.append(vis_tokens)

            # Stack and merge: [B, R, num_tokens, D]
            all_tokens = torch.stack(all_tokens_list, dim=1)

            # Masked average
            m = real_volume_mask.view(b, r, 1, 1).to(all_tokens.dtype)
            visual_tokens = (all_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

        elif self.fusion_mode == "late_attn":
            # Siamese pass with attention-based reconstruction fusion
            all_tokens_list = []

            for i in range(r):
                single_vol = images[:, i]
                enc_single = self.model.visual_transformer(single_vol)
                vis_tokens = vis_proj_layer(enc_single)
                all_tokens_list.append(vis_tokens)

            # Stack: [B, R, num_tokens, D]
            all_tokens = torch.stack(all_tokens_list, dim=1)

            # Use attention-based pooling to merge reconstructions
            if self.pooling_strategy == "simple_attn":
                visual_tokens = self.model.recon_pool(all_tokens, mask=real_volume_mask)

            elif self.pooling_strategy == "cross_attn":
                if text_latents is not None:
                    # text_latents should be [B, D] for cross-attention
                    visual_tokens = self.model.recon_pool(all_tokens, text_latents, mask=real_volume_mask)
                else:
                    # Fallback to masked average
                    m = real_volume_mask.view(b, r, 1, 1).to(all_tokens.dtype)
                    visual_tokens = (all_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

            elif self.pooling_strategy == "gated":
                if text_latents is not None:
                    visual_tokens = self.model.recon_pool(all_tokens, text_latents, mask=real_volume_mask)
                else:
                    # Fallback to masked average
                    m = real_volume_mask.view(b, r, 1, 1).to(all_tokens.dtype)
                    visual_tokens = (all_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

            else:
                # Default: masked average
                m = real_volume_mask.view(b, r, 1, 1).to(all_tokens.dtype)
                visual_tokens = (all_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        return l2norm(visual_tokens)

    def _text_guided_pool(self, visual_tokens, text_query_latent):
        """Pool visual tokens guided by text query."""
        if text_query_latent is None:
            return visual_tokens.mean(dim=1)
        text_norm = l2norm(text_query_latent)
        visual_norm = l2norm(visual_tokens)
        sim = einsum('b d, b n d -> b n', text_norm, visual_norm)
        attn = F.softmax(sim, dim=-1)
        return einsum('b n, b n d -> b d', attn, visual_tokens)

    def infer(self, batch_size=1):
        """
        Run inference for pathology classification.

        Args:
            batch_size: batch size for inference
        """
        device = self.device

        # 1. Define Pathologies
        pathologies = [
            'Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
            'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',
            'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
            'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening',
            'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening'
        ]

        # 2. Pre-compute Text Embeddings
        print("Pre-computing text embeddings for pathology prompts...")

        # Structure: [Pos1, Neg1, Pos2, Neg2, ...]
        prompts = []
        for p in pathologies:
            prompts.append(f"There is {p}.")
            prompts.append(f"There is no {p}.")

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                # Encode text to latent space [36, dim_latent]
                text_latents = self._encode_text_latents(prompts, device=device, use_extra_proj=False)
                text_latents = text_latents.to(dtype=torch.bfloat16)

                # Pre-fetch temperature
                logit_temp = self.model.logit_temperature.exp().clamp(max=100.0)

        # 3. Inference Loop
        print(f"Starting batched inference (Batch Size: {batch_size}, Fusion Mode: {self.fusion_mode}, Pooling: {self.pooling_strategy})...")

        eval_loader = DataLoader(
            self.ds,
            batch_size=batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

        all_predicted_scores = []
        all_real_labels = []
        all_accessions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(eval_loader)):
                # Unpack batch - CTReportDatasetinfer returns 5 items:
                # (images, texts, labels, names, real_volume_mask)
                imgs, texts, onehot_labels, acc_names, real_volume_mask = batch

                # imgs: [B, R, C, D, H, W]
                # real_volume_mask: [B, R]
                real_volume_mask = real_volume_mask.to(device)

                # Move to GPU in bfloat16
                imgs = imgs.to(device, dtype=torch.bfloat16)

                with autocast(dtype=torch.bfloat16):
                    # --- A. Encode Visual Tokens with Attention Pooling ---
                    # For cross_attn and gated, we need a text query for reconstruction pooling
                    # Use mean of positive prompts as a generic query
                    if self.pooling_strategy in ["cross_attn", "gated"]:
                        # Use positive prompt embeddings as query (indices 0, 2, 4, ...)
                        positive_indices = list(range(0, len(prompts), 2))
                        mean_text_query = text_latents[positive_indices].mean(dim=0, keepdim=True)
                        mean_text_query = mean_text_query.expand(imgs.shape[0], -1)  # [B, D]

                        visual_tokens = self._encode_visual_tokens_with_attention(
                            imgs, real_volume_mask,
                            text_latents=mean_text_query,
                            use_extra_proj=False
                        )
                    else:
                        visual_tokens = self._encode_visual_tokens_with_attention(
                            imgs, real_volume_mask,
                            text_latents=None,
                            use_extra_proj=False
                        )

                    # --- B. Text-Guided Pooling for Each Pathology Query ---
                    # visual_tokens: [B, N, D]
                    # text_latents: [P, D] where P=36

                    B, N, D = visual_tokens.shape
                    P = text_latents.shape[0]

                    # Compute attention scores: [B, N, P]
                    sim_scores = torch.einsum('b n d, p d -> b n p', visual_tokens, text_latents)

                    # Softmax to get attention weights
                    attn_weights = F.softmax(sim_scores, dim=1)  # [B, N, P]

                    # Weighted sum pooling for each query: [B, P, D]
                    pooled_visual = torch.einsum('b n p, b n d -> b p d', attn_weights, visual_tokens)
                    pooled_visual = l2norm(pooled_visual)

                    # --- C. Compute Final Similarity Scores ---
                    # [B, P] - dot product between pooled visual and text
                    final_sim = torch.einsum('b p d, p d -> b p', pooled_visual, text_latents)
                    final_sim = final_sim * logit_temp

                    # --- D. Calculate Deltas (Yes - No) ---
                    # Reshape to [B, 18, 2] where dim 2 is (Yes, No)
                    sim_reshaped = final_sim.view(-1, 18, 2)

                    # Score = Logit(Yes) - Logit(No)
                    final_scores = sim_reshaped[:, :, 0] - sim_reshaped[:, :, 1]

                    all_predicted_scores.append(final_scores.float().cpu().numpy())
                    all_real_labels.append(onehot_labels.numpy())
                    print(all_predicted_scores[-1])
                    print(all_real_labels[-1])
                    print(f"\nPredicted scores shape: {all_predicted_scores[-1].shape}")
                    print(f"Labels shape: {all_real_labels[-1].shape}")
                    all_accessions.extend(acc_names)

                # Memory cleanup
                del imgs, visual_tokens, pooled_visual, final_sim

        # 4. Save Results
        final_preds = np.concatenate(all_predicted_scores, axis=0)
        final_labels = np.concatenate(all_real_labels, axis=0)

        plotdir = self.result_folder_txt
        print(f"\nSaving results to {plotdir}...")

        np.savez(f"{plotdir}predicted_weights.npz", data=final_preds)
        np.savez(f"{plotdir}labels_weights.npz", data=final_labels)

        with open(f"{plotdir}accessions.txt", "w") as file:
            for item in all_accessions:
                file.write(item + "\n")

        print("Running Evaluation (AUROC)...")
        dfs = evaluate_internal(final_preds, final_labels, pathologies, plotdir)

        writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')
        dfs.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.close()
        print('Inference complete.')

        return {
            'predictions': final_preds,
            'labels': final_labels,
            'accessions': all_accessions,
            'aurocs': dfs
        }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RAD-RATE Multi-Reconstruction Inference')
    parser.add_argument('--fusion_mode', type=str, required=True,
                        choices=['early', 'mid_cnn', 'late', 'late_attn'],
                        help='Fusion strategy used during training')
    parser.add_argument('--pooling_strategy', type=str, default='simple_attn',
                        choices=['simple_attn', 'cross_attn', 'gated'],
                        help='Pooling strategy for late_attn fusion mode')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--results_folder', type=str, default='./inference_results',
                        help='Folder to save results')

    # Data paths (with defaults)
    parser.add_argument('--data_folder', type=str,
                        default='/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/valid_fixed/dataset/valid_fixed/',
                        help='Path to CT scan data folder')
    parser.add_argument('--reports_file', type=str,
                        default='/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/radiology_text_reports/validation_reports.csv',
                        help='Path to radiology reports CSV')
    parser.add_argument('--meta_file', type=str,
                        default='/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/metadata/validation_metadata.csv',
                        help='Path to metadata CSV')
    parser.add_argument('--labels_file', type=str,
                        default='/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/valid_labels.csv',
                        help='Path to labels CSV')

    args = parser.parse_args()

    # Initialize components
    print(f"\n{'='*60}")
    print(f"RAD-RATE Multi-Reconstruction Inference")
    print(f"{'='*60}")
    print(f"Fusion Mode: {args.fusion_mode}")
    print(f"Pooling Strategy: {args.pooling_strategy}")
    print(f"{'='*60}\n")

    print(f"--- Initializing VJEPA2 Image Encoder ---")
    image_encoder = VJEPA2Encoder(
        model_name="facebook/vjepa2-vitg-fpc64-384",
        input_channels=(3 if args.fusion_mode == "early" else 1),
        freeze_backbone=True,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
    )

    print(f"\n--- Initializing RAD-RATE Model ---")
    clip = RADRATE(
        image_encoder=image_encoder,
        dim_image=image_encoder.output_dim,
        dim_text=768,
        dim_latent=512,
        fusion_mode=args.fusion_mode,
        pooling_strategy=args.pooling_strategy,
        use_gradient_checkpointing=False  # Disable for inference
    ).cuda()

    print(f"Loading Weights: {args.weights_path}")
    clip.load(args.weights_path)

    # Merge LoRA for speed
    print("Merging LoRA weights to speed up inference...")
    try:
        if hasattr(image_encoder, "model") and hasattr(image_encoder.model, "merge_and_unload"):
            image_encoder.model.merge_and_unload()
            print("LoRA merged successfully.")
    except Exception as e:
        print(f"LoRA merge skipped: {e}")

    # Convert to bfloat16 for inference speed
    print("Converting model to bfloat16...")
    clip.to(torch.bfloat16)

    inference = RadRateInferenceMultiReconstruction(
        clip,
        data_folder=args.data_folder,
        reports_file=args.reports_file,
        meta_file=args.meta_file,
        labels=args.labels_file,
        results_folder=args.results_folder,
        fusion_mode=args.fusion_mode,
        pooling_strategy=args.pooling_strategy
    )

    # Run inference
    results = inference.infer(batch_size=args.batch_size)

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.results_folder}")