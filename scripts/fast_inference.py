# fast_inference_multireconstruction.py
import os
import torch
import gc
from pathlib import Path
import numpy as np
import tqdm
import pandas as pd
from einops import rearrange
from torch import nn
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
            accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        # Initialize Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)

        self.model = model
        self.fusion_mode = fusion_mode
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
        #return text_latents

    def _encode_visual_tokens(self, images, real_volume_mask, use_extra_proj=False):
        """
        Encode multi-reconstruction images to visual tokens.

        Args:
            images: [B, R, 1, D, H, W] where R is number of reconstructions
            real_volume_mask: [B, R] boolean mask for valid reconstructions
            use_extra_proj: whether to use extra projection layer

        Returns:
            visual_tokens: [B, num_tokens, dim_latent] UN-normalized tokens (matches training)
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
            # Siamese pass - process volumes separately and merge with masked average
            all_tokens_list = []

            for i in range(r):
                single_vol = images[:, i]
                enc_single = self.model.visual_transformer(single_vol)
                vis_tokens = vis_proj_layer(enc_single)
                all_tokens_list.append(vis_tokens)

            # Stack: [B, R, num_tokens, D]
            all_tokens = torch.stack(all_tokens_list, dim=1)

            # Masked average for "late" mode
            m = real_volume_mask.view(b, r, 1, 1).to(all_tokens.dtype)
            visual_tokens = (all_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

        elif self.fusion_mode == "late_attn":
            # Siamese pass with learned attention pooling (matches training)
            all_tokens_list = []

            for i in range(r):
                single_vol = images[:, i]
                enc_single = self.model.visual_transformer(single_vol)
                vis_tokens = vis_proj_layer(enc_single)
                all_tokens_list.append(vis_tokens)

            # Stack: [B, R, num_tokens, D]
            all_tokens = torch.stack(all_tokens_list, dim=1)

            # Use the learned attention pooling module (matches training)
            visual_tokens = self.model.recon_pool(all_tokens, mask=real_volume_mask)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # Return UN-normalized tokens (normalization done separately for attention vs pooling)
        return visual_tokens

    def infer(self, batch_size=1):
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
        print(f"Starting batched inference (Batch Size: {batch_size}, Fusion Mode: {self.fusion_mode})...")

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
            for batch in tqdm.tqdm(eval_loader):
                # Unpack batch - CTReportDatasetinfer returns 5 items:
                # (images, texts, labels, names, real_volume_mask)
                imgs, texts, onehot_labels, acc_names, real_volume_mask = batch

                # imgs: [B, R, C, D, H, W]
                # real_volume_mask: [B, R]
                real_volume_mask = real_volume_mask.to(device)

                # Move to GPU in bfloat16
                imgs = imgs.to(device, dtype=torch.bfloat16)

                with autocast(dtype=torch.bfloat16):
                    # --- A. Encode Visual Tokens (UN-normalized, matches training) ---
                    # [B, num_tokens, dim_latent]
                    visual_tokens = self._encode_visual_tokens(
                        imgs,
                        real_volume_mask,
                        use_extra_proj=False
                    )

                    # --- B. Text-Guided Pooling for Each Pathology Query ---
                    # Matches training: use normalized tokens for attention, un-normalized for pooling
                    # visual_tokens: [B, N, D]
                    # text_latents: [P, D] where P=36

                    B, N, D = visual_tokens.shape
                    P = text_latents.shape[0]

                    # Normalize tokens ONLY for attention score computation (matches training)
                    visual_tokens_norm = l2norm(visual_tokens)

                    # Compute attention scores using NORMALIZED tokens: [B, N, P]
                    sim_scores = torch.einsum('b n d, p d -> b n p', visual_tokens_norm, text_latents)

                    # Softmax to get attention weights
                    attn_weights = F.softmax(sim_scores, dim=1)  # [B, N, P]

                    # Weighted sum pooling using UN-NORMALIZED tokens (matches training!)
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
                    all_accessions.extend(acc_names)

                # Memory cleanup
                del imgs, visual_tokens, visual_tokens_norm, pooled_visual, final_sim

        # 4. Save Results
        final_preds = np.concatenate(all_predicted_scores, axis=0)
        final_labels = np.concatenate(all_real_labels, axis=0)

        plotdir = self.result_folder_txt
        print(f"Saving results to {plotdir}...")

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

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RAD-RATE Multi-Reconstruction Inference')
    parser.add_argument('--fusion_mode', type=str, required=True,
                        choices=['early', 'mid_cnn', 'late', 'late_attn'],
                        help='Fusion strategy used during training')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--results_folder', type=str, default='./inference_results',
                        help='Folder to save results')
    args = parser.parse_args()

    # Initialize components
    print(f"\n--- Initializing VJEPA2 Image Encoder (Fusion: {args.fusion_mode}) ---")
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
        data_folder='/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/valid_fixed/dataset/valid_fixed/',
        reports_file="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
        meta_file="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/metadata/validation_metadata.csv",
        labels="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/valid_labels.csv",
        results_folder=args.results_folder,
        fusion_mode=args.fusion_mode
    )

    # Run inference
    inference.infer(batch_size=args.batch_size)