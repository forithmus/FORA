import argparse
from transformers import BertTokenizer, BertModel
from rad_rate import RADRATE
from rad_rate_trainer import RadRateTrainer
from vision_encoder import VJEPA2Encoder
import torch
import torch.nn as nn

# --- Helper to convert BatchNorm to SyncBatchNorm ---
def convert_bn_to_syncbn(module):
    """Convert all BatchNorm layers to SyncBatchNorm for proper distributed training."""
    return nn.SyncBatchNorm.convert_sync_batchnorm(module)

# Parse command line arguments
parser = argparse.ArgumentParser(description='RAD-RATE Training with different fusion modes and pooling strategies')
parser.add_argument('--fusion_mode', type=str, required=True,
                    choices=['early', 'mid_cnn', 'late', 'late_attn'],
                    help='Fusion strategy: early, mid_cnn, late, or late_attn')
parser.add_argument('--pooling_strategy', type=str, default='simple_attn',
                    choices=['simple_attn', 'cross_attn', 'gated'],
                    help='Pooling strategy for late_attn mode: simple_attn, cross_attn, or gated')
args = parser.parse_args()

FUSION_MODE = args.fusion_mode
POOLING_STRATEGY = args.pooling_strategy

print(f"--- Configuration ---")
print(f"Fusion Mode: {FUSION_MODE}")
print(f"Pooling Strategy: {POOLING_STRATEGY}")

print("\n--- Initializing Tokenizer and Text Encoder ---")
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

print(f"\n--- Initializing VJEPA2 Image Encoder for Strategy: {FUSION_MODE} ---")
image_encoder = VJEPA2Encoder(
    input_channels=(3 if FUSION_MODE == "early" else 1),
    freeze_backbone=True,
    use_lora=True,
    lora_r=32,
    lora_alpha=64
)

print(f"\n--- Initializing RAD-RATE Model ---")
clip = RADRATE(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    dim_text=768,
    dim_image=image_encoder.output_dim,
    dim_latent=512,
    fusion_mode=FUSION_MODE,
    pooling_strategy=POOLING_STRATEGY,
    use_gradient_checkpointing=True
)

# Convert all BatchNorm layers to SyncBatchNorm for proper distributed training
clip = convert_bn_to_syncbn(clip)
print("Converted BatchNorm layers to SyncBatchNorm") 
print("\n--- Initializing Trainer ---")
clip.load("/iopsstor/scratch/cscs/ihamamci/ct_clip/ct_clip_windows_full_data/scripts/nocosine_syncbatchnorm_results_late_simple_attn_alldata_10fixed/CTClip.3500.pt")
trainer = RadRateTrainer(
    clip,
    # CT-RATE dataset paths
    data_train="/iopsstor/scratch/cscs/ihamamci/data_ctrate/train_fixed/train_fixed/",
    reports_file_train="/iopsstor/scratch/cscs/ihamamci/ct_clip/reports_processed_final_merged.jsonl",
    train_meta_file="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/metadata/train_metadata.csv",
    # Dataset inclusion (enable all 5 sources)
    include_ctrate=True,
    include_merlin=True,
    include_mimic=True,
    include_chexpert=True,
    include_padchest=True,
    # Training parameters
    batch_size=1,  # Local batch size 1 with SyncBatchNorm for distributed training
    num_train_steps=100001,
    lr=1e-5,
    warmup_steps=500,
    save_model_every=500,
    results_folder=f"nocosine_syncbatchnorm_results_{FUSION_MODE}_{POOLING_STRATEGY}_alldata_11fixed"
)


print("\n--- Starting Training ---")
trainer.train()