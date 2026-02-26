# data_inference_nii.py
import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from functools import partial
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from collections import defaultdict
import random


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing using trilinear interpolation.
    """
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


# Window settings: (window_center, window_width) -> (hu_min, hu_max)
WINDOW_SETTINGS = {
    "full": (-1000, 1000),       # Full range: HU min=-1000, max=1000
    "mediastinal": (-150, 350),  # Mediastinal: center=40, width=400 -> (-160, 240), using (-150, 350) for better coverage
    "lung": (-1400, -200),       # Lung: center=-600, width=1500 -> (-1350, 150), using (-1400, -200) for lung tissue
}


class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, reports_file, meta_file, min_slices=20, labels="labels.csv", max_reconstructions=2):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.max_reconstructions = max_reconstructions

        # --- EXCLUSION LOGIC ---
        exclusion_file_path = "no_chest_valid.txt"
        self.exclusion_list = self._load_exclusion_list(exclusion_file_path)
        # -----------------------

        # Load metadata
        print(f"Loading metadata from {meta_file}...")
        self.meta_df = pd.read_csv(meta_file)
        # Helper for grouping: valid_1_a_1.nii.gz -> valid_1_a
        self.meta_df['scan_id'] = self.meta_df['VolumeName'].apply(
            lambda x: x.replace('.nii.gz', '').rsplit('_', 1)[0]
        )

        self.accession_to_text = self.load_accession_text(reports_file)

        self.samples = self.prepare_samples()
        print(f"Found {len(self.samples)} total valid scans (grouped reconstructions).")

    def _load_exclusion_list(self, file_path):
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        print(f"Warning: Exclusion file not found at {file_path}. No images will be excluded.")
        return set()

    def load_accession_text(self, reports_file):
        df = pd.read_csv(reports_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = (row["Findings_EN"], row['Impressions_EN'])
        return accession_to_text

    def prepare_samples(self):
        """Group reconstructions by scan_id, similar to training dataloader."""
        samples = []

        # Load labels
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        # Create a mapping from volume name to labels
        volume_to_labels = {}
        for _, row in test_df.iterrows():
            volume_to_labels[row['VolumeName']] = row['one_hot_labels']

        # Find all nii files
        all_nii_paths = glob.glob(os.path.join(self.data_folder, '**/*.nii.gz'), recursive=True)

        # Group by scan_id
        grouped_scans = defaultdict(list)

        for path in tqdm(all_nii_paths, desc="Preparing samples"):
            relative_path = os.path.relpath(path, self.data_folder)
            relative_path_for_exclusion = "valid/" + str(relative_path)

            if relative_path_for_exclusion in self.exclusion_list:
                continue

            volume_name = os.path.basename(path)

            # Check if we have text for this volume
            if volume_name not in self.accession_to_text:
                continue

            # Check if we have labels for this volume
            if volume_name not in volume_to_labels:
                continue

            # Get scan_id for grouping (e.g., valid_1_a_1.nii.gz -> valid_1_a)
            scan_id = volume_name.replace('.nii.gz', '').rsplit('_', 1)[0]

            # Get text
            impression_text = self.accession_to_text[volume_name]
            text_final = ""
            for text in list(impression_text):
                text = str(text)
                if text == "Not given.":
                    text = ""
                text_final = text_final + text

            # Clean text
            text_final = text_final.replace('"', '').replace("'", '').replace('(', '').replace(')', '')

            grouped_scans[scan_id].append({
                "volume_name": volume_name,
                "image_path": path,
                "text": text_final,
                "labels": volume_to_labels[volume_name]
            })

        # Create samples from grouped scans
        for scan_id, volumes in grouped_scans.items():
            # Use first volume's text and labels (they should be the same for all reconstructions)
            samples.append({
                "scan_id": scan_id,
                "volumes": volumes,
                "text": volumes[0]["text"],
                "labels": volumes[0]["labels"]
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def load_and_resample_nii(self, path):
        """
        Load NIfTI file and resample to target spacing.
        Returns the resampled data in original HU values (not normalized).
        """
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata().astype(np.float32)

        file_name = path.split("/")[-1]
        row = self.meta_df[self.meta_df['VolumeName'] == file_name]
        if row.empty:
            raise RuntimeError(f"Could not find metadata for volume: {file_name}")

        # --- Step 1: Resample to Target Spacing ---
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
        current_spacing = (z_spacing, xy_spacing, xy_spacing)
        target_spacing = (1.5, 0.75, 0.75)

        # Transpose to (Depth, H, W) for processing
        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        resampled_data = resize_array(tensor, current_spacing, target_spacing)[0, 0]  # Back to (D, H, W) numpy

        return resampled_data  # Return raw HU values

    def apply_window_and_normalize(self, resampled_data, hu_min, hu_max):
        """
        Apply windowing and normalize to [-1, 1] range.
        """
        # --- Step 2: HU Clipping and Normalization ---
        clipped_data = np.clip(resampled_data, hu_min, hu_max)
        # Normalize to [-1, 1] range
        normalized_data = ((clipped_data - hu_min) / (hu_max - hu_min)) * 2.0 - 1.0
        tensor = torch.from_numpy(normalized_data.astype(np.float32))

        # --- Step 3: Center Crop / Pad to 480x480x256 ---
        target_shape_intermediate = (256, 480, 480)  # (D, H, W)
        d, h, w = tensor.shape
        td, th, tw = target_shape_intermediate

        # Crop if larger
        d_start = max((d - td) // 2, 0)
        h_start = max((h - th) // 2, 0)
        w_start = max((w - tw) // 2, 0)
        tensor = tensor[d_start:d_start + td, h_start:h_start + th, w_start:w_start + tw]

        # Pad if smaller
        pad_d_before = (td - tensor.size(0)) // 2
        pad_d_after = td - tensor.size(0) - pad_d_before
        pad_h_before = (th - tensor.size(1)) // 2
        pad_h_after = th - tensor.size(1) - pad_h_before
        pad_w_before = (tw - tensor.size(2)) // 2
        pad_w_after = tw - tensor.size(2) - pad_w_before

        # Pad with -1 (normalized minimum value)
        tensor = F.pad(tensor, (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after), value=-1)

        # --- Step 4: Resize to 384x384x256 ---
        tensor = tensor.unsqueeze(1)  # Shape: (D, 1, H, W) -> (256, 1, 480, 480)
        resized_tensor = F.interpolate(tensor, size=(384, 384), mode='bilinear', align_corners=False)
        final_tensor = resized_tensor.squeeze(1)  # Shape: (256, 384, 384)

        # Final shape: [1, D, H, W]
        final_tensor = final_tensor.unsqueeze(0)

        return final_tensor.to(torch.bfloat16)

    def __getitem__(self, index):
        sample_meta = self.samples[index]
        vols = sample_meta["volumes"]

        # Determine reconstruction selection based on available volumes
        num_vols = len(vols)

        if num_vols == 1:
            # Use same reconstruction for all 3 windows
            selected_vols = [vols[0], vols[0], vols[0]]
        elif num_vols == 2:
            # Use vol[0] for one window, vol[1] for the other two (deterministically)
            # For inference, we use a deterministic pattern instead of random
            selected_vols = [vols[0], vols[1], vols[1]]
        elif num_vols >= 3:
            # Use first 3 reconstructions deterministically
            selected_vols = vols[:3]

        # Window names in order: full, mediastinal, lung
        window_names = ["full", "mediastinal", "lung"]
        video_tensors = []
        real_volume_mask = []

        for vol, window_name in zip(selected_vols, window_names):
            # Load and resample the volume (raw HU)
            resampled_data = self.load_and_resample_nii(vol["image_path"])

            # Apply the specific window
            hu_min, hu_max = WINDOW_SETTINGS[window_name]
            tensor = self.apply_window_and_normalize(resampled_data, hu_min, hu_max)

            video_tensors.append(tensor)
            real_volume_mask.append(True)  # All 3 windows are always valid

        # Stack: [3, 1, D, H, W] where 3 = number of windows
        video_stack = torch.stack(video_tensors, dim=0)

        input_text = sample_meta["text"]
        onehotlabels = sample_meta["labels"]
        name_acc = sample_meta["scan_id"]

        return (
            video_stack,  # [3, 1, D, H, W]
            input_text,
            onehotlabels,
            name_acc,
            torch.tensor(real_volume_mask, dtype=torch.bool)  # [3]
        )


def collate_fn_infer(batch):
    """Collate function for inference that handles the multi-window format."""
    images, texts, labels, names, masks = [], [], [], [], []

    for img, text, label, name, mask in batch:
        images.append(img)
        texts.append(text)
        labels.append(label)
        names.append(name)
        masks.append(mask)

    # Stack images: [B, 3, C, D, H, W] where 3 = number of windows
    images = torch.stack(images, dim=0)
    # Stack masks: [B, 3]
    masks = torch.stack(masks, dim=0)
    # Labels as numpy array
    labels = np.array(labels)

    return images, texts, labels, names, masks