import os
import glob
import torch
import pandas as pd
import numpy as np
import json
import random
import pickle
from torch.utils.data import Dataset
from functools import partial
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from collections import defaultdict
from PIL import Image


def cycle(dl):
    """Helper to infinitely loop through a DataLoader."""
    while True:
        for data in dl:
            yield data

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing using trilinear interpolation.
    """
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


# Window settings for chest CT (CT-RATE)
CHEST_WINDOW_SETTINGS = {
    "full": (-1000, 1000),
    "mediastinal": (-150, 350),
    "lung": (-1400, -200),
    "bone": (-500, 2000),
}

# Window settings for abdominal CT (Merlin)
ABDOMINAL_WINDOW_SETTINGS = {
    "full": (-1000, 1000),
    "soft_tissue": (-125, 225),
    "liver": (-30, 170),
    "bone": (-500, 2000),
}


class CTReportDataset(Dataset):
    """
    Unified dataset for CT-RATE (chest CT), Merlin (abdominal CT), and MIMIC-CXR (X-ray) data.
    
    Balanced sampling: Uses round-robin iteration across datasets.
    - Each dataset is fully iterated through
    - Smaller datasets cycle/repeat until the largest dataset completes one full pass
    - Epoch length = max(dataset_size) * num_datasets
    
    Example with 5 datasets of sizes [100, 200, 500, 300, 150]:
    - index 0 -> dataset 0, sample 0
    - index 1 -> dataset 1, sample 0
    - index 2 -> dataset 2, sample 0
    - index 3 -> dataset 3, sample 0
    - index 4 -> dataset 4, sample 0
    - index 5 -> dataset 0, sample 1
    - index 6 -> dataset 1, sample 1
    - ...
    - Smaller datasets wrap around when exhausted
    """
    
    def __init__(
        self,
        # CT-RATE parameters
        data_folder=None,
        jsonl_file=None,
        meta_file=None,
        # Merlin parameters
        merlin_data_folder="/capstor/store/cscs/swissai/a135/RadVLM_project/hamamci_experiments/merlin_data/merlinabdominalctdataset/merlin_data/",
        merlin_jsonl_file="/capstor/store/cscs/swissai/a135/RadVLM_project/hamamci_experiments/merlin_data/merlinabdominalctdataset/extracted_sentences/merged.jsonl",
        merlin_csv_file="/capstor/store/cscs/swissai/a135/RadVLM_project/hamamci_experiments/merlin_data/merlinabdominalctdataset/train.csv",
        # MIMIC-CXR parameters
        mimic_data_folder="/capstor/store/cscs/swissai/a135/RadVLM_project/data/MIMIC-CXR-JPG/files",
        mimic_jsonl_file="/iopsstor/scratch/cscs/ihamamci/xray_datasets_reports/mimic_extracted_sentences.jsonl",
        mimic_csv_file="/iopsstor/scratch/cscs/ihamamci/xray_datasets_reports/mimic_maplocation.csv",
        # CheXpert parameters
        chexpert_data_folder="/capstor/store/cscs/swissai/a135/RadVLM_project/data/CheXpert",
        chexpert_jsonl_file="/iopsstor/scratch/cscs/ihamamci/xray_datasets_reports/chexpert_extracted_sentences.jsonl",
        # PadChest parameters
        padchest_data_folder="/capstor/store/cscs/swissai/a135/RadVLM_project/data/PadChest/images-224",
        padchest_jsonl_file="/iopsstor/scratch/cscs/ihamamci/xray_datasets_reports/padchest_extracted_sentences.jsonl",
        # Cache file parameters (for fast loading of X-ray datasets)
        mimic_cache_file="/iopsstor/scratch/cscs/ihamamci/xray_datasets_reports/mimic_cache.pkl",
        chexpert_cache_file="/iopsstor/scratch/cscs/ihamamci/xray_datasets_reports/chexpert_cache.pkl",
        padchest_cache_file="/iopsstor/scratch/cscs/ihamamci/xray_datasets_reports/padchest_cache.pkl",
        # Common parameters
        min_slices=20,
        max_sentences_per_image=34,
        include_ctrate=True,
        include_merlin=True,
        include_mimic=False,
        include_chexpert=False,
        include_padchest=False,
    ):
        self.min_slices = min_slices
        self.max_sentences = max_sentences_per_image
        
        # CT-RATE specific
        self.data_folder = data_folder
        self.meta_df = None
        self.volume_to_sentences = {}
        
        # Merlin specific
        self.merlin_data_folder = merlin_data_folder
        self.merlin_meta_df = None
        self.merlin_study_to_sentences = {}
        
        # MIMIC-CXR specific
        self.mimic_data_folder = mimic_data_folder
        self.mimic_accession_to_sentences = {}
        self.mimic_accession_to_path = {}
        
        # CheXpert specific
        self.chexpert_data_folder = chexpert_data_folder
        self.chexpert_accession_to_sentences = {}
        
        # PadChest specific
        self.padchest_data_folder = padchest_data_folder
        self.padchest_accession_to_sentences = {}
        
        # Samples organized by source for balanced sampling
        self.samples_by_source = {
            'ctrate': [],
            'merlin': [],
            'mimic': [],
            'chexpert': [],
            'padchest': [],
        }
        
        # Load CT-RATE data
        if include_ctrate and data_folder and jsonl_file and meta_file:
            self._load_ctrate_data(data_folder, jsonl_file, meta_file)
        
        # Load Merlin data
        if include_merlin and merlin_data_folder and merlin_jsonl_file and merlin_csv_file:
            self._load_merlin_data(merlin_data_folder, merlin_jsonl_file, merlin_csv_file)
        
        # Load MIMIC-CXR data (prefer cache for fast loading)
        if include_mimic:
            self._load_mimic_data(mimic_data_folder, mimic_jsonl_file, mimic_csv_file, mimic_cache_file)

        # Load CheXpert data (prefer cache for fast loading)
        if include_chexpert:
            self._load_chexpert_data(chexpert_data_folder, chexpert_jsonl_file, chexpert_cache_file)

        # Load PadChest data (prefer cache for fast loading)
        if include_padchest:
            self._load_padchest_data(padchest_data_folder, padchest_jsonl_file, padchest_cache_file)
        
        # Get list of available sources (non-empty) - maintain consistent order
        self.available_sources = [src for src in ['ctrate', 'merlin', 'mimic', 'chexpert', 'padchest'] 
                                  if len(self.samples_by_source[src]) > 0]
        self.num_sources = len(self.available_sources)
        
        # Calculate max dataset size and effective epoch length
        self.source_sizes = {src: len(self.samples_by_source[src]) for src in self.available_sources}
        self.max_source_size = max(self.source_sizes.values()) if self.source_sizes else 0
        self.total_samples = sum(self.source_sizes.values())
        
        # Epoch length = max_dataset_size * num_datasets
        # This ensures the largest dataset is fully covered, smaller ones cycle
        self._epoch_length = self.max_source_size * self.num_sources
        
        print(f"[CTReportDataset] Total samples across all sources: {self.total_samples}")
        print(f"[CTReportDataset] Available sources: {self.available_sources}")
        print(f"[CTReportDataset] Max source size: {self.max_source_size}")
        print(f"[CTReportDataset] Epoch length (balanced): {self._epoch_length}")
        for src in self.available_sources:
            count = self.source_sizes[src]
            repeats = self.max_source_size / count if count > 0 else 0
            print(f"  - {src}: {count} samples (repeats {repeats:.2f}x per epoch)")
        
        # Noise settings
        self.noise_images = False
        self.noise_prob = 0
        self.noise_mode = "normal"
        self.noise_mix = 0
        self.noise_seed = int(0)

        self.random_masks = True
        self.random_mask_prob = 1.0
        self.random_mask_num_labels = 1

        # One-time visualization flags - DISABLED
        self._saved_example_ctrate = True
        self._saved_example_merlin = True
        self._saved_example_mimic = True
        self._saved_example_chexpert = True
        self._saved_example_padchest = True
        self._example_save_dir = "./window_examples"
    
    # ==================== CT-RATE Loading ====================
    
    def _load_ctrate_data(self, data_folder, jsonl_file, meta_file):
        """Load CT-RATE dataset."""
        exclusion_file_path = "no_chest_train.txt"
        self.exclusion_list = self._load_exclusion_list(exclusion_file_path)
        
        print(f"[CT-RATE] Loading metadata from {meta_file}...")
        self.meta_df = pd.read_csv(meta_file)
        self.meta_df['scan_id'] = self.meta_df['VolumeName'].apply(
            lambda x: x.replace('.nii.gz', '').rsplit('_', 1)[0]
        )
        
        print(f"[CT-RATE] Loading sentences from {jsonl_file}...")
        self.volume_to_sentences = self._load_ctrate_jsonl(jsonl_file)
        
        ctrate_samples = self._prepare_ctrate_samples(data_folder)
        self.samples_by_source['ctrate'] = ctrate_samples
        print(f"[CT-RATE] Found {len(ctrate_samples)} valid scans.")
    
    def _load_exclusion_list(self, file_path):
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        return set()
    
    def _load_ctrate_jsonl(self, jsonl_path):
        mapping = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('valid_json', False) and len(data.get('extracted_sentences', [])) > 0:
                        mapping[data['volume_name']] = data['extracted_sentences']
                except:
                    continue
        return mapping
    
    def _prepare_ctrate_samples(self, data_folder):
        all_nii_paths = glob.glob(os.path.join(data_folder, '**/*.nii.gz'), recursive=True)
        grouped_scans = defaultdict(list)
        
        for path in tqdm(all_nii_paths, desc="[CT-RATE] Preparing samples"):
            relative_path = os.path.relpath(path, data_folder)
            relative_path_for_exclusion = "train/" + str(relative_path)
            if relative_path_for_exclusion in self.exclusion_list:
                continue
            volume_name = os.path.basename(path)
            if volume_name in self.volume_to_sentences:
                scan_id = volume_name.replace('.nii.gz', '').rsplit('_', 1)[0]
                grouped_scans[scan_id].append({
                    "volume_name": volume_name,
                    "image_path": path,
                    "sentences": self.volume_to_sentences[volume_name]
                })
        
        samples = []
        for scan_id, volumes in grouped_scans.items():
            samples.append({
                "source": "ctrate",
                "scan_id": scan_id,
                "volumes": volumes,
            })
        return samples
    
    # ==================== Merlin Loading ====================
    
    def _load_merlin_data(self, data_folder, jsonl_file, csv_file):
        """Load Merlin dataset."""
        print(f"[Merlin] Loading metadata from {csv_file}...")
        self.merlin_meta_df = pd.read_csv(csv_file)
        self.merlin_meta_df.columns = self.merlin_meta_df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        print(f"[Merlin] Loading sentences from {jsonl_file}...")
        self.merlin_study_to_sentences = self._load_merlin_jsonl(jsonl_file)
        
        merlin_samples = self._prepare_merlin_samples(data_folder)
        self.samples_by_source['merlin'] = merlin_samples
        print(f"[Merlin] Found {len(merlin_samples)} valid samples.")
    
    def _load_merlin_jsonl(self, jsonl_path):
        mapping = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    study_id = data.get('study_id') or data.get('accession') or data.get('volume_name')
                    sentences = data.get('extracted_sentences', [])
                    valid = data.get('valid_json', True)
                    
                    if study_id and valid and len(sentences) > 0:
                        mapping[str(study_id)] = sentences
                except:
                    continue
        return mapping
    
    def _prepare_merlin_samples(self, data_folder):
        samples = []
        
        for idx, row in tqdm(self.merlin_meta_df.iterrows(), total=len(self.merlin_meta_df), 
                            desc="[Merlin] Preparing samples"):
            study_id = str(row.get('study_id', ''))
            
            if not study_id or study_id not in self.merlin_study_to_sentences:
                continue
            
            nii_path = self._find_merlin_nii_path(data_folder, study_id, row)
            
            if nii_path is None:
                continue
            
            samples.append({
                "source": "merlin",
                "study_id": study_id,
                "image_path": nii_path,
                "sentences": self.merlin_study_to_sentences[study_id],
                "row_data": row.to_dict(),
            })
        
        return samples
    
    def _find_merlin_nii_path(self, data_folder, study_id, row):
        """Find NIfTI file path for Merlin sample."""
        possible_paths = [
            os.path.join(data_folder, f"{study_id}.nii.gz"),
            os.path.join(data_folder, f"{study_id}.nii"),
            os.path.join(data_folder, study_id, f"{study_id}.nii.gz"),
        ]
        
        for col in ['file_path', 'filename', 'nifti_path', 'path']:
            if col in row and pd.notna(row[col]):
                possible_paths.insert(0, os.path.join(data_folder, str(row[col])))
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        pattern = os.path.join(data_folder, f"**/*{study_id}*.nii*")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
        
        return None
    
    # ==================== MIMIC-CXR Loading ====================

    def _load_mimic_data(self, data_folder, jsonl_file, csv_file, cache_file=None):
        """Load MIMIC-CXR dataset. Uses cache file if available for fast loading."""
        # Try to load from cache first
        if cache_file and os.path.exists(cache_file):
            print(f"[MIMIC-CXR] Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                mimic_samples = pickle.load(f)
            self.samples_by_source['mimic'] = mimic_samples
            print(f"[MIMIC-CXR] Loaded {len(mimic_samples)} samples from cache.")
            return

        # Fall back to live validation (slow on network filesystems)
        print(f"[MIMIC-CXR] WARNING: Cache file not found. Falling back to live validation.")
        print(f"[MIMIC-CXR] This may be slow. Consider running: python generate_xray_cache.py --mimic")

        print(f"[MIMIC-CXR] Loading path mapping from {csv_file}...")
        mimic_map_df = pd.read_csv(csv_file)

        for _, row in tqdm(mimic_map_df.iterrows(), total=len(mimic_map_df),
                         desc="[MIMIC-CXR] Building path mapping"):
            accession = str(row['accession'])
            path = str(row['path'])

            if path.startswith('./'):
                path = path[2:]
            elif path.startswith('.'):
                path = path[1:]

            if path.endswith('.txt'):
                path = path[:-4]

            folder_path = os.path.join(data_folder, path)
            self.mimic_accession_to_path[accession] = folder_path

        print(f"[MIMIC-CXR] Loading sentences from {jsonl_file}...")
        self.mimic_accession_to_sentences = self._load_mimic_jsonl(jsonl_file)

        mimic_samples = self._prepare_mimic_samples()
        self.samples_by_source['mimic'] = mimic_samples
        print(f"[MIMIC-CXR] Found {len(mimic_samples)} valid samples.")
    
    def _load_mimic_jsonl(self, jsonl_path):
        """Load MIMIC-CXR sentences from JSONL file."""
        mapping = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    accession_no = str(data.get('accession_no', ''))
                    sentences = data.get('extracted_sentences', [])
                    valid = data.get('valid_json', True)
                    
                    if accession_no and valid and len(sentences) > 0:
                        mapping[accession_no] = sentences
                except:
                    continue
        return mapping
    
    def _prepare_mimic_samples(self):
        """Prepare MIMIC-CXR samples by matching accessions with sentences and paths."""
        samples = []
        
        common_accessions = set(self.mimic_accession_to_path.keys()) & set(self.mimic_accession_to_sentences.keys())
        
        for accession in tqdm(common_accessions, desc="[MIMIC-CXR] Preparing samples"):
            folder_path = self.mimic_accession_to_path[accession]
            
            if not os.path.exists(folder_path):
                continue
            
            jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
            if len(jpg_files) == 0:
                continue
            
            samples.append({
                "source": "mimic",
                "accession": accession,
                "image_folder": folder_path,
                "image_paths": sorted(jpg_files),
                "sentences": self.mimic_accession_to_sentences[accession],
            })
        
        return samples
    
    # ==================== CheXpert Loading ====================

    def _load_chexpert_data(self, data_folder, jsonl_file, cache_file=None):
        """Load CheXpert dataset. Uses cache file if available for fast loading."""
        # Try to load from cache first
        if cache_file and os.path.exists(cache_file):
            print(f"[CheXpert] Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                chexpert_samples = pickle.load(f)
            self.samples_by_source['chexpert'] = chexpert_samples
            print(f"[CheXpert] Loaded {len(chexpert_samples)} samples from cache.")
            return

        # Fall back to live validation (slow on network filesystems)
        print(f"[CheXpert] WARNING: Cache file not found. Falling back to live validation.")
        print(f"[CheXpert] This may be slow. Consider running: python generate_xray_cache.py --chexpert")

        print(f"[CheXpert] Loading sentences from {jsonl_file}...")
        self.chexpert_accession_to_sentences = self._load_chexpert_jsonl(jsonl_file)

        chexpert_samples = self._prepare_chexpert_samples(data_folder)
        self.samples_by_source['chexpert'] = chexpert_samples
        print(f"[CheXpert] Found {len(chexpert_samples)} valid samples.")
    
    def _load_chexpert_jsonl(self, jsonl_path):
        """Load CheXpert sentences from JSONL file."""
        mapping = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    accession_no = str(data.get('accession_no', ''))
                    sentences = data.get('extracted_sentences', [])
                    valid = data.get('valid_json', True)
                    
                    if accession_no and valid and len(sentences) > 0:
                        mapping[accession_no] = sentences
                except:
                    continue
        return mapping
    
    def _parse_chexpert_accession(self, accession_no):
        parts = accession_no.split('_')
        if len(parts) >= 3:
            split = parts[0]
            patient_id = parts[1]
            study_id = '_'.join(parts[2:])
            return split, patient_id, study_id
        return None, None, None
    
    def _prepare_chexpert_samples(self, data_folder):
        """Prepare CheXpert samples by parsing accession numbers to construct paths."""
        samples = []
        
        for accession_no, sentences in tqdm(self.chexpert_accession_to_sentences.items(), 
                                            desc="[CheXpert] Preparing samples"):
            split, patient_id, study_id = self._parse_chexpert_accession(accession_no)
            
            if split is None:
                continue
            
            folder_path = os.path.join(data_folder, split, patient_id, study_id)
            
            if not os.path.exists(folder_path):
                continue
            
            jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
            if len(jpg_files) == 0:
                continue
            
            samples.append({
                "source": "chexpert",
                "accession": accession_no,
                "image_folder": folder_path,
                "image_paths": sorted(jpg_files),
                "sentences": sentences,
            })
        
        return samples
    
    # ==================== PadChest Loading ====================

    def _load_padchest_data(self, data_folder, jsonl_file, cache_file=None):
        """Load PadChest dataset. Uses cache file if available for fast loading."""
        # Try to load from cache first
        if cache_file and os.path.exists(cache_file):
            print(f"[PadChest] Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                padchest_samples = pickle.load(f)
            self.samples_by_source['padchest'] = padchest_samples
            print(f"[PadChest] Loaded {len(padchest_samples)} samples from cache.")
            return

        # Fall back to live validation (slow on network filesystems)
        print(f"[PadChest] WARNING: Cache file not found. Falling back to live validation.")
        print(f"[PadChest] This may be slow. Consider running: python generate_xray_cache.py --padchest")

        print(f"[PadChest] Loading sentences from {jsonl_file}...")
        self.padchest_accession_to_sentences = self._load_padchest_jsonl(jsonl_file)

        padchest_samples = self._prepare_padchest_samples(data_folder)
        self.samples_by_source['padchest'] = padchest_samples
        print(f"[PadChest] Found {len(padchest_samples)} valid samples.")
    
    def _load_padchest_jsonl(self, jsonl_path):
        """Load PadChest sentences from JSONL file."""
        mapping = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    accession_no = str(data.get('accession_no', ''))
                    sentences = data.get('extracted_sentences', [])
                    valid = data.get('valid_json', True)
                    
                    if accession_no and valid and len(sentences) > 0:
                        mapping[accession_no] = sentences
                except:
                    continue
        return mapping
    
    def _prepare_padchest_samples(self, data_folder):
        """Prepare PadChest samples - accession_no is the image filename directly."""
        samples = []
        
        for accession_no, sentences in tqdm(self.padchest_accession_to_sentences.items(), 
                                            desc="[PadChest] Preparing samples"):
            image_path = os.path.join(data_folder, accession_no)
            
            if not os.path.exists(image_path):
                continue
            
            samples.append({
                "source": "padchest",
                "accession": accession_no,
                "image_path": image_path,
                "sentences": sentences,
            })
        
        return samples
    
    # ==================== Common Methods ====================
    
    def __len__(self):
        """
        Returns epoch length = max_source_size * num_sources.
        This ensures:
        - The largest dataset is fully covered once per epoch
        - Smaller datasets cycle/repeat to maintain balance
        """
        return self._epoch_length
    
    def _get_sample_by_index(self, index):
        """
        Round-robin sampling across datasets.
        
        index 0 -> source 0, sample 0
        index 1 -> source 1, sample 0
        index 2 -> source 2, sample 0
        ...
        index num_sources -> source 0, sample 1
        index num_sources+1 -> source 1, sample 1
        ...
        
        Smaller datasets wrap around when exhausted.
        """
        # Which source (round-robin)
        source_idx = index % self.num_sources
        source = self.available_sources[source_idx]
        
        # Which sample within that source (wraps around for smaller datasets)
        within_source_idx = (index // self.num_sources) % len(self.samples_by_source[source])
        
        return self.samples_by_source[source][within_source_idx]
    
    def _make_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_mode == "uniform":
            noise = torch.rand(x.shape, dtype=torch.float32) * 2.0 - 1.0
        elif self.noise_mode == "normal":
            noise = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        else:
            raise ValueError(f"Unknown noise_mode={self.noise_mode}. Use 'uniform' or 'normal'.")
        return noise.to(dtype=x.dtype)
    
    def load_and_resample_nii_ctrate(self, path):
        """Load and resample NIfTI for CT-RATE (uses metadata CSV for spacing)."""
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata().astype(np.float32)
        # Replace NaN/Inf from corrupted scans with 0 (air HU)
        np.nan_to_num(img_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        file_name = path.split("/")[-1]
        row = self.meta_df[self.meta_df['VolumeName'] == file_name]
        
        if row.empty:
            raise RuntimeError(f"Could not find metadata for volume: {file_name}")
        
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
        current_spacing = (z_spacing, xy_spacing, xy_spacing)
        target_spacing = (1.5, 0.75, 0.75)
        
        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)
        resampled_data = resize_array(tensor, current_spacing, target_spacing)[0, 0]
        
        return resampled_data
    
    def load_and_resample_nii_merlin(self, path):
        """Load and resample NIfTI for Merlin (uses NIfTI header for spacing)."""
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata().astype(np.float32)
        # Replace NaN/Inf from corrupted scans with 0 (air HU)
        np.nan_to_num(img_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        header = nii_img.header
        voxel_sizes = header.get_zooms()
        
        if len(voxel_sizes) >= 3:
            xy_spacing = float(voxel_sizes[0])
            z_spacing = float(voxel_sizes[2])
        else:
            xy_spacing = 1.0
            z_spacing = 1.0
        
        current_spacing = (z_spacing, xy_spacing, xy_spacing)
        target_spacing = (1.5, 0.75, 0.75)
        
        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)
        resampled_data = resize_array(tensor, current_spacing, target_spacing)[0, 0]
        
        return resampled_data
    
    def apply_window_and_normalize(self, resampled_data, hu_min, hu_max):
        """Apply windowing and normalize to [-1, 1] range."""
        clipped_data = np.clip(resampled_data, hu_min, hu_max)
        normalized_data = ((clipped_data - hu_min) / (hu_max - hu_min)) * 2.0 - 1.0
        tensor = torch.from_numpy(normalized_data.astype(np.float32))
        
        target_shape_intermediate = (256, 480, 480)
        d, h, w = tensor.shape
        td, th, tw = target_shape_intermediate
        
        d_start, h_start, w_start = max((d - td) // 2, 0), max((h - th) // 2, 0), max((w - tw) // 2, 0)
        tensor = tensor[d_start:d_start + td, h_start:h_start + th, w_start:w_start + tw]
        
        pad_d_before = (td - tensor.size(0)) // 2
        pad_d_after = td - tensor.size(0) - pad_d_before
        pad_h_before = (th - tensor.size(1)) // 2
        pad_h_after = th - tensor.size(1) - pad_h_before
        pad_w_before = (tw - tensor.size(2)) // 2
        pad_w_after = tw - tensor.size(2) - pad_w_before
        
        tensor = F.pad(tensor, (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after), value=-1)
        tensor = tensor.unsqueeze(1)
        resized_tensor = F.interpolate(tensor, size=(384, 384), mode='bilinear', align_corners=False)
        final_tensor = resized_tensor.squeeze(1).unsqueeze(0)
        return final_tensor.to(torch.bfloat16)
    
    def load_and_normalize_xray(self, image_path):
        """Load X-ray image (JPG, 8-bit) and normalize to [-1, 1] range."""
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        
        img_min = img_array.min()
        img_max = img_array.max()
        
        if img_max - img_min > 0:
            normalized = ((img_array - img_min) / (img_max - img_min)) * 2.0 - 1.0
        else:
            normalized = np.zeros_like(img_array)
        
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(tensor, size=(384, 384), mode='bilinear', align_corners=False)
        final_tensor = resized_tensor.unsqueeze(2)
        
        return final_tensor.squeeze(0).to(torch.bfloat16)
    
    def load_and_normalize_png_16bit(self, image_path):
        """Load 16-bit PNG image (PadChest) and normalize to [-1, 1] range."""
        img = Image.open(image_path)
        img_array = np.array(img, dtype=np.float32)
        
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, 0]
            elif img_array.shape[2] == 3:
                img_array = np.mean(img_array, axis=2)
            else:
                img_array = img_array[:, :, 0]
        
        img_min = img_array.min()
        img_max = img_array.max()
        
        if img_max - img_min > 0:
            normalized = ((img_array - img_min) / (img_max - img_min)) * 2.0 - 1.0
        else:
            normalized = np.zeros_like(img_array)
        
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(tensor, size=(384, 384), mode='bilinear', align_corners=False)
        final_tensor = resized_tensor.unsqueeze(2)
        
        return final_tensor.squeeze(0).to(torch.bfloat16)
    
    def __getitem__(self, index):
        sample = self._get_sample_by_index(index)
        source = sample['source']
        
        if source == 'ctrate':
            return self._getitem_ctrate(sample)
        elif source == 'merlin':
            return self._getitem_merlin(sample)
        elif source == 'mimic':
            return self._getitem_mimic(sample)
        elif source == 'chexpert':
            return self._getitem_chexpert(sample)
        elif source == 'padchest':
            return self._getitem_padchest(sample)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _getitem_ctrate(self, sample):
        """Get item for CT-RATE sample."""
        vols = sample["volumes"]
        all_sentences = vols[0]["sentences"]
        
        num_vols = len(vols)
        if num_vols == 1:
            selected_vols = [vols[0], vols[0], vols[0], vols[0]]
        elif num_vols == 2:
            indices = [0, 0, 1, 1]
            random.shuffle(indices)
            selected_vols = [vols[i] for i in indices]
        elif num_vols == 3:
            indices = [0, 1, 2, random.choice([0, 1, 2])]
            random.shuffle(indices)
            selected_vols = [vols[i] for i in indices]
        elif num_vols == 4:
            selected_vols = random.sample(vols, 4)
        else:
            selected_vols = random.sample(vols, 4)
        
        window_names = ["full", "mediastinal", "lung", "bone"]
        video_tensors = []
        real_volume_mask = []
        
        for i, (vol, window_name) in enumerate(zip(selected_vols, window_names)):
            resampled_data = self.load_and_resample_nii_ctrate(vol["image_path"])
            hu_min, hu_max = CHEST_WINDOW_SETTINGS[window_name]
            tensor = self.apply_window_and_normalize(resampled_data, hu_min, hu_max)
            
            if self.noise_images and (random.random() < self.noise_prob):
                noise = self._make_noise_like(tensor)
                if self.noise_mix >= 1.0:
                    tensor = noise
                elif self.noise_mix > 0.0:
                    tensor = (1.0 - self.noise_mix) * tensor + self.noise_mix * noise
            
            video_tensors.append(tensor)
            real_volume_mask.append(True)
        
        video_stack = torch.stack(video_tensors, dim=0)
        
        if not self._saved_example_ctrate:
            self._save_example_visualization_ctrate(video_stack, sample.get("scan_id", "unknown"))
            self._saved_example_ctrate = True
        
        n = len(all_sentences)
        if n >= self.max_sentences:
            selected_sentences = random.sample(all_sentences, self.max_sentences)
            mask = [1] * self.max_sentences
        else:
            padding_count = self.max_sentences - n
            selected_sentences = all_sentences + [""] * padding_count
            mask = [1] * n + [0] * padding_count
        
        return video_stack, selected_sentences, torch.tensor(mask, dtype=torch.bool), torch.tensor(real_volume_mask, dtype=torch.bool)
    
    def _getitem_merlin(self, sample):
        """Get item for Merlin sample."""
        resampled_data = self.load_and_resample_nii_merlin(sample["image_path"])
        resampled_data = resampled_data[:, :, ::-1]

        all_sentences = sample["sentences"]
        
        window_names = ["full", "soft_tissue", "liver", "bone"]
        video_tensors = []
        real_volume_mask = []
        
        for window_name in window_names:
            hu_min, hu_max = ABDOMINAL_WINDOW_SETTINGS[window_name]
            tensor = self.apply_window_and_normalize(resampled_data, hu_min, hu_max)

            if self.noise_images and (random.random() < self.noise_prob):
                noise = self._make_noise_like(tensor)
                if self.noise_mix >= 1.0:
                    tensor = noise
                elif self.noise_mix > 0.0:
                    tensor = (1.0 - self.noise_mix) * tensor + self.noise_mix * noise
            
            video_tensors.append(tensor)
            real_volume_mask.append(True)
        
        video_stack = torch.stack(video_tensors, dim=0)
        
        if not self._saved_example_merlin:
            self._save_example_visualization_merlin(video_stack, sample.get("study_id", "unknown"))
            self._saved_example_merlin = True
        
        n = len(all_sentences)
        if n >= self.max_sentences:
            selected_sentences = random.sample(all_sentences, self.max_sentences)
            mask = [1] * self.max_sentences
        else:
            padding_count = self.max_sentences - n
            selected_sentences = all_sentences + [""] * padding_count
            mask = [1] * n + [0] * padding_count
        
        return video_stack, selected_sentences, torch.tensor(mask, dtype=torch.bool), torch.tensor(real_volume_mask, dtype=torch.bool)
    
    def _getitem_mimic(self, sample):
        """Get item for MIMIC-CXR sample."""
        image_paths = sample["image_paths"]
        all_sentences = sample["sentences"]

        video_tensors = []
        real_volume_mask = []

        for image_path in image_paths:
            tensor = self.load_and_normalize_xray(image_path)

            if self.noise_images and (random.random() < self.noise_prob):
                noise = self._make_noise_like(tensor)
                if self.noise_mix >= 1.0:
                    tensor = noise
                elif self.noise_mix > 0.0:
                    tensor = (1.0 - self.noise_mix) * tensor + self.noise_mix * noise

            video_tensors.append(tensor)
            real_volume_mask.append(True)

        # Pad to exactly 4 volumes for consistency with CT data (required for DDP)
        while len(video_tensors) < 4:
            # Repeat last image as padding
            video_tensors.append(video_tensors[-1].clone())
            real_volume_mask.append(False)  # Mark padded volumes as not real

        # If more than 4 images, take first 4
        video_tensors = video_tensors[:4]
        real_volume_mask = real_volume_mask[:4]

        video_stack = torch.stack(video_tensors, dim=0)
        
        if not self._saved_example_mimic:
            self._save_example_visualization_mimic(video_stack, sample.get("accession", "unknown"))
            self._saved_example_mimic = True
        
        n = len(all_sentences)
        if n >= self.max_sentences:
            selected_sentences = random.sample(all_sentences, self.max_sentences)
            mask = [1] * self.max_sentences
        else:
            padding_count = self.max_sentences - n
            selected_sentences = all_sentences + [""] * padding_count
            mask = [1] * n + [0] * padding_count
        
        return video_stack, selected_sentences, torch.tensor(mask, dtype=torch.bool), torch.tensor(real_volume_mask, dtype=torch.bool)
    
    def _getitem_chexpert(self, sample):
        """Get item for CheXpert sample."""
        image_paths = sample["image_paths"]
        all_sentences = sample["sentences"]

        video_tensors = []
        real_volume_mask = []

        for image_path in image_paths:
            tensor = self.load_and_normalize_xray(image_path)

            if self.noise_images and (random.random() < self.noise_prob):
                noise = self._make_noise_like(tensor)
                if self.noise_mix >= 1.0:
                    tensor = noise
                elif self.noise_mix > 0.0:
                    tensor = (1.0 - self.noise_mix) * tensor + self.noise_mix * noise

            video_tensors.append(tensor)
            real_volume_mask.append(True)

        # Pad to exactly 4 volumes for consistency with CT data (required for DDP)
        while len(video_tensors) < 4:
            video_tensors.append(video_tensors[-1].clone())
            real_volume_mask.append(False)  # Padded volumes excluded from loss

        video_tensors = video_tensors[:4]
        real_volume_mask = real_volume_mask[:4]

        video_stack = torch.stack(video_tensors, dim=0)
        
        if not self._saved_example_chexpert:
            self._save_example_visualization_chexpert(video_stack, sample.get("accession", "unknown"))
            self._saved_example_chexpert = True
        
        n = len(all_sentences)
        if n >= self.max_sentences:
            selected_sentences = random.sample(all_sentences, self.max_sentences)
            mask = [1] * self.max_sentences
        else:
            padding_count = self.max_sentences - n
            selected_sentences = all_sentences + [""] * padding_count
            mask = [1] * n + [0] * padding_count
        
        return video_stack, selected_sentences, torch.tensor(mask, dtype=torch.bool), torch.tensor(real_volume_mask, dtype=torch.bool)
    
    def _getitem_padchest(self, sample):
        """Get item for PadChest sample (single 16-bit PNG image)."""
        image_path = sample["image_path"]
        all_sentences = sample["sentences"]

        tensor = self.load_and_normalize_png_16bit(image_path)

        if self.noise_images and (random.random() < self.noise_prob):
            noise = self._make_noise_like(tensor)
            if self.noise_mix >= 1.0:
                tensor = noise
            elif self.noise_mix > 0.0:
                tensor = (1.0 - self.noise_mix) * tensor + self.noise_mix * noise

        # Pad to exactly 4 volumes for consistency with CT data (required for DDP)
        video_tensors = [tensor]
        real_volume_mask = [True]

        while len(video_tensors) < 4:
            video_tensors.append(tensor.clone())
            real_volume_mask.append(False)  # Padded volumes excluded from loss

        video_stack = torch.stack(video_tensors, dim=0)
        
        if not self._saved_example_padchest:
            self._save_example_visualization_padchest(video_stack, sample.get("accession", "unknown"))
            self._saved_example_padchest = True
        
        n = len(all_sentences)
        if n >= self.max_sentences:
            selected_sentences = random.sample(all_sentences, self.max_sentences)
            mask = [1] * self.max_sentences
        else:
            padding_count = self.max_sentences - n
            selected_sentences = all_sentences + [""] * padding_count
            mask = [1] * n + [0] * padding_count
        
        return video_stack, selected_sentences, torch.tensor(mask, dtype=torch.bool), torch.tensor(real_volume_mask, dtype=torch.bool)
    
    # ==================== Visualization Methods (Disabled) ====================
    
    def _save_example_visualization_ctrate(self, video_stack, scan_id):
        import matplotlib.pyplot as plt
        os.makedirs(self._example_save_dir, exist_ok=True)
        video_np = video_stack.float().numpy()
        window_names = ["full", "mediastinal", "lung", "bone"]
        window_ranges = [(-1000, 1000), (-150, 350), (-1400, -200), (-500, 2000)]
        num_slices = video_np.shape[2]
        mid_slice = num_slices // 2
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, (name, (hu_min, hu_max)) in enumerate(zip(window_names, window_ranges)):
            slice_data = video_np[i, 0, mid_slice, :, :]
            axes[i].imshow(slice_data, cmap='gray', vmin=-1, vmax=1)
            axes[i].set_title(f"{name.capitalize()} Window\nHU: [{hu_min}, {hu_max}]")
            axes[i].axis('off')
        plt.suptitle(f"CT-RATE (Chest) | Scan: {scan_id} | Slice: {mid_slice}/{num_slices}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self._example_save_dir, "ctrate_window_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_example_visualization_merlin(self, video_stack, study_id):
        import matplotlib.pyplot as plt
        os.makedirs(self._example_save_dir, exist_ok=True)
        video_np = video_stack.float().numpy()
        window_names = ["full", "soft_tissue", "liver", "bone"]
        window_ranges = [(-1000, 1000), (-125, 225), (-30, 170), (-500, 2000)]
        num_slices = video_np.shape[2]
        mid_slice = num_slices // 2
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, (name, (hu_min, hu_max)) in enumerate(zip(window_names, window_ranges)):
            slice_data = video_np[i, 0, mid_slice, :, :]
            axes[i].imshow(slice_data, cmap='gray', vmin=-1, vmax=1)
            axes[i].set_title(f"{name.capitalize()} Window\nHU: [{hu_min}, {hu_max}]")
            axes[i].axis('off')
        plt.suptitle(f"Merlin (Abdominal) | Study: {study_id} | Slice: {mid_slice}/{num_slices}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self._example_save_dir, "merlin_window_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_example_visualization_mimic(self, video_stack, accession):
        import matplotlib.pyplot as plt
        os.makedirs(self._example_save_dir, exist_ok=True)
        video_np = video_stack.float().numpy()
        num_images = video_np.shape[0]
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        if num_images == 1:
            axes = [axes]
        for i in range(num_images):
            img_data = video_np[i, 0, 0, :, :]
            axes[i].imshow(img_data, cmap='gray', vmin=-1, vmax=1)
            axes[i].set_title(f"Image {i+1}")
            axes[i].axis('off')
        plt.suptitle(f"MIMIC-CXR (X-ray) | Accession: {accession} | {num_images} image(s)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self._example_save_dir, "mimic_xray_example.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_example_visualization_chexpert(self, video_stack, accession):
        import matplotlib.pyplot as plt
        os.makedirs(self._example_save_dir, exist_ok=True)
        video_np = video_stack.float().numpy()
        num_images = video_np.shape[0]
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        if num_images == 1:
            axes = [axes]
        for i in range(num_images):
            img_data = video_np[i, 0, 0, :, :]
            axes[i].imshow(img_data, cmap='gray', vmin=-1, vmax=1)
            axes[i].set_title(f"Image {i+1}")
            axes[i].axis('off')
        plt.suptitle(f"CheXpert (X-ray) | Accession: {accession} | {num_images} image(s)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self._example_save_dir, "chexpert_xray_example.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_example_visualization_padchest(self, video_stack, accession):
        import matplotlib.pyplot as plt
        os.makedirs(self._example_save_dir, exist_ok=True)
        video_np = video_stack.float().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        img_data = video_np[0, 0, 0, :, :]
        ax.imshow(img_data, cmap='gray', vmin=-1, vmax=1)
        ax.set_title("Image")
        ax.axis('off')
        plt.suptitle(f"PadChest (X-ray) | Accession: {accession[:50]}...", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self._example_save_dir, "padchest_xray_example.png"), dpi=150, bbox_inches='tight')
        plt.close()


def collate_fn(batch):
    images, flat_sentences, flat_masks, real_vols = [], [], [], []
    for img, sentences, mask, r_mask in batch:
        images.append(img)
        flat_sentences.extend(sentences)
        flat_masks.append(mask)
        real_vols.append(r_mask)
    return torch.stack(images), flat_sentences, torch.cat(flat_masks), torch.stack(real_vols)