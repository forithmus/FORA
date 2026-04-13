"""
SFT dataset for report generation.

Wraps FORA's existing CTReportDataset for volume loading (all 5 sources,
4-window processing) but reads ground-truth reports from a reports CSV
(Findings_EN + Impressions_EN) rather than the sentence JSONL used for
contrastive pretraining.

For non-CT-RATE sources (Merlin, MIMIC, CheXpert, PadChest), reports are
constructed by joining the extracted sentences since those datasets don't
have a separate reports CSV.
"""

import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset

# Import CTReportDataset from FORA's scripts/data.py without name collision
import importlib.util as _ilu

_fora_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scripts_data_path = os.path.join(_fora_root, "scripts", "data.py")
_spec = _ilu.spec_from_file_location("fora_scripts_data", _scripts_data_path)
_fora_data = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fora_data)
CTReportDataset = _fora_data.CTReportDataset


def _load_reports_csv(csv_path: str) -> dict:
    """Load VolumeName -> report text mapping from reports CSV.

    Format: "Findings: {findings} Impression: {impression}"
    This matches the format used in the original autoregressive training.
    """
    mapping = {}
    df = pd.read_csv(csv_path)
    df["Findings_EN"] = df["Findings_EN"].fillna("").astype(str)
    df["Impressions_EN"] = df["Impressions_EN"].fillna("").astype(str)

    for _, row in df.iterrows():
        vol = row["VolumeName"]
        findings = row["Findings_EN"].strip()
        impression = row["Impressions_EN"].strip()
        report = f"Findings: {findings} Impression: {impression}".strip()
        mapping[vol] = report

    print(f"[Reports CSV] Loaded {len(mapping)} reports from {csv_path}")
    return mapping


class SFTReportDataset(Dataset):
    """Wraps CTReportDataset for SFT report generation.

    For CT-RATE: reads ground-truth reports from a reports CSV
    (Findings_EN + Impressions_EN), keyed by VolumeName.

    For other sources (Merlin, MIMIC, CheXpert, PadChest): joins
    the extracted sentence list into a report string.

    Returns per sample:
        (video_stack [4,1,D,H,W], report_text: str, real_volume_mask [4])

    If ``single_source`` is set, only samples from that source are used.
    """

    def __init__(
        self,
        ct_dataset: CTReportDataset,
        reports_csv: str = None,
        single_source: str = None,
    ):
        self.ct_dataset = ct_dataset

        # Load reports CSV for CT-RATE
        self.volume_to_report = {}
        if reports_csv and os.path.exists(reports_csv):
            self.volume_to_report = _load_reports_csv(reports_csv)

        # Single-source filtering
        if single_source is not None:
            if single_source not in ct_dataset.samples_by_source:
                raise ValueError(
                    f"Unknown source '{single_source}'. "
                    f"Available: {list(ct_dataset.samples_by_source.keys())}"
                )
            if len(ct_dataset.samples_by_source[single_source]) == 0:
                raise ValueError(
                    f"Source '{single_source}' has no samples. "
                    f"Make sure it is enabled in the dataset config."
                )
            self._samples = ct_dataset.samples_by_source[single_source]
            self._single_source = single_source
        else:
            self._samples = None
            self._single_source = None

    def __len__(self):
        if self._samples is not None:
            return len(self._samples)
        return len(self.ct_dataset)

    def _get_report_for_sample(self, sample, sentences, sentence_mask):
        """Get the report text for a sample.

        CT-RATE: look up from reports CSV by VolumeName.
        Others: join the extracted sentence list.
        """
        source = sample.get("source", "ctrate")

        # CT-RATE: use reports CSV (Findings + Impression)
        if source == "ctrate" and self.volume_to_report:
            volumes = sample.get("volumes", [])
            if volumes:
                vol_name = volumes[0].get("volume_name", "")
                report = self.volume_to_report.get(vol_name)
                if report:
                    return report

        # Fallback for all sources: join valid sentences
        valid_sentences = [
            s for s, m in zip(sentences, sentence_mask.tolist())
            if m and s.strip()
        ]
        return " ".join(valid_sentences)

    def __getitem__(self, idx):
        if self._samples is not None:
            # Single-source mode
            sample = self._samples[idx]
            source = sample["source"]
            getter = getattr(self.ct_dataset, f"_getitem_{source}")
            video_stack, sentences, sentence_mask, real_volume_mask = getter(sample)
        else:
            # Multi-source mode (round-robin)
            sample = self.ct_dataset._get_sample_by_index(idx)
            video_stack, sentences, sentence_mask, real_volume_mask = self.ct_dataset[idx]

        report_text = self._get_report_for_sample(sample, sentences, sentence_mask)

        return video_stack, report_text, real_volume_mask


def sft_collate_fn(batch):
    """Collate: stack volumes and masks, collect texts as a list."""
    videos, texts, masks = zip(*batch)
    videos = torch.stack(videos, dim=0)   # [B, 4, 1, D, H, W]
    masks = torch.stack(masks, dim=0)     # [B, 4]
    return videos, list(texts), masks
