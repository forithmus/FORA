"""
RadRateTrainer: Distributed trainer for RAD-RATE with multi-source medical imaging data.

Supports training on:
- CT-RATE (chest CT scans with 4 window settings)
- Merlin (abdominal CT scans with 4 window settings)
- MIMIC-CXR (chest X-rays)
- CheXpert (chest X-rays)
- PadChest (chest X-rays)

Uses VL-CABS contrastive learning with sentence-level supervision.
"""

from pathlib import Path
from shutil import rmtree
from datetime import timedelta
import math

from vision_encoder.optimizer import get_optimizer
from transformers import BertTokenizer

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import CTReportDataset, collate_fn, cycle

import numpy as np

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import torch.optim.lr_scheduler as lr_scheduler
from rad_rate import RADRATE


def exists(val):
    return val is not None


def noop(*args, **kwargs):
    pass


def yes_or_no(question):
    """Ask yes/no question. Returns False if not running interactively (e.g., batch job)."""
    import sys
    if not sys.stdin.isatty():
        # Non-interactive mode (batch job), default to No to avoid blocking
        print(f"{question} (y/n) -> Defaulting to 'n' (non-interactive mode)")
        return False
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup and cosine annealing restarts."""

    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))
        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma


class RadRateTrainer(nn.Module):
    """
    Distributed trainer for RAD-RATE model.

    Supports multi-source training with balanced sampling across:
    CT-RATE, Merlin, MIMIC-CXR, CheXpert, and PadChest datasets.

    Data format per batch:
    - images: [B, 4, 1, D, H, W] - 4 windows/views per sample
    - sentences: List of B * max_sentences strings (flattened)
    - sentence_mask: [B * max_sentences] - which sentences are valid
    - real_volume_mask: [B, 4] - which of the 4 volumes are real (vs padded)
    """

    def __init__(
            self,
            model: RADRATE,
            *,
            num_train_steps,
            batch_size=1,
            gradient_accumulation_steps=1,

            # CT-RATE dataset parameters
            data_train=None,
            reports_file_train=None,
            train_meta_file=None,

            # Merlin dataset parameters (uses defaults from CTReportDataset if None)
            merlin_data_folder=None,
            merlin_jsonl_file=None,
            merlin_csv_file=None,

            # MIMIC-CXR dataset parameters (uses defaults from CTReportDataset if None)
            mimic_data_folder=None,
            mimic_jsonl_file=None,
            mimic_csv_file=None,
            mimic_cache_file=None,

            # CheXpert dataset parameters (uses defaults from CTReportDataset if None)
            chexpert_data_folder=None,
            chexpert_jsonl_file=None,
            chexpert_cache_file=None,

            # PadChest dataset parameters (uses defaults from CTReportDataset if None)
            padchest_data_folder=None,
            padchest_jsonl_file=None,
            padchest_cache_file=None,

            # Dataset inclusion flags
            include_ctrate=True,
            include_merlin=True,
            include_mimic=True,
            include_chexpert=True,
            include_padchest=True,

            # Common dataset parameters
            max_sentences_per_image=34,

            # Training parameters
            tokenizer=None,
            lr=5e-5,
            wd=5e-2,
            max_grad_norm=0.5,
            warmup_steps=500,
            save_results_every=1000,
            save_model_every=1000,
            results_folder='./ctclip/',
            num_workers=8,
            accelerate_kwargs: dict = dict()
    ):
        super().__init__()

        # Initialize accelerator for distributed training
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            **accelerate_kwargs
        )

        self.model = model

        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                'microsoft/BiomedVLP-CXR-BERT-specialized',
                do_lower_case=True
            )

        self.register_buffer('steps', torch.Tensor([0]))
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.lr = lr

        # Initialize optimizer
        all_parameters = set(model.parameters())
        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        # Initialize learning rate scheduler (currently disabled in train_step)
        self.scheduler = CosineAnnealingWarmUpRestarts(
            self.optim,
            T_0=num_train_steps,
            T_warmup=warmup_steps,
            eta_max=lr
        )

        # Build dataset kwargs - only include non-None values to use defaults
        dataset_kwargs = {
            'max_sentences_per_image': max_sentences_per_image,
            'include_ctrate': include_ctrate,
            'include_merlin': include_merlin,
            'include_mimic': include_mimic,
            'include_chexpert': include_chexpert,
            'include_padchest': include_padchest,
        }

        # CT-RATE parameters
        if data_train is not None:
            dataset_kwargs['data_folder'] = data_train
        if reports_file_train is not None:
            dataset_kwargs['jsonl_file'] = reports_file_train
        if train_meta_file is not None:
            dataset_kwargs['meta_file'] = train_meta_file

        # Merlin parameters
        if merlin_data_folder is not None:
            dataset_kwargs['merlin_data_folder'] = merlin_data_folder
        if merlin_jsonl_file is not None:
            dataset_kwargs['merlin_jsonl_file'] = merlin_jsonl_file
        if merlin_csv_file is not None:
            dataset_kwargs['merlin_csv_file'] = merlin_csv_file

        # MIMIC-CXR parameters
        if mimic_data_folder is not None:
            dataset_kwargs['mimic_data_folder'] = mimic_data_folder
        if mimic_jsonl_file is not None:
            dataset_kwargs['mimic_jsonl_file'] = mimic_jsonl_file
        if mimic_csv_file is not None:
            dataset_kwargs['mimic_csv_file'] = mimic_csv_file
        if mimic_cache_file is not None:
            dataset_kwargs['mimic_cache_file'] = mimic_cache_file

        # CheXpert parameters
        if chexpert_data_folder is not None:
            dataset_kwargs['chexpert_data_folder'] = chexpert_data_folder
        if chexpert_jsonl_file is not None:
            dataset_kwargs['chexpert_jsonl_file'] = chexpert_jsonl_file
        if chexpert_cache_file is not None:
            dataset_kwargs['chexpert_cache_file'] = chexpert_cache_file

        # PadChest parameters
        if padchest_data_folder is not None:
            dataset_kwargs['padchest_data_folder'] = padchest_data_folder
        if padchest_jsonl_file is not None:
            dataset_kwargs['padchest_jsonl_file'] = padchest_jsonl_file
        if padchest_cache_file is not None:
            dataset_kwargs['padchest_cache_file'] = padchest_cache_file

        # Initialize dataset
        self.print(f"[Trainer] Initializing dataset...")
        self.ds = CTReportDataset(**dataset_kwargs)

        self.print(f"[Trainer] Dataset initialized with {len(self.ds)} samples per epoch")
        self.print(f"[Trainer] Max sentences per image: {self.ds.max_sentences}")

        # Barrier: ensure all ranks have loaded the dataset before proceeding
        self.accelerator.wait_for_everyone()
        self.print(f"[Trainer] All ranks synchronized after dataset init")

        # Initialize dataloader
        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        # Move model to device
        self.device = self.accelerator.device
        self.model.to(self.device)

        # Prepare for distributed training
        # IMPORTANT: Prepare DataLoader FIRST, then wrap in cycle()
        (
            self.dl,
            self.model,
            self.optim,
            self.scheduler
        ) = self.accelerator.prepare(
            self.dl,
            self.model,
            self.optim,
            self.scheduler
        )

        # Create infinite iterator AFTER prepare()
        self.dl_iter = cycle(self.dl)

        # Force optimizer LR to desired value (scheduler __init__ and prepare() set it to 0)
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        # Setup results folder
        self.results_folder = Path(results_folder)

        if self.is_main:
            if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no(
                'Do you want to clear previous experiment checkpoints and results?'
            ):
                rmtree(str(self.results_folder))
            self.results_folder.mkdir(parents=True, exist_ok=True)

        # Synchronize all processes before training
        self.accelerator.wait_for_everyone()

    def save(self, path):
        """Save model checkpoint."""
        if not self.accelerator.is_local_main_process:
            return
        pkg = dict(
            model=self.accelerator.get_state_dict(self.model),
            optim=self.optim.state_dict(),
            scheduler=self.scheduler.state_dict(),
            steps=self.steps.item()
        )
        torch.save(pkg, path)

    def load(self, path):
        """Load model checkpoint."""
        path = Path(path)
        assert path.exists(), f"Checkpoint not found: {path}"
        pkg = torch.load(path, map_location=self.device)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])

        if 'scheduler' in pkg:
            self.scheduler.load_state_dict(pkg['scheduler'])
        if 'steps' in pkg:
            self.steps.fill_(pkg['steps'])

    def print(self, msg):
        """Print message on main process only."""
        self.accelerator.print(msg)

    @property
    def is_main(self):
        """Check if this is the main process."""
        return self.accelerator.is_main_process

    def train_step(self):
        """Execute a single training step."""
        device = self.device
        steps = int(self.steps.item())

        self.model.train()
        logs = {}

        # DEBUG: Print before data loading
        if steps == 0:
            self.print(f"[DEBUG] Step {steps}: Loading batch...")

        # Get batch: [B, 4, 1, D, H, W], sentences, [B*max_sentences], [B, 4]
        images, sentences, masks, real_vols = next(self.dl_iter)

        if steps == 0:
            self.print(f"[DEBUG] Step {steps}: Batch loaded. images={images.shape}, masks={masks.shape}")

        images = images.to(device)
        masks = masks.to(device)
        real_vols = real_vols.to(device)

        # Tokenize sentences
        tok = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        if steps == 0:
            self.print(f"[DEBUG] Step {steps}: Starting forward pass...")

        # Forward pass with mixed precision
        with self.accelerator.autocast():
            loss = self.model(
                text_input=tok,
                image=images,
                num_sentences_per_image=self.ds.max_sentences,
                sentence_mask=masks,
                real_volume_mask=real_vols,
                return_loss=True,
                device=device,
                debug=(steps == 0)  # Enable debug prints on first step
            )
            loss = loss / self.gradient_accumulation_steps

        if steps == 0:
            self.print(f"[DEBUG] Step {steps}: Forward done, loss={loss.item():.4f}. Starting backward...")

        # NaN/Inf guard: skip backward + optimizer step to prevent model corruption
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            self.print(f"[WARNING] Step {steps}: loss is {loss_val}, skipping backward/optimizer step")
            self.optim.zero_grad()
            accum_log(logs, {'loss': float('nan')})
            self.print(f"Step {steps}: loss=nan, lr={self.optim.param_groups[0]['lr']:.2e}")
            self.steps += 1
            return logs

        # Backward pass
        self.accelerator.backward(loss)

        if steps == 0:
            self.print(f"[DEBUG] Step {steps}: Backward done.")

        # Accumulate loss for logging
        accum_log(logs, {'loss': loss.item() * self.gradient_accumulation_steps})

        # Update weights every gradient_accumulation_steps
        if (steps + 1) % self.gradient_accumulation_steps == 0:
            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            self.optim.step()
            # self.scheduler.step()  # Disabled: use constant LR for now
            self.optim.zero_grad()

        # Log progress
        current_lr = self.optim.param_groups[0]['lr']
        self.print(f"Step {steps}: loss={logs['loss']:.4f}, lr={current_lr:.2e}")

        # Save checkpoint
        if self.is_main and steps > 0 and (steps % self.save_model_every == 0):
            model_path = str(self.results_folder / f'RadRate.{steps}.pt')
            state_dict = self.accelerator.get_state_dict(self.model, unwrap=False)
            self.accelerator.save(state_dict, model_path)
            self.print(f"Saved checkpoint: {model_path}")

        self.steps += 1
        return logs

    def train(self, log_fn=noop):
        """Run the full training loop."""
        self.print(f"Starting training for {self.num_train_steps} steps...")
        self.print(f"Batch size: {self.batch_size}, Grad accumulation: {self.gradient_accumulation_steps}")
        self.print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes}")

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('Training complete!')