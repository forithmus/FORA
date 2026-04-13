"""
SFT Trainer for report generation.

Accelerate-based distributed training loop that:
1. Passes CT/X-ray volumes through the visual encoder
2. Prepends visual embeddings to tokenized report text
3. Trains with causal LM cross-entropy loss (labels=-100 for visual tokens)
4. Logs to W&B (loss, LR, grad norms, generated sample reports)
5. Supports checkpoint save/resume
"""

import os
import re
import math
import logging
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


class SFTTrainer:
    def __init__(
        self,
        visual_encoder: nn.Module,
        llm: nn.Module,
        tokenizer,
        train_dataset,
        collate_fn,
        config: dict,
    ):
        self.config = config
        train_cfg = config["training"]
        model_cfg = config["model"]
        wandb_cfg = config.get("wandb", {})

        # Accelerate
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(
            mixed_precision="bf16" if train_cfg.get("bf16", True) else "no",
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            kwargs_handlers=[ddp_kwargs, init_kwargs],
        )
        self.device = self.accelerator.device

        # Models
        self.visual_encoder = visual_encoder.to(self.device)
        self.llm = llm.to(self.device)
        self.tokenizer = tokenizer

        # Training config
        self.max_steps = train_cfg["max_steps"]
        self.max_report_length = train_cfg.get("max_report_length", 512)
        self.save_steps = train_cfg.get("save_steps", 500)
        self.logging_steps = train_cfg.get("logging_steps", 10)
        self.generate_every = train_cfg.get("generate_every", 500)
        self.num_generate_samples = train_cfg.get("num_generate_samples", 4)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.output_dir = Path(train_cfg.get("output_dir", "./sft_results"))
        self.freeze_visual = model_cfg.get("freeze_visual", False)
        self.local_batch_size = train_cfg.get("per_device_batch_size", 1)

        # EOS token — critical for non-stopping reports
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id. The model cannot learn to stop generating."
            )
        self.print(f"EOS token: '{tokenizer.eos_token}' (id={self.eos_token_id})")

        # DataLoader
        self.dl = DataLoader(
            train_dataset,
            batch_size=train_cfg.get("per_device_batch_size", 1),
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=train_cfg.get("dataloader_num_workers", 4),
            prefetch_factor=2 if train_cfg.get("dataloader_num_workers", 4) > 0 else None,
            persistent_workers=train_cfg.get("dataloader_num_workers", 4) > 0,
        )

        # Optimizer: separate param groups for LLM LoRA vs visual encoder
        llm_lr = train_cfg.get("learning_rate", 1e-5)
        visual_lr = train_cfg.get("visual_lr", 1e-6)

        llm_params = [p for p in self.llm.parameters() if p.requires_grad]
        param_groups = [{"params": llm_params, "lr": llm_lr}]

        if not self.freeze_visual:
            visual_params = [p for p in self.visual_encoder.parameters() if p.requires_grad]
            if visual_params:
                param_groups.append({"params": visual_params, "lr": visual_lr})
                self.print(f"Visual encoder: {len(visual_params)} trainable param tensors, lr={visual_lr}")
            else:
                self.print("WARNING: freeze_visual=False but no trainable visual params found.")
        else:
            self.visual_encoder.freeze()
            self.print("Visual encoder: FROZEN")

        self.optim = AdamW(
            param_groups,
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        # Scheduler: cosine with warmup
        warmup_steps = train_cfg.get("warmup_steps", 500)
        pct_start = min(warmup_steps / self.max_steps, 0.3)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=self.max_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

        # Prepare with accelerate
        self.llm, self.optim, self.dl, self.scheduler = self.accelerator.prepare(
            self.llm, self.optim, self.dl, self.scheduler
        )
        # Visual encoder: prepare separately (no DDP if frozen)
        if not self.freeze_visual:
            self.visual_encoder = self.accelerator.prepare_model(self.visual_encoder)

        # Infinite iterator
        self._dl_iter = self._cycle(self.dl)

        # Step counter
        self.global_step = 0

        # Resume
        self.resume = train_cfg.get("resume", False)
        if self.resume:
            self._auto_resume()

        # Setup output dir
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        self.use_wandb = bool(wandb_cfg.get("project")) and wandb is not None
        if self.use_wandb and self.is_main:
            resumed_step = self.global_step
            wandb_resume = "must" if (self.resume and resumed_step > 0) else "allow"
            run_id = None
            wandb_id_file = self.output_dir / "wandb_run_id.txt"
            if self.resume and wandb_id_file.exists():
                run_id = wandb_id_file.read_text().strip()
                self.print(f"[W&B] Resuming run {run_id}")
            wandb.init(
                project=wandb_cfg["project"],
                name=wandb_cfg.get("run_name"),
                config=config,
                resume=wandb_resume,
                id=run_id,
                tags=wandb_cfg.get("tags", []),
            )
            wandb_id_file.write_text(wandb.run.id)

        self.accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cycle(dl):
        while True:
            for batch in dl:
                yield batch

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # ------------------------------------------------------------------
    # Batch preparation — this is where EOS handling happens
    # ------------------------------------------------------------------

    def _prepare_batch(self, volumes, report_texts, real_volume_masks):
        """Prepare a training batch.

        1. Encode volumes -> visual embeddings [B, V, D]
        2. Tokenize reports WITH EOS appended
        3. Build inputs_embeds = [visual_embeds | text_embeds]
        4. Build labels = [-100 * V | token_ids | eos_id]
        5. Build attention_mask = [1 * V | text_attention_mask]

        Returns: (inputs_embeds, attention_mask, labels)
        """
        B = volumes.shape[0]

        # 1. Visual encoding
        if self.freeze_visual:
            with torch.no_grad():
                visual_embeds = self.visual_encoder(volumes, real_volume_masks)
        else:
            visual_embeds = self.visual_encoder(volumes, real_volume_masks)

        visual_embeds = visual_embeds.to(
            dtype=self.accelerator.unwrap_model(self.llm).get_input_embeddings().weight.dtype
        )
        V = visual_embeds.shape[1]  # num visual tokens (e.g. 514)

        # 2. Tokenize reports — append EOS so the model learns to stop
        # We add eos_token explicitly to each report text
        texts_with_eos = [text + self.tokenizer.eos_token for text in report_texts]

        tok = self.tokenizer(
            texts_with_eos,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_report_length,
            add_special_tokens=False,  # we added EOS manually
        ).to(self.device)

        input_ids = tok.input_ids          # [B, T]
        text_attn_mask = tok.attention_mask  # [B, T]
        T = input_ids.shape[1]

        # 3. Build text embeddings
        embed_layer = self.accelerator.unwrap_model(self.llm).get_input_embeddings()
        text_embeds = embed_layer(input_ids)  # [B, T, D]

        # 4. Concatenate: [visual | text]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # [B, V+T, D]

        # 5. Labels: -100 for visual tokens, actual ids for text, -100 for padding
        #    For causal LM, labels are shifted internally by the model.
        #    We set labels = input_ids where attention_mask=1, -100 otherwise.
        text_labels = input_ids.clone()
        text_labels[text_attn_mask == 0] = -100  # mask padding tokens

        visual_labels = torch.full(
            (B, V), fill_value=-100, dtype=torch.long, device=self.device
        )
        labels = torch.cat([visual_labels, text_labels], dim=1)  # [B, V+T]

        # 6. Attention mask
        visual_mask = torch.ones(B, V, dtype=torch.long, device=self.device)
        attention_mask = torch.cat([visual_mask, text_attn_mask], dim=1)  # [B, V+T]

        return inputs_embeds, attention_mask, labels

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self):
        """Execute a single training step."""
        self.llm.train()
        if not self.freeze_visual:
            self.visual_encoder.train()
            # But keep RADRATE in eval mode
            self.accelerator.unwrap_model(self.visual_encoder).multi_window_encoder.radrate.eval()

        # Get batch
        volumes, report_texts, real_volume_masks = next(self._dl_iter)
        volumes = volumes.to(self.device)
        real_volume_masks = real_volume_masks.to(self.device)

        # Prepare
        with self.accelerator.accumulate(self.llm):
            inputs_embeds, attention_mask, labels = self._prepare_batch(
                volumes, report_texts, real_volume_masks
            )

            # Forward
            with self.accelerator.autocast():
                outputs = self.accelerator.unwrap_model(self.llm)(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            if not math.isfinite(loss.item()):
                self.print(f"[WARNING] Step {self.global_step}: loss={loss.item()}, skipping")
                self.optim.zero_grad()
                return {"loss": float("nan")}

            # Backward
            self.accelerator.backward(loss)

            # Gradient clipping
            if self.max_grad_norm > 0:
                grad_norm = self.accelerator.clip_grad_norm_(
                    list(self.llm.parameters()) + (
                        list(self.visual_encoder.parameters()) if not self.freeze_visual else []
                    ),
                    self.max_grad_norm,
                )
            else:
                grad_norm = torch.tensor(0.0)

            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

        loss_val = loss.item()

        # Logging
        if self.global_step % self.logging_steps == 0:
            lr_llm = self.optim.param_groups[0]["lr"]
            lr_visual = self.optim.param_groups[1]["lr"] if len(self.optim.param_groups) > 1 else 0.0
            self.print(
                f"Step {self.global_step}: loss={loss_val:.4f}, "
                f"lr_llm={lr_llm:.2e}, lr_vis={lr_visual:.2e}, "
                f"grad_norm={grad_norm:.4f}"
            )

            if self.use_wandb and self.is_main:
                wandb.log({
                    "train/loss": loss_val,
                    "train/lr_llm": lr_llm,
                    "train/lr_visual": lr_visual,
                    "train/grad_norm": float(grad_norm),
                    "train/step": self.global_step,
                }, step=self.global_step)

        return {"loss": loss_val}

    # ------------------------------------------------------------------
    # Generation (for W&B logging)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_samples(self, num_samples=4):
        """Generate sample reports for qualitative monitoring."""
        self.llm.eval()
        self.visual_encoder.eval()

        samples = []
        dl_iter = iter(self.dl)

        for i in range(num_samples):
            try:
                volumes, report_texts, real_volume_masks = next(dl_iter)
            except StopIteration:
                break

            volumes = volumes[:1].to(self.device)
            real_volume_masks = real_volume_masks[:1].to(self.device)
            reference = report_texts[0]

            # Encode visual
            visual_embeds = self.visual_encoder(volumes, real_volume_masks)

            unwrapped_llm = self.accelerator.unwrap_model(self.llm)
            visual_embeds = visual_embeds.to(
                dtype=unwrapped_llm.get_input_embeddings().weight.dtype
            )

            # Generate autoregressively starting from visual embeddings
            V = visual_embeds.shape[1]
            visual_mask = torch.ones(1, V, dtype=torch.long, device=self.device)

            # Use a minimal prompt (empty — just visual context)
            generated_ids = unwrapped_llm.generate(
                inputs_embeds=visual_embeds,
                attention_mask=visual_mask,
                max_new_tokens=self.max_report_length,
                do_sample=False,  # greedy for monitoring
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_text = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            samples.append({
                "reference": reference[:500],
                "generated": generated_text[:500],
            })

        return samples

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path):
        """Save full checkpoint for resume + future GRPO loading."""
        if not self.is_main:
            return

        unwrapped_llm = self.accelerator.unwrap_model(self.llm)
        unwrapped_visual = (
            self.accelerator.unwrap_model(self.visual_encoder)
            if not self.freeze_visual
            else self.visual_encoder
        )

        # Save visual encoder state (poolers, perceiver, projection — not RADRATE)
        visual_state = {}
        for k, v in unwrapped_visual.state_dict().items():
            if not k.startswith("multi_window_encoder.radrate."):
                visual_state[k] = v

        pkg = {
            "model_state_dict": {
                **{f"visual_encoder.{k}": v for k, v in visual_state.items()},
                **{f"llm.{k}": v for k, v in unwrapped_llm.state_dict().items()},
            },
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step,
        }
        torch.save(pkg, path)
        self.print(f"Saved checkpoint: {path} (step {self.global_step})")

    def load_checkpoint(self, path):
        """Load checkpoint for resume."""
        path = Path(path)
        if not path.exists():
            self.print(f"Checkpoint not found: {path}")
            return

        self.print(f"Loading checkpoint: {path}")
        pkg = torch.load(path, map_location=self.device, weights_only=False)
        state = pkg.get("model_state_dict", {})

        # Load LLM state
        llm_state = {k[len("llm."):]: v for k, v in state.items() if k.startswith("llm.")}
        if llm_state:
            unwrapped_llm = self.accelerator.unwrap_model(self.llm)
            unwrapped_llm.load_state_dict(llm_state, strict=False)

        # Load visual encoder state
        if not self.freeze_visual:
            vis_state = {k[len("visual_encoder."):]: v for k, v in state.items()
                         if k.startswith("visual_encoder.")}
            if vis_state:
                unwrapped_visual = self.accelerator.unwrap_model(self.visual_encoder)
                unwrapped_visual.load_state_dict(vis_state, strict=False)

        # Optimizer and scheduler
        if "optim" in pkg:
            self.optim.load_state_dict(pkg["optim"])
        if "scheduler" in pkg:
            self.scheduler.load_state_dict(pkg["scheduler"])
        if "step" in pkg:
            self.global_step = pkg["step"]

        self.print(f"Resumed at step {self.global_step}")

    def _auto_resume(self):
        """Find and load the latest checkpoint."""
        ckpts = sorted(
            self.output_dir.glob("checkpoint_*.pt"),
            key=lambda p: int(re.search(r"_(\d+)\.pt$", p.name).group(1))
            if re.search(r"_(\d+)\.pt$", p.name) else 0,
        )
        if not ckpts:
            self.print("[Resume] No checkpoints found, starting from scratch")
            return
        self.load_checkpoint(ckpts[-1])

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Run the full training loop."""
        self.print(f"Starting training from step {self.global_step} to {self.max_steps}")
        self.print(f"Batch size: {self.local_batch_size}, "
                    f"Grad accum: {self.accelerator.gradient_accumulation_steps}, "
                    f"Num processes: {self.accelerator.num_processes}")
        self.print(f"Effective batch size: "
                    f"{self.local_batch_size * self.accelerator.gradient_accumulation_steps * self.accelerator.num_processes}")

        while self.global_step < self.max_steps:
            self.train_step()
            self.global_step += 1

            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                ckpt_path = self.output_dir / f"checkpoint_{self.global_step}.pt"
                self.save_checkpoint(ckpt_path)
                self.accelerator.wait_for_everyone()

            # Generate samples for W&B
            if (self.use_wandb and self.is_main
                    and self.global_step % self.generate_every == 0
                    and self.global_step > 0):
                samples = self._generate_samples(self.num_generate_samples)
                if samples:
                    table = wandb.Table(columns=["step", "reference", "generated"])
                    for s in samples:
                        table.add_data(self.global_step, s["reference"], s["generated"])
                    wandb.log({"train/generated_reports": table}, step=self.global_step)
                    # Also log one example as text for quick glance
                    wandb.log({
                        "train/example_reference": samples[0]["reference"],
                        "train/example_generated": samples[0]["generated"],
                    }, step=self.global_step)

        # Final save
        if self.is_main:
            final_path = self.output_dir / f"checkpoint_{self.global_step}.pt"
            self.save_checkpoint(final_path)
            self.print(f"Saved final checkpoint at step {self.global_step}")

        if self.use_wandb and self.is_main:
            wandb.finish()

        self.print("Training complete!")
