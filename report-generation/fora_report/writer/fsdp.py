"""FSDP2 (HSDP) helpers for training the report writer (Gemma-4-31B) (Methods, "Report generation").

Layout: a 2-D DeviceMesh ("replicate", "shard") = (nodes, GPUs per node). Every Gemma decoder layer is
its own ``fully_shard`` unit; the root (report generator: embeddings/lm-head (tied), final norm,
resampler, projections, finding projector, ...) is one more unit. Frozen bf16 base weights are sharded
like the trainable LoRA/connector tensors; only tensors with gradients are reduce-scattered (shard
dim) and all-reduced (replicate dim) -> the same mean gradient as DDP over the world.

Checkpoints use a plain format: ``trainable_model`` = {name: full CPU tensor} and ``optimizer`` = a
plain ``torch.optim`` state dict with full (unsharded) state tensors and integer parameter ids in
``model.named_parameters()`` order, so inference and DPO read it without FSDP.

Every function that gathers (``full_tensor``) or scatters is called by all ranks in the same order.
"""
from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist
from torch import nn


def fsdp_requested(llm_family: str, flag: str = "auto") -> bool:
    """``flag`` 1/0 forces; ``auto`` (default) enables FSDP for Gemma-4."""
    value = str(flag).strip().lower()
    if value in {"1", "true", "on"}:
        return True
    if value in {"0", "false", "off"}:
        return False
    if value in {"auto", ""}:
        return llm_family == "gemma4"
    raise ValueError(f"fsdp flag must be 0, 1 or auto, got {value!r}")


def is_dtensor(value) -> bool:
    from torch.distributed.tensor import DTensor
    return isinstance(value, DTensor)


def build_mesh(device_type: str, world_size: int, local_world: int):
    from torch.distributed.device_mesh import init_device_mesh
    if local_world <= 0 or world_size % local_world:
        raise ValueError(f"world={world_size} not divisible by local_world={local_world}")
    return init_device_mesh(device_type, (world_size // local_world, local_world),
                            mesh_dim_names=("replicate", "shard"))


def train_mode_for_checkpointing(llm: nn.Module, lora_dropout_active: bool = False) -> dict:
    """Put the LLM in train mode so HF gradient checkpointing is actually active.

    ``from_pretrained`` returns the model in eval mode; HF's ``GradientCheckpointingLayer`` only
    checkpoints when ``self.training`` is true. LoRA dropout is set to p=0 unless ``lora_dropout_active``
    (the paper runs train without LoRA dropout).
    """
    llm.train()
    changed = 0
    if not lora_dropout_active:
        for name, module in llm.named_modules():
            if isinstance(module, nn.Dropout) and "lora_dropout" in name and module.p != 0.0:
                module.p = 0.0
                changed += 1
    other = [n for n, m in llm.named_modules()
             if isinstance(m, nn.Dropout) and "lora_dropout" not in n and m.p != 0.0]
    return {"lora_dropout_zeroed": changed, "other_active_dropout": other[:5]}


def shard_report_model(model: nn.Module, layers, mesh, *, reshard_after_forward: bool = True):
    """fully_shard each decoder layer, then the root report model (HSDP over ``mesh``)."""
    from torch.distributed.fsdp import fully_shard
    for layer in layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    return model


class FSDPSyncAdapter:
    """DDP-like wrapper: ``adapter(...)`` = model forward, ``no_sync()`` = no reduction.

    Gradient accumulation: in ``no_sync`` micro-steps FSDP keeps unsharded gradients locally
    (set_requires_gradient_sync(False)); the boundary micro-step reduces the accumulated sum (the mean over the
    world, as DDP no_sync accumulation).
    """

    def __init__(self, module: nn.Module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextlib.contextmanager
    def no_sync(self):
        self.module.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            self.module.set_requires_gradient_sync(True)


# ------------------------------------------------------------------------------------ state
def full_trainable_state(module: nn.Module, main: bool) -> dict:
    """{name: full CPU tensor} of trainable parameters (collective for DTensors; result on ``main`` only)."""
    out = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach()
        if is_dtensor(value):
            value = value.full_tensor()
        if main:
            out[name] = value.cpu()
    return out


def _to_param_layout(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A full tensor -> the local layout of ``like`` (DTensor shard or plain tensor)."""
    if is_dtensor(like):
        from torch.distributed.tensor import distribute_tensor
        return distribute_tensor(value.to(device=like.device, dtype=like.dtype), like.device_mesh,
                                 like.placements, src_data_rank=None)
    return value.to(device=like.device)


def load_full_trainable_state(module: nn.Module, state: dict) -> None:
    """Inverse of :func:`full_trainable_state` (every rank holds the full CPU state)."""
    parameters = dict(module.named_parameters())
    missing = [n for n in state if n not in parameters]
    if missing:
        raise RuntimeError(f"Checkpoint has unknown trainable keys: {missing[:10]}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"{name}: checkpoint {tuple(value.shape)} model {tuple(parameter.shape)}")
            parameter.copy_(_to_param_layout(value, parameter))


def full_optimizer_state(optimizer: torch.optim.Optimizer, main: bool) -> dict | None:
    """Plain ``optimizer.state_dict()`` with every DTensor state gathered to a full CPU tensor."""
    sd = optimizer.state_dict()
    state = {}
    for index in sorted(sd["state"]):
        entry = {}
        for key in sorted(sd["state"][index]):
            value = sd["state"][index][key]
            if is_dtensor(value):
                value = value.full_tensor()
            if torch.is_tensor(value):
                value = value.detach().cpu() if main else None
            entry[key] = value
        state[index] = entry
    if not main:
        return None
    return {"state": state, "param_groups": sd["param_groups"]}


def load_full_optimizer_state(optimizer: torch.optim.Optimizer, package: dict) -> None:
    """Load a plain optimizer state dict into an optimizer over DTensor parameters."""
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    state = {}
    for index, entry in package["state"].items():
        parameter = parameters[int(index)]
        converted = {}
        for key, value in entry.items():
            if torch.is_tensor(value) and value.ndim > 0 and tuple(value.shape) == tuple(parameter.shape):
                value = _to_param_layout(value, parameter)
            converted[key] = value
        state[index] = converted
    optimizer.load_state_dict({"state": state, "param_groups": package["param_groups"]})


def trainable_checksum(module: nn.Module, device) -> torch.Tensor:
    """(sum, sum of squares) over full trainable tensors; identical on every rank (collective)."""
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach()
        if is_dtensor(value):
            value = value.full_tensor()
        value = value.float()
        total[0] += value.sum().double()
        total[1] += value.square().sum().double()
    return total


def clip_grad_norm(parameters, max_norm: float) -> torch.Tensor:
    """clip_grad_norm_ that also returns a plain tensor for DTensor gradients."""
    parameters = [p for p in parameters if p.grad is not None]
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if is_dtensor(norm):
        norm = norm.full_tensor()
    return norm


def describe_memory(device) -> str:
    if device.type != "cuda":
        return "cpu"
    return (f"alloc={torch.cuda.memory_allocated(device) / 2**30:.1f}GiB "
            f"max_alloc={torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB "
            f"reserved={torch.cuda.memory_reserved(device) / 2**30:.1f}GiB")
