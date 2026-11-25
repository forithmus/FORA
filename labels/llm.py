"""vLLM chat wrapper and small helpers shared by every LLM pass of the labels stage.

One process per GPU (tensor parallel 1), greedy decoding (temperature 0, top-p 1), prefix caching on (every pass sends a
large byte-identical system prompt). Sharding over processes is ``rows[rank::world]``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

ChatMessages = list[dict[str, str]]

STAGE_DIR = Path(__file__).resolve().parent
GREEDY_TEMPERATURE = 0.0
GREEDY_TOP_P = 1.0
CONTEXT_LENGTH_SAFETY_TOKENS = 64
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_TOO_LONG = "too_long"


def stage_path(p: str | Path) -> Path:
    """Paths in configs are relative to the labels/ folder unless absolute."""
    p = Path(p)
    return p if p.is_absolute() else STAGE_DIR / p


def read_prompt(p: str | Path) -> str:
    return stage_path(p).read_text(encoding="utf-8")


def load_config(p: str | Path) -> dict:
    cfg = json.loads(stage_path(p).read_text(encoding="utf-8"))
    cfg.pop("_comment", None)
    return cfg


def rank_world(rank: int | None = None, world: int | None = None) -> tuple[int, int]:
    """Explicit --rank/--world, else SLURM_PROCID/SLURM_NTASKS, else 0/1."""
    r = rank if rank is not None else int(os.environ.get("SLURM_PROCID", 0))
    w = world if world is not None else int(os.environ.get("SLURM_NTASKS", 1))
    return r, w


class VLLMInferer:
    """Thin facade over ``vllm.LLM`` for chat-style batched generation."""

    def __init__(self, model_path: str, *, max_model_len: int, gpu_memory_utilization: float, seed: int, dtype: str,
                 tensor_parallel_size: int = 1, enable_prefix_caching: bool = True,
                 log_fn: Callable[[str], None] | None = None) -> None:
        from vllm import LLM

        self._log = log_fn or (lambda m: print(m, flush=True))
        self._seed = seed
        self.max_model_len = max_model_len
        t0 = time.time()
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, dtype=dtype, seed=seed,
                       enable_prefix_caching=enable_prefix_caching, limit_mm_per_prompt={"image": 0, "audio": 0})
        self.tokenizer = self.llm.get_tokenizer()
        self._log(f"model loaded in {time.time() - t0:.1f}s")

    def render(self, messages: ChatMessages) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_prompt_tokens(self, messages: ChatMessages) -> int:
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def prompt_fits(self, messages: ChatMessages, max_new_tokens: int, *, safety_tokens: int = 0) -> tuple[bool, int]:
        n = self.count_prompt_tokens(messages)
        return n <= self.max_model_len - max_new_tokens - safety_tokens, n

    def chat(self, messages_batch: Sequence[ChatMessages], sampling_params: Any) -> list[str]:
        if not messages_batch:
            return []
        outs = self.llm.generate([self.render(m) for m in messages_batch], sampling_params)
        return [o.outputs[0].text for o in outs]

    def make_sampling(self, *, temperature: float, max_tokens: int, top_p: float = 1.0, seed: int | None = None) -> Any:
        from vllm import SamplingParams

        return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                              seed=self._seed if seed is None else seed)


def inferer_from_config(model: str, cfg: dict, log_fn: Callable[[str], None]) -> VLLMInferer:
    return VLLMInferer(model, max_model_len=cfg["max_model_len"], gpu_memory_utilization=cfg["gpu_memory_utilization"],
                       seed=cfg["seed"], dtype=cfg["dtype"], tensor_parallel_size=1, log_fn=log_fn)


def vocab_lines(leaves: Sequence[dict]) -> str:
    """One ``- canonical — syn; syn`` line per leaf (the closed vocabulary block of every prompt)."""
    lines = []
    for lf in leaves:
        syn = (lf.get("syn") or "").strip()
        lines.append(f"- {lf['name']}" + (f" — {syn}" if syn else ""))
    return "\n".join(lines)


def parse_first_last_brace(raw: str):
    """JSON between the first '{' and the last '}' (recall and malignancy passes)."""
    s = raw.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(s[i:j + 1])
    except Exception:
        return None
