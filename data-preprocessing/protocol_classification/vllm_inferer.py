"""Thin vLLM wrapper for chat-style batched inference (model loading, chat template, generation, sampling).

Assumes GPU isolation is handled outside the process (e.g. SLURM's
`--gpus-per-task=1` plus `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` exported
in the runner script), so this class itself does not touch
`CUDA_VISIBLE_DEVICES`.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence


# Chat message shape used across this codebase: a list of
# {"role": "system"|"user"|"assistant", "content": "..."} dicts.
ChatMessages = list[dict[str, str]]


class VLLMInferer:
    """Thin, task-agnostic facade over `vllm.LLM` for chat-style inference."""

    def __init__(
        self,
        model_path: str,
        *,
        max_model_len: int = 61440,
        gpu_memory_utilization: float = 0.85,
        seed: int = 42,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        from vllm import LLM  # local import: keeps unit tests vLLM-optional

        self._log = log_fn or (lambda msg: print(msg, flush=True))
        self._seed = seed
        self.max_model_len = max_model_len

        self._log(f"VLLMInferer: loading model from {model_path} ...")
        t0 = time.time()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            seed=seed,
            # Every request shares a long system prompt; prefix caching reuses its KV blocks
            # (it does not change outputs).
            enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt={"image": 0, "audio": 0},
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._log(f"VLLMInferer: model loaded in {time.time() - t0:.1f}s "
                  f"(prefix_caching={enable_prefix_caching})")

    # ── chat-template rendering ────────────────────────────────────────────
    def render(self, messages: ChatMessages) -> str:
        """Apply the model's chat template, with a generation prompt appended."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def count_prompt_tokens(self, messages: ChatMessages) -> int:
        """Token length of the rendered chat prompt (including generation prompt)."""
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)

    def max_prompt_tokens(
        self,
        max_new_tokens: int,
        *,
        safety_tokens: int = 0,
    ) -> int:
        """Maximum allowed prompt tokens for a single generate call."""
        return self.max_model_len - max_new_tokens - safety_tokens

    def prompt_fits(
        self,
        messages: ChatMessages,
        max_new_tokens: int,
        *,
        safety_tokens: int = 0,
    ) -> tuple[bool, int]:
        """Return (fits, prompt_token_count) for prompt + max_new_tokens."""
        prompt_tokens = self.count_prompt_tokens(messages)
        limit = self.max_prompt_tokens(max_new_tokens, safety_tokens=safety_tokens)
        return prompt_tokens <= limit, prompt_tokens

    # ── batched generation ─────────────────────────────────────────────────
    def chat(
        self,
        messages_batch: Sequence[ChatMessages],
        sampling_params: Any,
    ) -> list[str]:
        """Render each message list to a prompt string and run vLLM on the batch.

        Returns the generated text for the FIRST output of each request,
        in the same order as `messages_batch`.
        """
        if not messages_batch:
            return []
        prompts = [self.render(m) for m in messages_batch]
        outs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outs]

    # ── sampling factory ───────────────────────────────────────────────────
    def make_sampling(
        self,
        *,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> Any:
        """Build a deterministic `SamplingParams` with this inferer's seed."""
        from vllm import SamplingParams

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=self._seed if seed is None else seed,
        )
