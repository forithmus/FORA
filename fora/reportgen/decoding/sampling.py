"""Decoding settings of the CT report writer (paper section "Grounded report generation", decoding paragraph).

The CT writer samples ONE report per study with nucleus sampling (temperature 0.8, top-p 0.9, top-k 0)
and a per-study seed, so that a study's report does not depend on how studies are placed on GPUs:

    seed = int.from_bytes(sha256("temperature_v1|<uid>|42")[:8], "big") % 2**31
    torch.manual_seed(seed)   # immediately before generate()

`greedy` is kept for the chest X-ray writer and for ablations; the CT paper rows use `t08`.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

SEED_TAG = "temperature_v1"
DEFAULT_SEED_INDEX = 42


def stable_seed(uid: str, index: int = DEFAULT_SEED_INDEX) -> int:
    """Per-study torch seed: first 8 bytes (big-endian) of sha256('temperature_v1|uid|index'), mod 2**31."""
    digest = hashlib.sha256((SEED_TAG + "|" + str(uid) + "|" + str(index)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)


@dataclass(frozen=True)
class SamplingConfig:
    """`do_sample=False` = greedy (temperature/top_p/top_k ignored)."""

    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    seed_index: int = DEFAULT_SEED_INDEX

    @classmethod
    def named(cls, name: str) -> "SamplingConfig":
        if name == "t08":
            return cls()
        if name == "greedy":
            return cls(do_sample=False)
        raise ValueError(f"unknown sampling preset {name!r} (t08 | greedy)")

    def generate_kwargs(self) -> dict:
        """Keyword arguments for HF `generate` (one beam; top_k 0 = no top-k filter)."""
        if not self.do_sample:
            return {"do_sample": False, "num_beams": 1}
        return {"do_sample": True, "temperature": float(self.temperature), "top_p": float(self.top_p),
                "top_k": int(self.top_k), "num_beams": 1}

    def seed_for(self, uid: str) -> int | None:
        return stable_seed(uid, self.seed_index) if self.do_sample else None

    def record(self, uid: str) -> dict:
        """`decode_settings` field of a generation record."""
        if not self.do_sample:
            return {"do_sample": False}
        out = asdict(self)
        out.pop("do_sample")
        out["per_study_seed"] = self.seed_for(uid)
        return out
