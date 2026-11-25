"""Predicted diagnostic plan built from frozen, calibrated MIL predictions (paper: "Diagnostic plan").

Each study's MIL logits are mapped through per-class Platt curves. Classes with calibrated probability at or above the
probable cutoff (calibrated p >= 0.5, never below the train Youden cutoff) form the confident tier; classes between
the possible cutoff (train Youden) and the probable cutoff form the possible tier. For CT the possible tier is capped
at 20 items ranked by calibrated probability. Each planned finding is written as
``<name> (calibrated probability p, Nx the base rate of X%; <evidence description>) <evidence tokens><image tokens>``
and the plan is inserted into the writer prompt (Supplementary Fig. 11; fixed text in ``plan_templates.txt`` next to this module).

The same builder serves the CT writer (``dataset_kind="ct"``) and the chest X-ray writer (``dataset_kind="cxr"``);
the two differ only in wording (X-ray: "Probable findings", location before probability, radiograph sentences)
and in the possible-tier cap (X-ray: uncapped).

Reference labels never enter :meth:`DiagnosticPlanBuilder.plan_ids`; they are used only by
:meth:`DiagnosticPlanBuilder.consistency_targets`, an output-side target vector.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

TEMPLATE_FILE = Path(__file__).resolve().parent / "plan_templates.txt"

# Gemma-4 single reserved tokens that the text pipeline never emits. Each finding-evidence token slot in the plan
# is FINDING_PLACEHOLDER (replaced by a projected evidence feature inside the writer); each zoomed evidence image
# slot is FINDING_IMAGE_PLACEHOLDER (replaced by a pooled Gemma-4 vision-tower token).
FINDING_PLACEHOLDER = "<|video|>"
FINDING_IMAGE_PLACEHOLDER = "<|audio|>"
FAMILY_FINDING_PLACEHOLDER = {"gemma4": "<|video|>"}

# Paper settings per dataset kind (CT: train-Platt calibration, cap 20 by calibrated p, six zoomed evidence images
# per finding, evidence descriptions; X-ray: uncapped, no images, no CT descriptions).
DEFAULTS = {
    "ct": dict(plan_possible_cap=20, plan_cap_rank="prob", plan_prob=True, fimg_tokens=6, evidence_descriptions=True),
    "cxr": dict(plan_possible_cap=0, plan_cap_rank="fold", plan_prob=True, fimg_tokens=0, evidence_descriptions=False),
}


def set_finding_placeholder(family: str) -> str:
    """Select the LLM family's evidence placeholder (only Gemma-4 is used by the final models)."""
    global FINDING_PLACEHOLDER
    FINDING_PLACEHOLDER = FAMILY_FINDING_PLACEHOLDER[str(family)]
    return FINDING_PLACEHOLDER


_FRAGMENT = re.compile(r"^\[\[([a-z0-9_]+)\]\]\n(.*?)\n\[\[/\1\]\]$", re.S | re.M)


def load_templates(path: str | Path | None = None) -> dict[str, str]:
    """Parse ``[[name]]\\n<text>\\n[[/name]]`` fragments from the plan template file."""
    text = Path(path or TEMPLATE_FILE).read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in _FRAGMENT.finditer(text)}


@dataclass(frozen=True)
class PlanSelection:
    """Thresholded plan entries plus the MIL class supplying each entry."""

    probability: torch.Tensor
    source_class_indices: torch.Tensor
    confident: tuple[int, ...]
    possible: tuple[int, ...]


def label_order_sha256(labels: Sequence[str]) -> str:
    payload = "\n".join(map(str, labels)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DiagnosticPlanBuilder:
    """Build text plans exclusively from frozen-MIL predictions.

    ``artifact_path`` is a calibration JSON written by :mod:`fora_report.plan.calibration_tiers`. Settings left at
    ``None`` take the paper value for ``dataset_kind`` (see ``DEFAULTS``).
    """

    def __init__(
        self,
        artifact_path: str | Path,
        labels: Sequence[str],
        tokenizer,
        mode: str = "all_classes",
        dataset_kind: str = "cxr",
        *,
        plan_possible_cap: int | None = None,
        plan_cap_rank: str | None = None,
        plan_prob: bool | None = None,
        fimg_tokens: int | None = None,
        evidence_descriptions: bool | str | None = None,
        templates: str | Path | None = None,
    ) -> None:
        if mode != "all_classes":
            raise ValueError("the final plans use mode='all_classes'")
        if dataset_kind not in DEFAULTS:
            raise ValueError("dataset kind must be ct or cxr")
        self.path = Path(artifact_path)
        self.artifact = json.loads(self.path.read_text())
        self.labels = list(map(str, labels))
        if self.artifact["labels"] != self.labels:
            raise ValueError("MIL calibration label order does not match the dataset")
        if self.artifact.get("label_order_sha256") != label_order_sha256(self.labels):
            raise ValueError("MIL calibration label-order checksum differs")
        if self.artifact.get("clinical_plan_constraints", {}).get("suppress_normal_with_abnormal_confident"):
            raise ValueError("clinical_plan_constraints are not supported (no released calibration uses them)")
        self.mode = mode
        self.dataset_kind = dataset_kind
        self.tokenizer = tokenizer
        defaults = DEFAULTS[dataset_kind]
        self.plan_possible_cap = int(defaults["plan_possible_cap"] if plan_possible_cap is None else plan_possible_cap)
        self.plan_cap_rank = str(defaults["plan_cap_rank"] if plan_cap_rank is None else plan_cap_rank)
        if self.plan_cap_rank not in {"prob", "fold"}:
            raise ValueError("plan_cap_rank must be prob or fold")
        self.plan_prob = bool(defaults["plan_prob"] if plan_prob is None else plan_prob)
        self.fimg_tokens = int(defaults["fimg_tokens"] if fimg_tokens is None else fimg_tokens)
        self.evidence_descriptions = bool(defaults["evidence_descriptions"] if evidence_descriptions is None else evidence_descriptions)
        self.templates = load_templates(templates)

        self.slope = torch.tensor(self.artifact["class_slope"], dtype=torch.float32)
        self.intercept = torch.tensor(self.artifact["class_intercept"], dtype=torch.float32)
        self.class_threshold = torch.tensor(self.artifact["class_plan_threshold"], dtype=torch.float32)
        self.class_possible_threshold = torch.tensor(
            self.artifact.get("class_possible_threshold", self.artifact["class_plan_threshold"]),
            dtype=torch.float32,
        )
        self.class_names = list(self.artifact["canonical_plan_labels"])
        # Base rate for "Nx the base rate of X%": (positives + 0.5) / (studies + 1) of the Platt split. The keys
        # class_val_positive / n_calibration_studies hold the TRAIN counts.
        npos = self.artifact.get("class_val_positive")
        ncal = float(self.artifact.get("n_calibration_studies") or 0)
        self.class_prevalence = (
            torch.tensor([(float(k) + 0.5) / (ncal + 1.0) for k in npos], dtype=torch.float32)
            if npos is not None and ncal > 0 else None
        )
        self.names = self.class_names

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        slope = self.slope.to(logits.device)
        intercept = self.intercept.to(logits.device)
        return torch.sigmoid(logits.float() * slope.unsqueeze(0) + intercept.unsqueeze(0))

    def plan_selection(self, logits: torch.Tensor) -> PlanSelection:
        probability = self.probabilities(logits).reshape(-1)
        source_class_indices = torch.arange(probability.numel(), dtype=torch.long, device=probability.device)
        threshold = self.class_threshold.to(probability.device)
        possible_threshold = self.class_possible_threshold.to(probability.device)
        confident = torch.nonzero(probability >= threshold, as_tuple=False).flatten()
        possible = torch.nonzero(
            (probability >= possible_threshold) & (probability < threshold), as_tuple=False
        ).flatten()
        confident_tuple = tuple(map(int, confident.tolist()))
        possible_tuple = tuple(map(int, possible.tolist()))
        # Cap of the POSSIBLE tier only (the confident tier is never capped): keep the K possible items with the
        # largest calibrated p (rank "prob") or fold change p / base rate (rank "fold"); ties -> higher p, then
        # lower class id; the kept items stay in class order.
        cap = self.plan_possible_cap
        if cap > 0 and len(possible_tuple) > cap:
            p = probability.detach().float().cpu()
            den = (self.class_prevalence if (self.plan_cap_rank == "fold" and self.class_prevalence is not None)
                   else torch.ones_like(p))
            ranked = sorted(
                possible_tuple,
                key=lambda i: (-float(p[i]) / max(float(den[i]), 1e-12), -float(p[i]), int(i)),
            )
            keep = set(ranked[:cap])
            possible_tuple = tuple(i for i in possible_tuple if i in keep)
        return PlanSelection(
            probability=probability,
            source_class_indices=source_class_indices,
            confident=confident_tuple,
            possible=possible_tuple,
        )

    def thresholds(self, index: int) -> tuple[float, float]:
        """Confident and possible thresholds (calibrated probability) of one class."""
        return float(self.class_threshold[index]), float(self.class_possible_threshold[index])

    def probability_text(self, index: int, p: float) -> str:
        """``0.42, 3.1x the base rate of 13.5%`` (base-rate part only when the calibration stores counts)."""
        ptxt = "<0.01" if p < 0.01 else ">0.99" if p > 0.99 else f"{p:.2f}"
        if self.class_prevalence is not None:
            prev = float(self.class_prevalence[int(index)])
            fold = p / max(prev, 1e-6)
            ftxt = f"{fold:.1f}" if fold < 10 else f"{fold:.0f}"
            prtxt = f"{100 * prev:.2f}%" if prev < 0.01 else f"{100 * prev:.1f}%"
            ptxt += f", {ftxt}x the base rate of {prtxt}"
        return ptxt

    def plan_content(
        self,
        logits: torch.Tensor,
        location_descriptions: Mapping[int, str] | None = None,
        finding_token_classes=None,
    ) -> str:
        """The user-turn text of the writer prompt (before the chat template).

        ``location_descriptions`` maps class index -> evidence description; key ``-1`` holds the per-study
        automatic measurement line. ``finding_token_classes`` is an iterable of classes that have evidence
        tokens, or a mapping class -> number of evidence tokens (pooled + focus tokens).
        """
        T = self.templates
        selection = self.plan_selection(logits)
        locations = dict(location_descriptions or {})
        study_line = locations.pop(-1, "")
        token_classes = frozenset(map(int, finding_token_classes)) if finding_token_classes else frozenset()
        token_counts = (
            {int(k): int(v) for k, v in finding_token_classes.items()}
            if isinstance(finding_token_classes, Mapping) else {}
        )
        calibrated = self.probabilities(logits).reshape(-1) if self.plan_prob else None

        def described(index: int) -> str:
            text = self.names[index]
            location = locations.get(index)
            if calibrated is not None:
                ptxt = self.probability_text(index, float(calibrated[int(index)]))
                if self.dataset_kind == "cxr":
                    location = (f"{location}; " if location else "") + f"calibrated probability {ptxt}"
                else:
                    location = f"calibrated probability {ptxt}" + (f"; {location}" if location else "")
            if location:
                text += f" ({location})"
            if index in token_classes:
                text += " " + FINDING_PLACEHOLDER * token_counts.get(int(index), 1)
                if self.fimg_tokens > 0:
                    text += FINDING_IMAGE_PLACEHOLDER * self.fimg_tokens
            return text

        confident_text = "; ".join(described(i) for i in selection.confident) if selection.confident else "none"
        possible_text = "; ".join(described(i) for i in selection.possible) if selection.possible else "none"
        image_name = "chest radiograph" if self.dataset_kind == "cxr" else "CT"
        content = T["findings_instruction"].format(image_name=image_name)
        if study_line:
            content += T["study_measurements"].format(study_line=study_line)
        content += T["plan_header"]
        content += T["confident_cxr" if self.dataset_kind == "cxr" else "confident_ct"].format(confident=confident_text)
        content += T["possible"].format(possible=possible_text)
        content += T["plan_footer"]
        if self.dataset_kind == "cxr":
            if locations or token_classes or calibrated is not None:
                if calibrated is not None and locations:
                    content += T["cxr_locations_probability"]
                elif calibrated is not None:
                    content += T["cxr_probability"]
                if token_classes:
                    content += T["cxr_evidence_token"]
        elif locations or token_classes:
            if self.evidence_descriptions:
                content += T["ct_descriptions_probability" if calibrated is not None else "ct_descriptions"]
            else:
                content += T["ct_attention_locations"]
            if token_classes and token_counts:
                content += T["ct_evidence_tokens_foci"]
                if self.fimg_tokens > 0:
                    content += T["ct_evidence_images"]
            elif token_classes:
                content += T["ct_evidence_token"]
        return content

    def plan_ids(
        self,
        logits: torch.Tensor,
        device: torch.device,
        location_descriptions: Mapping[int, str] | None = None,
        repair_draft: str | None = None,
        report_writer_mode: str = "full",
        writer_supported_plan: bool = False,
        hide_classifier_candidates: bool = False,
        finding_token_classes=None,
        location_style: str = "evidence",
    ) -> torch.Tensor:
        """Chat-templated prompt token ids for one study.

        The keyword arguments ``repair_draft``, ``report_writer_mode``, ``writer_supported_plan``,
        ``hide_classifier_candidates`` and ``location_style`` select the plan prompt; the paper writers use the defaults
        (no draft, full writer, no writer-supported plan, classifier candidates shown, evidence descriptions as location
        text) and other values are rejected.
        """
        if (repair_draft is not None or report_writer_mode != "full" or writer_supported_plan
                or hide_classifier_candidates or location_style != "evidence"):
            raise ValueError("only the paper plan prompt (full writer, evidence descriptions, no draft) is supported")
        content = self.plan_content(logits, location_descriptions, finding_token_classes)
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
        return torch.tensor(ids, dtype=torch.long, device=device)

    def planned_finding_order(
        self,
        logits: torch.Tensor,
        finding_token_classes,
        *,
        hide_classifier_candidates: bool = False,
    ) -> list[int]:
        """Class indices in the exact order their placeholders appear in :meth:`plan_ids`:
        confident findings first, then possible findings, each in class order, restricted to classes that have
        an evidence feature."""
        if hide_classifier_candidates or not finding_token_classes:
            return []
        token_classes = frozenset(map(int, finding_token_classes))
        selection = self.plan_selection(logits)
        return [int(i) for i in (*selection.confident, *selection.possible) if int(i) in token_classes]

    def consistency_targets(
        self,
        positive_class_indices: torch.Tensor,
        device: torch.device,
        *,
        negative_class_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-class target vector: 1 = asserted/uncertain, 0 = explicitly denied, -1 = unmentioned (masked).

        Without ``negative_class_indices`` every non-positive class is 0 (dense form used by the X-ray trainers).
        """
        if negative_class_indices is None:
            target = torch.zeros(len(self.labels), dtype=torch.float32, device=device)
        else:
            target = torch.full((len(self.labels),), -1.0, dtype=torch.float32, device=device)
            if negative_class_indices.numel():
                target[negative_class_indices.long()] = 0.0
        if positive_class_indices.numel():
            target[positive_class_indices.long()] = 1.0
        return target
