"""Minimal public entrypoint for the autonomous study and two validations."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def resolve_attr(obj, path: str):
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def load_checkpoint(config: dict):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if config.get("attn_implementation"):
        kwargs["attn_implementation"] = config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(config["target_model"], **kwargs)
    model.eval()
    return model, tokenizer


def main() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    stage = config.get("study_stage")
    if stage == "flint_authored_neuron_study":
        from .autonomous_lab import main as stage_main
    elif stage == "goalkeeper_neuron_validation":
        from .goalkeeper_validation import main as stage_main
    elif stage == "goalkeeper_ood_validation":
        from .goalkeeper_ood_validation import main as stage_main
    else:
        raise ValueError(f"Unsupported public study stage: {stage!r}")
    stage_main()


if __name__ == "__main__":
    main()
