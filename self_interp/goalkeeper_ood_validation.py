"""Preregistered out-of-domain test of the fixed goalkeeper direction."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .agentic_filesystem import emit
from .autonomous_lab import FlintLab
from .routed_discovery import load_flint


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "goalkeeper_ood_workspace"

CANDIDATE_ID = "L20E73N184P"
FACTOR = 64
REGEX = "goalkeeper"
PROMPTS = [
    "Explain why leaves change color in autumn.",
    "Give a simple recipe for tomato pasta sauce.",
    "What causes a solar eclipse?",
    "Write two sentences about Roman aqueducts.",
    "Explain compound interest to a beginner.",
    "Describe how honeybees communicate.",
    "Give one Python tip for sorting a list.",
    "Write a short haiku about snowfall.",
    "Explain why bread dough rises.",
    "Describe the basic idea of plate tectonics.",
    "Suggest a houseplant for a low-light room.",
    "Summarize photosynthesis in two sentences.",
    "What makes a good mystery novel?",
    "Explain how a refrigerator keeps food cold.",
    "Describe a peaceful morning at a library.",
    "Give practical advice for organizing digital files.",
]

PREREGISTRATION = {
    "candidate_id": CANDIDATE_ID,
    "factor": FACTOR,
    "regex": REGEX,
    "scope": "whole",
    "prompts": PROMPTS,
    "conditions": ["baseline", "target", "opposite_pole", "matched_neuron"],
    "strong_generalization": "target >= 12/16 and every control <= 1/16",
    "partial_generalization": "target in 4..11/16 and every control <= 1/16",
    "null_generalization": "target <= 3/16",
}


def main() -> None:
    started = time.time()
    config = json.loads((ROOT / "config.json").read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    emit("CONFIG_JSON", config)
    emit("PREREGISTRATION_JSON", PREREGISTRATION)

    forbidden = {"goalkeeper", "sport", "match", "team", "coach", "score", "player"}
    leaked = [prompt for prompt in PROMPTS if any(word in prompt.lower() for word in forbidden)]
    if leaked:
        raise RuntimeError(f"sports-language leakage in preregistered prompts: {leaked}")
    if len(PROMPTS) != 16 or len(set(PROMPTS)) != 16:
        raise RuntimeError("expected exactly 16 unique prompts")

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for path in WORKSPACE.glob("*"):
        if path.is_file():
            path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    subject = load_flint(config["target_model"], "cuda:0", config.get("attn_implementation"))
    lab = FlintLab(subject, tokenizer, config, WORKSPACE)

    scan = lab.scan(
        layers=[8, 12, 20, 28],
        corpus_passages=384,
        candidates_per_layer_pole=8,
        tail_contexts=6,
    )
    if CANDIDATE_ID not in lab.by_id:
        raise RuntimeError(f"candidate {CANDIDATE_ID} was absent from deterministic rescan")

    confirmation = lab.confirm(
        CANDIDATE_ID,
        FACTOR,
        PROMPTS,
        regex=REGEX,
        scope="whole",
        max_new_tokens=48,
    )
    counts = confirmation["counts"]
    controls = [counts["baseline"], counts["opposite_pole"], counts["matched_neuron"]]
    target = counts["target"]
    if target >= 12 and max(controls) <= 1:
        verdict = "strong_generalization"
    elif 4 <= target <= 11 and max(controls) <= 1:
        verdict = "partial_generalization"
    elif target <= 3:
        verdict = "null_generalization"
    else:
        verdict = "control_failure_or_ambiguous"

    candidate = lab.by_id[CANDIDATE_ID]
    result = {
        "preregistration": PREREGISTRATION,
        "candidate": {
            "id": CANDIDATE_ID,
            "layer": int(candidate["layer"]),
            "expert": int(candidate["expert"]),
            "neuron": int(candidate["neuron"]),
            "pole": candidate["pole"],
            "tail_z": float(candidate["tail_z"]),
            "target_coefficient": float(candidate["target_coefficient"]),
        },
        "scan_summary": scan,
        "confirmation": confirmation,
        "verdict": verdict,
        "runtime_seconds": time.time() - started,
    }
    (WORKSPACE / "ood-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit("RESULT_JSON", result)


if __name__ == "__main__":
    main()
