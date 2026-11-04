"""Independent fixed replication of the autonomous goalkeeper-direction finding."""

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
WORKSPACE = ROOT / "goalkeeper_validation_workspace"

PROMPTS_A = [
    "The final shot was stopped by the last defender.",
    "The team won the match in overtime.",
    "The stadium erupted after the winning goal.",
    "The coach gave a halftime speech.",
    "The player made a key pass.",
    "The referee blew the final whistle.",
    "The crowd chanted during the match.",
    "The match ended in a tie.",
]

PROMPTS_B = [
    "The final whistle ended the match.",
    "The team celebrated after the decisive goal.",
    "The stadium went silent after the save.",
    "The coach drew up a new play.",
    "The player checked the scoreboard.",
    "The referee reviewed the incident.",
    "The fans left the stadium early.",
    "The match was postponed due to weather.",
]


def main() -> None:
    started = time.time()
    config = json.loads((ROOT / "config.json").read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    emit("CONFIG_JSON", config)

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
    candidate_id = "L20E73N184P"
    if candidate_id not in lab.by_id:
        raise RuntimeError(f"candidate {candidate_id} was absent from deterministic rescan")

    first = lab.confirm(
        candidate_id,
        64,
        PROMPTS_A,
        regex="goalkeeper",
        scope="whole",
        max_new_tokens=48,
    )
    second = lab.confirm(
        candidate_id,
        64,
        PROMPTS_B,
        regex="goalkeeper",
        scope="whole",
        max_new_tokens=48,
    )
    candidate = lab.by_id[candidate_id]
    result = {
        "candidate": {
            "id": candidate_id,
            "layer": int(candidate["layer"]),
            "expert": int(candidate["expert"]),
            "neuron": int(candidate["neuron"]),
            "pole": candidate["pole"],
            "tail_z": float(candidate["tail_z"]),
            "target_coefficient": float(candidate["target_coefficient"]),
            "contexts": candidate["examples"],
        },
        "scan_summary": scan,
        "confirmations": [first, second],
        "runtime_seconds": time.time() - started,
    }
    (WORKSPACE / "replication-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit("RESULT_JSON", result)


if __name__ == "__main__":
    main()
