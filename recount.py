"""Independently recount exact-word goalkeeper matches from published rows."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATTERN = re.compile(r"\bgoalkeeper\b", re.IGNORECASE)


def recount(path: Path) -> dict[str, tuple[int, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    conditions = ("target", "baseline", "opposite_pole", "matched_neuron")
    return {
        condition: (
            sum(bool(PATTERN.search(row["output"])) for row in rows if row["condition"] == condition),
            sum(row["condition"] == condition for row in rows),
        )
        for condition in conditions
    }


def line(label: str, counts: dict[str, tuple[int, int]]) -> str:
    target_hits, target_total = counts["target"]
    control_hits = sum(counts[name][0] for name in counts if name != "target")
    control_total = sum(counts[name][1] for name in counts if name != "target")
    return f"{label}: target {target_hits}/{target_total}; controls {control_hits}/{control_total}"


fixed = recount(ROOT / "data" / "fixed-replication-rows.json")
ood = recount(ROOT / "data" / "out-of-domain-rows.json")

print(line("fixed replication", fixed))
print(line("out-of-domain", ood))

assert fixed == {
    "target": (15, 16),
    "baseline": (0, 16),
    "opposite_pole": (0, 16),
    "matched_neuron": (0, 16),
}
assert ood == {
    "target": (16, 16),
    "baseline": (0, 16),
    "opposite_pole": (0, 16),
    "matched_neuron": (0, 16),
}
