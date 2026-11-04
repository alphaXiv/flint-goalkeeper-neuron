"""Filesystem-agent Flint investigates another Flint through fixed lab equipment.

The harness owns activation collection and interventions.  Researcher Flint owns
file navigation, candidate triage, causal-pattern discovery, and the final audit.
Every model response, tool call, and tool result is printed to the run log.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from json_repair import repair_json
from transformers import AutoTokenizer

from .discovery import generate_text
from .routed_discovery import (
    build_candidates,
    first_word,
    inject_neuron,
    load_diverse_corpus,
    load_flint,
    scan_routed_neurons,
)
from .run import resolve_attr


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "agent_workspace"

SCREEN_PROMPTS = [
    "Tell me one surprising fact.",
    "Choose any topic and teach me something interesting.",
    "Write a vivid two-sentence miniature encyclopedia entry.",
    "What should we talk about today? Answer with something concrete.",
]

CONFIRM_PROMPTS = [
    "Name an overlooked object and explain why it matters.",
    "Give a concise piece of practical advice.",
    "Describe a historical event in two sentences.",
    "Write the opening of a friendly email.",
    "Explain a scientific idea to a curious teenager.",
    "Invent a title and premise for a short story.",
    "Describe a meal that someone might remember.",
    "Offer one observation about how cities work.",
    "Introduce an animal that deserves more attention.",
    "Complete this thought naturally: The most important detail is",
    "Give an example of a useful everyday habit.",
    "Write two sentences about any craft or skill.",
]


def emit(marker: str, payload: dict) -> None:
    print(marker + " " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class WorkspaceTools:
    """Small auditable filesystem interface; no hidden analytical helpers."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("path escapes workspace")
        return path

    def call(self, name: str, args: dict) -> str:
        if name == "list_files":
            rows = []
            for path in sorted(self.root.glob("**/*")):
                if path.is_file():
                    rows.append({
                        "path": str(path.relative_to(self.root)),
                        "bytes": path.stat().st_size,
                        "lines": sum(1 for _ in path.open(encoding="utf-8")),
                    })
            return json.dumps(rows, ensure_ascii=False)
        if name == "read_lines":
            path = self._path(str(args["path"]))
            start = max(1, int(args.get("start", 1)))
            end = min(start + 39, int(args.get("end", start + 19)))
            lines = path.read_text(encoding="utf-8").splitlines()
            return "\n".join(f"{index}: {lines[index - 1]}" for index in range(start, min(end, len(lines)) + 1))
        if name == "grep":
            path = self._path(str(args["path"]))
            pattern = str(args["pattern"])
            if len(pattern) > 160:
                raise ValueError("pattern too long")
            regex = re.compile(pattern, re.IGNORECASE)
            matches = []
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{index}: {line}")
                if len(matches) >= min(40, int(args.get("max_results", 20))):
                    break
            return "\n".join(matches) if matches else "NO MATCHES"
        if name == "write_note":
            path = self._path(str(args.get("path", "research_notes.md")))
            if path.suffix not in {".md", ".txt", ".json"}:
                raise ValueError("notes must be md, txt, or json")
            content = str(args["content"])
            if len(content) > 6000:
                raise ValueError("note too long")
            mode = "a" if bool(args.get("append", True)) else "w"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(content + ("\n" if not content.endswith("\n") else ""))
            return f"WROTE {len(content)} characters to {path.name}"
        raise ValueError(f"unknown tool {name}")


def parse_agent_action(text: str) -> tuple[str, dict]:
    # Flint emits visible <think> blocks even when thinking is disabled.  Those
    # blocks may quote the protocol verbatim, so they must never be executable.
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in clean:  # incomplete reasoning was truncated before an action
        raise ValueError("reasoning block was not closed before the action")
    # Normalize harmless wrapper duplication seen from the hosted checkpoint.
    clean = re.sub(r"(?:<tool>\s*){2,}", "<tool>", clean)
    clean = re.sub(r"(?:<final>\s*){2,}", "<final>", clean)
    clean = re.sub(r"<(tool|final)>\s*<(?=\{)", r"<\1>", clean)
    actions = list(re.finditer(r"<(tool|final)>\s*(\{.*?\})\s*</\1>", clean, re.DOTALL))
    if actions:
        # Explanatory prose sometimes quotes earlier attempted actions before
        # ending with the single action Flint intends to execute. Execute only
        # the last complete action, never more than one.
        match = actions[-1]
        return match.group(1), json.loads(repair_json(match.group(2)))
    # Hosted Flint occasionally emits a complete JSON action after a single
    # opening marker but omits only the redundant XML closing tag. Accept that
    # narrow case while still rejecting ambiguous/multiple actions.
    markers = list(re.finditer(r"<(tool|final)>\s*", clean))
    if len(markers) == 1:
        marker = markers[0]
        tail = clean[marker.end():].lstrip()
        if tail.startswith("{"):
            try:
                payload, _ = json.JSONDecoder().raw_decode(tail)
                if isinstance(payload, dict):
                    return marker.group(1), payload
            except json.JSONDecodeError:
                pass
    # Flint also sometimes preserves the closing action tag while dropping the
    # opener inside its <commands> wrapper. Recover one unambiguous JSON object
    # immediately preceding a single closing action tag.
    closers = list(re.finditer(r"</(tool|final)>", clean))
    if not markers and len(closers) == 1:
        prefix = clean[:closers[0].start()]
        for start in (match.start() for match in re.finditer(r"\{", prefix)):
            try:
                payload, consumed = json.JSONDecoder().raw_decode(prefix[start:])
            except json.JSONDecodeError:
                continue
            remainder = prefix[start + consumed:].strip()
            if isinstance(payload, dict) and remainder in {"", ">"}:
                return closers[0].group(1), payload
    # Finally accept a single bare action object. Some hosted completions drop
    # all XML markers while preserving the exact JSON tool/final schema.
    bare_actions: list[dict] = []
    for start in (match.start() for match in re.finditer(r"\{", clean)):
        try:
            payload, _ = json.JSONDecoder().raw_decode(clean[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if "name" in payload and "args" in payload:
            bare_actions.append(payload)
    if bare_actions:
        payload = bare_actions[-1]
        return "tool", payload
    raise ValueError("response must contain one <tool> JSON </tool> or <final> JSON </final>")


def model_turn(endpoint: str, token: str, messages: list[dict], max_new_tokens: int) -> str:
    """Call hosted Researcher Flint with the complete agent history."""
    payload = json.dumps({
        "model": "RecursiveSI/flint",
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "stream": False,
        "reasoning_effort": "minimal",
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Researcher endpoint returned HTTP {exc.code}: {detail[:2000]}") from exc
    message = result["choices"][0]["message"]
    content = message.get("content")
    if content:
        return str(content).strip()
    # Some Flint-serving stacks place the entire completion in
    # ``reasoning_content`` when visible content is empty.  Preserve it as the
    # model's response rather than silently turning a valid action into an
    # empty formatting failure.
    return str(message.get("reasoning_content") or "").strip()


def run_agent(
    stage: str,
    endpoint: str,
    token: str,
    mission: str,
    validator,
    config: dict,
) -> dict:
    tools = WorkspaceTools(WORKSPACE)
    system = """You are Researcher Flint investigating a separate subject Flint checkpoint.
You work autonomously inside a persistent filesystem workspace. Inspect evidence rather than
guessing from filenames. Available tools:
- list_files: {"name":"list_files","args":{}}
- read_lines: {"name":"read_lines","args":{"path":"file","start":1,"end":20}}
- grep: {"name":"grep","args":{"path":"file","pattern":"regex","max_results":20}}
- write_note: {"name":"write_note","args":{"path":"research_notes.md","content":"...","append":true}}
Reply with exactly one tool call as <tool>{JSON}</tool>, or finish as <final>{JSON}</final>.
Use multiple tool calls as needed. Do not claim evidence you did not inspect."""
    system += "\nYour complete conversation history is retained. Use research_notes.md to organize evidence and comparisons."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": mission}]
    corrections = 0
    for step in range(1, int(config["agent_max_steps"]) + 1):
        raw = model_turn(endpoint, token, messages, int(config["agent_max_new_tokens"]))
        emit("AGENT_TRACE_JSON", {"stage": stage, "step": step, "kind": "model_response", "content": raw})
        messages.append({"role": "assistant", "content": raw})
        try:
            kind, payload = parse_agent_action(raw)
            if kind == "final":
                validated = validator(payload)
                emit("AGENT_FINAL_JSON", {"stage": stage, "step": step, "result": validated})
                return validated
            tool_name = str(payload["name"])
            tool_args = payload.get("args", {})
            emit("AGENT_TRACE_JSON", {"stage": stage, "step": step, "kind": "tool_call", "name": tool_name, "args": tool_args})
            result = tools.call(tool_name, tool_args)
            emit("AGENT_TRACE_JSON", {"stage": stage, "step": step, "kind": "tool_result", "name": tool_name, "content": result})
            messages.append({"role": "user", "content": f"TOOL RESULT ({tool_name}):\n{result}"})
        except Exception as exc:
            corrections += 1
            if corrections > int(config["agent_max_corrections"]):
                raise RuntimeError(f"Researcher Flint exceeded correction budget in {stage}: {exc}") from exc
            error = f"Invalid action/result: {exc}. Inspect files further or emit a valid schema. No fallback will choose for you."
            emit("AGENT_TRACE_JSON", {"stage": stage, "step": step, "kind": "correction", "content": error})
            messages.append({"role": "user", "content": error})
    raise RuntimeError(f"Researcher Flint did not finish {stage} within the step budget")


def validate_shortlist(payload: dict, allowed: set[str], count: int) -> dict:
    ids = payload.get("selected_ids")
    if not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count or not set(ids) <= allowed:
        raise ValueError(f"selected_ids must contain exactly {count} distinct catalog IDs")
    if not str(payload.get("analysis", "")).strip():
        raise ValueError("analysis is required")
    return {"selected_ids": ids, "analysis": str(payload["analysis"]), "candidate_notes": payload.get("candidate_notes", {})}


def validate_patterns(payload: dict, allowed_ids: set[str], factors: set[float], count: int) -> dict:
    finalists = payload.get("finalists")
    if not isinstance(finalists, list) or len(finalists) != count:
        raise ValueError(f"finalists must contain exactly {count} hypotheses")
    seen = set()
    clean = []
    for row in finalists:
        candidate_id = str(row.get("candidate_id", ""))
        factor = float(row.get("factor", -1))
        scope = str(row.get("scope", ""))
        regex = str(row.get("regex", ""))
        if candidate_id not in allowed_ids or candidate_id in seen:
            raise ValueError("finalist IDs must be distinct screened candidates")
        if factor not in factors or scope not in {"start", "whole"} or not regex or len(regex) > 120:
            raise ValueError("each finalist needs a tested factor, start/whole scope, and short regex")
        try:
            re.compile(regex, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        seen.add(candidate_id)
        clean.append({
            "candidate_id": candidate_id,
            "factor": factor,
            "scope": scope,
            "regex": regex,
            "prediction": str(row.get("prediction", "")),
            "rationale": str(row.get("rationale", "")),
        })
    return {"finalists": clean, "analysis": str(payload.get("analysis", ""))}


def validate_audit(payload: dict) -> dict:
    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != 2:
        raise ValueError("audit requires exactly two verdicts")
    for row in verdicts:
        if row.get("verdict") not in {"supported", "failed"}:
            raise ValueError("each audit verdict must be supported or failed")
    if not str(payload.get("bottom_line", "")).strip():
        raise ValueError("bottom_line required")
    return payload


def causal_rows(subject, tokenizer, config: dict, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    baselines = []
    for prompt in SCREEN_PROMPTS:
        output = generate_text(subject, tokenizer, prompt, config["generation_max_new_tokens"])
        baselines.append({"condition": "baseline", "prompt": prompt, "output": output, "first_word": first_word(output)})
    rows = []
    for index, candidate in enumerate(candidates, 1):
        for factor in config["screen_factors"]:
            for prompt in SCREEN_PROMPTS:
                with inject_neuron(subject, config, candidate, float(factor)):
                    output = generate_text(subject, tokenizer, prompt, config["generation_max_new_tokens"])
                rows.append({
                    "candidate_id": candidate["id"], "factor": float(factor), "prompt": prompt,
                    "output": output, "first_word": first_word(output),
                })
        print(f"AGENTIC_SCREEN_PROGRESS candidates={index}/{len(candidates)}", flush=True)
    return baselines, rows


def regex_match(text: str, regex: str, scope: str) -> bool:
    if scope == "start":
        return re.match(regex, text, re.IGNORECASE) is not None
    return re.search(regex, text, re.IGNORECASE) is not None


def matched_control(subject, config: dict, target: dict, pool: list[dict]) -> dict:
    experts = resolve_attr(subject, config["experts_path"].format(layer=target["layer"]))
    target_norm = float(experts.down_proj[target["expert"], :, target["neuron"]].float().norm())
    same_layer = [x for x in pool if x["layer"] == target["layer"] and x["id"] != target["id"]]
    return min(same_layer, key=lambda x: abs(float(experts.down_proj[x["expert"], :, x["neuron"]].float().norm()) - target_norm))


def confirm_hypotheses(subject, tokenizer, config: dict, hypotheses: list[dict], candidates: list[dict]) -> list[dict]:
    by_id = {x["id"]: x for x in candidates}
    results = []
    for hypothesis in hypotheses:
        target = by_id[hypothesis["candidate_id"]]
        control = matched_control(subject, config, target, candidates)
        opposite = {**target, "reverse_sign": True}
        rows = []
        for prompt in CONFIRM_PROMPTS:
            for name, candidate, factor in (
                ("target", target, hypothesis["factor"]),
                ("opposite_sign", opposite, hypothesis["factor"]),
                ("matched_neuron", control, hypothesis["factor"]),
            ):
                with inject_neuron(subject, config, candidate, float(factor)):
                    output = generate_text(subject, tokenizer, prompt, config["generation_max_new_tokens"])
                rows.append({"condition": name, "prompt": prompt, "output": output, "match": regex_match(output, hypothesis["regex"], hypothesis["scope"])})
            baseline = generate_text(subject, tokenizer, prompt, config["generation_max_new_tokens"])
            rows.append({"condition": "baseline", "prompt": prompt, "output": baseline, "match": regex_match(baseline, hypothesis["regex"], hypothesis["scope"])})
        counts = {name: sum(x["match"] for x in rows if x["condition"] == name) for name in ("target", "baseline", "opposite_sign", "matched_neuron")}
        success = counts["target"] >= int(config["success_target"]) and all(counts[x] <= int(config["success_control"]) for x in ("baseline", "opposite_sign", "matched_neuron"))
        results.append({"hypothesis": hypothesis, "control_candidate_id": control["id"], "counts": counts, "success": success, "rows": rows})
    return results


def main() -> None:
    started = time.time()
    config = json.loads((ROOT / "config.json").read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    emit("CONFIG_JSON", config)
    if torch.cuda.device_count() < 1:
        raise RuntimeError("filesystem-agent study requires one visible GPU for subject Flint")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for path in WORKSPACE.glob("*"):
        if path.is_file():
            path.unlink()
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    researcher_endpoint = os.environ["FLINT_RESEARCHER_ENDPOINT"]
    researcher_token = os.environ["HF_TOKEN"]
    subject = load_flint(config["target_model"], "cuda:0", config.get("attn_implementation"))
    emit("TWO_FLINT_ARCHITECTURE_JSON", {
        "researcher": "hosted RecursiveSI/flint",
        "subject": "cuda:0",
        "separate_model_objects": True,
        "researcher_interface": "persistent filesystem agent",
        "researcher_context_policy": "complete unabridged message history",
    })

    layers = list(config["candidate_layers"])
    passages = load_diverse_corpus(config)
    scan = scan_routed_neurons(config, layers, passages, subject, tokenizer)
    candidates = build_candidates(config, scan, passages, tokenizer)
    catalog = []
    compact_catalog = []
    candidate_cards = []
    for candidate in candidates:
        record = {
            "id": candidate["id"], "layer": candidate["layer"], "expert": candidate["expert"],
            "neuron": candidate["neuron"], "pole": candidate["pole"], "tail_z": round(candidate["tail_z"], 4),
            "target_coefficient": round(candidate["target_coefficient"], 6),
            "contexts": [{"coefficient": x["coefficient"], "token": x["token"], "context": x["context"]} for x in candidate["examples"]],
        }
        catalog.append(record)
        compact_catalog.append(
            f"{record['id']} | layer={record['layer']} expert={record['expert']} neuron={record['neuron']} "
            f"pole={record['pole']} tail_z={record['tail_z']} | "
            + " || ".join(
                f"token={x['token']!r} context={x['context'][:180]}" for x in record["contexts"]
            )
        )
        candidate_cards.append(
            f"{record['id']} | L{record['layer']} E{record['expert']} N{record['neuron']} {record['pole']} "
            f"z={record['tail_z']} | tokens="
            + ", ".join(repr(x["token"]) for x in record["contexts"])
            + " | snippets="
            + " || ".join(re.sub(r"\s+", " ", x["context"])[:72] for x in record["contexts"])
        )
    write_jsonl(WORKSPACE / "candidates.jsonl", catalog)
    (WORKSPACE / "candidates_compact.txt").write_text("\n".join(compact_catalog) + "\n", encoding="utf-8")
    (WORKSPACE / "candidate_cards.txt").write_text("\n".join(candidate_cards) + "\n", encoding="utf-8")
    (WORKSPACE / "README.md").write_text(
        "This workspace contains raw activation-tail candidates from a subject Flint. "
        "Each JSONL record is one signed expert-neuron pole and its strongest natural contexts.\n",
        encoding="utf-8",
    )
    emit("WORKSPACE_FILE_JSON", {"path": "candidates.jsonl", "records": len(catalog), "compact_view": "candidates_compact.txt", "card_index": "candidate_cards.txt"})

    shortlist_mission = f"""Investigate this workspace as a research analyst. There are {len(catalog)} signed
expert-neuron poles. Use filesystem tools to inspect broadly, compare contexts, and choose exactly
{config['screen_candidates']} candidates worth causal intervention. Prefer narrow, recurring,
falsifiable patterns rather than impressive-sounding guesses. You are not being shown candidates
one by one: decide how to navigate the catalog. candidates.jsonl is the untouched evidence and
candidates_compact.txt is a lossless-ID compact reading view with truncated passages. Your full
agent conversation is retained; use research_notes.md to build a deliberate evidence table.
First inspect candidate_cards.txt in six non-overlapping pages: 1-10, 11-20, 21-30, 31-40,
41-50, and 51-60. After each page, append your interpretations and promising IDs to
research_notes.md. Do not reread README or a completed page. Use the larger files afterward only
to verify candidates you choose. The harness does not rank or interpret any card.
Final schema: {{"selected_ids":[exact IDs],"analysis":"selection reasoning", "candidate_notes":{{"ID":"note"}}}}"""
    allowed = {x["id"] for x in candidates}
    shortlist = run_agent(
        "candidate_triage", researcher_endpoint, researcher_token, shortlist_mission,
        lambda x: validate_shortlist(x, allowed, int(config["screen_candidates"])), config,
    )
    (WORKSPACE / "shortlist.json").write_text(json.dumps(shortlist, ensure_ascii=False, indent=2), encoding="utf-8")

    by_id = {x["id"]: x for x in candidates}
    baselines, screen = causal_rows(subject, tokenizer, config, [by_id[x] for x in shortlist["selected_ids"]])
    write_jsonl(WORKSPACE / "screen_baselines.jsonl", baselines)
    write_jsonl(WORKSPACE / "causal_results.jsonl", screen)
    grouped = []
    for candidate_id in shortlist["selected_ids"]:
        grouped.append({
            "candidate_id": candidate_id,
            "generations": [
                {"factor": row["factor"], "prompt": row["prompt"], "output": row["output"][:240]}
                for row in screen if row["candidate_id"] == candidate_id
            ],
        })
    write_jsonl(WORKSPACE / "causal_cards.jsonl", grouped)
    emit("WORKSPACE_FILE_JSON", {"path": "causal_results.jsonl", "records": len(screen)})

    pattern_mission = f"""The fixed harness has now executed the candidates you selected at factors
{config['screen_factors']} on four generic prompts. Raw generations are in causal_results.jsonl;
causal_cards.jsonl groups the same generations by candidate for efficient reading; baselines are
in screen_baselines.jsonl; your earlier notes, shortlist.json, and candidate catalog remain.
Independently investigate these files. Find exactly {config['confirm_candidates']} distinct, robust,
machine-testable causal effects worth held-out confirmation. Do not assume the effect is a prefix,
topic, or style: discover it from outputs. For each, choose an actually tested factor and specify a
Python regular expression plus scope `start` or `whole`. The fixed confirmation criterion will be
target >= {config['success_target']}/12 and every control <= {config['success_control']}/12.
Inspect every selected candidate's grouped causal card exactly once and maintain your own comparison
table in research_notes.md. The harness supplied no causal labels, scores, or pattern detector.
Final schema: {{"finalists":[{{"candidate_id":"ID","factor":80,"scope":"start|whole","regex":"...","prediction":"...","rationale":"..."}}],"analysis":"..."}}"""
    patterns = run_agent(
        "causal_pattern_discovery", researcher_endpoint, researcher_token, pattern_mission,
        lambda x: validate_patterns(x, set(shortlist["selected_ids"]), set(map(float, config["screen_factors"])), int(config["confirm_candidates"])), config,
    )

    confirmations = confirm_hypotheses(subject, tokenizer, config, patterns["finalists"], candidates)
    confirmation_rows = []
    for result in confirmations:
        for row in result["rows"]:
            confirmation_rows.append({**row, "candidate_id": result["hypothesis"]["candidate_id"], "regex": result["hypothesis"]["regex"], "scope": result["hypothesis"]["scope"]})
    write_jsonl(WORKSPACE / "confirmation_results.jsonl", confirmation_rows)
    confirmation_summary = [
        {"hypothesis": x["hypothesis"], "control_candidate_id": x["control_candidate_id"], "counts": x["counts"], "success": x["success"]}
        for x in confirmations
    ]
    (WORKSPACE / "confirmation_summary.json").write_text(json.dumps(confirmation_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("AGENTIC_CONFIRMATION_JSON", {"results": [{"hypothesis": x["hypothesis"], "control_candidate_id": x["control_candidate_id"], "counts": x["counts"], "success": x["success"]} for x in confirmations]})

    audit_mission = """The fixed harness completed held-out confirmation. Inspect confirmation_summary.json
and confirmation_results.jsonl. Audit both hypotheses against their target, baseline, opposite-sign,
and matched-neuron conditions. Explain what the evidence establishes and what it does not.
Final schema: {"verdicts":[{"candidate_id":"ID","verdict":"supported|failed","evidence":"...","limits":"..."}],"bottom_line":"..."}"""
    audit = run_agent("result_audit", researcher_endpoint, researcher_token, audit_mission, validate_audit, config)
    summary = {
        "study_stage": config["study_stage"], "target_model": config["target_model"],
        "candidate_records": len(candidates), "researcher_selected": shortlist["selected_ids"],
        "researcher_hypotheses": patterns["finalists"],
        "successful_candidates": [x["hypothesis"]["candidate_id"] for x in confirmations if x["success"]],
        "researcher_audit": audit, "no_candidate_fallback": True,
        "no_mechanical_pattern_miner": True, "runtime_seconds": time.time() - started,
    }
    emit("AGENTIC_SUMMARY_JSON", summary)
    emit("RESULT_JSON", {"config": config, "shortlist": shortlist, "patterns": patterns, "confirmations": confirmations, "audit": audit, "summary": summary})


if __name__ == "__main__":
    main()
