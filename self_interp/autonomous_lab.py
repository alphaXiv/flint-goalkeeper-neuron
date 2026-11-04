"""Researcher Flint writes and runs a neuron-level study on subject Flint.

The operator supplies only stable laboratory primitives: checkpoint loading,
activation scanning, one-token write-vector interventions, generation, and
immutable logging.  Researcher Flint owns the study program, search choices,
prompts, hypotheses, confirmation endpoint, analysis, and report.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from json_repair import repair_json
from transformers import AutoTokenizer

from .agentic_filesystem import emit, model_turn, parse_agent_action
from .discovery import generate_text
from .routed_discovery import (
    build_candidates,
    inject_neuron,
    load_diverse_corpus,
    load_flint,
    scan_routed_neurons,
)
from .run import resolve_attr


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "flint_research_workspace"


def jsonable_candidate(candidate: dict) -> dict:
    return {
        "id": candidate["id"],
        "layer": int(candidate["layer"]),
        "expert": int(candidate["expert"]),
        "neuron": int(candidate["neuron"]),
        "pole": candidate["pole"],
        "tail_z": round(float(candidate["tail_z"]), 5),
        "target_coefficient": round(float(candidate["target_coefficient"]), 7),
        "contexts": [
            {
                "coefficient": round(float(row["coefficient"]), 7),
                "token": row["token"],
                "context": row["context"],
            }
            for row in candidate["examples"]
        ],
    }


class FlintLab:
    """Stateful, deliberately narrow laboratory API exposed to Researcher Flint."""

    allowed_layers = {8, 12, 20, 28, 32, 36, 40}

    def __init__(self, subject, tokenizer, config: dict, workspace: Path):
        self.subject = subject
        self.tokenizer = tokenizer
        self.config = dict(config)
        self.workspace = workspace
        self.candidates: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.scan_count = 0
        self.call_count = 0
        self.confirmations: list[dict] = []
        self.intervened_candidate_ids: set[str] = set()
        self.high_dose_candidate_ids: set[str] = set()
        self.log_path = workspace / "lab_calls.jsonl"

    def _record(self, operation: str, args: dict, result: Any) -> Any:
        self.call_count += 1
        row = {
            "call_id": self.call_count,
            "operation": operation,
            "args": args,
            "result": result,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        emit("FLINT_LAB_CALL_JSON", row)
        return result

    def scan(
        self,
        layers: list[int],
        corpus_passages: int = 384,
        candidates_per_layer_pole: int = 8,
        tail_contexts: int = 6,
    ) -> dict:
        """Scan natural activations and write human-readable candidate cards."""
        clean_layers = [int(x) for x in layers]
        if not clean_layers or len(clean_layers) > 4 or not set(clean_layers) <= self.allowed_layers:
            raise ValueError(f"choose 1-4 distinct layers from {sorted(self.allowed_layers)}")
        if len(set(clean_layers)) != len(clean_layers):
            raise ValueError("layers must be distinct")
        if not 192 <= int(corpus_passages) <= 512:
            raise ValueError("corpus_passages must be between 192 and 512")
        if not 4 <= int(candidates_per_layer_pole) <= 12:
            raise ValueError("candidates_per_layer_pole must be between 4 and 12")
        if not 4 <= int(tail_contexts) <= 10:
            raise ValueError("tail_contexts must be between 4 and 10")
        if self.scan_count >= 2:
            raise ValueError("the resource budget permits at most two scans")

        local = dict(self.config)
        local["candidate_layers"] = clean_layers
        local["corpus_passages"] = int(corpus_passages)
        local["candidates_per_layer_pole"] = int(candidates_per_layer_pole)
        local["tail_contexts"] = int(tail_contexts)
        passages = load_diverse_corpus(local)
        scan = scan_routed_neurons(local, clean_layers, passages, self.subject, self.tokenizer)
        candidates = build_candidates(local, scan, passages, self.tokenizer)
        self.scan_count += 1
        # A second scan extends the available pool without silently replacing evidence.
        for candidate in candidates:
            if candidate["id"] not in self.by_id:
                self.candidates.append(candidate)
                self.by_id[candidate["id"]] = candidate

        records = [jsonable_candidate(x) for x in candidates]
        jsonl = self.workspace / f"activation_candidates_scan_{self.scan_count}.jsonl"
        cards = self.workspace / f"activation_cards_scan_{self.scan_count}.txt"
        with jsonl.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        cards.write_text(
            "\n".join(
                f"{row['id']} | z={row['tail_z']} | tokens="
                + ", ".join(repr(x["token"]) for x in row["contexts"])
                + " | contexts="
                + " || ".join(re.sub(r"\s+", " ", x["context"])[:110] for x in row["contexts"])
                for row in records
            )
            + "\n",
            encoding="utf-8",
        )
        result = {
            "scan": self.scan_count,
            "layers": clean_layers,
            "candidate_count": len(records),
            "jsonl": jsonl.name,
            "cards": cards.name,
        }
        return self._record("scan", {
            "layers": clean_layers,
            "corpus_passages": int(corpus_passages),
            "candidates_per_layer_pole": int(candidates_per_layer_pole),
            "tail_contexts": int(tail_contexts),
        }, result)

    def _prompts(self, prompts: list[str]) -> list[str]:
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= 16:
            raise ValueError("prompts must be a list of 1-16 strings")
        clean = [str(x).strip() for x in prompts]
        if any(not x or len(x) > 500 for x in clean) or len(set(clean)) != len(clean):
            raise ValueError("prompts must be distinct, nonempty, and at most 500 characters")
        return clean

    def _candidate(self, candidate_id: str) -> dict:
        if candidate_id not in self.by_id:
            raise ValueError(f"unknown candidate {candidate_id}; call scan first")
        return self.by_id[candidate_id]

    def baseline(self, prompts: list[str], max_new_tokens: int = 48) -> list[dict]:
        clean = self._prompts(prompts)
        limit = max(16, min(96, int(max_new_tokens)))
        rows = [
            {"prompt": prompt, "output": generate_text(self.subject, self.tokenizer, prompt, limit)}
            for prompt in clean
        ]
        return self._record("baseline", {"prompts": clean, "max_new_tokens": limit}, rows)

    def intervene(
        self,
        candidate_id: str,
        factor: float,
        prompts: list[str],
        reverse: bool = False,
        max_new_tokens: int = 48,
    ) -> list[dict]:
        candidate = self._candidate(str(candidate_id))
        self.intervened_candidate_ids.add(candidate["id"])
        clean = self._prompts(prompts)
        factor = float(factor)
        if not 0.25 <= abs(factor) <= 120:
            raise ValueError("absolute factor must be between 0.25 and 120")
        if abs(factor) >= 16:
            self.high_dose_candidate_ids.add(candidate["id"])
        limit = max(16, min(96, int(max_new_tokens)))
        chosen = {**candidate, "reverse_sign": bool(reverse)}
        rows = []
        for prompt in clean:
            with inject_neuron(self.subject, self.config, chosen, factor):
                output = generate_text(self.subject, self.tokenizer, prompt, limit)
            rows.append({"prompt": prompt, "output": output})
        return self._record("intervene", {
            "candidate_id": str(candidate_id), "factor": factor, "reverse": bool(reverse),
            "prompts": clean, "max_new_tokens": limit,
        }, rows)

    def matched_control(self, candidate_id: str) -> str:
        target = self._candidate(str(candidate_id))
        experts = resolve_attr(self.subject, self.config["experts_path"].format(layer=target["layer"]))
        target_norm = float(experts.down_proj[target["expert"], :, target["neuron"]].float().norm())
        pool = [x for x in self.candidates if x["layer"] == target["layer"] and x["id"] != target["id"]]
        if not pool:
            raise ValueError("no same-layer candidate is available for a matched control")
        control = min(
            pool,
            key=lambda x: abs(
                float(experts.down_proj[x["expert"], :, x["neuron"]].float().norm()) - target_norm
            ),
        )
        result = {"target": target["id"], "control": control["id"]}
        self._record("matched_control", {"candidate_id": target["id"]}, result)
        return control["id"]

    @staticmethod
    def _matches(text: str, regex: str, scope: str) -> bool:
        return (re.match if scope == "start" else re.search)(regex, text, re.IGNORECASE) is not None

    def confirm(
        self,
        candidate_id: str,
        factor: float,
        prompts: list[str],
        regex: str,
        scope: str = "whole",
        max_new_tokens: int = 48,
    ) -> dict:
        """Run fresh target, baseline, opposite-pole, and matched-neuron conditions."""
        clean = self._prompts(prompts)
        if len(clean) < 8:
            raise ValueError("confirmation requires at least 8 distinct prompts")
        if scope not in {"start", "whole"} or not regex or len(regex) > 160:
            raise ValueError("scope must be start/whole and regex must contain 1-160 characters")
        compiled = re.compile(regex, re.IGNORECASE)
        if any(compiled.search(prompt) for prompt in clean):
            raise ValueError(
                "confirmation prompts must not contain text matched by the scoring regex"
            )
        target = self._candidate(str(candidate_id))
        control_id = self.matched_control(str(candidate_id))
        control = self._candidate(control_id)
        factor = float(factor)
        if not 0.25 <= abs(factor) <= 120:
            raise ValueError("absolute factor must be between 0.25 and 120")
        limit = max(16, min(96, int(max_new_tokens)))
        rows = []
        conditions = [
            ("target", target),
            ("opposite_pole", {**target, "reverse_sign": True}),
            ("matched_neuron", control),
        ]
        for prompt in clean:
            baseline = generate_text(self.subject, self.tokenizer, prompt, limit)
            rows.append({"condition": "baseline", "prompt": prompt, "output": baseline,
                         "match": self._matches(baseline, regex, scope)})
            for name, candidate in conditions:
                with inject_neuron(self.subject, self.config, candidate, factor):
                    output = generate_text(self.subject, self.tokenizer, prompt, limit)
                rows.append({"condition": name, "prompt": prompt, "output": output,
                             "match": self._matches(output, regex, scope)})
        counts = {
            name: sum(int(row["match"]) for row in rows if row["condition"] == name)
            for name in ("target", "baseline", "opposite_pole", "matched_neuron")
        }
        result = {
            "candidate_id": target["id"],
            "matched_control_id": control_id,
            "factor": factor,
            "regex": regex,
            "scope": scope,
            "prompt_count": len(clean),
            "counts": counts,
            "rows": rows,
        }
        self.confirmations.append(result)
        return self._record("confirm", {
            "candidate_id": target["id"], "factor": factor, "prompts": clean,
            "regex": regex, "scope": scope, "max_new_tokens": limit,
        }, result)

    def public_summary(self) -> dict:
        return {
            "scans": self.scan_count,
            "lab_calls": self.call_count,
            "candidate_pool": len(self.candidates),
            "intervened_candidates": sorted(self.intervened_candidate_ids),
            "high_dose_candidates": sorted(self.high_dose_candidate_ids),
            "confirmations": [
                {key: value for key, value in row.items() if key != "rows"}
                for row in self.confirmations
            ],
        }


class ResearchWorkspace:
    """Auditable tools for Flint-authored code and notes."""

    def __init__(self, root: Path, lab: FlintLab):
        self.root = root.resolve()
        self.lab = lab

    def _path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("path escapes workspace")
        return path

    def list_files(self, _args: dict) -> str:
        rows = []
        for path in sorted(self.root.glob("**/*")):
            if path.is_file():
                rows.append({
                    "path": str(path.relative_to(self.root)),
                    "bytes": path.stat().st_size,
                    "lines": sum(1 for _ in path.open(encoding="utf-8")),
                })
        return json.dumps(rows, ensure_ascii=False)

    def read_lines(self, args: dict) -> str:
        path = self._path(str(args["path"]))
        start = max(1, int(args.get("start", 1)))
        end = min(start + 99, int(args.get("end", start + 49)))
        lines = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, min(end, len(lines)) + 1))

    def grep(self, args: dict) -> str:
        path = self._path(str(args["path"]))
        regex = re.compile(str(args["pattern"]), re.IGNORECASE)
        out = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if regex.search(line):
                out.append(f"{i}: {line}")
            if len(out) >= min(100, int(args.get("max_results", 40))):
                break
        return "\n".join(out) if out else "NO MATCHES"

    def write_file(self, args: dict) -> str:
        path = self._path(str(args["path"]))
        if path.suffix not in {".py", ".md", ".txt", ".json"}:
            raise ValueError("allowed file extensions: .py, .md, .txt, .json")
        content = str(args["content"])
        if len(content) > 24000:
            raise ValueError("file write exceeds 24000 characters")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"WROTE {len(content)} characters to {path.relative_to(self.root)}"

    @staticmethod
    def _validate_code(tree: ast.AST) -> None:
        forbidden_nodes = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
        forbidden_names = {"open", "eval", "exec", "compile", "__import__", "input", "help", "breakpoint"}
        for node in ast.walk(tree):
            if isinstance(node, forbidden_nodes):
                raise ValueError(f"forbidden syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                raise ValueError(f"forbidden name: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError("private attributes are unavailable")

    def run_python(self, args: dict) -> str:
        path = self._path(str(args["path"]))
        if path.suffix != ".py":
            raise ValueError("run_python requires a .py file")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path.name)
        self._validate_code(tree)
        safe_builtins = {
            "print": print, "len": len, "range": range, "enumerate": enumerate,
            "sorted": sorted, "min": min, "max": max, "sum": sum, "any": any,
            "all": all, "round": round, "float": float, "int": int, "str": str,
            "list": list, "dict": dict, "set": set, "tuple": tuple, "zip": zip,
            "abs": abs, "bool": bool, "isinstance": isinstance, "Exception": Exception,
        }
        globals_dict = {
            "__builtins__": safe_builtins,
            "lab": self.lab,
            "json": json,
            "re": re,
            "math": math,
        }
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree, path.name, "exec"), globals_dict, globals_dict)
        output = stdout.getvalue()
        if len(output) > 20000:
            output = output[:10000] + "\n...[OUTPUT TRUNCATED; full lab results are in lab_calls.jsonl]...\n" + output[-10000:]
        digest = hashlib.sha256(source.encode()).hexdigest()
        return json.dumps({
            "path": str(path.relative_to(self.root)),
            "sha256": digest,
            "stdout": output,
            "lab_state": self.lab.public_summary(),
        }, ensure_ascii=False)

    def call(self, name: str, args: dict) -> str:
        if name == "list_files":
            return self.list_files(args)
        if name == "read_lines":
            return self.read_lines(args)
        if name == "grep":
            return self.grep(args)
        if name == "write_file":
            return self.write_file(args)
        if name == "run_python":
            return self.run_python(args)
        raise ValueError(f"unknown tool: {name}")


def validate_final(payload: dict, workspace: Path, lab: FlintLab) -> dict:
    status = "null" if payload.get("status") is None else str(payload.get("status", ""))
    payload["status"] = status
    if status not in {"confirmed", "null"}:
        raise ValueError("status must be confirmed or null")
    for required in ("study.py", "REPORT.md"):
        if not (workspace / required).is_file():
            raise ValueError(f"you must author {required} before finishing")
    if lab.scan_count < 1 or lab.call_count < 3:
        raise ValueError("the study must execute a scan and causal laboratory calls")
    if status == "confirmed":
        index = int(payload.get("primary_confirmation_index", -1))
        if not 0 <= index < len(lab.confirmations):
            raise ValueError("confirmed status requires a valid primary_confirmation_index")
        chosen = lab.confirmations[index]
        if chosen["prompt_count"] < 8:
            raise ValueError("primary confirmation requires at least 8 prompts")
        strongest_control = max(
            chosen["counts"]["baseline"],
            chosen["counts"]["opposite_pole"],
            chosen["counts"]["matched_neuron"],
        )
        if chosen["counts"]["target"] < 6 or strongest_control > 2:
            raise ValueError(
                "confirmed status requires >=6/8 target matches and <=2/8 for every control"
            )
        same_candidate_replications = [
            row for row in lab.confirmations
            if row["candidate_id"] == chosen["candidate_id"]
            and row["counts"]["target"] > max(
                row["counts"]["baseline"],
                row["counts"]["opposite_pole"],
                row["counts"]["matched_neuron"],
            )
        ]
        if len(same_candidate_replications) < 2:
            raise ValueError(
                "confirmed status requires two separated confirmations of the same candidate"
            )
    else:
        if lab.scan_count < 2:
            raise ValueError("null status requires using both available activation scans")
        if len(lab.intervened_candidate_ids) < 6:
            raise ValueError(
                "null status requires causally screening at least six distinct candidates"
            )
        if len(lab.high_dose_candidate_ids) < 3:
            raise ValueError(
                "null status requires high-dose screening (|factor| >= 16) of at least three candidates"
            )
    if not str(payload.get("headline", "")).strip():
        raise ValueError("headline is required")
    return payload


def run_researcher(endpoint: str, token: str, lab: FlintLab, config: dict) -> dict:
    tools = ResearchWorkspace(WORKSPACE, lab)
    system = """You are Researcher Flint. You have a blank coding workspace and access to a
separate subject Flint checkpoint through a stable low-level lab API. You—not the operator—must
design, write, execute, debug, and report a neuron-level mechanistic-interpretability study.

Workspace tools (emit exactly one per turn):
- <tool>{"name":"list_files","args":{}}</tool>
- <tool>{"name":"read_lines","args":{"path":"LAB_API.md","start":1,"end":100}}</tool>
- <tool>{"name":"grep","args":{"path":"file","pattern":"regex","max_results":40}}</tool>
- <tool>{"name":"write_file","args":{"path":"study.py","content":"..."}}</tool>
- <tool>{"name":"run_python","args":{"path":"study.py"}}</tool>

Keep analysis plus plan under 250 words. Do not recap the mission, quote example action JSON, or
deliberate past the point of a decision. Emit the single tool/final action promptly.

Write Python programs that use the prebound object `lab`; imports and direct filesystem access are
disabled. The laboratory automatically preserves every raw call and generation. You decide layers,
scan budget, candidates, prompts, doses, hypotheses, regex/endpoints, controls, and stopping rule.
Do not merely narrate code: run it, inspect evidence files, revise it, and test fresh prompts. Treat
activation contexts as hypotheses, not proof. Prefer a narrow causal result with explicit controls.
Do not stop after the first plausible candidate: use inexpensive `lab.intervene` calls to causally
screen multiple self-selected candidates and doses before spending full controls on the strongest
effects. Explore a logarithmic dose range rather than assuming factors 1-2 are sufficient; before a
null, test |factor| >= 16 on at least three candidates and inspect coherence. A confirmed result must
survive two fresh-prompt confirmations. Report a null honestly only after using both scans and screening
at least six distinct candidates. The operator will not pick
candidates or repair your scientific analysis. Before finishing, author study.py and REPORT.md.

Finish with exactly:
<final>{"status":"confirmed|unconfirmed|null","headline":"...","primary_confirmation_index":0,
"report_path":"REPORT.md","limitations":"..."}</final>
Use status `confirmed` or `null` in the actual final JSON. All claims must come from files you inspected."""
    mission = """Independently investigate the subject Flint for an interpretable native neuron-pole
whose causal intervention produces a repeatable, human-readable effect. Begin by reading LAB_API.md.
You have up to 100 agent turns and two activation scans. Build the strongest auditable study you can
within the available compute, including fresh confirmation prompts and baseline, opposite-pole, and
matched-neuron controls. The complete agent conversation and every version of your code will be shared."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": mission}]
    corrections = 0
    for step in range(1, int(config["agent_max_steps"]) + 1):
        raw = model_turn(endpoint, token, messages, int(config["agent_max_new_tokens"]))
        emit("RESEARCHER_FLINT_TRACE_JSON", {"step": step, "kind": "model_response", "content": raw})
        messages.append({"role": "assistant", "content": raw})
        try:
            kind, payload = parse_agent_action(raw)
            if kind == "final":
                validated = validate_final(payload, WORKSPACE, lab)
                emit("RESEARCHER_FLINT_FINAL_JSON", {"step": step, "result": validated})
                return validated
            name = str(payload["name"])
            args = payload.get("args", {})
            emit("RESEARCHER_FLINT_TRACE_JSON", {"step": step, "kind": "tool_call", "name": name, "args": args})
            result = tools.call(name, args)
            emit("RESEARCHER_FLINT_TRACE_JSON", {"step": step, "kind": "tool_result", "name": name, "content": result})
            messages.append({"role": "user", "content": f"TOOL RESULT ({name}):\n{result}"})
        except Exception as exc:
            corrections += 1
            if corrections > int(config["agent_max_corrections"]):
                raise RuntimeError(f"Researcher Flint exceeded correction budget: {exc}") from exc
            error = f"Invalid action or execution: {exc}. Correct your own code/action and continue."
            emit("RESEARCHER_FLINT_TRACE_JSON", {"step": step, "kind": "correction", "content": error})
            messages.append({"role": "user", "content": error})
    raise RuntimeError("Researcher Flint exhausted the agent-turn budget")


def write_api_docs() -> None:
    (WORKSPACE / "LAB_API.md").write_text("""# Subject Flint laboratory API

Your Python file runs with these prebound names: `lab`, `json`, `re`, and `math`.
Imports and direct filesystem access are intentionally unavailable.

## `lab.scan(layers, corpus_passages=384, candidates_per_layer_pole=8, tail_contexts=6)`
Collect natural routed-expert neuron activations from Wikipedia and write full candidate contexts to
`activation_candidates_scan_N.jsonl` plus compact `activation_cards_scan_N.txt`. Choose 1–4 distinct
layers from `[8, 12, 20, 28, 32, 36, 40]`. At most two scans; 192–512 passages per scan.

## `lab.baseline(prompts, max_new_tokens=48)`
Generate unmodified subject outputs for 1–16 distinct prompts.

## `lab.intervene(candidate_id, factor, prompts, reverse=False, max_new_tokens=48)`
While predicting the first generated token, add the chosen signed neuron's unit-normalized output
direction at `factor` strength. `reverse=True` applies the opposite pole. This bypasses natural expert
routing and is a causal write-direction intervention, not proof of the neuron's ordinary function.

## `lab.matched_control(candidate_id)`
Return the same-layer scanned candidate with closest output-vector norm.

## `lab.confirm(candidate_id, factor, prompts, regex, scope="whole", max_new_tokens=48)`
On at least eight prompts, run target, baseline, opposite-pole, and matched-neuron conditions. You
choose the candidate, factor, fresh prompts, regex, scope, and interpretation. Returns raw outputs and
match counts; it imposes no success threshold. Confirmation prompts must not themselves contain text
matched by the scoring regex. The matched control is also returned.

Every operation and raw output is appended immutably to `lab_calls.jsonl`. Use ordinary Python
analysis in your script, and use workspace read/grep tools to inspect the generated evidence.
""", encoding="utf-8")


def main() -> None:
    started = time.time()
    config = json.loads((ROOT / "config.json").read_text())
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    emit("CONFIG_JSON", config)
    if torch.cuda.device_count() < 1:
        raise RuntimeError("subject Flint requires one visible GPU")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for path in WORKSPACE.glob("*"):
        if path.is_file():
            path.unlink()
    write_api_docs()

    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    subject = load_flint(config["target_model"], "cuda:0", config.get("attn_implementation"))
    lab = FlintLab(subject, tokenizer, config, WORKSPACE)
    endpoint = str(config["researcher_endpoint"])
    token = os.environ["HF_TOKEN"]
    emit("TWO_FLINT_ARCHITECTURE_JSON", {
        "researcher": "hosted RecursiveSI/flint",
        "subject": "separate RecursiveSI/flint on cuda:0",
        "researcher_role": "writes and executes study code",
        "operator_role": "low-level lab API and immutable logging only",
        "separate_model_processes": True,
    })

    final = run_researcher(endpoint, token, lab, config)
    study_source = (WORKSPACE / "study.py").read_text(encoding="utf-8")
    report = (WORKSPACE / "REPORT.md").read_text(encoding="utf-8")
    result = {
        "config": config,
        "researcher_final": final,
        "lab_summary": lab.public_summary(),
        "study_source": study_source,
        "study_sha256": hashlib.sha256(study_source.encode()).hexdigest(),
        "report": report,
        "runtime_seconds": time.time() - started,
        "operator_selected_candidates": False,
        "operator_supplied_hypotheses": False,
        "operator_supplied_prompts": False,
        "operator_supplied_regex": False,
    }
    emit("AUTONOMOUS_LAB_SUMMARY_JSON", {
        "researcher_final": final,
        "lab_summary": lab.public_summary(),
        "study_sha256": result["study_sha256"],
        "runtime_seconds": result["runtime_seconds"],
    })
    emit("RESULT_JSON", result)


if __name__ == "__main__":
    main()
