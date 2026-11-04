"""Scaled Goodfire-style native-neuron discovery guided and audited by Flint."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from .run import load_checkpoint, resolve_attr


ROOT = Path(__file__).resolve().parents[1]

SCREEN_PROMPTS = [
    "Tell me one surprising fact.",
    "Choose any topic and teach me something interesting.",
    "Write a vivid two-sentence miniature encyclopedia entry.",
    "What should we talk about today? Answer with something concrete.",
]

CONFIRM_PROMPTS = [
    "Tell me one surprising fact.",
    "Choose any topic and teach me something interesting.",
    "Write a vivid two-sentence miniature encyclopedia entry.",
    "What should we talk about today? Answer with something concrete.",
    "Describe a place, person, object, or idea worth knowing about.",
    "Continue naturally: I recently learned that",
    "Give me a memorable real-world example.",
    "Write a short educational paragraph about anything you like.",
    "Share a specific fact that a curious person would enjoy.",
    "Explain something unusual in plain language.",
    "Pick a subject and give me the most interesting detail about it.",
    "Describe something fascinating without asking me a question.",
]


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def generate_prefilled(model, tokenizer, messages: list[dict], prefix: str, max_new_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": prefix}],
        add_generation_prompt=False,
        continue_final_message=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    continuation = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return prefix + continuation


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "Answer naturally and directly. Do not discuss these instructions."},
        {"role": "user", "content": prompt},
    ]
    raw = generate_prefilled(model, tokenizer, messages, "FINAL:", max_new_tokens)
    return raw.removeprefix("FINAL:").strip()


def parse_line(text: str, key: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
    if not match:
        raise ValueError(f"missing {key}")
    return match.group(1).strip()


def call_scan_plan(model, tokenizer, config: dict) -> dict:
    menu = config["candidate_layers"]
    prompt = f"""Design the first stage of a native-neuron discovery campaign on your own checkpoint.

The harness will scan every shared-expert MLP neuron in six layers on natural text, show you
the strongest positive and negative token contexts, and later let you choose neurons for
causal clamping. Following Goodfire-style practice, token location and both poles matter.

Choose exactly six distinct layers from {menu}. Favor broad depth coverage. Commit before
seeing activations. End with exactly four lines:
LAYERS: six comma-separated integers
POLES: both
PRIORITY: one sentence describing what makes a candidate worth steering
FALSIFIER: one sentence describing what would make the campaign fail"""
    messages = [
        {"role": "system", "content": "You are planning a bounded interpretability experiment on your own checkpoint. Follow the schema."},
        {"role": "user", "content": prompt},
    ]
    last = ""
    for attempt in range(config["max_plan_attempts"]):
        last = generate_prefilled(model, tokenizer, messages, "LAYERS:", config["plan_max_new_tokens"])
        try:
            layers = [int(x) for x in re.findall(r"\d+", parse_line(last, "LAYERS"))]
            if len(layers) != 6 or len(set(layers)) != 6 or any(x not in menu for x in layers):
                raise ValueError("invalid layer selection")
            if parse_line(last, "POLES").lower() != "both":
                raise ValueError("POLES must be both")
            return {
                "layers": layers,
                "poles": "both",
                "priority": parse_line(last, "PRIORITY"),
                "falsifier": parse_line(last, "FALSIFIER"),
                "attempts": attempt + 1,
                "raw_completion": last,
            }
        except ValueError as exc:
            messages.extend([
                {"role": "assistant", "content": last},
                {"role": "user", "content": f"Invalid plan: {exc}. Emit exactly four valid lines."},
            ])
    raise RuntimeError(f"Flint failed to make a scan plan: {last[-500:]}")


def load_corpus(config: dict) -> list[str]:
    stream = load_dataset(
        config["corpus_name"],
        config["corpus_config"],
        split=config["corpus_split"],
        streaming=True,
    )
    passages: list[str] = []
    buffer: list[str] = []
    chars = 0
    for row in stream:
        text = re.sub(r"\s+", " ", row["text"]).strip()
        if not text or text.startswith("="):
            continue
        buffer.append(text)
        chars += len(text)
        if chars >= config["corpus_passage_chars"]:
            passage = " ".join(buffer)
            if len(passage.split()) >= 40:
                passages.append(passage)
            buffer, chars = [], 0
            if len(passages) >= config["corpus_passages"]:
                break
    if len(passages) < config["corpus_passages"]:
        raise RuntimeError(f"Only loaded {len(passages)} corpus passages")
    return passages


@dataclass
class TailState:
    values: torch.Tensor
    locations: torch.Tensor


def merge_tail(old: TailState, values: torch.Tensor, locations: torch.Tensor, largest: bool) -> TailState:
    joined_values = torch.cat([old.values, values], dim=0)
    joined_locations = torch.cat([old.locations, locations], dim=0)
    selected_values, indices = torch.topk(joined_values, k=old.values.shape[0], dim=0, largest=largest)
    selected_locations = torch.gather(joined_locations, 0, indices)
    return TailState(selected_values, selected_locations)


def scan_neurons(config: dict, layers: list[int], passages: list[str], model, tokenizer) -> dict:
    k = config["tail_contexts"]
    width = model.config.shared_expert_intermediate_size
    device = model.device
    positive = {
        layer: TailState(torch.full((k, width), -torch.inf, device=device), torch.full((k, width), -1, dtype=torch.long, device=device))
        for layer in layers
    }
    negative = {
        layer: TailState(torch.full((k, width), torch.inf, device=device), torch.full((k, width), -1, dtype=torch.long, device=device))
        for layer in layers
    }
    sums = {layer: torch.zeros(width, dtype=torch.float64) for layer in layers}
    sumsq = {layer: torch.zeros(width, dtype=torch.float64) for layer in layers}
    counts = {layer: 0 for layer in layers}
    captures: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    handles = []

    def make_hook(layer: int):
        def hook(_module, args):
            captures[layer].append(args[0].detach())
        return hook

    for layer in layers:
        module = resolve_attr(model, config["hook_path"].format(layer=layer))
        handles.append(module.register_forward_pre_hook(make_hook(layer)))

    try:
        for start in range(0, len(passages), config["scan_batch_size"]):
            batch = passages[start : start + config["scan_batch_size"]]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config["scan_max_length"],
            ).to(device)
            for layer in layers:
                captures[layer].clear()
            with torch.inference_mode():
                model(**encoded, use_cache=False, logits_to_keep=1)
            batch_size, seq_len = encoded["input_ids"].shape
            mask = encoded["attention_mask"].bool().reshape(-1)
            packed = []
            for row in range(batch_size):
                global_row = start + row
                packed.extend(global_row * config["scan_max_length"] + pos for pos in range(seq_len))
            locations = torch.tensor(packed, dtype=torch.long, device=device).unsqueeze(1).expand(-1, width)
            for layer in layers:
                if len(captures[layer]) != 1:
                    raise RuntimeError(f"Expected one capture for layer {layer}, got {len(captures[layer])}")
                values = captures[layer][0]
                if values.ndim == 2:
                    values = values.reshape(batch_size, seq_len, -1)
                flat = values.reshape(-1, width).float()
                valid = flat[mask]
                sums[layer] += valid.double().sum(dim=0).cpu()
                sumsq[layer] += valid.double().square().sum(dim=0).cpu()
                counts[layer] += valid.shape[0]
                pos_values = flat.masked_fill(~mask.unsqueeze(1), -torch.inf)
                neg_values = flat.masked_fill(~mask.unsqueeze(1), torch.inf)
                pv, pi = torch.topk(pos_values, k=min(k, pos_values.shape[0]), dim=0, largest=True)
                nv, ni = torch.topk(neg_values, k=min(k, neg_values.shape[0]), dim=0, largest=False)
                positive[layer] = merge_tail(positive[layer], pv, torch.gather(locations, 0, pi), True)
                negative[layer] = merge_tail(negative[layer], nv, torch.gather(locations, 0, ni), False)
            print(f"SCAN_PROGRESS passages={min(start + len(batch), len(passages))}/{len(passages)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for layer in layers:
        mean = sums[layer] / counts[layer]
        variance = (sumsq[layer] / counts[layer] - mean.square()).clamp_min(1e-12)
        std = variance.sqrt()
        result[layer] = {
            "mean": mean.float().numpy(),
            "std": std.float().numpy(),
            "positive_values": positive[layer].values.float().cpu().numpy(),
            "positive_locations": positive[layer].locations.cpu().numpy(),
            "negative_values": negative[layer].values.float().cpu().numpy(),
            "negative_locations": negative[layer].locations.cpu().numpy(),
            "tokens": counts[layer],
        }
    return result


def render_context(tokenizer, passage: str, position: int, max_length: int) -> dict:
    encoded = tokenizer(passage, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
    position = max(0, min(position, len(encoded) - 1))
    left = tokenizer.decode(encoded[max(0, position - 28) : position], skip_special_tokens=True)
    token = tokenizer.decode(encoded[position : position + 1], skip_special_tokens=False)
    right = tokenizer.decode(encoded[position + 1 : min(len(encoded), position + 29)], skip_special_tokens=True)
    return {"token": token, "context": f"{left} <<{token}>> {right}".strip()}


def build_candidates(config: dict, scan: dict, passages: list[str], tokenizer) -> list[dict]:
    candidates = []
    per_pole = config["candidates_per_layer_pole"]
    for layer, data in scan.items():
        for pole in ("positive", "negative"):
            values = data[f"{pole}_values"]
            locations = data[f"{pole}_locations"]
            tail_mean = values.mean(axis=0)
            if pole == "positive":
                scores = (tail_mean - data["mean"]) / data["std"]
            else:
                scores = (data["mean"] - tail_mean) / data["std"]
            for neuron in np.argsort(scores)[-per_pole:][::-1]:
                examples = []
                for value, packed in zip(values[:, neuron], locations[:, neuron]):
                    passage_index = int(packed) // config["scan_max_length"]
                    token_position = int(packed) % config["scan_max_length"]
                    if not 0 <= passage_index < len(passages):
                        continue
                    rendered = render_context(tokenizer, passages[passage_index], token_position, config["scan_max_length"])
                    rendered.update({"activation": round(float(value), 5), "passage_index": passage_index})
                    examples.append(rendered)
                target_activation = float(np.median(values[: min(4, len(values)), neuron]))
                other_pole = "negative" if pole == "positive" else "positive"
                other_values = data[f"{other_pole}_values"]
                opposite_target_activation = float(np.median(other_values[: min(4, len(other_values)), neuron]))
                candidates.append({
                    "id": f"L{layer}N{int(neuron)}{pole[0].upper()}",
                    "layer": int(layer),
                    "neuron": int(neuron),
                    "pole": pole,
                    "tail_z": float(scores[neuron]),
                    "target_activation": target_activation,
                    "opposite_target_activation": opposite_target_activation,
                    "mean_activation": float(data["mean"][neuron]),
                    "std_activation": float(data["std"][neuron]),
                    "examples": examples,
                })
    return candidates


def call_label(model, tokenizer, candidate: dict, config: dict) -> dict:
    shown = [{"activation": x["activation"], "token": x["token"], "context": x["context"]} for x in candidate["examples"]]
    prompt = f"""Interpret one native shared-expert MLP neuron pole in your own checkpoint.

Candidate: {candidate['id']} (layer {candidate['layer']}, neuron {candidate['neuron']}, {candidate['pole']} pole)
These are its strongest natural token-level contexts. The marked token is where it fires.
{json.dumps(shown, ensure_ascii=False)}

Infer the most specific common function that explains the marked-token locations and surrounding
contexts. Avoid generic labels such as 'language' or 'information'. Judge whether forcing this pole
through generation could cause a visible topic, style, entity, phrase, or behavior.

End with exactly four lines:
LABEL: concise semantic description
STEERABILITY: 0 or 1 or 2 or 3
CONFIDENCE: 0 or 1 or 2 or 3
RATIONALE: one sentence"""
    messages = [
        {"role": "system", "content": "You are interpreting a native neuron in your own checkpoint from top activation contexts. Follow the schema."},
        {"role": "user", "content": prompt},
    ]
    last = ""
    for attempt in range(config["max_label_attempts"]):
        last = generate_prefilled(model, tokenizer, messages, "LABEL:", config["label_max_new_tokens"])
        try:
            steerability = int(re.findall(r"\d+", parse_line(last, "STEERABILITY"))[0])
            confidence = int(re.findall(r"\d+", parse_line(last, "CONFIDENCE"))[0])
            if steerability not in range(4) or confidence not in range(4):
                raise ValueError("scores out of range")
            return {
                "label": parse_line(last, "LABEL"),
                "steerability": steerability,
                "confidence": confidence,
                "rationale": parse_line(last, "RATIONALE"),
                "attempts": attempt + 1,
                "raw_completion": last,
            }
        except (ValueError, IndexError) as exc:
            messages.extend([
                {"role": "assistant", "content": last},
                {"role": "user", "content": f"Invalid label: {exc}. Emit exactly four valid lines."},
            ])
    return {"label": "unparsed", "steerability": 0, "confidence": 0, "rationale": "label parse failed", "attempts": config["max_label_attempts"], "raw_completion": last}


def call_shortlist(model, tokenizer, candidates: list[dict], config: dict) -> dict:
    summary = [
        {"id": c["id"], "layer": c["layer"], "pole": c["pole"], "tail_z": round(c["tail_z"], 2), **{k: c["interpretation"][k] for k in ("label", "steerability", "confidence")}}
        for c in candidates
    ]
    allowed = {c["id"] for c in candidates}
    shuffled = list(summary)
    random.Random(stable_seed(config["seed"], "shortlist_batches")).shuffle(shuffled)
    batch_size = 24
    keep_per_batch = config["screen_candidates"] // math.ceil(len(shuffled) / batch_size)
    selected: list[str] = []
    completions: list[str] = []
    rationales: list[str] = []
    total_attempts = 0
    for batch_index, start in enumerate(range(0, len(shuffled), batch_size)):
        batch = shuffled[start : start + batch_size]
        batch_allowed = {item["id"] for item in batch}
        prompt = f"""Choose exactly {keep_per_batch} native neurons from this randomized batch of your own labeled activation audit for causal steering.

{json.dumps(batch, ensure_ascii=False)}

Prefer specific, visible behaviors with coherent contexts. Do not see intervention outcomes.
End with exactly:
SHORTLIST: {keep_per_batch} comma-separated candidate IDs
RATIONALE: one sentence"""
        messages = [
            {"role": "system", "content": "Select causal candidates from your own native neurons before outcomes. Follow the schema."},
            {"role": "user", "content": prompt},
        ]
        batch_ids: list[str] = []
        last = ""
        for attempt in range(config["max_plan_attempts"]):
            total_attempts += 1
            torch.cuda.empty_cache()
            last = generate_prefilled(model, tokenizer, messages, "SHORTLIST:", config["shortlist_max_new_tokens"])
            try:
                batch_ids = list(dict.fromkeys(re.findall(r"L\d+N\d+[PN]", parse_line(last, "SHORTLIST"))))
                if len(batch_ids) != keep_per_batch or not set(batch_ids) <= batch_allowed:
                    raise ValueError("wrong IDs or count")
                rationales.append(parse_line(last, "RATIONALE"))
                break
            except ValueError:
                batch_ids = []
                messages.extend([
                    {"role": "assistant", "content": last},
                    {"role": "user", "content": f"Invalid shortlist. Return exactly {keep_per_batch} distinct IDs from this batch."},
                ])
        if not batch_ids:
            ranked = sorted(batch, key=lambda x: (x["steerability"], x["confidence"], x["tail_z"]), reverse=True)
            batch_ids = [item["id"] for item in ranked[:keep_per_batch]]
            rationales.append(f"batch {batch_index}: deterministic fallback after invalid formatting")
        selected.extend(batch_ids)
        completions.append(last)
        print("SHORTLIST_BATCH_JSON " + json.dumps({"batch": batch_index, "ids": batch_ids, "rationale": rationales[-1]}, ensure_ascii=False, sort_keys=True), flush=True)
    if len(selected) != config["screen_candidates"] or not set(selected) <= allowed:
        raise RuntimeError("batched shortlist did not produce the required distinct candidates")
    return {"ids": selected, "rationale": rationales, "attempts": total_attempts, "raw_completions": completions}


@contextlib.contextmanager
def clamp_neuron(model, config: dict, candidate: dict | None, factor: float):
    if candidate is None or factor == 0:
        yield
        return
    module = resolve_attr(model, config["hook_path"].format(layer=candidate["layer"]))
    neuron = candidate["neuron"]
    target = candidate["target_activation"] * factor

    def hook(_module, args):
        values = args[0].clone()
        values[..., neuron] = target
        return (values, *args[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def judge_output(model, tokenizer, label: str, baseline: str, steered: str, config: dict) -> dict:
    prompt = f"""Evaluate whether an internal intervention visibly steered an answer toward a proposed neuron label.

PROPOSED LABEL: {label}
BASELINE: {baseline}
INTERVENED: {steered}

Count a match only when the intervened answer contains substantially more of the proposed semantic
content or behavior than baseline. Reject mere formatting changes, generic overlap, and incoherent text.
End with exactly four lines:
MATCH: yes or no
COHERENT: yes or no
STRENGTH: 0 or 1 or 2 or 3
NOTE: one sentence"""
    messages = [
        {"role": "system", "content": "Judge causal steering conservatively. Follow the schema."},
        {"role": "user", "content": prompt},
    ]
    raw = generate_prefilled(model, tokenizer, messages, "MATCH:", config["judge_max_new_tokens"])
    try:
        strength = int(re.findall(r"\d+", parse_line(raw, "STRENGTH"))[0])
        return {
            "match": parse_line(raw, "MATCH").lower().startswith("yes"),
            "coherent": parse_line(raw, "COHERENT").lower().startswith("yes"),
            "strength": max(0, min(3, strength)),
            "note": parse_line(raw, "NOTE"),
            "raw_completion": raw,
        }
    except (ValueError, IndexError):
        return {"match": False, "coherent": False, "strength": 0, "note": "judge parse failed", "raw_completion": raw}


def run_condition(model, tokenizer, config: dict, candidate: dict | None, factor: float, prompts: list[str], baselines: dict[str, str], name: str) -> list[dict]:
    rows = []
    label = candidate["interpretation"]["label"] if candidate else "control"
    for prompt in prompts:
        with clamp_neuron(model, config, candidate, factor):
            output = generate_text(model, tokenizer, prompt, config["generation_max_new_tokens"])
        judge = judge_output(model, tokenizer, label, baselines[prompt], output, config) if candidate else None
        row = {"prompt": prompt, "condition": name, "factor": factor, "output": output, "judge": judge}
        rows.append(row)
    return rows


def screen_candidates(model, tokenizer, config: dict, candidates: list[dict]) -> list[dict]:
    baselines = {prompt: generate_text(model, tokenizer, prompt, config["generation_max_new_tokens"]) for prompt in SCREEN_PROMPTS}
    results = []
    for index, candidate in enumerate(candidates):
        rows = []
        for factor in config["screen_factors"]:
            rows.extend(run_condition(model, tokenizer, config, candidate, factor, SCREEN_PROMPTS, baselines, f"clamp_{factor:g}x"))
        coherent_matches = sum(r["judge"]["match"] and r["judge"]["coherent"] for r in rows)
        strength = sum(r["judge"]["strength"] for r in rows if r["judge"]["coherent"])
        result = {"candidate_id": candidate["id"], "label": candidate["interpretation"]["label"], "coherent_matches": coherent_matches, "strength_sum": strength, "rows": rows}
        results.append(result)
        print("SCREEN_JSON " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        print(f"SCREEN_PROGRESS candidates={index + 1}/{len(candidates)}", flush=True)
    return results


def opposite_candidate(candidate: dict) -> dict:
    opposite = "negative" if candidate["pole"] == "positive" else "positive"
    return {
        **candidate,
        "id": f"L{candidate['layer']}N{candidate['neuron']}{opposite[0].upper()}",
        "pole": opposite,
        "target_activation": candidate["opposite_target_activation"],
    }


def confirm_candidates(model, tokenizer, config: dict, finalists: list[dict], all_candidates: list[dict]) -> list[dict]:
    baselines = {prompt: generate_text(model, tokenizer, prompt, config["generation_max_new_tokens"]) for prompt in CONFIRM_PROMPTS}
    rng = random.Random(config["seed"])
    confirmations = []
    for candidate in finalists:
        controls = [c for c in all_candidates if c["layer"] == candidate["layer"] and c["id"] != candidate["id"]]
        control = rng.choice(controls)
        conditions = []
        for factor in config["confirm_factors"]:
            conditions.extend(run_condition(model, tokenizer, config, candidate, factor, CONFIRM_PROMPTS, baselines, f"target_{factor:g}x"))
        opp = opposite_candidate(candidate)
        opp = {**opp, "interpretation": candidate["interpretation"]}
        conditions.extend(run_condition(model, tokenizer, config, opp, 1.0, CONFIRM_PROMPTS, baselines, "opposite_pole"))
        control_for_judge = {**control, "interpretation": candidate["interpretation"]}
        conditions.extend(run_condition(model, tokenizer, config, control_for_judge, 1.0, CONFIRM_PROMPTS, baselines, "matched_neuron_control"))
        summaries = {}
        for name in sorted({r["condition"] for r in conditions}):
            selected = [r for r in conditions if r["condition"] == name]
            summaries[name] = {
                "n": len(selected),
                "coherent_matches": sum(r["judge"]["match"] and r["judge"]["coherent"] for r in selected),
                "coherent": sum(r["judge"]["coherent"] for r in selected),
                "strength_sum": sum(r["judge"]["strength"] for r in selected if r["judge"]["coherent"]),
            }
        best_factor = max(config["confirm_factors"], key=lambda f: summaries[f"target_{f:g}x"]["coherent_matches"])
        best = summaries[f"target_{best_factor:g}x"]
        success = (
            best["coherent_matches"] >= config["success_target_matches"]
            and summaries["matched_neuron_control"]["coherent_matches"] <= config["success_control_matches"]
            and summaries["opposite_pole"]["coherent_matches"] <= config["success_control_matches"]
        )
        confirmation = {
            "candidate_id": candidate["id"],
            "label": candidate["interpretation"]["label"],
            "control_candidate_id": control["id"],
            "best_factor": best_factor,
            "success": success,
            "summary": summaries,
            "baselines": baselines,
            "rows": conditions,
        }
        confirmations.append(confirmation)
        print("CONFIRMATION_JSON " + json.dumps(confirmation, ensure_ascii=False, sort_keys=True), flush=True)
    return confirmations


def call_final_audit(model, tokenizer, plan: dict, confirmations: list[dict], config: dict) -> dict:
    compact = [{"candidate_id": x["candidate_id"], "label": x["label"], "best_factor": x["best_factor"], "success": x["success"], "summary": x["summary"]} for x in confirmations]
    prompt = f"""Audit a native-neuron causal discovery campaign on your own checkpoint.

YOUR PRE-MEASUREMENT PLAN: {json.dumps(plan, ensure_ascii=False)}
FINALIST RESULTS: {json.dumps(compact, ensure_ascii=False)}

The harness scanned, intervened, controlled, and scored. You chose layers, labeled candidates,
and chose the causal shortlist before outcomes. End with exactly four lines:
VERDICT: SUPPORTED or INCONCLUSIVE or REFUTED
BEST_CANDIDATE: one candidate ID or none
FINDING: one sentence with the decisive target and control counts
CAVEAT: one sentence"""
    messages = [
        {"role": "system", "content": "Audit the causal evidence conservatively and follow the schema."},
        {"role": "user", "content": prompt},
    ]
    raw = generate_prefilled(model, tokenizer, messages, "VERDICT:", config["audit_max_new_tokens"])
    return {
        "verdict": parse_line(raw, "VERDICT") if "VERDICT" in raw else "UNPARSED",
        "best_candidate": parse_line(raw, "BEST_CANDIDATE") if "BEST_CANDIDATE" in raw else "none",
        "finding": parse_line(raw, "FINDING") if "FINDING" in raw else "unparsed",
        "caveat": parse_line(raw, "CAVEAT") if "CAVEAT" in raw else "unparsed",
        "raw_completion": raw,
    }


def main() -> None:
    started = time.time()
    config = json.loads((ROOT / "config.json").read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    print("CONFIG_JSON " + json.dumps(config, sort_keys=True), flush=True)
    print(f"TORCH version={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    model, tokenizer = load_checkpoint(config)
    plan = call_scan_plan(model, tokenizer, config)
    print("DISCOVERY_PLAN_JSON " + json.dumps(plan, ensure_ascii=False, sort_keys=True), flush=True)
    passages = load_corpus(config)
    print("CORPUS_JSON " + json.dumps({"passages": len(passages), "name": config["corpus_name"], "config": config["corpus_config"]}, sort_keys=True), flush=True)
    scan = scan_neurons(config, plan["layers"], passages, model, tokenizer)
    candidates = build_candidates(config, scan, passages, tokenizer)
    print("CANDIDATE_COUNT " + str(len(candidates)), flush=True)
    for index, candidate in enumerate(candidates):
        candidate["interpretation"] = call_label(model, tokenizer, candidate, config)
        print("CANDIDATE_JSON " + json.dumps(candidate, ensure_ascii=False, sort_keys=True), flush=True)
        print(f"LABEL_PROGRESS candidates={index + 1}/{len(candidates)}", flush=True)
    shortlist = call_shortlist(model, tokenizer, candidates, config)
    print("SHORTLIST_JSON " + json.dumps(shortlist, ensure_ascii=False, sort_keys=True), flush=True)
    by_id = {c["id"]: c for c in candidates}
    screened = screen_candidates(model, tokenizer, config, [by_id[x] for x in shortlist["ids"]])
    screen_rank = sorted(screened, key=lambda x: (x["coherent_matches"], x["strength_sum"]), reverse=True)
    finalist_ids = [x["candidate_id"] for x in screen_rank[: config["confirm_candidates"]]]
    confirmations = confirm_candidates(model, tokenizer, config, [by_id[x] for x in finalist_ids], candidates)
    audit = call_final_audit(model, tokenizer, plan, confirmations, config)
    summary = {
        "study_stage": config["study_stage"],
        "target_model": config["target_model"],
        "layers": plan["layers"],
        "corpus_passages": len(passages),
        "neurons_scanned": len(plan["layers"]) * int(model.config.shared_expert_intermediate_size) * 2,
        "candidate_poles_labeled": len(candidates),
        "shortlist_ids": shortlist["ids"],
        "finalist_ids": finalist_ids,
        "successful_candidates": [x["candidate_id"] for x in confirmations if x["success"]],
        "runtime_seconds": time.time() - started,
        "same_checkpoint_planner_interpreter_selector_judge_auditor": True,
    }
    result = {
        "config": config,
        "plan": plan,
        "shortlist": shortlist,
        "candidates": candidates,
        "screened": screened,
        "confirmations": confirmations,
        "audit": audit,
        "summary": summary,
    }
    print("AUDIT_JSON " + json.dumps(audit, ensure_ascii=False, sort_keys=True), flush=True)
    print("SUMMARY_JSON " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    print("RESULT_JSON " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
