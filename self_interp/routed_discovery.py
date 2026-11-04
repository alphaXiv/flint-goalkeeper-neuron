"""Goodfire-style discovery and steering of Flint's routed MoE neurons."""

from __future__ import annotations

import contextlib
from collections import Counter
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .discovery import (
    CONFIRM_PROMPTS,
    SCREEN_PROMPTS,
    generate_prefilled,
    generate_text,
    judge_output,
    parse_line,
    render_context,
    stable_seed,
)
from .run import resolve_attr


ROOT = Path(__file__).resolve().parents[1]


def load_flint(model_name: str, device: str, attn_implementation: str | None):
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": device},
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def call_scan_plan(model, tokenizer, config: dict) -> dict:
    menu = config["candidate_layers"]
    prompt = f"""Plan a native-neuron survey on a separate Flint checkpoint.

The harness will inspect every routed expert-specific MLP neuron (layer / expert / neuron),
not the always-on shared MLP. It will collect both activation poles from diverse natural text,
then let you interpret and causally test a blinded shortlist by adding each neuron's native
write vector. Do not favor any semantic topic, entity, style, or expected public appeal.

Choose exactly three distinct layers from {menu}. Commit before seeing activations. End with exactly four lines:
LAYERS: three comma-separated integers
POLES: both
PRIORITY: one sentence describing neutral selection criteria based on specificity, confidence, and falsifiability
FALSIFIER: one sentence describing failure"""
    messages = [
        {"role": "system", "content": "You are Flint directing an interpretability survey on a separate Flint checkpoint. Follow the schema exactly and do not optimize for a preferred finding."},
        {"role": "user", "content": prompt},
    ]
    last = ""
    for attempt in range(config["max_plan_attempts"]):
        last = generate_prefilled(model, tokenizer, messages, "LAYERS:", config["plan_max_new_tokens"])
        try:
            layers = [int(x) for x in re.findall(r"\d+", parse_line(last, "LAYERS"))]
            if len(layers) != 3 or len(set(layers)) != 3 or any(x not in menu for x in layers):
                raise ValueError("choose exactly three allowed layers")
            # The harness always scans both signed poles. Flint sometimes describes them
            # numerically despite the requested literal schema; require the line, not wording.
            parse_line(last, "POLES")
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
    raise RuntimeError(f"Flint failed to plan routed-neuron scan: {last[-500:]}")


def load_diverse_corpus(config: dict) -> list[str]:
    """Reservoir-sample assembled passages instead of adjacent WikiText rows."""
    stream = load_dataset(
        config["corpus_name"], config["corpus_config"], split=config["corpus_split"], streaming=True
    )
    rng = random.Random(stable_seed(config["seed"], "corpus_reservoir"))
    reservoir: list[str] = []
    buffer: list[str] = []
    chars = 0
    seen = 0
    for row in stream:
        text = re.sub(r"\s+", " ", row["text"]).strip()
        if not text or text.startswith("="):
            continue
        buffer.append(text)
        chars += len(text)
        if chars < config["corpus_passage_chars"]:
            continue
        passage = " ".join(buffer)
        buffer, chars = [], 0
        if len(passage.split()) < 40:
            continue
        seen += 1
        if len(reservoir) < config["corpus_passages"]:
            reservoir.append(passage)
        else:
            slot = rng.randrange(seen)
            if slot < len(reservoir):
                reservoir[slot] = passage
        if seen >= config["corpus_source_passages"]:
            break
    if len(reservoir) < config["corpus_passages"]:
        raise RuntimeError(f"Only found {len(reservoir)} passages from {seen} assembled passages")
    rng.shuffle(reservoir)
    return reservoir


def empty_scan_state(layers: list[int], experts: int, width: int, k: int, device) -> dict:
    shape = (k, experts, width)
    return {
        layer: {
            "positive_values": torch.full(shape, -torch.inf, device=device),
            "positive_locations": torch.full(shape, -1, dtype=torch.long, device=device),
            "negative_values": torch.full(shape, torch.inf, device=device),
            "negative_locations": torch.full(shape, -1, dtype=torch.long, device=device),
            "sum": torch.zeros((experts, width), device=device),
            "sumsq": torch.zeros((experts, width), device=device),
            "count": torch.zeros(experts, dtype=torch.long, device=device),
        }
        for layer in layers
    }


def merge_expert_tail(state: dict, pole: str, expert: int, values: torch.Tensor, locations: torch.Tensor, k: int) -> None:
    largest = pole == "positive"
    old_values = state[f"{pole}_values"][:, expert]
    old_locations = state[f"{pole}_locations"][:, expert]
    take = min(k, values.shape[0])
    new_values, new_indices = torch.topk(values, k=take, dim=0, largest=largest)
    expanded_locations = locations[:, None].expand(-1, values.shape[1])
    new_locations = torch.gather(expanded_locations, 0, new_indices)
    joined_values = torch.cat((old_values, new_values), dim=0)
    joined_locations = torch.cat((old_locations, new_locations), dim=0)
    kept_values, kept_indices = torch.topk(joined_values, k=k, dim=0, largest=largest)
    state[f"{pole}_values"][:, expert] = kept_values
    state[f"{pole}_locations"][:, expert] = torch.gather(joined_locations, 0, kept_indices)


def scan_routed_neurons(config: dict, layers: list[int], passages: list[str], model, tokenizer) -> dict:
    experts = int(model.config.num_experts)
    width = int(model.config.moe_intermediate_size)
    k = int(config["tail_contexts"])
    device = model.device
    state = empty_scan_state(layers, experts, width, k, device)
    batch_context: dict[str, torch.Tensor | None] = {"valid": None, "locations": None}
    handles = []

    def make_hook(layer: int):
        def hook(module, args):
            hidden_states, selected_experts, routing_weights = args[:3]
            valid = batch_context["valid"]
            locations = batch_context["locations"]
            if valid is None or locations is None:
                raise RuntimeError("batch context was not initialized")
            expert_ids = torch.unique(selected_experts[valid]).tolist()
            with torch.inference_mode():
                for expert in expert_ids:
                    token_idx, top_k_pos = torch.where(selected_experts == expert)
                    keep = valid[token_idx]
                    token_idx, top_k_pos = token_idx[keep], top_k_pos[keep]
                    if token_idx.numel() == 0:
                        continue
                    gate, up = F.linear(hidden_states[token_idx], module.gate_up_proj[expert]).chunk(2, dim=-1)
                    activations = module.act_fn(gate) * up
                    coefficients = activations.float() * routing_weights[token_idx, top_k_pos, None].float()
                    layer_state = state[layer]
                    layer_state["sum"][expert] += coefficients.sum(dim=0)
                    layer_state["sumsq"][expert] += coefficients.square().sum(dim=0)
                    layer_state["count"][expert] += coefficients.shape[0]
                    token_locations = locations[token_idx]
                    merge_expert_tail(layer_state, "positive", expert, coefficients, token_locations, k)
                    merge_expert_tail(layer_state, "negative", expert, coefficients, token_locations, k)
        return hook

    for layer in layers:
        module = resolve_attr(model, config["experts_path"].format(layer=layer))
        handles.append(module.register_forward_pre_hook(make_hook(layer)))
    try:
        for start in range(0, len(passages), config["scan_batch_size"]):
            batch = passages[start : start + config["scan_batch_size"]]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=config["scan_max_length"]
            ).to(device)
            batch_size, seq_len = encoded["input_ids"].shape
            batch_context["valid"] = encoded["attention_mask"].bool().reshape(-1)
            rows = torch.arange(start, start + batch_size, device=device)[:, None]
            positions = torch.arange(seq_len, device=device)[None, :]
            batch_context["locations"] = (rows * config["scan_max_length"] + positions).reshape(-1)
            with torch.inference_mode():
                model(**encoded, use_cache=False, logits_to_keep=1)
            print(f"ROUTED_SCAN_PROGRESS passages={min(start + len(batch), len(passages))}/{len(passages)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for layer in layers:
        item = state[layer]
        count = item["count"].clamp_min(1).float()[:, None]
        mean = item["sum"] / count
        variance = (item["sumsq"] / count - mean.square()).clamp_min(1e-10)
        result[layer] = {
            "mean": mean.cpu().numpy(),
            "std": variance.sqrt().cpu().numpy(),
            "count": item["count"].cpu().numpy(),
            "positive_values": item["positive_values"].cpu().numpy(),
            "positive_locations": item["positive_locations"].cpu().numpy(),
            "negative_values": item["negative_values"].cpu().numpy(),
            "negative_locations": item["negative_locations"].cpu().numpy(),
        }
    return result


def build_candidates(config: dict, scan: dict, passages: list[str], tokenizer) -> list[dict]:
    candidates = []
    per_pole = int(config["candidates_per_layer_pole"])
    for layer, data in scan.items():
        for pole in ("positive", "negative"):
            values = data[f"{pole}_values"]
            locations = data[f"{pole}_locations"]
            tail_mean = values.mean(axis=0)
            score = (tail_mean - data["mean"]) / data["std"] if pole == "positive" else (data["mean"] - tail_mean) / data["std"]
            score[data["count"] < config["min_expert_activations"], :] = -np.inf
            flat_order = np.argsort(score.reshape(-1))[-per_pole:][::-1]
            for flat in flat_order:
                expert, neuron = np.unravel_index(int(flat), score.shape)
                examples = []
                for value, packed in zip(values[:, expert, neuron], locations[:, expert, neuron]):
                    passage_index = int(packed) // config["scan_max_length"]
                    token_position = int(packed) % config["scan_max_length"]
                    if not 0 <= passage_index < len(passages):
                        continue
                    rendered = render_context(tokenizer, passages[passage_index], token_position, config["scan_max_length"])
                    rendered.update({"coefficient": round(float(value), 6), "passage_index": passage_index})
                    examples.append(rendered)
                opposite = "negative" if pole == "positive" else "positive"
                candidates.append({
                    "id": f"L{layer}E{expert}N{neuron}{pole[0].upper()}",
                    "layer": int(layer),
                    "expert": int(expert),
                    "neuron": int(neuron),
                    "pole": pole,
                    "tail_z": float(score[expert, neuron]),
                    "target_coefficient": float(np.median(values[:4, expert, neuron])),
                    "opposite_target_coefficient": float(np.median(data[f"{opposite}_values"][:4, expert, neuron])),
                    "expert_activation_count": int(data["count"][expert]),
                    "examples": examples,
                })
    return candidates


def call_label(model, tokenizer, candidate: dict, config: dict) -> dict:
    shown = [{"coefficient": x["coefficient"], "token": x["token"], "context": x["context"]} for x in candidate["examples"]]
    prompt = f"""Interpret a native routed MoE neuron in your own checkpoint.

Candidate: {candidate['id']} (layer {candidate['layer']}, expert {candidate['expert']}, neuron {candidate['neuron']}, {candidate['pole']} pole)
The marked token is where this expert-specific neuron has its strongest routed contribution:
{json.dumps(shown, ensure_ascii=False)}

Infer the narrowest common concept or function supported by the examples. Token location matters.
Do not favor any category of interpretation, and do not infer beyond the shown contexts.
End with exactly four lines:
LABEL: concise description
STEERABILITY: 0 or 1 or 2 or 3
CONFIDENCE: 0 or 1 or 2 or 3
RATIONALE: one sentence"""
    messages = [
        {"role": "system", "content": "You are Flint interpreting an expert-specific neuron in a separate Flint checkpoint. Follow the schema and report only what the contexts support."},
        {"role": "user", "content": prompt},
    ]
    last = ""
    for attempt in range(config["max_label_attempts"]):
        last = generate_prefilled(model, tokenizer, messages, "LABEL:", config["label_max_new_tokens"])
        try:
            steerability = int(re.findall(r"\d+", parse_line(last, "STEERABILITY"))[0])
            confidence = int(re.findall(r"\d+", parse_line(last, "CONFIDENCE"))[0])
            if steerability not in range(4) or confidence not in range(4):
                raise ValueError("score out of range")
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
    return {"label": "unparsed", "steerability": 0, "confidence": 0, "rationale": "parse failed", "attempts": config["max_label_attempts"], "raw_completion": last}


def call_shortlist(model, tokenizer, candidates: list[dict], config: dict) -> dict:
    summary = [
        {"id": c["id"], "tail_z": round(c["tail_z"], 2), "label": c["interpretation"]["label"], "steerability": c["interpretation"]["steerability"], "confidence": c["interpretation"]["confidence"]}
        for c in candidates
    ]
    allowed = {x["id"] for x in summary}
    shuffled = list(summary)
    random.Random(stable_seed(config["seed"], "routed_shortlist")).shuffle(shuffled)
    # Keep each decision small enough for Flint to return an auditable exact-ID
    # choice. The prior 24-item batches repeatedly produced malformed lists.
    batch_size = 10
    batches = (len(shuffled) + batch_size - 1) // batch_size
    base_keep = config["screen_candidates"] // batches
    remainder = config["screen_candidates"] % batches
    selected: list[str] = []
    raw = []
    for batch_index, start in enumerate(range(0, len(shuffled), batch_size)):
        batch = shuffled[start : start + batch_size]
        keep = base_keep + (1 if batch_index < remainder else 0)
        prompt = f"""Choose exactly {keep} expert-specific native neurons from this blinded batch for causal steering.
{json.dumps(batch, ensure_ascii=False)}
Select by activation specificity, interpretive confidence, and whether the label admits a clear
causal falsification. Do not prefer any topic or style. You have not seen intervention outcomes.
The assistant response is already prefixed with `SHORTLIST:`. Continue it with exactly {keep}
comma-separated IDs from the batch and nothing else."""
        messages = [
            {"role": "system", "content": "You are Flint selecting neurons in a separate Flint checkpoint for a neutral causal survey. Follow the schema."},
            {"role": "user", "content": prompt},
        ]
        batch_allowed = {x["id"] for x in batch}
        completion = ""
        ids = []
        attempts = []
        for attempt in range(config["max_shortlist_attempts"]):
            continued = [*messages, {"role": "assistant", "content": "SHORTLIST:"}]
            inputs = tokenizer.apply_chat_template(
                continued,
                add_generation_prompt=False,
                continue_final_message=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    use_cache=True,
                )
            completion = "SHORTLIST:" + tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            attempts.append(completion)
            try:
                ids = list(dict.fromkeys(re.findall(
                    r"L\d+E\d+N\d+[PN]", parse_line(completion, "SHORTLIST")
                )))
            except ValueError:
                ids = []
            if len(ids) == keep and set(ids) <= batch_allowed:
                break
            messages.append({"role": "user", "content": (
                f"Your prior answer was invalid. Continue the supplied SHORTLIST prefix with "
                f"exactly {keep} distinct IDs from this batch and no other text."
            )})
        if len(ids) != keep or not set(ids) <= batch_allowed:
            print("ROUTED_SHORTLIST_FAILURE_JSON " + json.dumps({
                "batch": batch_index,
                "required": keep,
                "allowed": sorted(batch_allowed),
                "raw_attempts": attempts,
                "fallback_used": False,
            }, ensure_ascii=False, sort_keys=True), flush=True)
            raise RuntimeError(
                f"Researcher Flint failed to choose shortlist batch {batch_index}; no fallback selection is permitted"
            )
        selected.extend(ids)
        raw.extend(attempts)
        print("ROUTED_SHORTLIST_BATCH_JSON " + json.dumps({
            "batch": batch_index, "ids": ids, "attempts": len(attempts),
            "chosen_by_researcher_flint": True, "fallback_used": False,
        }, sort_keys=True), flush=True)
    if len(selected) != config["screen_candidates"] or not set(selected) <= allowed:
        raise RuntimeError("invalid routed shortlist")
    return {"ids": selected, "raw_completions": raw}


@contextlib.contextmanager
def inject_neuron(model, config: dict, candidate: dict | None, factor: float):
    if candidate is None or factor == 0:
        yield
        return
    mlp = resolve_attr(model, config["mlp_path"].format(layer=candidate["layer"]))
    experts = resolve_attr(model, config["experts_path"].format(layer=candidate["layer"]))
    column = experts.down_proj[candidate["expert"], :, candidate["neuron"]].detach().float()
    pole_sign = 1.0 if candidate["pole"] == "positive" else -1.0
    if candidate.get("reverse_sign"):
        pole_sign *= -1.0
    write_vector = column * (pole_sign * float(factor) / float(column.norm()))
    calls = 0

    def hook(_module, _args, output):
        nonlocal calls
        if calls > 0:
            calls += 1
            return output
        changed = output.clone()
        write = write_vector.to(device=output.device, dtype=output.dtype)
        if changed.ndim == 3 and changed.shape[1] > 1:
            changed[:, -1, :] += write
        else:
            changed += write
        calls += 1
        return changed

    handle = mlp.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def first_word(text: str) -> str:
    match = re.match(r"^[\s\W_]*([A-Za-z]+)", text)
    return match.group(1).lower() if match else ""


def run_condition(subject, researcher, tokenizer, config: dict, candidate: dict, factor: float, prompts: list[str], baselines: dict[str, str], name: str, judge_semantic: bool = True) -> list[dict]:
    rows = []
    for prompt in prompts:
        with inject_neuron(subject, config, candidate, factor):
            output = generate_text(subject, tokenizer, prompt, config["generation_max_new_tokens"])
        judge = judge_output(researcher, tokenizer, candidate["interpretation"]["label"], baselines[prompt], output, config) if judge_semantic else None
        rows.append({"prompt": prompt, "condition": name, "factor": factor, "output": output, "first_word": first_word(output), "judge": judge})
    return rows


def best_button(rows: list[dict], baselines: dict[str, str]) -> dict:
    excluded = {"the", "and", "this", "that", "final", "user", "answer", "here", "with", "from"}
    best = {"prefix": "", "hits": 0, "baseline_hits": 0, "factor": 0.0}
    factors = sorted({float(row["factor"]) for row in rows})
    baseline_words = [first_word(text) for text in baselines.values()]
    for factor in factors:
        chosen = [row for row in rows if float(row["factor"]) == factor]
        stems = [row["first_word"][:3] for row in chosen if len(row["first_word"]) >= 4 and row["first_word"] not in excluded]
        if not stems:
            continue
        prefix, hits = Counter(stems).most_common(1)[0]
        baseline_hits = sum(word.startswith(prefix) for word in baseline_words)
        candidate = {"prefix": prefix, "hits": hits, "baseline_hits": baseline_hits, "factor": factor}
        if (hits - baseline_hits, hits, factor) > (best["hits"] - best["baseline_hits"], best["hits"], best["factor"]):
            best = candidate
    return best


def screen_candidates(subject, researcher, tokenizer, config: dict, candidates: list[dict]) -> list[dict]:
    baselines = {prompt: generate_text(subject, tokenizer, prompt, config["generation_max_new_tokens"]) for prompt in SCREEN_PROMPTS}
    results = []
    for index, candidate in enumerate(candidates):
        rows = []
        for factor in config["screen_factors"]:
            rows.extend(run_condition(subject, researcher, tokenizer, config, candidate, factor, SCREEN_PROMPTS, baselines, f"inject_{factor:g}", judge_semantic=False))
        button = best_button(rows, baselines)
        result = {"candidate_id": candidate["id"], "label": candidate["interpretation"]["label"], "button": button, "rows": rows}
        results.append(result)
        print("ROUTED_SCREEN_JSON " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        print(f"ROUTED_SCREEN_PROGRESS candidates={index + 1}/{len(candidates)}", flush=True)
    return results


def call_button_finalists(model, tokenizer, positive: list[dict], config: dict) -> dict:
    keep = min(int(config["confirm_candidates"]), len(positive))
    summaries = []
    for item in positive:
        factor = item["button"]["factor"]
        chosen = [row for row in item["rows"] if float(row["factor"]) == float(factor)]
        summaries.append({
            "id": item["candidate_id"],
            "activation_label": item["label"],
            "causal_prefix": item["button"]["prefix"],
            "hits": item["button"]["hits"],
            "factor": factor,
            "openings": [row["output"][:180] for row in chosen],
        })
    prompt = f"""You are Researcher Flint choosing which causal buttons in another Flint deserve held-out confirmation.
These candidates were selected and intervened on only after your activation-context analysis:
{json.dumps(summaries, ensure_ascii=False)}

Choose exactly {keep} candidates whose repeated first-word effect is clearest, most specific, and easiest to falsify.
Your response is already prefixed with FINALISTS:. Continue with exactly {keep} comma-separated candidate IDs and no prose."""
    messages = [
        {"role": "system", "content": "You are Flint making a frozen causal follow-up choice. Select only from supplied IDs."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "FINALISTS:"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, continue_final_message=True,
        enable_thinking=False, return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=96, do_sample=False, use_cache=True)
    raw = "FINALISTS:" + tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    ids = list(dict.fromkeys(re.findall(r"L\d+E\d+N\d+[PN]", raw)))
    allowed = {item["candidate_id"] for item in positive}
    if len(ids) != keep or not set(ids) <= allowed:
        raise RuntimeError(f"Researcher Flint failed causal-button selection: {raw}")
    print("ROUTED_BUTTON_SELECTION_JSON " + json.dumps({"ids": ids, "raw_completion": raw, "fallback_used": False}, ensure_ascii=False), flush=True)
    return {"ids": ids, "raw_completion": raw}


def confirm_buttons(subject, tokenizer, config: dict, finalists: list[dict], screened_by_id: dict, all_candidates: list[dict]) -> list[dict]:
    prompts = CONFIRM_PROMPTS
    baselines = {prompt: generate_text(subject, tokenizer, prompt, config["generation_max_new_tokens"]) for prompt in prompts}
    confirmations = []
    for candidate in finalists:
        screen = screened_by_id[candidate["id"]]
        prefix = screen["button"]["prefix"]
        factor = float(screen["button"]["factor"])
        experts = resolve_attr(subject, config["experts_path"].format(layer=candidate["layer"]))
        target_norm = float(experts.down_proj[candidate["expert"], :, candidate["neuron"]].float().norm())
        pool = [x for x in all_candidates if x["layer"] == candidate["layer"] and x["id"] != candidate["id"]]
        control = min(pool, key=lambda x: abs(float(experts.down_proj[x["expert"], :, x["neuron"]].float().norm()) - target_norm))
        opposite = {**candidate, "reverse_sign": True}
        target_rows = run_condition(subject, None, tokenizer, config, candidate, factor, prompts, baselines, "target", judge_semantic=False)
        opposite_rows = run_condition(subject, None, tokenizer, config, opposite, factor, prompts, baselines, "opposite_sign", judge_semantic=False)
        control_rows = run_condition(subject, None, tokenizer, config, control, factor, prompts, baselines, "matched_neuron", judge_semantic=False)
        counts = {
            "target": sum(row["first_word"].startswith(prefix) for row in target_rows),
            "baseline": sum(first_word(text).startswith(prefix) for text in baselines.values()),
            "opposite_sign": sum(row["first_word"].startswith(prefix) for row in opposite_rows),
            "matched_neuron": sum(row["first_word"].startswith(prefix) for row in control_rows),
        }
        success = counts["target"] >= int(config["button_success_target"]) and all(counts[key] <= int(config["button_success_control"]) for key in ("baseline", "opposite_sign", "matched_neuron"))
        result = {"candidate_id": candidate["id"], "activation_label": candidate["interpretation"]["label"], "prefix": prefix, "factor": factor, "control_candidate_id": control["id"], "counts": counts, "success": success, "baselines": baselines, "target_rows": target_rows, "opposite_rows": opposite_rows, "control_rows": control_rows}
        confirmations.append(result)
        print("ROUTED_BUTTON_CONFIRMATION_JSON " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return confirmations


def confirm_candidates(subject, researcher, tokenizer, config: dict, finalists: list[dict], all_candidates: list[dict]) -> list[dict]:
    prompts = CONFIRM_PROMPTS[: config["confirm_prompts"]]
    baselines = {prompt: generate_text(subject, tokenizer, prompt, config["generation_max_new_tokens"]) for prompt in prompts}
    rng = random.Random(stable_seed(config["seed"], "routed_controls"))
    confirmations = []
    for candidate in finalists:
        pool = [x for x in all_candidates if x["layer"] == candidate["layer"] and x["id"] != candidate["id"]]
        control = {**rng.choice(pool), "interpretation": candidate["interpretation"]}
        opposite = {**candidate, "id": candidate["id"][:-1] + ("N" if candidate["pole"] == "positive" else "P"), "target_coefficient": candidate["opposite_target_coefficient"]}
        rows = []
        for factor in config["confirm_factors"]:
            rows.extend(run_condition(subject, researcher, tokenizer, config, candidate, factor, prompts, baselines, f"target_{factor:g}x"))
        rows.extend(run_condition(subject, researcher, tokenizer, config, opposite, config["control_factor"], prompts, baselines, "opposite_pole"))
        rows.extend(run_condition(subject, researcher, tokenizer, config, control, config["control_factor"], prompts, baselines, "matched_neuron_control"))
        summaries = {}
        for name in sorted({x["condition"] for x in rows}):
            chosen = [x for x in rows if x["condition"] == name]
            summaries[name] = {"n": len(chosen), "coherent_matches": sum(x["judge"]["match"] and x["judge"]["coherent"] for x in chosen), "coherent": sum(x["judge"]["coherent"] for x in chosen), "strength_sum": sum(x["judge"]["strength"] for x in chosen if x["judge"]["coherent"])}
        best_factor = max(config["confirm_factors"], key=lambda x: (summaries[f"target_{x:g}x"]["coherent_matches"], summaries[f"target_{x:g}x"]["strength_sum"]))
        best = summaries[f"target_{best_factor:g}x"]
        success = best["coherent_matches"] >= config["success_target_matches"] and summaries["matched_neuron_control"]["coherent_matches"] <= config["success_control_matches"] and summaries["opposite_pole"]["coherent_matches"] <= config["success_control_matches"]
        result = {"candidate_id": candidate["id"], "label": candidate["interpretation"]["label"], "control_candidate_id": control["id"], "best_factor": best_factor, "success": success, "summary": summaries, "baselines": baselines, "rows": rows}
        confirmations.append(result)
        print("ROUTED_CONFIRMATION_JSON " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return confirmations


def main() -> None:
    started = time.time()
    config = json.loads((ROOT / "config.json").read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    print("CONFIG_JSON " + json.dumps(config, sort_keys=True), flush=True)
    print(f"TORCH version={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two-Flint routed survey requires two visible GPUs")
    tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    researcher = load_flint(config["target_model"], "cuda:0", config.get("attn_implementation"))
    subject = load_flint(config["target_model"], "cuda:1", config.get("attn_implementation"))
    print("TWO_FLINT_ARCHITECTURE_JSON " + json.dumps({"researcher": "cuda:0", "subject": "cuda:1", "separate_model_objects": True, "communication": "structured survey tools"}, sort_keys=True), flush=True)
    plan = call_scan_plan(researcher, tokenizer, config)
    print("ROUTED_PLAN_JSON " + json.dumps(plan, ensure_ascii=False, sort_keys=True), flush=True)
    passages = load_diverse_corpus(config)
    print("ROUTED_CORPUS_JSON " + json.dumps({"selected": len(passages), "source_passages": config["corpus_source_passages"]}, sort_keys=True), flush=True)
    scan = scan_routed_neurons(config, plan["layers"], passages, subject, tokenizer)
    candidates = build_candidates(config, scan, passages, tokenizer)
    for index, candidate in enumerate(candidates):
        candidate["interpretation"] = call_label(researcher, tokenizer, candidate, config)
        print("ROUTED_CANDIDATE_JSON " + json.dumps(candidate, ensure_ascii=False, sort_keys=True), flush=True)
        print(f"ROUTED_LABEL_PROGRESS candidates={index + 1}/{len(candidates)}", flush=True)
    shortlist = call_shortlist(researcher, tokenizer, candidates, config)
    by_id = {x["id"]: x for x in candidates}
    screened = screen_candidates(subject, researcher, tokenizer, config, [by_id[x] for x in shortlist["ids"]])
    ranked = sorted(screened, key=lambda x: (x["button"]["hits"] - x["button"]["baseline_hits"], x["button"]["hits"], x["button"]["factor"]), reverse=True)
    positive = [x for x in ranked if x["button"]["hits"] >= int(config["button_screen_hits"]) and x["button"]["baseline_hits"] <= int(config["button_screen_baseline_hits"])]
    selection = call_button_finalists(researcher, tokenizer, positive, config) if positive else {"ids": [], "raw_completion": "no screen positives"}
    finalist_ids = selection["ids"]
    screened_by_id = {x["candidate_id"]: x for x in screened}
    confirmations = confirm_buttons(subject, tokenizer, config, [by_id[x] for x in finalist_ids], screened_by_id, candidates) if finalist_ids else []
    summary = {
        "study_stage": config["study_stage"],
        "target_model": config["target_model"],
        "layers": plan["layers"],
        "corpus_passages": len(passages),
        "routed_neuron_poles_scanned": len(plan["layers"]) * int(subject.config.num_experts) * int(subject.config.moe_intermediate_size) * 2,
        "candidate_poles_labeled": len(candidates),
        "shortlist_ids": shortlist["ids"],
        "screen_positive_ids": [x["candidate_id"] for x in positive],
        "finalist_ids": finalist_ids,
        "successful_candidates": [x["candidate_id"] for x in confirmations if x["success"]],
        "button_endpoint": "three-letter first-word prefix under one-token unit-normalized write pulse",
        "separate_researcher_and_subject_flints": True,
        "runtime_seconds": time.time() - started,
    }
    result = {"config": config, "plan": plan, "candidates": candidates, "shortlist": shortlist, "screened": screened, "button_selection": selection, "confirmations": confirmations, "summary": summary}
    print("ROUTED_SUMMARY_JSON " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    print("RESULT_JSON " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
