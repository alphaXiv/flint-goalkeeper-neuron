# Flint finds a goalkeeper-associated direction in Flint

An autonomous `RecursiveSI/flint` researcher was given a file workspace and a small activation/intervention API connected to a separately loaded Flint checkpoint. It selected a routed-expert neuron-pole, interpreted its natural activation contexts, screened causal interventions, and wrote two controlled confirmation tests.

The selected coordinate was `L20E73N184P`. Its strongest natural contexts were field-hockey goalkeeper biographies. At factor 64, the two Flint-authored confirmations produced `goalkeeper` matches in 7/8 and 8/8 target continuations; baseline, opposite-pole, and norm-matched-neuron controls produced 0/48 matches.

We subsequently froze the coordinate, dose, scoring rule, decoding setup, and controls, then tested 16 operator-authored prompts unrelated to sports. The intervention produced `goalkeeper` in 16/16 raw continuations; all three controls again produced 0/48.

## Results

| Study | Target | Baseline | Opposite pole | Matched neuron |
|---|---:|---:|---:|---:|
| Flint confirmation A | 7/8 | 0/8 | 0/8 | 0/8 |
| Flint confirmation B | 8/8 | 0/8 | 0/8 | 0/8 |
| Out-of-domain follow-up | 16/16 | 0/16 | 0/16 | 0/16 |

These are literal, case-insensitive whole-continuation matches for `goalkeeper`. Several high-dose outputs are repetitive or malformed. The result supports a goalkeeper-associated causal write direction under this intervention setup, not a perfectly monosemantic neuron during ordinary inference.

## Division of labor

- **Researcher Flint:** chose the scan, candidate, interpretation, factor, exploratory tests, two confirmation prompt sets, and scoring regex; wrote and executed its study code.
- **Subject Flint:** supplied the measured activations and every generated continuation.
- **Fixed harness:** collected activation tails, exposed the laboratory API, applied interventions, generated controls, and recorded calls.
- **Operator:** built the general harness and launched the search. After discovery, the operator authored the fixed replication and the 16-prompt out-of-domain follow-up.

See [`ROLE_SPLIT.md`](ROLE_SPLIT.md) and [`RESEARCH_PROMPT.md`](RESEARCH_PROMPT.md) for the precise boundary.

## Repository layout

- `self_interp/` — the runnable dependency closure for the autonomous study and its two fixed validations. Scientific routines are copied from the immutable experiment commits; `run.py` is a minimal public dispatcher with unrelated historical stages removed.
- `configs/` — the three configurations used here. The public discovery config replaces the hosted endpoint URL with a placeholder.
- `records/` — persisted run records. The discovery record is sanitized only to remove the hosted endpoint hostname.
- `data/` — normalized counts and raw validation rows extracted from the records.
- `recount.py` — independent exact-word recount of the two validation datasets.
- `viewer/` — Marimo source for the interactive evidence browser plus a static exported HTML preview.
- `SEARCH_LEDGER.md` — all six autonomous search outcomes, including five nulls.

## Recount the published outputs

```bash
python recount.py
```

Expected headline output:

```text
fixed replication: target 15/16; controls 0/48
out-of-domain: target 16/16; controls 0/48
```

The recount reads the published raw rows and does not trust the recorded aggregate counts.

To open the evidence browser:

```bash
uvx marimo run viewer/goalkeeper_direction.py
```

## Re-run inference

The subject model requires a CUDA GPU and substantial host memory. The original jobs used one GPU. Install the locked environment, select a configuration, and run the fixed entrypoint:

```bash
uv sync --locked
cp configs/out-of-domain-validation.json config.json
uv run --locked python -m self_interp.run
```

For the autonomous discovery stage, copy `configs/autonomous-discovery.json`, replace `REPLACE_WITH_FLINT_ENDPOINT_URL` with an OpenAI-compatible endpoint serving `RecursiveSI/flint`, and provide its authentication token as `HF_TOKEN`. The autonomous trajectory is stochastic; the fixed validation stages are the direct reproduction targets.

## Evidence and limitations

- [`REPORT.md`](REPORT.md) summarizes the experiment.
- [`ACTIVATION_EVIDENCE.md`](ACTIVATION_EVIDENCE.md) contains the natural activation contexts.
- [`AUDIT.md`](AUDIT.md) documents the independent recount and public-log redaction.
- [`SEARCH_LEDGER.md`](SEARCH_LEDGER.md) prevents the positive result from concealing the failed searches.

The intervention adds the selected neuron-pole's output direction directly to the routed MLP output, bypassing normal expert routing. Factor 64 is large. Accordingly, “goalkeeper-associated causal direction” is the intended description.
