# Researcher Flint prompt

The exact system and user messages are preserved in `source-snapshot/autonomous_lab.py`. The scientific mission sent to Researcher Flint was:

> Independently investigate the subject Flint for an interpretable native neuron-pole whose causal intervention produces a repeatable, human-readable effect. Begin by reading LAB_API.md. You have up to 100 agent turns and two activation scans. Build the strongest auditable study you can within the available compute, including fresh confirmation prompts and baseline, opposite-pole, and matched-neuron controls. The complete agent conversation and every version of your code will be shared.

The system message established that Researcher Flint—not the operator—must design, write, execute, debug, and report the study. It required the model to choose layers, scan budget, candidates, prompts, doses, hypotheses, regex/endpoints, controls, and stopping rule. It also required:

- causal screening of multiple self-selected candidates and doses;
- exploration of a logarithmic dose range;
- at least three candidates at $|\mathrm{factor}| \ge 16$ before a null;
- two fresh-prompt confirmations for a positive;
- honest null reporting after both scans and at least six candidates;
- `study.py` and `REPORT.md` before finishing.

The fixed validator—not the model's prose—enforced the positive gate: at least 6/8 target matches, at most 2/8 matches in every control, and two separated confirmations of the same candidate.

