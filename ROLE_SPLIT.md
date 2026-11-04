# Role split

## Researcher Flint

Researcher Flint received a general objective, the lab API documentation, a writable workspace, and the current lab state. In the successful run it chose the scan, read activation cards, screened candidates, chose `L20E73N184P`, chose factor 64, authored fresh prompts and the regex `goalkeeper`, executed the study, and wrote the report.

## Subject Flint

Subject Flint was a separate loaded `RecursiveSI/flint` model object. It supplied activations during scans and generated every baseline, target, opposite-pole, and matched-neuron continuation.

## Fixed harness

The harness translated Researcher Flint's lab calls into activation collection and interventions. It deterministically selected the norm-matched control, counted regex matches, recorded all calls, and rejected a claimed confirmation unless two separated 8-prompt tests each had at least 6 target matches and no more than 2 matches in every control.

## Operator

The operator wrote the general-purpose harness and launched the run. The successful run's machine-readable result records:

- `operator_selected_candidates: false`
- `operator_supplied_hypotheses: false`
- `operator_supplied_prompts: false`
- `operator_supplied_regex: false`

The operator did not intervene in the autonomous run's live scientific choices. Post-run work included recounting and packaging the evidence, writing the fixed replication, and authoring the later 16-prompt out-of-domain validation. That follow-up reused Flint's chosen candidate, dose, and regex; it does not add to the autonomy claim.
