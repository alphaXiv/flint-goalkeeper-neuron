# Independent recount

The fixed replication log contains two `confirm` records with disjoint prompt sets. No prompt contains the scoring word `goalkeeper`.

An independent case-insensitive recount of every raw continuation gives:

| Round | Target | Baseline | Opposite pole | Matched neuron |
|---|---:|---:|---:|---:|
| 1 | 7/8 | 0/8 | 0/8 | 0/8 |
| 2 | 8/8 | 0/8 | 0/8 | 0/8 |

The fixed replication exactly matches the autonomous discovery run's recorded aggregate counts. The subject generations are deterministic under this setup.

The output quality is mixed. Several target continuations are readable goalkeeper-themed prose; several are repetitive or contain instruction-like artifacts. The literal semantic injection is robust, but this should not be described as 15 polished answers.

The preregistered out-of-domain record contains 16 unique non-sports prompts. An independent exact-word recount gives 16/16 target matches, 0/16 baseline matches, 0/16 opposite-pole matches, and 0/16 matched-neuron matches.

The public autonomous record is sanitized: the hosted endpoint hostname was replaced with `<REDACTED_HF_ENDPOINT>`. No scientific content or result was removed. SHA-256 hashes for the private original and public sanitized record are documented in `records/README.md`.
