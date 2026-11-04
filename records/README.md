# Run records

These files preserve the full stdout records used for the published recount.

- `autonomous-discovery.sanitized.log` is the autonomous Researcher-Flint trajectory and result. One operational value—the hosted Hugging Face endpoint hostname—was replaced with `<REDACTED_HF_ENDPOINT>`. No scientific prompt, model response, lab call, activation context, generation, count, or configuration value was otherwise changed.
- `fixed-replication.log` is the deterministic fixed replication.
- `out-of-domain-validation.log` is the preregistered unrelated-prompt follow-up.

Original autonomous log SHA-256:

```text
3fdb9ec9623c540fd8a90b45e29c3ffc325c95aece299e3d244dcd2b3be1bccc
```

Sanitized public log SHA-256:

```text
467dd888e4fb3a0b1071847a8e3d40ec0af7348e48160652a22d649e26d5e04d
```
