# Contributing

Keep each change to one logical repair or optimization. Include the trigger,
before/after behavior, provenance and a regression that detects the original
failure. Attribute upstream fixes and link their commits/issues.

Numerical changes must declare arithmetic, shapes, dtype, layout and execution
mode. Compare logical state as well as output. Use common-input operator tests,
then forced-token integration replay. Separate performance timing from capture
and verify actual compiled graph use.

Distinguish **PROVED**, **RUNTIME-CHECKED**, **TESTED**, **ASSUMED**, **UNPROVED**.
Proofs cover stated models and implementation bindings; samples are not universal
proofs. Unknown sources, unsupported modes and failed negative controls fail closed.

Never submit private conversations, token IDs, KV/tensors, credentials or raw logs.
Use synthetic/public inputs or aggregate measurements. Keep the pinned profile's
admission constraints; add newly qualified profiles for other hardware/models.
See [verification](docs/VERIFICATION.md).
