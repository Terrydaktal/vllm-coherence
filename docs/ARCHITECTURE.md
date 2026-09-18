# Architecture

```text
Pi + Coherence extensions
  │ streaming request + chat/generation identity
  ▼
Loopback API / SSH tunnel → snapshot ABI admission
  ▼
Response-boundary scheduler + per-answer priority
  │ GPU bank ↔ pinned handover RAM ↔ shared CPU offload arena
  ▼
Corrected compiled target + D7 drafter
  ├─ immutable attention blocks → compressed disk objects
  └─ changing recurrent tail → RAM journal → verified disk checkpoint
```

Numerical repairs install before compilation and graph capture. Optimized
replacements check source, build and evidence identities; unsupported shapes use
their corrected fallback. Unknown artifacts fail admission. Intermediate BF16
casts are preserved, and the existing nine-slot GDN/conv layout remains in use.

Global-256 selects candidates from complete INT2 score rows and reranks BF16
weights. Candidate completeness is empirical. The full-BF16 control exposes the
complete vocabulary; draft-head approximation is a separate concern.

Snapshots bind chat, generation, prefix and compatible runtime identity. Immutable
blocks are reused. A tail stays in RAM until its threshold or forced flush. Only
verified objects can replace the published disk head. Compaction commits a new
conversation generation before retiring the old one. Disk usage and cumulative
completed writes are separate metrics.

At equal priority, scheduling hands over at completed responses/tool pauses. A
two-second tool grace avoids unnecessary handovers. Priority 1 retains ownership
until the answer ends, including tools; priority 2 requests immediate preemption.
Answer ownership spans multiple provider requests and waiting status explains the
owner and phase.

Conformance observes logical values, positions and state versions. Physical block
IDs may differ. Common-input operator replay isolates kernels; forced-token replay
prevents an early sampling difference hiding the first state error. Experimental
checked publication is separate from ordinary production inference: the normal
server does not reference-check every token in real time.
