# Radiance chat automatic continuation

`scripts/pi-remote-qwen-radiance` enables recovery from omitted tool announcements
for every chat using its Radiance model, regardless of session ID or working
directory. The earlier, broader repair of malformed structured output and
reasoning-only stops remains enabled for these sessions:

- `/home/lewis/tasks/searchtool` (session `01a02132-9518-735b-805a-7b66a18c023d`)
- `/home/lewis/tasks/money` (session `01a067d9-69cf-761f-8503-6554ccfd7703` or `01a064da-27e6-7108-992a-7aa6ef3195d1`)
- `/home/lewis/tasks/moneychat` (session `01a07934-a41f-7a80-8fb3-3b0e0b5d3ab5`)

That broader repair independently requires the listed directory to match Pi's
context and session header, with the corresponding session ID. Ordinary
announcement recovery instead checks the Radiance model and Qwen provider/API.
Restart Pi through the Radiance launcher to enable this in previously excluded
projects. An existing Pi window that already loaded this extension can use
`/reload` while idle.

The launcher explicitly loads `qwen-searchtool-recovery.mjs` because its
`--no-extensions` flag disables normal discovery. The dependency is checked
before any remote startup. The backend, tokenizer, model, tool schemas, sampling,
and snapshot ABI are unchanged. No old transcript or snapshot was edited.

## Behaviour

- A normal provider stop ending with a colon or a short action announcement,
  without a tool call, gets a queued internal continuation after `agent_end`. A colon in a
  streaming response is never a trigger. The final response is checked again at
  the end of the run, so a later tool step or completed answer cancels the pending
  recovery.
- A trailing colon is sufficient, including `Results:` and trailing whitespace.
  It does not need an action verb, a particular wording, or a short paragraph;
  this rule also applies inside unfinished code and quotations. Without a final
  colon, explicit announcements ending with a full stop still use the wording
  heuristic. That fallback handles balanced inline commands and Markdown emphasis
  but excludes quotations, fenced/indented code, headings, conditional promises,
  questions and permission requests.
- Recovery is bounded to two continuations per real user message. The model
  creates the complete call; no command or tool arguments are fabricated.
- The previously enabled structured-output and reasoning-only recovery follows
  the same bound within its existing session scope.
- Earlier text/reasoning stays in history. Failed partial tool entries cannot
  dispatch. The extension avoids Pi's older error-retry path, which removes the
  failed message. It does not hide the original failure or restart successful
  tool steps as a replay of the user turn.
- Identical completed tool calls (same name and canonicalized argument values)
  emitted by the recovering response are blocked before execution, even under
  new call IDs. Semantically equivalent commands with different arguments cannot
  be universally identified by this check. Recovery instructions also explicitly
  tell the model to use existing results and respect pending approval.
- Cancellation and queued/steered user input take precedence. A colon requests
  continuation, never grants user approval. Recovery still tells the model to
  respect pending approval and give a final answer if the task is already complete.
  The colon rule is a chosen recovery policy, not proof of the original stop's cause.
- Context/token limits, ordinary transport errors, and tool execution errors
  retain their existing handling; this extension does not blindly retry them.
- The continuation uses Pi's `sendMessage` with `display: false`, rather than
  `sendUserMessage`. No visible or persisted user `go ahead` entry is created.
  There is still a small durable **extension** entry containing the instruction,
  reason and attempt number. Reload restores its budget from these entries, and
  the stable appended context avoids removing a temporary instruction from the
  middle of a cached prefix on the next tool turn. It is hidden from the normal
  chat UI, not erased from diagnostics.
- Recovery uses a temporary working message; it does not pin a footer status.
  After two attempts a warning reports that recovery stopped.

This is a client continuation mechanism, distinct from the isolated parser and
generation-grammar candidates in `experiments/radiance-public`. It does not
claim to fix every malformed generation or prove backend numerical equivalence.

## Validation

```bash
node --test tests/test_searchtool_recovery.mjs tests/test_searchtool_recovery_sdk.mjs
shellcheck scripts/pi-remote-qwen-radiance
shfmt -d scripts/pi-remote-qwen-radiance
```

Tests exercise actual Pi 0.84.2 scheduling with a synthetic provider and number
lookup tool, including a new Radiance session outside the older scope, inline
commands, shorthand announcements, bare trailing colons, and repeated failure that exhausts the
budget. They verify earlier output survives, completed tool calls do not execute
twice, the provider prefix remains intact, and only the genuine user's prompt
is saved with role `user`. Saved-history checks read only transcripts created by
the tests. No network request, GPU generation, real search or private user-session
action is executed by these tests. A launcher dry run from a temporary project
also verifies that recovery is enabled outside the older directory list.
