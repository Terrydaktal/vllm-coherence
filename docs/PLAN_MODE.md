# Persistent plans and read-only planning

Pi tracks substantial tasks through the `manage_task_plan` tool. The saved plan contains
a goal, ordered steps with stable IDs, constraints, notes, relevant file paths
and source/evidence entry IDs. Normal execution can create and update this plan
while work proceeds; simple one-step work does not need a plan. Tracking a plan
does not add an approval step.

Read-only planning is an explicit user-selected mode:

| Command | Effect |
| --- | --- |
| `/plan` | Toggle read-only planning and ordinary execution. |
| `/plan on` | Enable read-only planning. |
| `/plan off` or `/plan execute` | Resume ordinary execution. |
| `/plan show` | Display the current filtered plan, up to 12,000 characters, without a model request. |
| `/plan clear` | Clear the active plan, preserve its history, and keep the current mode. |

The status line shows `plan: read only` in planning mode and `task plan` when
ordinary execution has a saved plan. The model cannot change execution mode
through `manage_task_plan`; resuming execution requires the user command.

## Plan updates

The model uses `manage_task_plan` actions `create`, `update`, `revise`, `show` and `clear`.
`create` requires a goal and steps, and cannot overwrite an existing plan.
`update` patches steps by their existing IDs. `revise` replaces the ordered step
list and can add or remove steps. New steps receive monotonically allocated
IDs such as `s1`; removed automatic IDs are not reused. Supplied constraints,
notes and file lists replace their corresponding fields.

Steps use `pending`, `in_progress`, `completed` or `blocked`, with at most one
step in progress and 30 steps total. For example, the model can create:

```json
{
  "action": "create",
  "goal": "Add the requested CLI option",
  "relevantFiles": ["src/cli.mjs"],
  "steps": [
    { "title": "Inspect option handling", "status": "in_progress" },
    { "title": "Implement the option" },
    { "title": "Verify the resulting behavior" }
  ]
}
```

New source links default to the latest user message. Explicit source and evidence
IDs must exist on the selected session branch. Completion statuses and evidence
notes are model-reported claims: an existing cited entry does not establish that
the result is correct or that a check passed. Use `manage_task_plan show` to expand a
bounded excerpt and `pi_session_search` to inspect its cited source entries.

## Persistence and context

Versioned `qwen-task-plan` custom entries store complete structured snapshots in
Pi's session JSONL. State is replayed from the selected root-to-leaf branch, so
resume, compaction and branch navigation recover that branch's plan. Sibling
plans are not imported, and a cleared plan is not resurrected from older entries.

When needed, the extension adds a bounded hidden plan message to active context.
An unchanged plan already represented by a hidden message or `manage_task_plan` result
is not injected again. This uses session metadata without repeated visible file
reads. The normal plan excerpt is 3,000 characters; set
`QWEN_PI_PLAN_CONTEXT_MAX_CHARS` to adjust it, up to 12,000. In-progress, blocked
and pending steps appear before completed steps, with explicit omission notices.

`/context` exclusions suppress linked goals, steps, constraints, notes, files and
evidence, including dependencies on tool groups and compacted ancestors. Old
plan context is withheld when its linked sources are excluded. Saved metadata
remains available for user recovery. Pi's loaded project instructions remain in
its system prompt, separate from plan records and compaction restoration.

## Read-only tools

Read-only mode permits known implementations of read/search/list tools,
`pi_session_search`, `rehydrate_tool_result` and `manage_task_plan` metadata updates.
Tool provenance is checked; a custom tool with the same name does not gain
permission automatically. File edits, writes, tests and unknown tools are blocked.

`bash` permits one parsed literal command from a small option allowlist:
`pwd`, `cat`, `ls`, `head`, `tail`, `wc`, `stat`, `tree`, `rg`, `fd`, and Git index
metadata through `git ls-files`. Commands are rewritten to absolute system
binaries with quoted arguments. Shell operators, pipelines, redirects,
expansions, escapes, newlines and unknown options are blocked. `git status` and
`git diff` are blocked because Git can execute repository helpers. User `!` and
`!!` shell commands are also blocked. An invalid saved plan fails closed.

The gate checks Pi tool dispatch; it is not an operating-system sandbox. Enter
`/plan execute` before implementation or verification commands. See
[COMPACTION_MEMORY.md](COMPACTION_MEMORY.md) for the bounded handoff of plans,
file snapshots and historical execution reminders.
