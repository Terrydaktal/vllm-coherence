# Pi and persistent sessions

The launcher verifies the Pi 0.84.2 lock and patches exact known runtime bytes.
It installs to `STATE/pi/0.84.2` and exposes `STATE/bin/pi` as a symlink, preserving
unrelated Pi installations. Progress, temperatures, tool condensation/rehydration,
compaction and cache identity load with the operating prompt. `--no-context-files`
keeps AGENTS files out of Pi's context. `--search-extension` selects your own search
integration. Internal loop interruption is not enabled by the release launcher.

Transcripts are model-neutral files in `WORKSPACE/.pi/sessions`. A fresh launch
starts a new chat; `--session last` resumes the latest in this workspace. Changing
models can reuse the transcript, but incompatible model/KV state needs computation.
Add `.pi/` to the workspace's `.gitignore` to keep transcripts private.

## Compaction transaction

1. Prepare a prefix-preserving checkpoint prompt.
2. Report preparation, admission, GPU queue ownership, restore, prefill and
   checkpoint generation as separate timed phases.
3. Validate required sections, the completion marker and finish status.
4. Flush cache state, commit the transcript generation, then retire old snapshots.

Failure retains the original transcript. The widget updates in place; the durable
compaction entry records total time. Rates use a three-second window. Editor input
is preserved. Buffered tool arguments report generation usage before application.

## Shared monitoring and remote access

One file-locked temperature probe per host/state serves all windows at one second.
Scheduler/cache metadata refreshes every 0.5 seconds without copying GPU buffers.
Missing measurements remain unavailable, never fabricated as zero. Disk-saved
coverage is separate from current GPU/RAM residency.

Remote launches reserve a free port and use SSH directly; they do not nest tmux.
SSH authentication must already work noninteractively. Terminal keybindings remain
the terminal owner's configuration.

## Recovery

Start `tools/coherence serve` on the GPU host if its API is unavailable. Retain
failing artifacts when verification fails; prepare into a fresh `--state` rather
than bypassing checks. `tools/coherence cache -- audit` diagnoses publication and
cleanup failures; `--help` documents explicit flush and generation cleanup.

A new Coherence installation uses its own snapshot namespace. It does not
automatically import another backend's KV files. State/arithmetic compatibility
must be established before such a migration can reuse model state.
