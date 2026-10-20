#!/usr/bin/env node
import { runGrep } from '../src/grep-runtime.mjs';

// This entry point consumes GNU grep argv verbatim, including --help/--version.
const status = await runGrep(process.argv.slice(2));
process.exitCode = status;
// A blocked consumer can keep a pending stdout write alive after cleanup.
// Cancellation intentionally discards unfinished output; ordinary output drains.
if (status === 129 || status === 130 || status === 143) process.exit(status);
