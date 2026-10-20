#!/usr/bin/env node
import { runFind } from '../src/find-runtime.mjs';

// Preserve GNU's option language, help and version rather than reserving flags.
process.exitCode = await runFind(process.argv.slice(2));
