#!/usr/bin/env node
import { withGrepWrapper } from '../src/grep-wrapper.mjs';

if (process.argv.length > 2) {
  process.stderr.write('Usage: grep2rg-shell-prefix\n');
  process.exitCode = 2;
} else {
  // Reuse the same non-exported function and byte guard as the Pi spawn hook.
  process.stdout.write(withGrepWrapper({ command: '' }, 'bash').command);
}
