#!/usr/bin/env node
import { withFindWrapper } from '../src/find-wrapper.mjs';

if (process.argv.length > 2) {
  process.stderr.write('Usage: find2fd-shell-prefix\n');
  process.exitCode = 2;
} else {
  process.stdout.write(withFindWrapper({ command: '' }, 'bash').command);
}
