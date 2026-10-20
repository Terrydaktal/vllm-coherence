#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chmodSync, renameSync, rmSync } from 'node:fs';
const root = fileURLToPath(new URL('../', import.meta.url));
const destination = `${root}bin/grep2rg-native`, temporary = `${destination}.build-${process.pid}`;
const compiler = process.env.CC || 'cc';
const result = spawnSync(compiler, ['-O2', '-std=c11', '-Wall', '-Wextra', '-Werror', `${root}native/grep2rg.c`, '-o', temporary], { encoding: 'utf8' });
if (result.error || result.status !== 0) {
  rmSync(temporary, { force: true });
  // A failed rebuild must not leave an older helper silently enabled.
  rmSync(destination, { force: true });
  process.stderr.write(`grep2rg: native acceleration was not built; GNU fallback remains available.\n${result.stderr || result.error?.message || ''}`);
  process.exitCode = process.argv.includes('--optional') ? 0 : 1;
} else { chmodSync(temporary, 0o755); renameSync(temporary, destination); process.stdout.write('Built native grep2rg dispatcher.\n'); }
