#!/usr/bin/env node
// Build on the executing host, into the fixed path used by CLI and Pi. A failed
// build disables stale local acceleration instead of retaining an old binary.
import { spawnSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const target = `${root}rust/target`, binary = `${target}/release/grep2rg-rust`;
const optional = process.argv.includes('--optional');
const offline = process.argv.includes('--offline') || process.env.npm_config_offline === 'true';
const result = spawnSync('cargo', ['build', '--release', '--locked', ...(offline ? ['--offline'] : []),
  '--manifest-path', `${root}rust/Cargo.toml`, '--target-dir', target],
  { env: { ...process.env, CARGO_TARGET_DIR: target }, encoding: 'utf8' });
if (result.error || result.status !== 0) {
  rmSync(binary, { force: true });
  process.stderr.write(`grep2rg: Rust acceleration was not built; C/GNU fallback remains available.\n${result.stderr || result.error?.message || ''}`);
  process.exitCode = optional ? 0 : 1;
} else {
  process.stdout.write('Built integrated Rust grep2rg engine.\n');
}
