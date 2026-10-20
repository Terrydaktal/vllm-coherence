#!/usr/bin/env node
// npm source packages contain no host-specific binary. Build the standard
// release target after installation; source-only installs remain GNU-compatible.
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const optional = process.argv.slice(2).length === 1 && process.argv[2] === '--optional';
if (process.argv.length > (optional ? 3 : 2)) {
  process.stderr.write('Usage: node tools/build-rust.mjs [--optional]\n');
  process.exit(2);
}
const root = fileURLToPath(new URL('../', import.meta.url));
const version = spawnSync('cargo', ['--version'], { cwd: root, encoding: 'utf8' });
if (optional && version.error?.code === 'ENOENT') {
  process.stderr.write('find2fd: Cargo is unavailable; using GNU find until npm run build:rust is run.\n');
  process.exit(0);
}
if (version.error || version.status !== 0) {
  process.stderr.write(`find2fd: Cargo is required to build the native release (${version.error?.message ?? version.status}).\n`);
  process.exit(1);
}
const result = spawnSync('cargo', ['build', '--release', '--locked', '--manifest-path',
  fileURLToPath(new URL('../rust/Cargo.toml', import.meta.url)), '--target-dir',
  fileURLToPath(new URL('../rust/target', import.meta.url))], { cwd: root, stdio: 'inherit' });
if (result.error) process.stderr.write(`find2fd: Rust release build failed: ${result.error.message}\n`);
process.exitCode = result.status ?? 1;
