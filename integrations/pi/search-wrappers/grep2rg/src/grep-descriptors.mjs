// Preserve auxiliary caller descriptors (process substitutions, /dev/fd/N),
// while Node's own CLOEXEC pipes, spool descriptors and event handles stay local.
import { readFileSync, readdirSync } from "node:fs";

// libuv can mark inherited low descriptors CLOEXEC during startup. A private
// Bash launcher snapshot therefore complements the flags visible inside Node.
const snapshot = new Set((process.env._GREP2RG_CALLER_FDS ?? "").split(",")
  .filter((value) => /^\d+$/.test(value)).map(Number).filter((value) => Number.isSafeInteger(value) && value > 2));
delete process.env._GREP2RG_CALLER_FDS;
const CLOSE_ON_EXEC = 0o2000000;

export function inheritedGrepStdio(base = ["inherit", "inherit", "inherit"]) {
  const stdio = [...base];
  let names;
  try { names = readdirSync("/proc/self/fdinfo"); } catch { return stdio; }
  for (const name of names) {
    if (!/^\d+$/.test(name)) continue;
    const descriptor = Number(name);
    if (descriptor <= 2) continue;
    try {
      const info = readFileSync(`/proc/self/fdinfo/${name}`, "utf8");
      const flags = /^flags:\s+([0-7]+)$/m.exec(info)?.[1];
      if (flags && (snapshot.has(descriptor) || !(Number.parseInt(flags, 8) & CLOSE_ON_EXEC))) stdio[descriptor] = descriptor;
    } catch {} // A descriptor may close between enumeration and inspection.
  }
  return stdio;
}
