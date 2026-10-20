import { readFileSync, readdirSync } from "node:fs";

// Linux reports the per-descriptor close-on-exec bit in /proc fdinfo flags.
// Node/libuv's own descriptors carry it; caller descriptors surviving exec do not.
const O_CLOEXEC = 0o2000000;
const launchSnapshot = process.env._FIND2FD_CALLER_FDS;
delete process.env._FIND2FD_CALLER_FDS;
const callerDescriptors = typeof launchSnapshot === "string" && /^(?:\d+(?:,\d+)*)?$/.test(launchSnapshot)
  ? launchSnapshot.split(",").filter(Boolean).map(Number).filter((descriptor) => Number.isSafeInteger(descriptor) && descriptor >= 3) : [];

export function inheritedStdio(base = ["inherit", "inherit", "inherit"], descriptors = []) {
  const stdio = [...base];
  for (const descriptor of [...callerDescriptors, ...descriptors]) {
    if (!Number.isSafeInteger(descriptor) || descriptor < 3) continue;
    try {
      readFileSync(`/proc/self/fdinfo/${descriptor}`);
      stdio[descriptor] = descriptor;
    } catch { /* A caller descriptor closed since launch cannot be inherited. */ }
  }
  let descriptorNames;
  try { descriptorNames = readdirSync("/proc/self/fd"); }
  catch { return stdio; }
  for (const name of descriptorNames) {
    if (!/^\d+$/.test(name)) continue;
    const descriptor = Number(name);
    if (descriptor < 3 || !Number.isSafeInteger(descriptor)) continue;
    try {
      const text = readFileSync(`/proc/self/fdinfo/${name}`, "utf8");
      const flags = /^flags:\s+([0-7]+)\s*$/m.exec(text);
      if (flags && !(Number.parseInt(flags[1], 8) & O_CLOEXEC)) stdio[descriptor] = descriptor;
    } catch { /* Descriptors closed during the snapshot are not inheritable. */ }
  }
  return stdio;
}

export function unusedInputFd(stdio = inheritedStdio()) {
  let descriptor = 3;
  while (stdio[descriptor] !== undefined && stdio[descriptor] !== null && stdio[descriptor] !== "ignore") descriptor++;
  return descriptor;
}
