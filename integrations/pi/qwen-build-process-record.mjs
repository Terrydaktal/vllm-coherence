// A bounded, content-free local interface for process inspectors. It never reads
// conversations or invokes application code to identify a build.
import { lstatSync, unlinkSync } from "node:fs";
import { lstat, mkdir, open, readFile, rename, unlink } from "node:fs/promises";
import { dirname, join } from "node:path";
import { randomBytes } from "node:crypto";

export function processStartTicks(text) {
  const end = text.lastIndexOf(")");
  const ticks = end >= 0 ? text.slice(end + 2).trim().split(/\s+/)[19] : undefined;
  if (!/^\d+$/.test(ticks ?? "")) throw new Error("Unknown process start identity");
  return ticks;
}

async function ownedDirectory(path, uid) {
  const info = await lstat(path);
  if (!info.isDirectory() || info.uid !== uid || (info.mode & 0o077)) throw new Error("Unsafe build-record directory");
}

export function createBuildProcessRecord({ runtimeDir = process.env.XDG_RUNTIME_DIR, pid = process.pid,
  uid = process.getuid?.(), startTicks } = {}) {
  let path, previous, running, pending = Promise.resolve(), stopped = false, error;
  function removeOwn() {
    if (path && previous) {
      try {
        const info = lstatSync(path);
        if (info.dev === previous[0] && info.ino === previous[1]) unlinkSync(path);
      } catch { /* A missing record is explicitly unknown to the reader. */ }
    }
  }
  async function initialize() {
    if (!runtimeDir || !runtimeDir.startsWith("/") || uid === undefined) return false;
    await ownedDirectory(runtimeDir, uid);
    const root = join(runtimeDir, "pi"), directory = join(root, "processes");
    for (const item of [root, directory]) {
      await mkdir(item, { mode: 0o700 }).catch((error) => { if (error.code !== "EEXIST") throw error; });
      await ownedDirectory(item, uid);
    }
    const ticks = startTicks ?? processStartTicks(await readFile(`/proc/${pid}/stat`, "utf8"));
    path = join(directory, `${pid}-${ticks}.json`);
    return { pid, uid, start_ticks: ticks };
  }
  const initialized = initialize().catch((failure) => {
    error = `runtime-directory: ${failure.code ?? failure.message}`;
    return false;
  });
  return {
    get path() { return path; },
    get error() { return error; },
    publish(identity) {
      // Serialize writes, retaining the first successful startup identity forever.
      pending = pending.catch(() => {}).then(async () => {
        const owner = await initialized;
        if (!owner || stopped) return;
        if (!running) running = structuredClone({ ...identity, latest: undefined });
        const value = { ...running, ...owner, schema: "coherence.pi.process-build.v1", latest: identity.latest };
        const bytes = JSON.stringify(value) + "\n";
        if (Buffer.byteLength(bytes) > 512 * 1024) throw new Error("Build record exceeds size bound");
        try {
          const info = await lstat(path);
          if (!info.isFile() || info.uid !== uid || (info.mode & 0o077)) throw new Error("Unsafe existing build record");
        } catch (error) { if (error.code !== "ENOENT") throw error; }
        const temporary = join(dirname(path), `.${pid}-${randomBytes(8).toString("hex")}.tmp`);
        const file = await open(temporary, "wx", 0o600);
        try {
          await file.writeFile(bytes);
          await file.close();
          if (stopped) return;
          await rename(temporary, path);
          const info = await lstat(path);
          previous = [info.dev, info.ino];
          if (stopped) removeOwn();
        } finally {
          await file.close().catch(() => {});
          await unlink(temporary).catch((error) => { if (error.code !== "ENOENT") throw error; });
        }
      });
      return pending;
    },
    async ready() { await pending; },
    stop() {
      stopped = true;
      // Exit hooks cannot await I/O. Remove only the inode this recorder published.
      removeOwn();
    },
  };
}
