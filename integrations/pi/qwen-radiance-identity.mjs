import { createHash } from "node:crypto";
import { lstatSync, readFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";

const digest = (value) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const HISTORY_SCHEMA = "urn:qwen-r9700:pi-project-history:v1";

// Cache identity is deliberately independent of UI/controller/telemetry modules.
// Keep these exact hashes stable across module extraction and import order.
export function radianceIdentityPath(sessionFile) {
  const actual = resolve(sessionFile);
  const sessionDirectory = dirname(actual);
  if (basename(sessionDirectory) !== "sessions" || basename(dirname(sessionDirectory)) !== ".pi") return actual;
  const indexPath = join(sessionDirectory, ".identity.json");
  let indexInfo;
  try { indexInfo = lstatSync(indexPath); }
  catch (error) {
    if (error?.code === "ENOENT") return actual;
    throw error;
  }
  if (!indexInfo.isFile() || indexInfo.isSymbolicLink() || indexInfo.uid !== process.getuid() ||
      indexInfo.nlink !== 1 || (indexInfo.mode & 0o022) !== 0)
    throw new Error("unsafe project history identity index");
  const index = JSON.parse(readFileSync(indexPath, "utf8"));
  if (index?.schema !== HISTORY_SCHEMA || typeof index.sessions !== "object" || index.sessions === null)
    throw new Error("invalid project history identity index");
  const record = index.sessions[basename(actual)];
  if (record === undefined) return actual;
  const info = lstatSync(actual);
  if (!info.isFile() || info.isSymbolicLink() || info.uid !== process.getuid() ||
      !Number.isSafeInteger(record.device) || !Number.isSafeInteger(record.inode) ||
      info.dev !== record.device || info.ino !== record.inode ||
      typeof record.identity_path !== "string" || !record.identity_path.startsWith("/"))
    throw new Error("project history identity does not match the session file");
  return resolve(record.identity_path);
}

export function radianceChatIdentity(ctx) {
  const manager = ctx.sessionManager;
  const sessionFile = manager.getSessionFile();
  if (!sessionFile) throw new Error("Radiance snapshots require a saved Pi session");
  // Use the latest committed compaction in the file, including when navigating
  // an older branch. Tree navigation must not resurrect a retired generation.
  const compaction = manager.getEntries().filter((entry) => entry.type === "compaction").at(-1);
  return { id: digest([radianceIdentityPath(sessionFile), manager.getSessionId()]),
    generation: digest(compaction?.id ?? "initial"),
    title: manager.getSessionName() ?? manager.getCwd(), cwd: manager.getCwd(),
    session_file: resolve(sessionFile) };
}

export function radiancePreCompactionIdentity(ctx) {
  const chat = radianceChatIdentity(ctx);
  const compactions = ctx.sessionManager.getEntries().filter((entry) => entry.type === "compaction");
  return { ...chat, generation: digest(compactions.at(-2)?.id ?? "initial") };
}
