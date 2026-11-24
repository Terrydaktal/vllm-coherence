import { randomUUID } from "node:crypto";
import { execFile } from "node:child_process";
import { lstat, mkdir, open, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { appendCompactionTiming, getCompactionProgress } from "./qwen-radiance-compaction-progress.mjs";
import { radianceBridgeRequest, radianceBridgeUrl } from "./qwen-radiance-bridge.mjs";
import installPriority from "./qwen-radiance-priority.mjs";
import installBackend from "./qwen-radiance-backend.mjs";
import { radianceChatIdentity, radiancePreCompactionIdentity } from "./qwen-radiance-identity.mjs";
export { radianceIdentityPath, radianceChatIdentity, radiancePreCompactionIdentity } from "./qwen-radiance-identity.mjs";

const exec = promisify(execFile);
const MODEL = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate";
const program = resolve(dirname(fileURLToPath(import.meta.url)), "../../scripts/qwen-radiance-cache");
const ACTIVITY_SCHEMA = "urn:qwen-r9700:radiance-active-pi:v1";
let activityFile;

export function withRadianceChat(payload, ctx) {
  if (ctx.model?.id !== MODEL || !process.env.QWEN_RADIANCE_CACHE_ABI) return payload;
  const chat = radianceChatIdentity(ctx);
  return { ...payload, cache_salt: `qwen-chat-cache-v1:${chat.id}:${chat.generation}`,
    kv_transfer_params: { ...payload.kv_transfer_params, qwen_chat: chat,
      qwen_snapshot_abi: process.env.QWEN_RADIANCE_CACHE_ABI } };
}

async function processStartTicks() {
  const stat = await readFile("/proc/self/stat", "utf8");
  const fields = stat.slice(stat.lastIndexOf(")") + 2).trim().split(/\s+/);
  if (!/^\d+$/.test(fields[19] ?? "")) throw new Error("cannot identify the Pi process start time");
  return fields[19];
}

export async function publishActivity(ctx) {
  const agentDirectory = process.env.PI_CODING_AGENT_DIR;
  const port = Number(process.env.QWEN_RADIANCE_LOCAL_PORT);
  if (!agentDirectory || !Number.isInteger(port) || port < 1 || port > 65535) return;
  const directory = join(agentDirectory, "radiance-active");
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const directoryInfo = await lstat(directory);
  if (!directoryInfo.isDirectory() || directoryInfo.isSymbolicLink() ||
      directoryInfo.uid !== process.getuid()) throw new Error("unsafe Radiance activity directory");
  const chat = radianceChatIdentity(ctx);
  const destination = join(directory, `${process.pid}.json`);
  const temporary = join(directory, `.${process.pid}.${randomUUID()}.tmp`);
  const record = { schema: ACTIVITY_SCHEMA, chat_id: chat.id, pid: process.pid, port,
    process_start_ticks: await processStartTicks(), updated_at: new Date().toISOString() };
  try {
    await writeFile(temporary, JSON.stringify(record) + "\n", { encoding: "utf8", mode: 0o600, flag: "wx" });
    await rename(temporary, destination);
    activityFile = destination;
  } finally {
    await rm(temporary, { force: true });
  }
}

export async function removeActivity() {
  if (!activityFile) return;
  const path = activityFile;
  activityFile = undefined;
  await rm(path, { force: true });
}

export async function cacheCommand(args, currentChat) {
  if (radianceBridgeUrl()) {
    if (args[0] === "list") {
      if (!currentChat?.id) throw new Error("current VM chat identity unavailable");
      return radianceBridgeRequest({ operation: "list", chat_ids: [currentChat.id] });
    }
    if (!["flush", "compact"].includes(args[0]) || args[1] !== "--identity-json" || args.length !== 3) {
      throw new Error("unsupported VM cache operation");
    }
    return radianceBridgeRequest({ operation: args[0], chat: JSON.parse(args[2]) });
  }
  const options = ["--host", process.env.QWEN_RADIANCE_CACHE_HOST ?? "ai",
    "--cache-root", process.env.QWEN_RADIANCE_CACHE_ROOT ?? "/home/lewis/.cache/qwen-radiance-public-clean-snapshot-v1"];
  // Inventory includes older ABIs so existing disk usage is not hidden.
  if (args[0] !== "list") options.push("--abi", process.env.QWEN_RADIANCE_CACHE_ABI);
  const { stdout } = await exec(program, [...options, ...args], { timeout: 120000, maxBuffer: 8 * 1024 * 1024 });
  return JSON.parse(stdout);
}

export async function flushRadianceSnapshotTail(chat, run = cacheCommand) {
  return run(["flush", "--identity-json", JSON.stringify(chat)]);
}

async function flushTranscript(chat, { allowUnwritten = false } = {}) {
  let file;
  try { file = await open(chat.session_file, "r"); }
  catch (error) {
    // Pi reserves a new session's path before creating the file on its first
    // assistant reply. Do not create an empty file: Pi uses exclusive creation.
    if (allowUnwritten && error?.code === "ENOENT") return false;
    throw error;
  }
  try { await file.sync(); } finally { await file.close(); }
  const directory = await open(dirname(chat.session_file), "r");
  try { await directory.sync(); } finally { await directory.close(); }
  return true;
}

export default function radianceCache(pi, {
  run = cacheCommand,
  activity = { publish: publishActivity, remove: removeActivity },
} = {}) {
  installPriority(pi, { identity: radianceChatIdentity });
  installBackend(pi);
  const applies = (ctx) => ctx.model?.id === MODEL && process.env.QWEN_RADIANCE_CACHE_ABI;
  const durableGenerations = new Map();
  const refreshActivity = async (ctx) => {
    if (!applies(ctx)) return;
    try { await activity.publish(ctx); } catch { /* Cache operation remains available. */ }
  };
  const onSession = async (_event, ctx) => {
    ctx.ui?.setStatus?.("qwen-cache", undefined);
    await refreshActivity(ctx);
  };
  pi.on("session_start", onSession);
  pi.on("session_switch", onSession);
  pi.on("before_provider_request", async (event, ctx) => {
    const payload = withRadianceChat(event.payload, ctx);
    if (applies(ctx)) {
      await refreshActivity(ctx);
      const chat = payload.kv_transfer_params.qwen_chat;
      if (durableGenerations.get(chat.id) !== chat.generation) {
        const allowUnwritten = !ctx.sessionManager.getEntries().some((entry) =>
          entry.type === "compaction" || (entry.type === "message" && entry.message?.role === "assistant"));
        // Retry after Pi persists the first reply; skipping is not a durable
        // commit. Existing conversations and compaction remain strict.
        if (await flushTranscript(chat, { allowUnwritten })) durableGenerations.set(chat.id, chat.generation);
      }
    }
    return payload;
  });
  pi.on("session_compact", async (event, ctx) => {
    if (!applies(ctx)) return;
    const progress = getCompactionProgress(ctx);
    let flushed = false;
    try {
      // Pi has appended the compaction entry. Flush it and its directory before
      // making the old snapshot unavailable, including on abrupt power loss.
      progress?.markAppended();
      progress?.update({ phase: "commit" });
      const chat = radianceChatIdentity(ctx);
      await flushTranscript(chat);
      flushed = true;
      progress?.markCommitted();
      if (event.compactionEntry?.details?.snapshotTailFlushed !== true) {
        await flushRadianceSnapshotTail(radiancePreCompactionIdentity(ctx), run);
      }
      durableGenerations.set(chat.id, chat.generation);
      progress?.update({ phase: "cleanup" });
      const result = await run(["compact", "--identity-json", JSON.stringify(chat)]);
      ctx.ui.setStatus("qwen-cache", undefined);
      progress?.update({ removedBytes: result.removed_file_bytes });
      await progress?.finish("complete");
    } catch (error) {
      ctx.ui.notify(`Snapshot cleanup pending: ${error.message}. The next model request retries generation retirement.`, "warning");
      await progress?.finish(flushed ? "cleanup_pending" : "commit_unconfirmed");
    }
    try {
      if (appendCompactionTiming(pi, event, ctx, progress)) {
        // The timing entry is context-free and does not change the cache
        // generation, but make its append durable for future session resumes.
        await flushTranscript(radianceChatIdentity(ctx));
      }
    } catch (error) {
      ctx.ui.notify(`Could not save completed compaction duration: ${error.message}`, "warning");
    }
  });
  pi.on("session_shutdown", async () => {
    try { await activity.remove(); } catch { /* A stale marker is rejected by PID identity. */ }
  });
  pi.registerCommand("cache", {
    description: "Show Radiance cache usage or force-flush this chat's RAM tail",
    handler: async (args, ctx) => {
      if (!applies(ctx)) return;
      try {
        if (args.trim() === "flush") {
          const result = await flushRadianceSnapshotTail(radianceChatIdentity(ctx), run);
          ctx.ui.notify(`Snapshot tail ${result.status.replaceAll("_", " ")} at ${result.tokens.toLocaleString("en-GB")} tokens.`, "info");
          return;
        }
        if (args.trim()) throw new Error("usage: /cache [flush]");
        const reports = await run(["list", "--json"], radianceBridgeUrl() ? radianceChatIdentity(ctx) : undefined);
        const lines = reports.flatMap((report) => report.chats.map((chat) =>
          `${chat.id.slice(0, 12)}  ${(chat.file_bytes / 1024 ** 3).toFixed(2)} GiB  ${chat.tokens} tok  ${chat.status}  ${chat.title || chat.cwd}`));
        const legacyBytes = reports.reduce((sum, report) => sum + report.legacy_unassigned_bytes, 0);
        lines.push(`Unassigned legacy cache: ${(legacyBytes / 1024 ** 3).toFixed(2)} GiB (preserved)`);
        ctx.ui.notify(lines.join("\n"), "info");
      } catch (error) { ctx.ui.notify(`Cannot read snapshot usage: ${error.message}`, "error"); }
    },
  });
}
