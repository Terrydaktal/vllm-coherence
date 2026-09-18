import { lstatSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { SchedulerTelemetry } from "./qwen-radiance-scheduler-telemetry.mjs";

const SCHEMA = "urn:qwen-r9700:cache-residency:v2";
const LEGACY_SCHEMA = "urn:qwen-r9700:cache-residency:v1";
const HEX = /^[0-9a-f]{64}$/;
const count = (value) => Number.isSafeInteger(value) && value >= 0;
const exactKeys = (value, keys) => value && !Array.isArray(value) &&
  Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));

export function parseResidencySample(text, abi, now = Date.now()) {
  const sample = JSON.parse(text);
  const fields = ["gpu_tokens", "ram_tokens", "disk_tokens", "input_tokens",
    ...(sample?.schema === SCHEMA ? ["disk_saved_tokens"] : [])];
  if (!exactKeys(sample, ["schema", "observed_at_ms", "abi", "live", "complete", "chats"]) ||
      ![SCHEMA, LEGACY_SCHEMA].includes(sample.schema) || sample.abi !== abi || !HEX.test(sample.abi) ||
      !count(sample.observed_at_ms) || now - sample.observed_at_ms > 5_000 || sample.observed_at_ms > now + 1_000 ||
      typeof sample.live !== "boolean" || typeof sample.complete !== "boolean" ||
      !Array.isArray(sample.chats) || sample.chats.length > 272 ||
      sample.chats.some(row => !exactKeys(row, ["chat_id", "generation", ...fields]) ||
        !HEX.test(row.chat_id) || !HEX.test(row.generation) ||
        fields.some(key => row[key] !== null && !count(row[key]))) ||
      new Set(sample.chats.map(row => `${row.chat_id}:${row.generation}`)).size !== sample.chats.length) {
    throw new Error("invalid or stale cache residency sample");
  }
  return sample;
}

export function cacheBreakdown(sample, chat, contextTokens) {
  const row = sample.chats.find(value => value.chat_id === chat.id && value.generation === chat.generation);
  const absent = sample.complete && sample.live;
  const gpuPrefix = sample.live ? (row ? row.gpu_tokens : absent ? 0 : null) : null;
  const ramPrefix = sample.live ? (row ? row.ram_tokens : absent ? 0 : null) : null;
  const diskPrefix = row ? row.disk_tokens : sample.complete ? 0 : null;
  // A request's tokenizer count is more recent than Pi's last completed usage.
  const total = count(contextTokens) || count(row?.input_tokens)
    ? Math.max(count(contextTokens) ? contextTokens : 0, row?.input_tokens ?? 0,
      count(row?.input_tokens) ? gpuPrefix ?? 0 : 0, count(row?.input_tokens) ? ramPrefix ?? 0 : 0)
    : null;
  const clamp = value => value === null ? null : total === null ? value : Math.min(value, total);
  const gpu = clamp(gpuPrefix);
  const ram = gpu === null ? null : total !== null && gpu >= total ? 0 :
    ramPrefix === null ? null : Math.max(0, clamp(ramPrefix) - gpu);
  const memory = gpu === null || ram === null ? null : gpu + ram;
  const disk = memory === null ? null : total !== null && memory >= total ? 0 :
    diskPrefix === null ? null : Math.max(0, clamp(diskPrefix) - memory);
  const knownPrefix = Math.max(gpuPrefix ?? 0, ramPrefix ?? 0, diskPrefix ?? 0);
  const cold = total === null ? null : knownPrefix >= total ? 0 :
    memory === null || disk === null ? null : total - memory - disk;
  // The four tiers partition estimated restore coverage. The verified disk
  // snapshot overlaps them and may trail a RAM-only tail or the latest output.
  // Legacy disk_tokens can include that tail, so it cannot stand in for backup.
  const diskSaved = sample.schema === SCHEMA
    ? row ? row.disk_saved_tokens : sample.complete ? 0 : null
    : null;
  return { gpu, ram, disk, cold, diskSaved };
}

export function formatCacheBreakdown(value) {
  const number = value => count(value) ? value.toLocaleString("en-GB") : "?";
  // Disk displays the verified backup, which can overlap GPU/RAM coverage.
  // Restore-only disk coverage remains internal to the Cold calculation.
  return `Cache ≈ GPU ${number(value?.gpu)} · RAM ${number(value?.ram)} · Disk ${number(value?.diskSaved)} · Cold ${number(value?.cold)} tok`;
}

export class CacheResidencyTelemetry extends SchedulerTelemetry {
  start(ctx) {
    if (ctx.model?.id !== "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate") {
      this.stop();
      return;
    }
    super.start(ctx);
  }

  bind(ctx) {
    if (ctx.model?.id !== "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate") {
      this.stop();
      return;
    }
    super.bind(ctx);
  }

  readBreakdown(contextTokens, now = Date.now()) {
    if (!this.config || !this.chat) return undefined;
    for (const name of ["cache-residency-v2.json", "cache-residency.json"]) {
      try {
        const path = join(this.config.stateDirectory, name);
        const details = lstatSync(path);
        if (!details.isFile() || details.isSymbolicLink() || details.nlink !== 1 ||
            details.uid !== process.getuid() || (details.mode & 0o022) || details.size > 65_536) {
          throw new Error("unsafe cache residency sample");
        }
        const sample = parseResidencySample(readFileSync(path, "utf8"), process.env.QWEN_RADIANCE_CACHE_ABI, now);
        return formatCacheBreakdown(cacheBreakdown(sample, this.chat, contextTokens));
      } catch {
        // An already running older monitor still provides residency, but cannot
        // authenticate durable coverage. Use it without inventing a backup count.
      }
    }
    this.ensure(now);
    return formatCacheBreakdown(undefined);
  }
}
