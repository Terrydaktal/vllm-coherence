import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { CacheResidencyTelemetry, cacheBreakdown, formatCacheBreakdown, parseResidencySample } from "../integrations/pi/qwen-cache-residency.mjs";

const abi = "a".repeat(64), chat = { id: "b".repeat(64), generation: "c".repeat(64) };
const sample = (row = {}) => ({
  schema: "urn:qwen-r9700:cache-residency:v2", observed_at_ms: Date.now(), abi,
  live: true, complete: true,
  chats: [{ chat_id: chat.id, generation: chat.generation,
    gpu_tokens: 30_000, ram_tokens: 40_000, disk_tokens: 50_000, input_tokens: 60_000, disk_saved_tokens: 40_000, ...row }],
});

test("cache tiers count each token once, in order of fastest available restore", () => {
  const value = parseResidencySample(JSON.stringify(sample()), abi);
  assert.deepEqual(cacheBreakdown(value, chat, 60_000), { gpu: 30_000, ram: 10_000, disk: 10_000, cold: 10_000, diskSaved: 40_000 });
  assert.deepEqual(cacheBreakdown(sample({ gpu_tokens: 60_000 }), chat, 60_000), { gpu: 60_000, ram: 0, disk: 0, cold: 0, diskSaved: 40_000 });
  assert.deepEqual(cacheBreakdown(sample({ gpu_tokens: 0, ram_tokens: 50_000 }), chat, 60_000), { gpu: 0, ram: 50_000, disk: 0, cold: 10_000, diskSaved: 40_000 });
  assert.equal(formatCacheBreakdown(cacheBreakdown(value, chat, 60_000)),
    "Cache ≈ GPU 30,000 · RAM 10,000 · Disk 40,000 · Cold 10,000 tok");
});

test("the newest request and output count is used during generation", () => {
  assert.deepEqual(cacheBreakdown(sample({ gpu_tokens: 61_500 }), chat, 50_000),
    { gpu: 61_500, ram: 0, disk: 0, cold: 0, diskSaved: 40_000 });
  assert.deepEqual(cacheBreakdown(sample({ input_tokens: null }), chat, 20_000),
    { gpu: 20_000, ram: 0, disk: 0, cold: 0, diskSaved: 40_000 });
  assert.equal(cacheBreakdown(sample({ input_tokens: null }), chat, null).cold, null);
});

test("a resumed prompt being rebuilt reports remaining prefill as cold, including while parked in RAM", () => {
  for (const parked of [false, true]) {
    const value = sample({ gpu_tokens: parked ? 0 : 16_384, ram_tokens: parked ? 16_384 : 0,
      disk_tokens: 16_384, input_tokens: 238_880 });
    const breakdown = cacheBreakdown(value, chat, 238_880);
    assert.equal(breakdown.disk, 0);
    assert.equal(breakdown.cold, 222_496);
    assert.match(formatCacheBreakdown(breakdown), /Disk 40,000 · Cold 222,496 tok/);
  }
});

test("compaction and other windows cannot borrow coverage from a different identity", () => {
  const value = sample();
  assert.deepEqual(cacheBreakdown(value, { ...chat, generation: "d".repeat(64) }, 4_000),
    { gpu: 0, ram: 0, disk: 0, cold: 4_000, diskSaved: 0 });
  assert.deepEqual(cacheBreakdown(value, { ...chat, id: "e".repeat(64) }, 2_000),
    { gpu: 0, ram: 0, disk: 0, cold: 2_000, diskSaved: 0 });
});

test("missing, stale or incomplete telemetry never becomes a false cold-fill claim", () => {
  assert.deepEqual(cacheBreakdown({ ...sample(), live: false }, chat, 60_000),
    { gpu: null, ram: null, disk: null, cold: null, diskSaved: 40_000 });
  assert.equal(cacheBreakdown(sample({ disk_tokens: null }), chat, 60_000).cold, null);
  assert.deepEqual(cacheBreakdown(sample({ gpu_tokens: 60_000, disk_tokens: null }), chat, 60_000),
    { gpu: 60_000, ram: 0, disk: 0, cold: 0, diskSaved: 40_000 });
  assert.equal(cacheBreakdown({ ...sample(), complete: false, chats: [] }, chat, 60_000).cold, null);
  for (const value of [
    { ...sample(), observed_at_ms: Date.now() - 6_000 },
    { ...sample(), abi: "d".repeat(64) },
    sample({ title: "no chat text in this sample" }),
    sample({ gpu_tokens: -1 }),
    sample({ disk_saved_tokens: -1 }),
    { ...sample(), chats: [...sample().chats, ...sample().chats] },
  ]) assert.throws(() => parseResidencySample(JSON.stringify(value), abi), /invalid or stale/);
});

test("retained idle RAM coverage resolves Cold while an unverified disk backup remains unknown", () => {
  const value = sample({ gpu_tokens: 0, ram_tokens: 35_207, disk_tokens: null, disk_saved_tokens: null, input_tokens: null });
  assert.equal(formatCacheBreakdown(cacheBreakdown(value, chat, 35_207)),
    "Cache ≈ GPU 0 · RAM 35,207 · Disk ? · Cold 0 tok");
  assert.deepEqual(cacheBreakdown({ ...value, live: false }, chat, 35_207),
    { gpu: null, ram: null, disk: null, cold: null, diskSaved: null });
});

test("disk backup overlaps GPU coverage and stays independent of a RAM-only tail or cold rebuild", () => {
  for (const row of [
    { gpu_tokens: 60_000, ram_tokens: 0, disk_tokens: 55_000 },
    { gpu_tokens: 0, ram_tokens: 60_000, disk_tokens: 55_000 },
    { gpu_tokens: 1000, ram_tokens: 0, disk_tokens: 1000 },
  ]) {
    const value = cacheBreakdown(sample(row), chat, 60_000);
    assert.equal(value.disk, 0);
    assert.equal(value.diskSaved, 40_000);
  }
});

test("legacy telemetry never labels restore coverage as a verified disk backup", () => {
  const modern = sample();
  const { disk_saved_tokens, ...row } = modern.chats[0];
  const legacy = parseResidencySample(JSON.stringify({ ...modern,
    schema: "urn:qwen-r9700:cache-residency:v1", chats: [row] }), abi);
  assert.equal(cacheBreakdown(legacy, chat, 60_000).diskSaved, null);
  assert.throws(() => parseResidencySample(JSON.stringify({ ...modern, chats: [row] }), abi), /invalid or stale/);
});

test("new Pi reads the shared v2 backup count and falls back safely to an older monitor", (t) => {
  const stateDirectory = mkdtempSync(join(tmpdir(), "qwen-residency-"));
  const previousAbi = process.env.QWEN_RADIANCE_CACHE_ABI;
  process.env.QWEN_RADIANCE_CACHE_ABI = abi;
  t.after(() => {
    if (previousAbi === undefined) delete process.env.QWEN_RADIANCE_CACHE_ABI;
    else process.env.QWEN_RADIANCE_CACHE_ABI = previousAbi;
    rmSync(stateDirectory, { recursive: true, force: true });
  });
  const modern = sample();
  const { disk_saved_tokens, ...row } = modern.chats[0];
  const legacy = { ...modern, schema: "urn:qwen-r9700:cache-residency:v1", chats: [row] };
  writeFileSync(join(stateDirectory, "cache-residency.json"), JSON.stringify(legacy), { mode: 0o600 });
  const telemetry = new CacheResidencyTelemetry();
  telemetry.config = { stateDirectory };
  telemetry.chat = chat;
  assert.match(telemetry.readBreakdown(60_000), /Disk \? · Cold 10,000 tok$/);
  writeFileSync(join(stateDirectory, "cache-residency-v2.json"), JSON.stringify(modern), { mode: 0o600 });
  assert.match(telemetry.readBreakdown(60_000), /Disk 40,000 · Cold 10,000 tok$/);
  modern.observed_at_ms -= 6000;
  writeFileSync(join(stateDirectory, "cache-residency-v2.json"), JSON.stringify(modern));
	assert.match(telemetry.readBreakdown(60_000), /GPU 30,000 .*Disk \? · Cold 10,000 tok$/);
});

test("cache residency consumes the combined telemetry snapshot without another file read", (t) => {
	const stateDirectory = mkdtempSync(join(tmpdir(), "qwen-combined-residency-"));
	const previousAbi = process.env.QWEN_RADIANCE_CACHE_ABI;
	process.env.QWEN_RADIANCE_CACHE_ABI = abi;
	t.after(() => {
		if (previousAbi === undefined) delete process.env.QWEN_RADIANCE_CACHE_ABI;
		else process.env.QWEN_RADIANCE_CACHE_ABI = previousAbi;
		rmSync(stateDirectory, { recursive: true, force: true });
	});
	const observed = Date.now();
	const combined = {
		schema: "urn:qwen-r9700:telemetry:v1", observed_at_ms: observed,
		scheduler: {}, worker: null, phases: null, cache: sample(), temperature: null,
	};
	writeFileSync(join(stateDirectory, "telemetry-v1.json"), JSON.stringify(combined), { mode: 0o600 });
	const telemetry = new CacheResidencyTelemetry();
	telemetry.config = { stateDirectory };
	telemetry.chat = chat;
	assert.match(telemetry.readBreakdown(60_000, observed), /GPU 30,000 .*Disk 40,000 · Cold 10,000 tok$/);
	combined.cache = sample({ gpu_tokens: 60_000 });
	combined.observed_at_ms = observed + 150;
	writeFileSync(join(stateDirectory, "telemetry-v1.json"), JSON.stringify(combined), { mode: 0o600 });
	assert.match(telemetry.readBreakdown(60_000, observed + 150), /GPU 60,000 .*Cold 0 tok$/);
});
