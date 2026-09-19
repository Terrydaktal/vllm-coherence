import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, mkdirSync, readdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
	parseSchedulerSample,
	parseCombinedTelemetry,
	parseRequestPhases,
	requestPhaseStatus,
	SchedulerTelemetry,
	schedulerStatusForChat,
} from "../integrations/pi/qwen-radiance-scheduler-telemetry.mjs";
import { radianceChatIdentity } from "../integrations/pi/qwen-radiance-cache.mjs";

const legacySchema = "urn:qwen-r9700:scheduler-telemetry:v1";
const schema = "urn:qwen-r9700:scheduler-telemetry:v2";
const chat = { id: "a".repeat(64), generation: "b".repeat(64) };
const otherChat = "c".repeat(64);

test("priority ownership names the whole answer and rejects malformed status", () => {
	const now = Date.now(), value = sample(now);
	value.backend.scheduler.priority_hold = { chat_id: otherChat, generation: "d".repeat(64), priority: 2 };
	const parsed = parseSchedulerSample(JSON.stringify(value), now);
	const observation = schedulerStatusForChat(parsed, chat);
	assert.equal(observation.priorityHold.priority, 2);
	assert.equal(schedulerStatusForChat(parsed, { id: otherChat, generation: "d".repeat(64) }).priorityHold, undefined);
	const status = requestPhaseStatus({ ...observation, requestPhase: {
		phase: "priority_wait", chat_id: chat.id, blocker: value.backend.scheduler.priority_hold,
		phase_elapsed_ms: 5000,
	} });
	assert.match(status.phase, /higher-priority/);
	assert.match(status.detail, /keeps the GPU through tools until its answer finishes/);
	for (const invalid of [null, { ...value.backend.scheduler.priority_hold, priority: 3 },
		{ ...value.backend.scheduler.priority_hold, prompt: "forbidden" }]) {
		assert.throws(() => parseSchedulerSample(JSON.stringify({ ...value, backend: {
			...value.backend, scheduler: { ...value.backend.scheduler, priority_hold: invalid },
		} }), now), /priority hold/);
	}
});

function scheduler(now) {
	return {
		pid: 1234,
		updated_at: now / 1000,
		quantum_seconds: 30,
		switches: 4,
		cached_chats: 2,
		max_cached_chats: 5,
		requests: [
			{
				chat_id: chat.id,
				generation: chat.generation,
				state: "paused",
				computed_tokens: 52_635,
				input_tokens: 60_000,
			},
			{
				chat_id: otherChat,
				generation: "d".repeat(64),
				state: "running",
				computed_tokens: 77_000,
				input_tokens: 70_000,
			},
		],
	};
}

function worker(now) {
	return {
		pid: 1234,
		updated_at: now / 1000 - 10,
		allocated_bytes: 10_000,
		reserved_capacity_bytes: 20_000,
		cached_chats: 2,
		switches: 4,
		last_transfer_bytes: 8_000,
		last_transfer_seconds: 0.4,
		transferred_bytes: 80_000,
		transfer_seconds: 4,
		last_allocation_bytes: 0,
		last_allocation_seconds: 0,
		allocation_events: 1,
		allocation_seconds: 2.5,
		generation_replacements: 1,
		last_handover: "swap",
		residency: {
			active: { chat_id: otherChat, generation: "d".repeat(64) },
			images: [
				{
					chat_id: chat.id,
					generation: chat.generation,
					data_bytes: 8_000,
					allocated_bytes: 10_000,
				},
			],
			free_buffer_bytes: 0,
			staging_buffer_bytes: 2_000,
		},
	};
}

function sample(now, { legacy = false } = {}) {
	if (legacy) {
		return { schema: legacySchema, observed_at_ms: now, scheduler: scheduler(now) };
	}
	return {
		schema,
		observed_at_ms: now,
		backend: { scheduler: scheduler(now), worker: worker(now) },
	};
}

test("scheduler telemetry validates content-free state and selects the active chat", () => {
	const now = Date.now();
	const parsed = parseSchedulerSample(JSON.stringify(sample(now)), now);
	const status = schedulerStatusForChat(parsed, chat);
	assert.equal(status.available, true);
	assert.equal(status.request.state, "paused");
	assert.equal(status.request.computed_tokens, 52_635);
	assert.equal(status.otherRunningChatId, otherChat);
	assert.equal(status.otherRunningRequest.computed_tokens, 77_000);
	assert.deepEqual(status.activeChat, { chat_id: otherChat, generation: "d".repeat(64) });
	assert.equal(status.cacheResidency, "ram");
	assert.equal(status.ramCacheBytes, 8_000);
	assert.equal(status.workerAvailable, true);
	assert.equal(status.quantumSeconds, 30);
});

test("combined telemetry preserves multi-rate sources behind one authenticated snapshot", () => {
	const now = Date.now();
	const value = {
		schema: "urn:qwen-r9700:telemetry:v1",
		observed_at_ms: now,
		scheduler: scheduler(now),
		worker: worker(now),
		phases: null,
		cache: { schema: "urn:qwen-r9700:cache-residency:v2" },
		temperature: { schema: "urn:qwen-r9700:gpu-temperature:v2" },
	};
	assert.deepEqual(parseCombinedTelemetry(JSON.stringify(value), now), value);
	for (const bad of [
		{ ...value, schema: "wrong" },
		{ ...value, temperature: [] },
		{ ...value, observed_at_ms: now - 6_000 },
		{ ...value, extra: "not allowed" },
	]) {
		assert.throws(() => parseCombinedTelemetry(JSON.stringify(bad), now), /combined telemetry/);
	}
});

test("legacy scheduler samples remain readable without physical-bank telemetry", () => {
	const now = Date.now();
	const parsed = parseSchedulerSample(JSON.stringify(sample(now, { legacy: true })), now);
	const status = schedulerStatusForChat(parsed, chat);
	assert.equal(status.available, true);
	assert.equal(status.workerAvailable, false);
	assert.equal(status.activeChat, undefined);
	assert.equal(status.cacheResidency, undefined);
});

test("zero quantum reports response-boundary scheduling with normal telemetry freshness", () => {
	const now = Date.now();
	const value = sample(now);
	value.backend.scheduler.quantum_seconds = 0;
	value.backend.scheduler.requests[0].state = "queued";
	value.backend.scheduler.updated_at = (now - 10_000) / 1000;
	const status = schedulerStatusForChat(parseSchedulerSample(JSON.stringify(value), now), chat);
	assert.equal(status.policy, "response_boundary");
	assert.equal(status.quantumSeconds, 0);
	assert.equal(status.request.state, "queued");
	value.backend.scheduler.quantum_seconds = -1;
	assert.throws(() => parseSchedulerSample(JSON.stringify(value), now), /payload/);
});

test("scheduler telemetry reads the combined snapshot on a 100 ms cadence", () => {
	const root = mkdtempSync(join(tmpdir(), "qwen-combined-scheduler-"));
	const state = join(root, "state");
	mkdirSync(state, { recursive: true, mode: 0o700 });
	chmodSync(root, 0o700);
	chmodSync(state, 0o700);
	const now = Date.now();
	const value = sample(now);
	const combined = {
		schema: "urn:qwen-r9700:telemetry:v1", observed_at_ms: now,
		scheduler: value.backend.scheduler, worker: value.backend.worker,
		phases: null, cache: null, temperature: null,
	};
	writeFileSync(join(state, "telemetry-v1.json"), JSON.stringify(combined), { mode: 0o600 });
	const telemetry = new SchedulerTelemetry();
	telemetry.config = { stateDirectory: state };
	telemetry.chat = chat;
	assert.equal(telemetry.read(now).request.state, "paused");
	combined.observed_at_ms = now + 50;
	combined.scheduler = { ...combined.scheduler, updated_at: (now + 50) / 1000,
		requests: combined.scheduler.requests.map((row) => ({ ...row, state: "running" })) };
	writeFileSync(join(state, "telemetry-v1.json"), JSON.stringify(combined), { mode: 0o600 });
	assert.equal(telemetry.read(now + 50).request.state, "paused");
	assert.equal(telemetry.read(now + 101).request.state, "running");
	telemetry.stop();
	rmSync(root, { recursive: true, force: true });
});

test("tool grace validates a bounded countdown without exposing tool contents", () => {
	const now = Date.now();
	const value = sample(now);
	value.backend.scheduler.tool_grace = {
		chat_id: otherChat, generation: "d".repeat(64), phase: "tool_grace", remaining_seconds: 1.5,
	};
	const parsed = parseSchedulerSample(JSON.stringify(value), now);
	assert.equal(schedulerStatusForChat(parsed, chat).toolGrace.remaining_seconds, 1.5);
	assert.equal(schedulerStatusForChat(parsed, { id: otherChat, generation: "d".repeat(64) }).toolGrace, undefined);
	for (const invalid of [null, { ...value.backend.scheduler.tool_grace, remaining_seconds: 100 },
		{ ...value.backend.scheduler.tool_grace, arguments: "must not be transported" }]) {
		assert.throws(() => parseSchedulerSample(JSON.stringify({ ...value, backend: {
			...value.backend, scheduler: { ...value.backend.scheduler, tool_grace: invalid },
		} }), now), /grace telemetry/);
	}
});

test("scheduler telemetry rejects stale, malformed, or content-bearing rows", () => {
	const now = Date.now();
	assert.throws(
		() => parseSchedulerSample(JSON.stringify({ ...sample(now), observed_at_ms: now - 6_000 }), now),
		/stale/,
	);
	assert.throws(
		() => parseSchedulerSample(JSON.stringify({
			...sample(now),
			backend: {
				...sample(now).backend,
				scheduler: { ...sample(now).backend.scheduler, updated_at: (now - 61_000) / 1000 },
			},
		}), now),
		/stale/,
	);
	assert.throws(
		() => parseSchedulerSample(JSON.stringify({
			...sample(now),
			backend: {
				...sample(now).backend,
				scheduler: {
					...sample(now).backend.scheduler,
					requests: [{ ...sample(now).backend.scheduler.requests[0], title: "must not be transported" }],
				},
			},
		}), now),
		/payload/,
	);
	assert.throws(
		() => parseSchedulerSample(JSON.stringify({
			...sample(now),
			backend: {
				...sample(now).backend,
				worker: { ...sample(now).backend.worker, title: "must not be transported" },
			},
		}), now),
		/worker telemetry payload/,
	);
});

test("scheduler telemetry lazy-starts on a request and binds its content-free sample", async (t) => {
	const root = mkdtempSync(join(tmpdir(), "qwen-scheduler-telemetry-"));
	const state = join(root, "state");
	const clients = join(state, "clients");
	mkdirSync(clients, { recursive: true, mode: 0o700 });
	chmodSync(root, 0o700);
	chmodSync(state, 0o700);
	chmodSync(clients, 0o700);
	const ctx = {
		mode: "tui",
		model: { provider: "qwen-r9700" },
		sessionManager: {
			getSessionFile: () => join(root, "synthetic-session.jsonl"),
			getEntries: () => [],
			getSessionId: () => "synthetic-session-id",
			getSessionName: () => "synthetic",
			getCwd: () => root,
		},
	};
	const identity = radianceChatIdentity(ctx);
	const now = Date.now();
	const payload = sample(now);
	payload.backend.scheduler.requests[0] = {
		...payload.backend.scheduler.requests[0],
		chat_id: identity.id,
		generation: identity.generation,
	};
	writeFileSync(join(state, "scheduler-v2.json"), JSON.stringify(payload), { mode: 0o600 });

	const previous = {
		state: process.env.QWEN_RADIANCE_GPU_TEMPERATURE_STATE,
		helper: process.env.QWEN_RADIANCE_SCHEDULER_HELPER,
		host: process.env.QWEN_RADIANCE_CACHE_HOST,
		abi: process.env.QWEN_RADIANCE_CACHE_ABI,
	};
	process.env.QWEN_RADIANCE_GPU_TEMPERATURE_STATE = state;
	process.env.QWEN_RADIANCE_SCHEDULER_HELPER = "/usr/bin/true";
	process.env.QWEN_RADIANCE_CACHE_HOST = "lewis@ai";
	process.env.QWEN_RADIANCE_CACHE_ABI = "1".repeat(64);
	t.after(() => {
		for (const [name, value] of Object.entries(previous)) {
			const key = {
				state: "QWEN_RADIANCE_GPU_TEMPERATURE_STATE",
				helper: "QWEN_RADIANCE_SCHEDULER_HELPER",
				host: "QWEN_RADIANCE_CACHE_HOST",
				abi: "QWEN_RADIANCE_CACHE_ABI",
			}[name];
			if (value === undefined) delete process.env[key];
			else process.env[key] = value;
		}
		rmSync(root, { recursive: true, force: true });
	});

	const telemetry = new SchedulerTelemetry();
	telemetry.bind(ctx);
	assert.equal(readdirSync(clients).length, 1);
	const status = telemetry.read(now);
	assert.equal(status.request.state, "paused");
	assert.equal(status.otherRunningChatId, otherChat);
		assert.equal(status.workerAvailable, true);
		let wakes = 0;
		const unsubscribe = telemetry.subscribe(() => wakes++);
		const updated = sample(Date.now());
		updated.backend.scheduler.requests[0] = { ...payload.backend.scheduler.requests[0], state: "running" };
		const stage = join(state, "next.tmp");
		writeFileSync(stage, JSON.stringify(updated), { mode: 0o600 });
		renameSync(stage, join(state, "scheduler-v2.json"));
		await new Promise((resolve) => setTimeout(resolve, 30));
		assert.ok(wakes > 0, "atomic replacement wakes the UI without a polling tick");
		assert.equal(telemetry.read(now + 50).request.state, "running", "file event invalidates the half-second cache");
		unsubscribe();
		telemetry.stop();
		assert.equal(telemetry.watcher, undefined);
	assert.equal(readdirSync(clients).length, 0);
});

test("phase telemetry accepts only numeric timings and opaque identities from the same engine", () => {
  const now = Date.now();
  const row = { chat_id: chat.id, generation: chat.generation, request_id: "e".repeat(64), phase: "cache_lookup",
    blocker: null, input_tokens: 150800, computed_tokens: 0, cached_tokens: null,
    elapsed_ms: 4.2, phase_elapsed_ms: 3, first_token_ms: null, timings_ms: { admission: 1.2, cache_lookup: 3 } };
  const value = { schema: "urn:qwen-r9700:request-phases:v1", pid: 1234, updated_at: now / 1000,
    requests: [row], recent: [] };
  assert.equal(parseRequestPhases(JSON.stringify(value), 1234, now).requests[0].phase, "cache_lookup");
  assert.throws(() => parseRequestPhases(JSON.stringify(value), 1235, now));
  assert.throws(() => parseRequestPhases(JSON.stringify(value), 1234, now + 31000));
  for (const bad of [{ ...row, arguments: "private" }, { ...row, timings_ms: { prompt: "private" } },
    { ...row, phase: "anything" }, { ...row, phase_elapsed_ms: -1 }, { ...row, blocker: { ...chat, text: "private" } }]) {
    assert.throws(() => parseRequestPhases(JSON.stringify({ ...value, requests: [bad] }), 1234, now));
  }
});

test("phase telemetry v2 exposes backend round timing and speculative acceptance", () => {
	const now = Date.now();
	const row = {
		chat_id: chat.id, generation: chat.generation, request_id: "e".repeat(64), phase: "generate",
		blocker: null, input_tokens: 150800, computed_tokens: 150801, cached_tokens: 150800,
		elapsed_ms: 1200, phase_elapsed_ms: 44.2, first_token_ms: 900,
		last_round_ms: 43.7, generation_rounds: 27, draft_tokens: 189, accepted_tokens: 102,
			acceptance_rate: 102 / 189, last_acceptance_rate: 5 / 7, acceptance_rate_3s: 0.61,
			timings_ms: { generate: 300 },
	};
	const value = {
		schema: "urn:qwen-r9700:request-phases:v2", pid: 1234, updated_at: now / 1000,
		requests: [row], recent: [],
	};
	const parsed = parseRequestPhases(JSON.stringify(value), 1234, now);
	assert.equal(parsed.requests[0].last_round_ms, 43.7);
	assert.equal(parsed.requests[0].generation_rounds, 27);
	assert.equal(parsed.requests[0].accepted_tokens, 102);
	assert.equal(parsed.requests[0].draft_tokens, 189);
	assert.equal(parsed.requests[0].acceptance_rate, 102 / 189);
	assert.equal(parsed.requests[0].last_acceptance_rate, 5 / 7);
	assert.equal(parsed.requests[0].acceptance_rate_3s, 0.61);
	for (const bad of [
		{ ...row, last_round_ms: -1 },
		{ ...row, accepted_tokens: 190 },
		{ ...row, acceptance_rate: 1.1 },
			{ ...row, acceptance_rate: "54%" },
			{ ...row, last_acceptance_rate: 1.1 },
			{ ...row, acceptance_rate_3s: 1.1 },
	]) {
		assert.throws(() => parseRequestPhases(JSON.stringify({ ...value, requests: [bad] }), 1234, now));
	}
});

test("prefill progress advances through only the uncached suffix", () => {
  const row = { phase: "prefill", input_tokens: 191632, computed_tokens: 2048,
    cached_tokens: 0, phase_elapsed_ms: 1000 };
  const sample = { requestPhase: row, phaseObservedAt: Date.now() };
  assert.match(requestPhaseStatus(sample).detail, /2,048 \/ 191,632 uncached tok processed · 0 reused/);
  row.computed_tokens = 140080;
  assert.match(requestPhaseStatus(sample).detail, /140,080 \/ 191,632 uncached tok processed/);
  row.cached_tokens = 186856;
  row.computed_tokens = 188904;
  assert.match(requestPhaseStatus(sample).detail, /2,048 \/ 4,776 uncached tok processed · 186,856 reused/);
  row.computed_tokens = 191640;
  assert.match(requestPhaseStatus(sample).detail, /4,776 \/ 4,776 uncached tok processed/);
  row.computed_tokens = 0;
  assert.match(requestPhaseStatus(sample).detail, /0 \/ 4,776 uncached tok processed/);
  row.cached_tokens = null;
  row.computed_tokens = 2048;
  assert.match(requestPhaseStatus(sample).detail, /2,048 \/ 191,632 prompt tok prepared · cached split pending/);
});
