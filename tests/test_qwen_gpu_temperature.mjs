import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import install, {
	formatTemperatureStatus,
	parseTemperatureSample,
} from "../integrations/pi/qwen-gpu-temperature.mjs";

const schema = "urn:qwen-r9700:gpu-temperature:v2";

test("temperature samples are validated and formatted with both AMD sensors", () => {
	const now = Date.now();
	const sample = parseTemperatureSample(JSON.stringify({
		schema,
		observed_at_ms: now,
		edge_millicelsius: 34_500,
		junction_millicelsius: 36_000,
		fan_percent: 25,
	}), now);
	assert.equal(formatTemperatureStatus(sample), "36°C · 34.5°C · 25%");
	assert.throws(() => parseTemperatureSample(JSON.stringify({ ...sample, observed_at_ms: now - 20_000 }), now),
		/stale/);
	assert.throws(() => parseTemperatureSample(JSON.stringify({ ...sample, fan_percent: 101 }), now), /invalid/);
	const legacy = parseTemperatureSample(JSON.stringify({
		...sample,
		schema: "urn:qwen-r9700:gpu-temperature:v1",
		fan_percent: undefined,
	}), now);
	assert.equal(formatTemperatureStatus(legacy), "36°C · 34.5°C");
});

test("extension uses one tmpfs client heartbeat and clears it on shutdown", (t) => {
	const root = mkdtempSync(join(tmpdir(), "qwen-gpu-temperature-extension-"));
	const state = join(root, "state");
	const clients = join(state, "clients");
	mkdirSync(clients, { recursive: true, mode: 0o700 });
	chmodSync(root, 0o700);
	chmodSync(state, 0o700);
	chmodSync(clients, 0o700);
	const helper = join(root, "helper");
	writeFileSync(helper, "#!/usr/bin/env bash\nexit 0\n", { mode: 0o700 });
	writeFileSync(join(state, "sample.json"), JSON.stringify({
		schema,
		observed_at_ms: Date.now(),
		edge_millicelsius: 35_000,
		junction_millicelsius: 40_000,
		fan_percent: 31,
	}), { mode: 0o600 });

	const previous = {
		state: process.env.QWEN_RADIANCE_GPU_TEMPERATURE_STATE,
		helper: process.env.QWEN_RADIANCE_GPU_TEMPERATURE_HELPER,
		host: process.env.QWEN_RADIANCE_CACHE_HOST,
	};
	process.env.QWEN_RADIANCE_GPU_TEMPERATURE_STATE = state;
	process.env.QWEN_RADIANCE_GPU_TEMPERATURE_HELPER = helper;
	process.env.QWEN_RADIANCE_CACHE_HOST = "lewis@ai";
	t.after(() => {
		for (const [name, value] of Object.entries(previous)) {
			const key = name === "state" ? "QWEN_RADIANCE_GPU_TEMPERATURE_STATE" :
				name === "helper" ? "QWEN_RADIANCE_GPU_TEMPERATURE_HELPER" : "QWEN_RADIANCE_CACHE_HOST";
			if (value === undefined) delete process.env[key]; else process.env[key] = value;
		}
		rmSync(root, { recursive: true, force: true });
	});

	const handlers = new Map();
	const statuses = [];
	let now = Date.now();
	t.mock.method(Date, "now", () => now);
	let refreshIntervalMs;
	let refreshCallback;
	t.mock.method(globalThis, "setInterval", (callback, milliseconds) => {
		refreshIntervalMs = milliseconds;
		refreshCallback = callback;
		return { unref() {} };
	});
	t.mock.method(globalThis, "clearInterval", () => {});
	let coverage = "Cache ≈ GPU 66,000 · RAM 0 · Disk 0 · Cold 160 tok";
	let bound = 0;
	install({ on(name, handler) { handlers.set(name, handler); } }, { residency: {
		start() {}, stop() {}, bind() { bound++; }, readBreakdown(tokens) {
			assert.equal(tokens, 66_160);
			return coverage;
		},
	} });
	const ctx = {
		mode: "tui",
		model: { provider: "qwen-r9700" },
		getContextUsage: () => ({ tokens: 66_160 }),
		ui: { setStatus(key, value) { statuses.push([key, value]); } },
	};
	handlers.get("session_start")({}, ctx);
	assert.equal(statuses.at(-1)[1], "40°C · 35°C · 31%");
	assert.equal(refreshIntervalMs, 100);
	assert.equal(readdirSync(clients).length, 1);
	assert.ok(statuses.some(([key, value]) => key === "qwen-cache-residency" && value === coverage));
	coverage = "Cache ≈ GPU 0 · RAM 66,000 · Disk 0 · Cold 160 tok";
	writeFileSync(join(state, "sample.json"), JSON.stringify({
		schema, observed_at_ms: now, edge_millicelsius: 36_000,
		junction_millicelsius: 41_000, fan_percent: 32,
	}));
	now += 500;
	refreshCallback();
	assert.deepEqual(statuses.at(-1), ["qwen-cache-residency", coverage]);
	assert.equal(statuses.filter(([key]) => key === "qwen-gpu-temperature").at(-1)[1], "40°C · 35°C · 31%");
	now += 500;
	refreshCallback();
	assert.deepEqual(statuses.at(-1), ["qwen-gpu-temperature", "41°C · 36°C · 32%"]);
	handlers.get("session_compact")({}, ctx);
	assert.equal(bound, 1, "compaction rebinds residency to the new generation");
	handlers.get("session_shutdown")();
	assert.equal(readdirSync(clients).length, 0);
	assert.deepEqual(statuses.at(-1), ["qwen-gpu-temperature", undefined]);
});
