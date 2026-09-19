import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import {
	closeSync,
	constants,
	lstatSync,
	openSync,
	readFileSync,
	unlinkSync,
	utimesSync,
	watch,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { radianceChatIdentity } from "./qwen-radiance-cache.mjs";

const TARGET_PROVIDER = "qwen-r9700";
const SAMPLE_SCHEMA_V1 = "urn:qwen-r9700:scheduler-telemetry:v1";
const SAMPLE_SCHEMA_V2 = "urn:qwen-r9700:scheduler-telemetry:v2";
export const COMBINED_TELEMETRY_SCHEMA = "urn:qwen-r9700:telemetry:v1";
const SAMPLE_STALE_MS = 5_000;
const SAMPLE_REFRESH_MS = 100;
const COMBINED_REFRESH_MS = 100;
const HEARTBEAT_MS = 5_000;
const ENSURE_RETRY_MS = 10_000;
const HEX_ID = /^[0-9a-f]{64}$/;
const REQUEST_PHASE_SCHEMA_V1 = "urn:qwen-r9700:request-phases:v1";
const REQUEST_PHASE_SCHEMA_V2 = "urn:qwen-r9700:request-phases:v2";
// All Pi extensions share this short-lived immutable read. Nested payloads
// retain their own timestamps, so a 100 ms consumer cadence never implies
// that a 1 Hz hardware probe or 0.5 Hz cache probe became fresher.
const combinedSnapshots = new Map();
export const REQUEST_PHASE_LABELS = {
	admission: "Preparing next response",
	gpu_queue: "Queued for GPU",
	priority_wait: "Waiting for higher-priority chat",
	priority_preempt: "Priority takeover requested",
	tool_grace: "Waiting for another chat's tool",
	cache_lookup: "Checking reusable context",
	cache_update: "Finishing previous cache update",
	cache_restore: "Loading cached context into GPU",
	ram_allocation: "Allocating RAM for chat handover",
	handover: "Switching active GPU chat cache",
	prefill: "Processing uncached prompt tokens",
	generate: "Generating model output",
	complete: "Response finished",
};

export function parseRequestPhases(text, pid, now = Date.now()) {
	const value = JSON.parse(text);
	const milliseconds = (n) => typeof n === "number" && Number.isFinite(n) && n >= 0;
	const phaseV2 = value.schema === REQUEST_PHASE_SCHEMA_V2;
	const identity = (n) => n && hasExactKeys(n, ["chat_id", "generation"]) &&
		HEX_ID.test(n.chat_id) && HEX_ID.test(n.generation);
	const row = (r) => {
		const baseKeys = ["chat_id", "generation", "request_id", "phase", "blocker",
			"input_tokens", "computed_tokens", "cached_tokens", "elapsed_ms", "phase_elapsed_ms", "first_token_ms"];
		const generationKeys = phaseV2
			? ["last_round_ms", "generation_rounds", "draft_tokens", "accepted_tokens", "acceptance_rate",
				...(Object.hasOwn(r, "last_acceptance_rate") ? ["last_acceptance_rate"] : []),
				...(Object.hasOwn(r, "acceptance_rate_3s") ? ["acceptance_rate_3s"] : [])]
			: [];
		return hasExactKeys(r, [...baseKeys, ...generationKeys, "timings_ms"]) &&
			HEX_ID.test(r.chat_id) && HEX_ID.test(r.generation) && HEX_ID.test(r.request_id) &&
			Object.hasOwn(REQUEST_PHASE_LABELS, r.phase) && (r.blocker === null || identity(r.blocker)) &&
			validCount(r.input_tokens) && validCount(r.computed_tokens) && (r.cached_tokens === null || validCount(r.cached_tokens)) &&
			milliseconds(r.elapsed_ms) && milliseconds(r.phase_elapsed_ms) &&
			(r.first_token_ms === null || milliseconds(r.first_token_ms)) &&
			(!phaseV2 || (
				(r.last_round_ms === null || milliseconds(r.last_round_ms)) &&
				validCount(r.generation_rounds) && validCount(r.draft_tokens) && validCount(r.accepted_tokens) &&
				r.accepted_tokens <= r.draft_tokens &&
				(r.acceptance_rate === null || (
					typeof r.acceptance_rate === "number" && Number.isFinite(r.acceptance_rate) &&
					r.acceptance_rate >= 0 && r.acceptance_rate <= 1
				)) &&
					(!Object.hasOwn(r, "last_acceptance_rate") || r.last_acceptance_rate === null || (
						typeof r.last_acceptance_rate === "number" && Number.isFinite(r.last_acceptance_rate) &&
						r.last_acceptance_rate >= 0 && r.last_acceptance_rate <= 1
					)) &&
					(!Object.hasOwn(r, "acceptance_rate_3s") || r.acceptance_rate_3s === null || (
						typeof r.acceptance_rate_3s === "number" && Number.isFinite(r.acceptance_rate_3s) &&
						r.acceptance_rate_3s >= 0 && r.acceptance_rate_3s <= 1
					))
			)) &&
			r.timings_ms !== null && typeof r.timings_ms === "object" && !Array.isArray(r.timings_ms) &&
			Object.entries(r.timings_ms).every(([key, ms]) => Object.hasOwn(REQUEST_PHASE_LABELS, key) && milliseconds(ms));
	};
	if (!hasExactKeys(value, ["schema", "pid", "updated_at", "requests", "recent"]) ||
		(!phaseV2 && value.schema !== REQUEST_PHASE_SCHEMA_V1) ||
		(phaseV2 && value.schema !== REQUEST_PHASE_SCHEMA_V2) || value.pid !== pid ||
		!milliseconds(value.updated_at) || now - value.updated_at * 1000 > 30_000 || value.updated_at * 1000 - now > 5_000 ||
		!Array.isArray(value.requests) || value.requests.length > 16 || !value.requests.every(row) ||
		!Array.isArray(value.recent) || value.recent.length > 16 || !value.recent.every(row)) {
		throw new Error("invalid or stale request phase telemetry");
	}
	return value;
}

export function requestPhaseStatus(observation) {
	const row = observation?.requestPhase;
	if (!row || row.phase === "complete") return undefined;
	const seconds = Math.max(0, row.phase_elapsed_ms + (Date.now() - (observation.phaseObservedAt ?? Date.now()))) / 1000;
	const blocker = row.blocker && (row.blocker.chat_id === row.chat_id
		? "an earlier request in this chat" : `another chat ${row.blocker.chat_id.slice(0, 12)}`);
	let detail = `${seconds.toFixed(1)}s`;
	if (row.phase === "gpu_queue") {
		const ownerPhase = observation.blockingRequestPhase?.phase;
		const work = ownerPhase ? `is ${REQUEST_PHASE_LABELS[ownerPhase].toLowerCase()}` : "owns the GPU";
		detail += ` · ${blocker ?? "another request"} ${work}; waiting for its response to finish`;
	}
	if (row.phase === "priority_wait") detail += ` · ${blocker ?? "another chat"} keeps the GPU through tools until its answer finishes`;
	if (row.phase === "priority_preempt") detail += ` · waiting for a safe GPU/cache handover from ${blocker ?? "the current owner"}`;
	if (row.phase === "tool_grace") detail += ` · ${toolGraceDescription(observation.toolGrace ?? {
		chat_id: row.blocker?.chat_id ?? row.chat_id, phase: "response_outcome",
	})}`;
	if (row.phase === "prefill" && row.cached_tokens !== null) {
		const uncached = Math.max(0, row.input_tokens - row.cached_tokens);
		const prepared = Math.min(uncached, Math.max(0, row.computed_tokens - row.cached_tokens));
		detail += ` · ${prepared.toLocaleString("en-US")} / ${uncached.toLocaleString("en-US")} uncached tok processed` +
			` · ${row.cached_tokens.toLocaleString("en-US")} reused`;
	} else if (row.phase === "prefill") {
		detail += ` · ${Math.min(row.computed_tokens, row.input_tokens).toLocaleString("en-US")} / ` +
			`${row.input_tokens.toLocaleString("en-US")} prompt tok prepared · cached split pending`;
	}
	return { phase: REQUEST_PHASE_LABELS[row.phase].toLowerCase(), detail };
}
const DEFAULT_HELPER = resolve(
	dirname(fileURLToPath(import.meta.url)),
	"../../scripts/qwen-radiance-scheduler-status",
);

function processStartTicks() {
	const stat = readFileSync("/proc/self/stat", "utf8");
	const commandEnd = stat.lastIndexOf(")");
	if (commandEnd < 0) throw new Error("cannot parse process identity");
	const value = stat.slice(commandEnd + 2).trim().split(/\s+/)[19];
	if (!/^[1-9][0-9]*$/.test(value ?? "")) throw new Error("cannot parse process start time");
	return value;
}

function validateDirectory(path) {
	const details = lstatSync(path);
	if (
		!details.isDirectory() ||
		details.isSymbolicLink() ||
		details.uid !== process.getuid() ||
		(details.mode & 0o777) !== 0o700
	) {
		throw new Error(`unsafe scheduler telemetry directory: ${path}`);
	}
}

function validateSampleFile(path) {
	const details = lstatSync(path);
	if (
		!details.isFile() ||
		details.isSymbolicLink() ||
		details.nlink !== 1 ||
		details.uid !== process.getuid() ||
		(details.mode & 0o022) !== 0 ||
		details.size > 64 * 1024
	) {
		throw new Error("unsafe scheduler telemetry sample");
	}
}

function validateCombinedSampleFile(path) {
	const details = lstatSync(path);
	if (
		!details.isFile() ||
		details.isSymbolicLink() ||
		details.nlink !== 1 ||
		details.uid !== process.getuid() ||
		(details.mode & 0o022) !== 0 ||
		details.size > 256 * 1024
	) {
		throw new Error("unsafe combined telemetry sample");
	}
}

function validCount(value) {
	return Number.isSafeInteger(value) && value >= 0;
}

function hasExactKeys(value, keys) {
	if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
	const actual = Object.keys(value).sort();
	const expected = [...keys].sort();
	return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

export function parseSchedulerSample(text, now = Date.now()) {
	const sample = JSON.parse(text);
	const legacy = sample?.schema === SAMPLE_SCHEMA_V1;
	const current = sample?.schema === SAMPLE_SCHEMA_V2;
	if (
		(!legacy && !current) ||
		!hasExactKeys(
			sample,
			legacy
				? ["schema", "observed_at_ms", "scheduler"]
				: ["schema", "observed_at_ms", "backend"],
		) ||
		!Number.isSafeInteger(sample.observed_at_ms) ||
		sample.observed_at_ms > now + 1_000 ||
		now - sample.observed_at_ms > SAMPLE_STALE_MS
	) {
		throw new Error("invalid or stale scheduler telemetry sample");
	}
	const backend = legacy ? { scheduler: sample.scheduler, worker: null } : sample.backend;
	if (!hasExactKeys(backend, ["scheduler", "worker"])) {
		throw new Error("invalid scheduler telemetry backend payload");
	}
	const scheduler = backend.scheduler;
	const quantum = scheduler?.quantum_seconds;
	const requests = scheduler?.requests;
	if (
		!hasExactKeys(scheduler, [
			"pid", "updated_at", "quantum_seconds", "switches", "cached_chats",
				"max_cached_chats", "requests",
				...(Object.hasOwn(scheduler ?? {}, "tool_grace") ? ["tool_grace"] : []),
				...(Object.hasOwn(scheduler ?? {}, "priority_hold") ? ["priority_hold"] : []),
		]) ||
		!Number.isSafeInteger(scheduler?.pid) ||
		scheduler.pid <= 0 ||
		typeof scheduler.updated_at !== "number" ||
		!Number.isFinite(scheduler.updated_at) ||
		typeof quantum !== "number" ||
		!Number.isFinite(quantum) ||
		(quantum !== 0 && quantum < 0.1) ||
		quantum > 120 ||
		!validCount(scheduler.switches) ||
		!validCount(scheduler.cached_chats) ||
		!validCount(scheduler.max_cached_chats) ||
		!Array.isArray(requests) ||
		requests.length > 16 ||
		requests.some(
			(row) =>
				!hasExactKeys(row, [
					"chat_id", "generation", "state", "computed_tokens", "input_tokens",
				]) ||
				!HEX_ID.test(row.chat_id ?? "") ||
				!HEX_ID.test(row.generation ?? "") ||
				!["running", "paused", "queued"].includes(row.state) ||
				!validCount(row.computed_tokens) ||
				!validCount(row.input_tokens),
		)
	) {
		throw new Error("invalid scheduler telemetry payload");
	}
	const grace = scheduler.tool_grace;
	if (grace !== undefined && (
		!hasExactKeys(grace, ["chat_id", "generation", "phase", "remaining_seconds"]) ||
		!HEX_ID.test(grace.chat_id ?? "") || !HEX_ID.test(grace.generation ?? "") ||
		!["tool_grace", "response_outcome"].includes(grace.phase) ||
		typeof grace.remaining_seconds !== "number" || !Number.isFinite(grace.remaining_seconds) ||
		grace.remaining_seconds < 0 || grace.remaining_seconds > 5
		)) throw new Error("invalid tool handover grace telemetry");
		const hold = scheduler.priority_hold;
		if (hold !== undefined && (!hasExactKeys(hold, ["chat_id", "generation", "priority"]) ||
			!HEX_ID.test(hold.chat_id ?? "") || !HEX_ID.test(hold.generation ?? "") ||
			![1, 2].includes(hold.priority))) throw new Error("invalid priority hold telemetry");
	const backendAge = now - scheduler.updated_at * 1_000;
	if (backendAge < -5_000 || backendAge > Math.max(5_000, (quantum || 15) * 2_000)) {
		throw new Error("stale scheduler telemetry payload");
	}
	const worker = backend.worker;
	if (worker !== null) {
		const numeric = [
			"allocated_bytes", "reserved_capacity_bytes", "cached_chats", "switches",
			"last_transfer_bytes", "transferred_bytes", "last_allocation_bytes",
			"allocation_events", "generation_replacements",
		];
		const seconds = [
			"updated_at", "last_transfer_seconds", "transfer_seconds",
			"last_allocation_seconds", "allocation_seconds",
		];
		const identity = (value) => value === null || (
			hasExactKeys(value, ["chat_id", "generation"]) &&
			HEX_ID.test(value.chat_id ?? "") && HEX_ID.test(value.generation ?? "")
		);
		const residency = worker?.residency;
		if (
			!hasExactKeys(worker, [
				"pid", "updated_at", "allocated_bytes", "reserved_capacity_bytes",
				"cached_chats", "switches", "last_transfer_bytes", "last_transfer_seconds",
				"transferred_bytes", "transfer_seconds", "last_allocation_bytes",
				"last_allocation_seconds", "allocation_events", "allocation_seconds",
				"generation_replacements", "last_handover", "residency",
			]) ||
			!Number.isSafeInteger(worker.pid) || worker.pid <= 0 ||
			numeric.some((key) => !validCount(worker[key])) ||
			seconds.some((key) =>
				typeof worker[key] !== "number" || !Number.isFinite(worker[key]) || worker[key] < 0) ||
			typeof worker.last_handover !== "string" || !/^[a-z-]{1,32}$/.test(worker.last_handover) ||
			!hasExactKeys(residency, ["active", "images", "free_buffer_bytes", "staging_buffer_bytes"]) ||
			!identity(residency.active) ||
			!Array.isArray(residency.images) || residency.images.length > 4 ||
			residency.images.some((image) =>
				!hasExactKeys(image, ["chat_id", "generation", "data_bytes", "allocated_bytes"]) ||
				!HEX_ID.test(image.chat_id ?? "") || !HEX_ID.test(image.generation ?? "") ||
				!validCount(image.data_bytes) || !validCount(image.allocated_bytes)) ||
			!validCount(residency.free_buffer_bytes) || !validCount(residency.staging_buffer_bytes)
		) {
			throw new Error("invalid scheduler worker telemetry payload");
		}
	}
	Object.defineProperty(scheduler, "worker", {
		value: worker?.pid === scheduler.pid ? worker : undefined,
		enumerable: false,
	});
	return scheduler;
}

export function parseCombinedTelemetry(text, now = Date.now()) {
	const sample = JSON.parse(text);
	if (
		!hasExactKeys(sample, ["schema", "observed_at_ms", "scheduler", "worker", "phases", "cache", "temperature"]) ||
		sample.schema !== COMBINED_TELEMETRY_SCHEMA ||
		!Number.isSafeInteger(sample.observed_at_ms) ||
		sample.observed_at_ms > now + 1_000 ||
		now - sample.observed_at_ms > SAMPLE_STALE_MS ||
		sample.scheduler === null || typeof sample.scheduler !== "object" || Array.isArray(sample.scheduler) ||
		(sample.worker !== null && (typeof sample.worker !== "object" || Array.isArray(sample.worker))) ||
		(sample.phases !== null && (typeof sample.phases !== "object" || Array.isArray(sample.phases))) ||
		(sample.cache !== null && (typeof sample.cache !== "object" || Array.isArray(sample.cache))) ||
		(sample.temperature !== null && (typeof sample.temperature !== "object" || Array.isArray(sample.temperature)))
	) {
		throw new Error("invalid or stale combined telemetry sample");
	}
	return sample;
}

export function schedulerStatusForChat(scheduler, chat) {
	const request = scheduler.requests.find(
		(row) => row.chat_id === chat.id && row.generation === chat.generation,
	);
	const otherRunning = scheduler.requests.find(
		(row) => row.state === "running" && (row.chat_id !== chat.id || row.generation !== chat.generation),
	);
	const matchesChat = (value) => value?.chat_id === chat.id && value?.generation === chat.generation;
	const activeChat = scheduler.worker?.residency.active;
	const ramImage = scheduler.worker?.residency.images.find(matchesChat);
	return {
		available: true,
		request,
		otherRunningChatId: otherRunning?.chat_id,
		otherRunningRequest: otherRunning,
		activeChat,
		cacheResidency: matchesChat(activeChat) ? "gpu" : ramImage ? "ram" : undefined,
		ramCacheBytes: ramImage?.data_bytes,
		workerAvailable: scheduler.worker !== undefined,
		quantumSeconds: scheduler.quantum_seconds,
		policy: scheduler.quantum_seconds === 0 ? "response_boundary" : "time_slice",
		...(scheduler.tool_grace && !matchesChat(scheduler.tool_grace)
			? { toolGrace: scheduler.tool_grace } : {}),
		...(scheduler.priority_hold && !matchesChat(scheduler.priority_hold)
			? { priorityHold: scheduler.priority_hold } : {}),
	};
}

export function toolGraceDescription(grace) {
	const owner = `chat ${grace.chat_id.slice(0, 12)}`;
	return grace.phase === "tool_grace"
		? `giving ${owner}'s tool time to finish · ${grace.remaining_seconds.toFixed(1)}s grace remaining`
		: `confirming whether ${owner} finished with a tool call`;
}

function attachRequestPhases(observation, phases, chat) {
	if (phases === undefined) return observation;
	const matches = (row) => row.chat_id === chat.id && row.generation === chat.generation;
	observation.requestPhase = phases.requests.find(matches);
	const blocker = observation.requestPhase?.blocker;
	observation.blockingRequestPhase = blocker && phases.requests.find((row) =>
		row.chat_id === blocker.chat_id && row.generation === blocker.generation);
	observation.lastRequestTiming = phases.recent.find(matches);
	observation.phaseObservedAt = phases.updated_at * 1000;
	return observation;
}

function configuration() {
	if (!process.env.QWEN_RADIANCE_CACHE_ABI) return undefined;
	const stateDirectory = process.env.QWEN_RADIANCE_GPU_TEMPERATURE_STATE;
	const helper = process.env.QWEN_RADIANCE_SCHEDULER_HELPER ?? DEFAULT_HELPER;
	const remoteHost = process.env.QWEN_RADIANCE_CACHE_HOST;
	if (
		!stateDirectory?.startsWith("/") ||
		!helper.startsWith("/") ||
		!remoteHost ||
		!/^[A-Za-z0-9_.@:-]{1,255}$/.test(remoteHost)
	) {
		return undefined;
	}
	return { stateDirectory, helper, remoteHost };
}

export class SchedulerTelemetry {
	constructor() {
		this.config = undefined;
		this.markerPath = undefined;
		this.heartbeatTimer = undefined;
        this.chat = undefined;
        this.lastEnsureAt = 0;
        this.lastReadAt = 0;
        this.cached = { available: false, configured: false };
		this.listeners = new Set();
		this.watcher = undefined;
	}

	subscribe(listener) {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	ensure(now, force = false) {
		if (!this.config || (!force && now - this.lastEnsureAt < ENSURE_RETRY_MS)) return;
		this.lastEnsureAt = now;
		try {
			const child = spawn(
				this.config.helper,
				["ensure", this.config.stateDirectory, this.config.remoteHost],
				{ detached: true, stdio: "ignore" },
			);
			child.unref();
		} catch {
			// A later heartbeat retries without interrupting the model request.
		}
	}

	heartbeat() {
		if (!this.config || !this.markerPath) return;
		const now = Date.now();
		try {
			utimesSync(this.markerPath, now / 1_000, now / 1_000);
			this.ensure(now);
		} catch {
			this.stop();
		}
	}

	start(ctx) {
		this.stop();
		const config = configuration();
		if (!config || ctx.mode !== "tui" || ctx.model?.provider !== TARGET_PROVIDER) return;
		try {
			validateDirectory(config.stateDirectory);
			const clients = join(config.stateDirectory, "clients");
			validateDirectory(clients);
			const random = randomBytes(8).toString("hex");
			const markerPath = join(
				clients,
				`${process.pid}-${processStartTicks()}-${random}.heartbeat`,
			);
			const descriptor = openSync(
				markerPath,
				constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
				0o600,
			);
			closeSync(descriptor);
				this.config = config;
				this.markerPath = markerPath;
				try {
				this.watcher = watch(config.stateDirectory, { persistent: false }, (_event, name) => {
						const fileName = String(name);
						if (!["telemetry-v1.json", "request-phases.json", "scheduler-v2.json", "scheduler.json"].includes(fileName)) return;
						if (fileName === "telemetry-v1.json") {
							combinedSnapshots.delete(join(config.stateDirectory, fileName));
						}
						this.lastReadAt = 0;
						for (const listener of this.listeners) listener();
					});
					this.watcher.on("error", () => { this.watcher?.close(); this.watcher = undefined; });
				} catch { /* The existing half-second reader remains the fallback. */ }
			this.bind(ctx);
			this.ensure(Date.now(), true);
			this.heartbeatTimer = setInterval(() => this.heartbeat(), HEARTBEAT_MS);
			this.heartbeatTimer.unref?.();
		} catch {
			this.stop();
		}
	}

    bind(ctx) {
        if (!this.config) {
            this.start(ctx);
            return;
        }
		try {
			this.chat = this.config ? radianceChatIdentity(ctx) : undefined;
		} catch {
			this.chat = undefined;
		}
        this.lastReadAt = 0;
        this.cached = { available: false, configured: true };
	}

	clear() {
        this.chat = undefined;
        this.lastReadAt = 0;
        this.cached = { available: false, configured: this.config !== undefined };
	}

	read(now = Date.now()) {
		if (!this.config) return { available: false, configured: false };
		if (!this.chat) return { available: false, configured: true };
		if (now - this.lastReadAt < SAMPLE_REFRESH_MS) return this.cached;
		this.lastReadAt = now;

		const combined = this.readCombinedSnapshot(now);
		if (combined !== undefined) {
			try {
				const scheduler = parseSchedulerSample(JSON.stringify({
					schema: SAMPLE_SCHEMA_V2,
					observed_at_ms: combined.observed_at_ms,
					backend: { scheduler: combined.scheduler, worker: combined.worker },
				}), now);
				let phases;
				if (combined.phases !== null) {
					try {
						phases = parseRequestPhases(JSON.stringify(combined.phases), scheduler.pid, now);
					} catch { /* The scheduler sample remains useful without a phase row. */ }
				}
				this.cached = attachRequestPhases(schedulerStatusForChat(scheduler, this.chat), phases, this.chat);
				return this.cached;
			} catch {
				// Fall through to the individually authenticated legacy files.
			}
		}

		for (const name of ["scheduler-v2.json", "scheduler.json"]) {
			try {
				const samplePath = join(this.config.stateDirectory, name);
				validateSampleFile(samplePath);
				const scheduler = parseSchedulerSample(readFileSync(samplePath, "utf8"), now);
				let phases;
				try {
					const path = join(this.config.stateDirectory, "request-phases.json");
					validateSampleFile(path);
					phases = parseRequestPhases(readFileSync(path, "utf8"), scheduler.pid, now);
				} catch { /* Old backends have no detailed phase feed. */ }
				this.cached = attachRequestPhases(schedulerStatusForChat(scheduler, this.chat), phases, this.chat);
				return this.cached;
			} catch {
				// A running old monitor has only scheduler.json; malformed or stale
				// v2 data likewise falls back to the authenticated legacy sample.
			}
		}
		this.cached = { available: false, configured: true };
		this.ensure(now);
		return this.cached;
	}

	readCombinedSnapshot(now = Date.now()) {
		if (!this.config) return undefined;
		const path = join(this.config.stateDirectory, "telemetry-v1.json");
		const previous = combinedSnapshots.get(path);
		if (previous !== undefined && now - previous.readAt < COMBINED_REFRESH_MS) return previous.sample;
		let sample;
		try {
			validateCombinedSampleFile(path);
			sample = parseCombinedTelemetry(readFileSync(path, "utf8"), now);
		} catch {
			sample = undefined;
		}
		combinedSnapshots.set(path, { readAt: now, sample });
		return sample;
	}

	stop() {
		if (this.config) combinedSnapshots.delete(join(this.config.stateDirectory, "telemetry-v1.json"));
		this.watcher?.close();
			this.watcher = undefined;
		if (this.heartbeatTimer !== undefined) {
			clearInterval(this.heartbeatTimer);
			this.heartbeatTimer = undefined;
		}
		if (this.markerPath !== undefined) {
			try {
				unlinkSync(this.markerPath);
			} catch (error) {
				if (error?.code !== "ENOENT") {
					// PID/start-time validation independently expires an inaccessible marker.
				}
			}
		}
		this.config = undefined;
		this.markerPath = undefined;
		this.chat = undefined;
		this.lastEnsureAt = 0;
		this.lastReadAt = 0;
        this.cached = { available: false, configured: false };
	}
}

export function createSchedulerTelemetry() {
	return new SchedulerTelemetry();
}
