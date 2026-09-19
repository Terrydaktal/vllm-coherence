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
} from "node:fs";
import { join } from "node:path";
import { CacheResidencyTelemetry } from "./qwen-cache-residency.mjs";

const STATUS_KEY = "qwen-gpu-temperature";
const TARGET_PROVIDER = "qwen-r9700";
const REFRESH_MS = 1_000;
const RESIDENCY_REFRESH_MS = 100;
const SAMPLE_STALE_MS = 15_000;
const ENSURE_RETRY_MS = 10_000;
const SAMPLE_SCHEMA_V1 = "urn:qwen-r9700:gpu-temperature:v1";
const SAMPLE_SCHEMA_V2 = "urn:qwen-r9700:gpu-temperature:v2";
const UNKNOWN_STATUS = Symbol("unknown GPU temperature status");

function processStartTicks() {
	const stat = readFileSync("/proc/self/stat", "utf8");
	const commandEnd = stat.lastIndexOf(")");
	if (commandEnd < 0) throw new Error("cannot parse process identity");
	const fieldsAfterCommand = stat.slice(commandEnd + 2).trim().split(/\s+/);
	const value = fieldsAfterCommand[19];
	if (!/^[1-9][0-9]*$/.test(value ?? "")) throw new Error("cannot parse process start time");
	return value;
}

function validateDirectory(path) {
	const details = lstatSync(path);
	if (!details.isDirectory() || details.isSymbolicLink() || details.uid !== process.getuid() ||
		(details.mode & 0o777) !== 0o700) {
		throw new Error(`unsafe GPU temperature directory: ${path}`);
	}
}

function validateSampleFile(path) {
	const details = lstatSync(path);
	if (!details.isFile() || details.isSymbolicLink() || details.nlink !== 1 ||
		details.uid !== process.getuid() || (details.mode & 0o022) !== 0 || details.size > 1_024) {
		throw new Error("unsafe GPU temperature sample");
	}
}

export function parseTemperatureSample(text, now = Date.now()) {
	const sample = JSON.parse(text);
	const legacy = sample?.schema === SAMPLE_SCHEMA_V1;
	if ((!legacy && sample?.schema !== SAMPLE_SCHEMA_V2) || !Number.isSafeInteger(sample.observed_at_ms) ||
		!Number.isSafeInteger(sample.edge_millicelsius) ||
		!Number.isSafeInteger(sample.junction_millicelsius) ||
		(!legacy && (!Number.isSafeInteger(sample.fan_percent) ||
			 sample.fan_percent < 0 || sample.fan_percent > 100)) ||
		sample.observed_at_ms > now + REFRESH_MS || now - sample.observed_at_ms > SAMPLE_STALE_MS ||
		sample.edge_millicelsius < 0 || sample.edge_millicelsius > 250_000 ||
		sample.junction_millicelsius < 0 || sample.junction_millicelsius > 250_000) {
		throw new Error("invalid or stale GPU temperature sample");
	}
	return sample;
}

function formatMilliCelsius(value) {
	const degrees = value / 1_000;
	return `${Number.isInteger(degrees) ? degrees.toFixed(0) : degrees.toFixed(1)}°C`;
}

export function formatTemperatureStatus(sample) {
	const temperatures = `${formatMilliCelsius(sample.junction_millicelsius)} · ` +
		formatMilliCelsius(sample.edge_millicelsius);
	return Number.isSafeInteger(sample.fan_percent)
		? `${temperatures} · ${sample.fan_percent}%`
		: temperatures;
}

function configuration() {
	const stateDirectory = process.env.QWEN_RADIANCE_GPU_TEMPERATURE_STATE;
	const helper = process.env.QWEN_RADIANCE_GPU_TEMPERATURE_HELPER;
	const remoteHost = process.env.QWEN_RADIANCE_CACHE_HOST;
	if (!stateDirectory?.startsWith("/") || !helper?.startsWith("/") ||
		!remoteHost || !/^[A-Za-z0-9_.@:-]{1,255}$/.test(remoteHost)) return undefined;
	return { stateDirectory, helper, remoteHost };
}

export default function qwenGpuTemperature(pi, { residency = new CacheResidencyTelemetry() } = {}) {
	let timer;
	let markerPath;
	let activeUi;
	let lastStatus = UNKNOWN_STATUS;
	let lastEnsureAt = 0;
	let activeContext;
	let lastResidency;
	let lastTemperatureRefreshAt;

	function refreshResidency() {
		const value = residency.readBreakdown(activeContext?.getContextUsage?.()?.tokens);
		if (activeUi && value !== lastResidency) {
			activeUi.setStatus?.("qwen-cache-residency", value);
			lastResidency = value;
		}
	}

	function setStatus(value) {
		if (!activeUi || value === lastStatus) return;
		activeUi.setStatus?.(STATUS_KEY, value);
		lastStatus = value;
	}

	function ensureMonitor(config, now, force = false) {
		if (!force && now - lastEnsureAt < ENSURE_RETRY_MS) return;
		lastEnsureAt = now;
		try {
			const child = spawn(config.helper, ["ensure", config.stateDirectory, config.remoteHost], {
				detached: true,
				stdio: "ignore",
			});
			child.unref();
		}
		catch {
			// A later heartbeat retries without disturbing the Pi session.
		}
	}

	function refresh(config) {
		const now = Date.now();
		refreshResidency();
		if (lastTemperatureRefreshAt !== undefined && now >= lastTemperatureRefreshAt &&
			now - lastTemperatureRefreshAt < REFRESH_MS) return;
		lastTemperatureRefreshAt = now;
		try {
			utimesSync(markerPath, now / 1_000, now / 1_000);
		}
		catch {
			setStatus(undefined);
			return;
		}
		try {
			let sample;
			const combined = residency.readCombinedSnapshot?.(now);
			if (combined?.temperature !== null && combined?.temperature !== undefined) {
				try {
					sample = parseTemperatureSample(JSON.stringify(combined.temperature), now);
				} catch {
					// Fall through to the standalone temperature monitor sample.
				}
			}
			if (!sample) {
				const samplePath = join(config.stateDirectory, "sample.json");
				validateSampleFile(samplePath);
				sample = parseTemperatureSample(readFileSync(samplePath, "utf8"), now);
			}
			setStatus(formatTemperatureStatus(sample));
		}
		catch {
			setStatus(undefined);
			ensureMonitor(config, now);
		}
	}

	function stop() {
		residency.stop();
		if (lastResidency !== undefined) activeUi?.setStatus?.("qwen-cache-residency", undefined);
		lastResidency = undefined;
		activeContext = undefined;
		if (timer !== undefined) {
			clearInterval(timer);
			timer = undefined;
		}
		if (markerPath !== undefined) {
			try {
				unlinkSync(markerPath);
			}
			catch (error) {
				if (error?.code !== "ENOENT") {
					// Stale heartbeats are independently rejected by PID/start-time identity.
				}
			}
			markerPath = undefined;
		}
		setStatus(undefined);
		activeUi = undefined;
		lastStatus = UNKNOWN_STATUS;
		lastEnsureAt = 0;
		lastTemperatureRefreshAt = undefined;
	}

	function start(ctx) {
		stop();
		const config = configuration();
		if (!config || ctx.mode !== "tui" || ctx.model?.provider !== TARGET_PROVIDER) return;
		try {
			validateDirectory(config.stateDirectory);
			const clients = join(config.stateDirectory, "clients");
			validateDirectory(clients);
			const random = randomBytes(8).toString("hex");
			markerPath = join(clients, `${process.pid}-${processStartTicks()}-${random}.heartbeat`);
			const descriptor = openSync(
				markerPath,
				constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
				0o600,
			);
			closeSync(descriptor);
		}
		catch {
			markerPath = undefined;
			return;
		}
		activeUi = ctx.ui;
		activeContext = ctx;
		residency.start(ctx);
		ensureMonitor(config, Date.now(), true);
		refresh(config);
		timer = setInterval(() => refresh(config), RESIDENCY_REFRESH_MS);
		timer.unref?.();
	}

	pi.on("session_start", (_event, ctx) => start(ctx));
	pi.on("session_switch", (_event, ctx) => start(ctx));
	for (const event of ["session_compact", "session_tree", "before_provider_request"]) {
		pi.on(event, (_event, ctx) => {
			activeContext = ctx;
			residency.bind(ctx);
			refreshResidency();
		});
	}
	pi.on("model_select", (_event, ctx) => start(ctx));
	pi.on("session_shutdown", () => stop());
}
