import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const STATUS_KEY = "qwen-250k-qualification";
const MANIFEST_ENV = "QWEN_PI_250K_MANIFEST";
const SAFE_MODEL = "qwen3.8-27b-frozenlock";
const SAFE_SYSTEM = "Public Pi qualification guard fallback. Do not use tools.";
const SAFE_USER = "Output only 0.";
const TARGET_PROMPT_TOKENS = 249_957;
const COMPLETION_CAP_TOKENS = 2_048;
const MODEL_CONTEXT_TOKENS = 253_792;
const MIN_OUTPUT_TOKENS = 512;
const MIN_RATE_WINDOW_SECONDS = 3;
const SUCCESS_STOP_REASONS = new Set(["stop", "length"]);
const RATE_DEFINITION =
	"(authoritative output tokens - output tokens reported with first data) / seconds after first data";

function sha256(text) {
	return createHash("sha256").update(text, "utf8").digest("hex");
}

function canonicalJson(value) {
	if (value === null || typeof value !== "object") return JSON.stringify(value);
	if (Array.isArray(value)) return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
	return `{${Object.keys(value)
		.sort()
		.map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
		.join(",")}}`;
}

function isSha256(value) {
	return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function isPositiveInteger(value) {
	return Number.isSafeInteger(value) && value > 0;
}

function isPositiveNumber(value) {
	return Number.isFinite(value) && value > 0;
}

function loadManifest() {
	try {
		const path = process.env[MANIFEST_ENV];
		if (typeof path !== "string" || path.length === 0) throw new Error("missing");
		const manifest = JSON.parse(readFileSync(path, "utf8"));
		if (manifest?.schema_version !== 1 || manifest.public_data_only !== true) throw new Error("schema");
		if (manifest.source_session !== null) throw new Error("source");
		if (manifest.safety?.original_session_read !== false) throw new Error("read safety");
		if (manifest.safety?.original_session_modified !== false) throw new Error("write safety");

		const recordedHash = manifest.manifest_payload_sha256;
		if (!isSha256(recordedHash)) throw new Error("hash field");
		const payload = { ...manifest };
		delete payload.manifest_payload_sha256;
		if (sha256(canonicalJson(payload)) !== recordedHash) throw new Error("manifest hash");

		const accounting = manifest.token_accounting;
		if (accounting?.target_prompt_tokens !== TARGET_PROMPT_TOKENS) throw new Error("target");
		if (accounting?.completion_cap_tokens !== COMPLETION_CAP_TOKENS) throw new Error("cap");
		if (accounting?.model_context_tokens !== MODEL_CONTEXT_TOKENS) throw new Error("context");
		if (accounting.predicted_prompt_tokens !== accounting.target_prompt_tokens) {
			throw new Error("prediction");
		}
		if (
			accounting.target_prompt_tokens + accounting.completion_cap_tokens >
			accounting.model_context_tokens
		) {
			throw new Error("overflow");
		}
		const sessions = manifest.files?.sessions;
		if (!Array.isArray(sessions) || sessions.length !== 3) throw new Error("sessions");
		const expectedPurposes = ["cold-prefix-prime", "warm-qualification", "warm-qualification"];
		const sessionIds = new Set();
		for (const [index, session] of sessions.entries()) {
			if (
				session?.clone !== index + 1 ||
				session.purpose !== expectedPurposes[index] ||
				typeof session.session_id !== "string" ||
				session.session_id.length === 0 ||
				!isSha256(session.sha256)
			) {
				throw new Error("session contract");
			}
			sessionIds.add(session.session_id);
		}
		if (sessionIds.size !== sessions.length) throw new Error("session identity");

		const contract = manifest.request_contract;
		if (contract?.provider !== "qwen-r9700" || contract.api !== "openai-completions") {
			throw new Error("provider");
		}
		if (contract.model !== SAFE_MODEL) throw new Error("model");
		if (canonicalJson(contract.message_roles) !== canonicalJson(["system", "user", "user"])) {
			throw new Error("roles");
		}
		if (contract.message_count !== 3 || contract.stream !== true) throw new Error("messages");
		for (const field of [
			"effective_system_prompt_sha256",
			"filler_message_sha256",
			"benchmark_prompt_sha256",
		]) {
			if (!isSha256(contract[field])) throw new Error("content hash");
		}
		const sampling = contract.sampling;
		if (sampling?.temperature !== 0 || sampling.top_p !== 1 || sampling.top_k !== 1) {
			throw new Error("sampling");
		}
		const kwargs = contract.chat_template_kwargs;
		if (
			kwargs?.enable_thinking !== true ||
			kwargs.preserve_thinking !== true ||
			kwargs.reasoning_effort !== "xhigh" ||
			Object.keys(kwargs).length !== 3
		) {
			throw new Error("thinking");
		}

		const qualification = manifest.qualification;
		if (![70, 100].includes(qualification?.target_output_rate_tps)) throw new Error("rate");
		if (qualification.interim_output_rate_tps !== 70) throw new Error("interim rate");
		if (qualification.destination_output_rate_tps !== 100) throw new Error("destination rate");
		if (qualification.rate_definition !== RATE_DEFINITION) throw new Error("rate definition");
		const expectedMilestone = qualification.target_output_rate_tps === 100 ? "destination" : "interim";
		if (qualification.configured_milestone !== expectedMilestone) throw new Error("milestone");
		if (!isPositiveNumber(qualification?.max_first_data_seconds)) throw new Error("first data");
		if (qualification.max_first_data_seconds > 30) throw new Error("first data ceiling");
		if (
			!isPositiveNumber(qualification?.min_rate_window_seconds) ||
			qualification.min_rate_window_seconds < MIN_RATE_WINDOW_SECONDS
		) {
			throw new Error("window");
		}
		if (
			!isPositiveInteger(qualification?.min_output_tokens) ||
			qualification.min_output_tokens < MIN_OUTPUT_TOKENS
		) {
			throw new Error("output");
		}
		if (qualification.min_output_tokens > accounting.completion_cap_tokens) {
			throw new Error("output cap");
		}
		return { manifest, error: undefined };
	} catch {
		return { manifest: undefined, error: "manifest unavailable or invalid" };
	}
}

function plainText(content) {
	if (typeof content === "string") return content;
	if (!Array.isArray(content) || content.length === 0) return undefined;
	const parts = [];
	for (const item of content) {
		if (
			item === null ||
			typeof item !== "object" ||
			item.type !== "text" ||
			typeof item.text !== "string"
		) {
			return undefined;
		}
		parts.push(item.text);
	}
	return parts.join("");
}

function fixedPublicFallback() {
	return {
		model: SAFE_MODEL,
		messages: [
			{ role: "system", content: SAFE_SYSTEM },
			{ role: "user", content: SAFE_USER },
		],
		stream: true,
		stream_options: { include_usage: true, continuous_usage_stats: true },
		max_completion_tokens: 1,
		temperature: 0,
		top_p: 1,
		top_k: 1,
		chat_template_kwargs: { enable_thinking: false, preserve_thinking: false },
	};
}

function verifyPayload(event, ctx, manifest) {
	if (manifest === undefined) return "manifest unavailable or invalid";
	const contract = manifest.request_contract;
	if (ctx?.model?.provider !== contract.provider || ctx.model.api !== contract.api) {
		return "provider or API mismatch";
	}
	const payload = event?.payload;
	if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
		return "payload is not an object";
	}
	if (payload.model !== contract.model || payload.stream !== true) return "model or stream mismatch";
	if (!Array.isArray(payload.messages) || payload.messages.length !== contract.message_count) {
		return "message count mismatch";
	}
	if (
		payload.messages.some((message, index) => message?.role !== contract.message_roles[index])
	) {
		return "message role mismatch";
	}
	const texts = payload.messages.map((message) => plainText(message?.content));
	if (texts.some((text) => text === undefined)) return "non-text message content";
	const expectedHashes = [
		contract.effective_system_prompt_sha256,
		contract.filler_message_sha256,
		contract.benchmark_prompt_sha256,
	];
	if (texts.some((value, index) => sha256(value) !== expectedHashes[index])) {
		return "public message hash mismatch";
	}
	if (payload.tools !== undefined && (!Array.isArray(payload.tools) || payload.tools.length !== 0)) {
		return "tools are enabled";
	}
	if (payload.tool_choice !== undefined || payload.parallel_tool_calls !== undefined) {
		return "tool controls are present";
	}
	const sampling = contract.sampling;
	if (
		payload.temperature !== sampling.temperature ||
		payload.top_p !== sampling.top_p ||
		payload.top_k !== sampling.top_k
	) {
		return "sampling mismatch";
	}
	if (canonicalJson(payload.chat_template_kwargs) !== canonicalJson(contract.chat_template_kwargs)) {
		return "thinking contract mismatch";
	}
	return undefined;
}

function identifySession(ctx, manifest) {
	const sessionId = ctx?.sessionManager?.getSessionId?.();
	if (typeof sessionId !== "string" || sessionId.length === 0) return undefined;
	return manifest?.files?.sessions?.find((session) => session.session_id === sessionId);
}

function authoritativePromptTokens(usage) {
	if (
		!Number.isSafeInteger(usage?.input) ||
		usage.input < 0 ||
		!Number.isSafeInteger(usage?.cacheRead) ||
		usage.cacheRead < 0 ||
		!Number.isSafeInteger(usage?.cacheWrite) ||
		usage.cacheWrite < 0
	) {
		return undefined;
	}
	return usage.input + usage.cacheRead + usage.cacheWrite;
}

function reportedOutputTokens(update) {
	const message =
		update?.type === "done"
			? update.message
			: update?.type === "error"
				? update.error
				: update?.partial;
	const output = message?.usage?.output;
	return Number.isSafeInteger(output) && output >= 0 ? output : undefined;
}

function monotonicNow() {
	return typeof globalThis.performance?.now === "function" ? globalThis.performance.now() : Date.now();
}

function setStatusSafe(ctx, message) {
	try {
		ctx?.ui?.setStatus?.(STATUS_KEY, message);
	} catch {
		// UI failures must never bypass the request-substitution safety boundary.
	}
}

export default function qwen250kQualification(pi) {
	const loaded = loadManifest();
	let accepted = false;
	let rejection = loaded.error;
	let turnStartedAt;
	let firstDataAt = 0;
	let firstDataOutputTokens;
	let finishedAt = 0;
	let liveOutputTokens;
	let activeSession;

	function reset() {
		accepted = false;
		rejection = loaded.error;
		turnStartedAt = monotonicNow();
		firstDataAt = 0;
		firstDataOutputTokens = undefined;
		finishedAt = 0;
		liveOutputTokens = undefined;
		activeSession = undefined;
	}

	pi.on("agent_start", () => reset());

	pi.on("session_before_compact", () => ({ cancel: true }));

	pi.on("before_provider_request", (event, ctx) => {
		try {
			firstDataAt = 0;
			firstDataOutputTokens = undefined;
			finishedAt = 0;
			liveOutputTokens = undefined;
			rejection = verifyPayload(event, ctx, loaded.manifest);
			activeSession = identifySession(ctx, loaded.manifest);
			if (rejection === undefined && activeSession === undefined) {
				rejection = "session identity mismatch";
			}
			accepted = rejection === undefined;
			if (!accepted) {
				setStatusSafe(
					ctx,
					`Pi 250K guard REJECTED: ${rejection}; fixed public 1-token request substituted`,
				);
				return fixedPublicFallback();
			}

			const runLabel =
				activeSession.clone === 1 ? "PRIME" : `WARM ${activeSession.clone - 1}/2`;
			setStatusSafe(ctx, `Pi 250K ${runLabel} public payload verified; qualification running`);
			const payload = event.payload;
			return {
				model: loaded.manifest.request_contract.model,
				messages: payload.messages,
				stream: true,
				stream_options: { include_usage: true, continuous_usage_stats: true },
				max_completion_tokens: loaded.manifest.token_accounting.completion_cap_tokens,
				temperature: loaded.manifest.request_contract.sampling.temperature,
				top_p: loaded.manifest.request_contract.sampling.top_p,
				top_k: loaded.manifest.request_contract.sampling.top_k,
				chat_template_kwargs: loaded.manifest.request_contract.chat_template_kwargs,
			};
		} catch {
			accepted = false;
			rejection = "guard internal validation failure";
			setStatusSafe(
				ctx,
				"Pi 250K guard REJECTED: internal validation failure; fixed public 1-token request substituted",
			);
			return fixedPublicFallback();
		}
	});

	pi.on("message_update", (event) => {
		if (!accepted) return;
		const update = event?.assistantMessageEvent;
		const outputTokens = reportedOutputTokens(update);
		if (outputTokens !== undefined && (liveOutputTokens === undefined || outputTokens >= liveOutputTokens)) {
			liveOutputTokens = outputTokens;
		}
		if (
			firstDataAt === 0 &&
			(update?.type === "thinking_delta" ||
				update?.type === "text_delta" ||
				update?.type === "toolcall_delta")
		) {
			firstDataAt = monotonicNow();
			firstDataOutputTokens = outputTokens;
		}
		if (update?.type === "done" || update?.type === "error") finishedAt = monotonicNow();
	});

	pi.on("message_end", (event, ctx) => {
		if (!accepted || event?.message?.role !== "assistant") return;
		finishedAt ||= monotonicNow();
		const usage = event.message.usage;
		const promptTokens = authoritativePromptTokens(usage);
		const outputTokens = Number.isSafeInteger(usage?.output) && usage.output >= 0 ? usage.output : liveOutputTokens;
		const target = loaded.manifest.token_accounting.target_prompt_tokens;
		const qualification = loaded.manifest.qualification;
		const firstDataSeconds =
			firstDataAt > 0 && turnStartedAt !== undefined ? (firstDataAt - turnStartedAt) / 1000 : undefined;
		const rateWindowSeconds = firstDataAt > 0 ? (finishedAt - firstDataAt) / 1000 : undefined;
		const rate =
			Number.isSafeInteger(outputTokens) &&
			Number.isSafeInteger(firstDataOutputTokens) &&
			outputTokens > firstDataOutputTokens &&
			rateWindowSeconds > 0
				? (outputTokens - firstDataOutputTokens) / rateWindowSeconds
				: undefined;

		const checks = {
			terminal: SUCCESS_STOP_REASONS.has(event.message.stopReason),
			prompt: promptTokens === target,
			firstData:
				firstDataSeconds !== undefined && firstDataSeconds < qualification.max_first_data_seconds,
			window:
				rateWindowSeconds !== undefined && rateWindowSeconds >= qualification.min_rate_window_seconds,
			output:
				Number.isSafeInteger(outputTokens) && outputTokens >= qualification.min_output_tokens,
			rate: rate !== undefined && rate >= qualification.target_output_rate_tps,
		};
		const passed = Object.values(checks).every(Boolean);
		const promptText = promptTokens === undefined ? "usage unavailable" : `${promptTokens.toLocaleString("en-US")} prompt tok`;
		const rateText = rate === undefined ? "rate unavailable" : `${rate.toFixed(1)} tok/s post-first`;
		const firstDataText =
			firstDataSeconds === undefined ? "first data unavailable" : `first data ${firstDataSeconds.toFixed(1)}s`;
		const outputText =
			outputTokens === undefined ? "output unavailable" : `${outputTokens.toLocaleString("en-US")} output tok`;
		const failedChecks = Object.entries(checks)
			.filter(([, passedCheck]) => !passedCheck)
			.map(([name]) => name)
			.join(",");
		const runLabel =
			activeSession.clone === 1 ? "PRIME" : `WARM ${activeSession.clone - 1}/2`;
		if (activeSession.clone === 1) {
			const primeChecks = { terminal: checks.terminal, prompt: checks.prompt };
			const primePassed = Object.values(primeChecks).every(Boolean);
			const failedPrimeChecks = Object.entries(primeChecks)
				.filter(([, passedCheck]) => !passedCheck)
				.map(([name]) => name)
				.join(",");
			setStatusSafe(
				ctx,
				`Pi 250K PRIME ${primePassed ? "COMPLETE" : `FAIL[${failedPrimeChecks}]`}: ${promptText} • ${rateText} • ${outputText} • ${firstDataText}`,
			);
			return;
		}
		setStatusSafe(
			ctx,
			`Pi 250K ${runLabel} ${qualification.configured_milestone.toUpperCase()} ${passed ? "QUALIFIED" : `FAIL[${failedChecks}]`}: ${promptText} • ${rateText} • ${outputText} • ${firstDataText}`,
		);
	});
}
