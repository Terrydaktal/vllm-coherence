import { closeSync, constants, fstatSync, fsyncSync, openSync, writeSync } from "node:fs";
import { isAbsolute } from "node:path";

const AUDIT_FILE_ENV = "QWEN_PI_LATENCY_AUDIT_FILE";

function monotonicNow() {
	return typeof globalThis.performance?.now === "function" ? globalThis.performance.now() : Date.now();
}

function openAuditFile(path) {
	if (!isAbsolute(path)) throw new Error(`${AUDIT_FILE_ENV} must be an absolute path`);
	let descriptor;
	try {
		descriptor = openSync(
			path,
			constants.O_WRONLY |
				constants.O_APPEND |
				constants.O_NOFOLLOW |
				constants.O_CREAT |
				constants.O_EXCL,
			0o600,
		);
	} catch (error) {
		if (error?.code !== "EEXIST") throw error;
		descriptor = openSync(path, constants.O_WRONLY | constants.O_APPEND | constants.O_NOFOLLOW);
	}
	try {
		const info = fstatSync(descriptor);
		if (
			!info.isFile() ||
			info.nlink !== 1 ||
			(typeof process.getuid === "function" && info.uid !== process.getuid()) ||
			(info.mode & 0o777) !== 0o600
		) {
			throw new Error("latency audit file must be an owned mode-0600 single-link regular file");
		}
		return descriptor;
	} catch (error) {
		closeSync(descriptor);
		throw error;
	}
}

function contentBytes(content) {
	if (!Array.isArray(content)) return undefined;
	let total = 0;
	for (const block of content) {
		if (block?.type !== "text" || typeof block.text !== "string") return undefined;
		total += Buffer.byteLength(block.text, "utf8");
	}
	return total;
}

function appendRecord(descriptor, payload) {
	const encoded = Buffer.from(`${JSON.stringify(payload)}\n`, "utf8");
	let offset = 0;
	while (offset < encoded.length) {
		const written = writeSync(descriptor, encoded, offset, encoded.length - offset);
		if (written <= 0) throw new Error("latency audit append made no progress");
		offset += written;
	}
}

function safeInteger(value) {
	return Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

export default function qwenLatencyAudit(pi) {
	const path = process.env[AUDIT_FILE_ENV];
	if (path === undefined || path === "") return;

	const descriptor = openAuditFile(path);
	let closed = false;
	let requestOrdinal = 0;
	let firstDataRecorded = false;

	function record(event, fields = {}) {
		if (closed) return;
		const payload = {
			event,
			monotonic_ms: monotonicNow(),
			request_ordinal: requestOrdinal,
			wall_time: new Date().toISOString(),
			...fields,
		};
		appendRecord(descriptor, payload);
	}

	function contextTokens(ctx) {
		return safeInteger(ctx?.getContextUsage?.()?.tokens);
	}

	pi.on("agent_start", (_event, ctx) => record("agent_start", { context_tokens: contextTokens(ctx) }));

	pi.on("before_provider_request", (_event, ctx) => {
		requestOrdinal += 1;
		firstDataRecorded = false;
		record("before_provider_request", { context_tokens: contextTokens(ctx) });
	});

	pi.on("after_provider_response", (_event, ctx) =>
		record("after_provider_response", { context_tokens: contextTokens(ctx) }),
	);

	pi.on("message_update", (event) => {
		if (firstDataRecorded) return;
		const update = event?.assistantMessageEvent;
		if (!update || !["thinking_delta", "text_delta", "toolcall_delta"].includes(update.type)) return;
		firstDataRecorded = true;
		record("first_model_data", {
			output_tokens: safeInteger(update?.partial?.usage?.output),
			update_type: update.type,
		});
	});

	pi.on("message_end", (event, ctx) => {
		const message = event?.message;
		record("message_end", {
			context_tokens: contextTokens(ctx),
			output_tokens: safeInteger(message?.usage?.output),
			role: typeof message?.role === "string" ? message.role : undefined,
			stop_reason: typeof message?.stopReason === "string" ? message.stopReason : undefined,
		});
	});

	pi.on("tool_execution_start", (event) =>
		record("tool_execution_start", {
			tool_call_id: typeof event?.toolCallId === "string" ? event.toolCallId : undefined,
			tool_name: typeof event?.toolName === "string" ? event.toolName : undefined,
		}),
	);

	pi.on("tool_result", (event) =>
		record("tool_result", {
			is_error: event?.isError === true,
			model_visible_bytes: contentBytes(event?.content),
			tool_call_id: typeof event?.toolCallId === "string" ? event.toolCallId : undefined,
			tool_name: typeof event?.toolName === "string" ? event.toolName : undefined,
		}),
	);

	pi.on("turn_end", (_event, ctx) => {
		record("turn_end", { context_tokens: contextTokens(ctx) });
		fsyncSync(descriptor);
	});

	pi.on("agent_end", (_event, ctx) => record("agent_end", { context_tokens: contextTokens(ctx) }));

	pi.on("session_shutdown", (_event, ctx) => {
		record("session_shutdown", { context_tokens: contextTokens(ctx) });
		fsyncSync(descriptor);
		closeSync(descriptor);
		closed = true;
	});
}
