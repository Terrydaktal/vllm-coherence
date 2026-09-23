import { spawnSync } from "node:child_process";
import { isDeepStrictEqual } from "node:util";

const STATUS_KEY = "qwen-tool-call-integrity";
const TARGET_API = "openai-completions";
const TARGET_PROVIDER = "qwen-r9700";
const EPHEMERAL_RECOVERY_REQUEST = Symbol.for("qwen-r9700:ephemeral-recovery-request:v1");
const INTERNAL_GUARD_RETRY = Symbol.for("qwen-r9700:internal-guard-retry:v1");
const RAW_COMPLETION_TOKEN_IDS = Symbol.for("qwen-r9700:raw-completion-token-ids:v1");
const TOOL_CALL_START_TOKEN_ID = 248058;
const TOOL_CALL_END_TOKEN_ID = 248059;
const TARGET_MODELS = new Set([
	"qwen3.8-27b-frozenlock",
	"qwen3.8-27b-hauhau-aggressive",
	"qwen3.8-27b-philbert-aggressive",
	"qwen3.8-27b-hauhau-delta-aggressive",
]);
const STOCHASTIC_RECOVERY = Object.freeze({
	temperature: 1,
	top_p: 0.95,
	top_k: 40,
});
const RECOVERY_INSTRUCTION =
	"The preceding assistant attempt was rejected because its structured tool arguments or shell syntax ended incomplete. Re-emit the complete tool call from scratch. Do not infer a transport or command-size limit. Ensure every required argument, JSON object, shell quote, and heredoc terminator is complete before ending the response.";
const RETRYABLE_ERROR =
	"Qwen tool-call integrity server error: incomplete structured arguments were blocked before execution; retrying once with corrective stochastic steering";
const TERMINAL_ERROR =
	"Qwen tool-call integrity stopped the response because incomplete structured arguments recurred during corrective recovery";

function isRecord(value) {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isTargetContext(ctx) {
	return ctx?.model?.provider === TARGET_PROVIDER && ctx.model.api === TARGET_API;
}

function isTargetPayload(payload) {
	return isRecord(payload) && TARGET_MODELS.has(payload.model) && payload.stream === true;
}

function isTargetMessage(message) {
	return (
		message?.role === "assistant" &&
		message.provider === TARGET_PROVIDER &&
		message.api === TARGET_API &&
		TARGET_MODELS.has(message.model)
	);
}

function toolBlockAt(partial, contentIndex) {
	const block = partial?.content?.[contentIndex];
	return block?.type === "toolCall" ? block : undefined;
}

function scrubToolCalls(message) {
	if (!Array.isArray(message.content) || !message.content.some((block) => block?.type === "toolCall")) {
		return message;
	}
	return {
		...message,
		content: message.content.filter((block) => block?.type !== "toolCall"),
	};
}

function rejectedMessage(message, retryable, detail) {
	return {
		...message,
		content: [],
		stopReason: "error",
		rawStopReason: "tool_call_integrity_violation",
		...(retryable ? { [INTERNAL_GUARD_RETRY]: true } : {}),
		errorMessage: `${retryable ? RETRYABLE_ERROR : TERMINAL_ERROR}: ${detail}`,
	};
}

function strictArguments(raw) {
	if (typeof raw !== "string" || raw.trim().length === 0) {
		return { error: "tool arguments contained no complete JSON object" };
	}
	try {
		const value = JSON.parse(raw);
		if (!isRecord(value)) return { error: "tool arguments were not a JSON object" };
		return { value };
	} catch {
		return { error: "tool arguments ended before a complete JSON object was emitted" };
	}
}

function validateBashSyntax(command) {
	if (typeof command !== "string") return "bash command was not a string";
	if (command.includes("\0")) return "bash command contained a NUL byte";
	const result = spawnSync("/bin/bash", ["--noprofile", "--norc", "-n"], {
		input: command,
		encoding: "utf8",
		env: { LANG: "C", LC_ALL: "C", PATH: "/usr/bin:/bin" },
		timeout: 2000,
		maxBuffer: 64 * 1024,
		stdio: ["pipe", "ignore", "pipe"],
	});
	if (result.error !== undefined) {
		return result.error.code === "ETIMEDOUT"
			? "bash syntax validation timed out"
			: "bash syntax validation could not run";
	}
	if (result.status !== 0) return "bash command has incomplete or invalid shell syntax";
	if (typeof result.stderr === "string" && result.stderr.trim().length > 0) {
		return "bash command produced a syntax warning, such as an unfinished heredoc";
	}
	return undefined;
}

function validateRawToolProtocol(message) {
	const tokenIds = message?.[RAW_COMPLETION_TOKEN_IDS];
	if (!Array.isArray(tokenIds)) return "provider did not expose the requested raw completion token IDs";
	if (!tokenIds.every((token) => Number.isSafeInteger(token) && token >= 0)) {
		return "provider exposed malformed raw completion token IDs";
	}
	if (Number.isSafeInteger(message?.usage?.output) && message.usage.output !== tokenIds.length) {
		return "raw completion token count differed from provider usage";
	}

	let active = false;
	let rawCalls = 0;
	for (const token of tokenIds) {
		if (token === TOOL_CALL_START_TOKEN_ID) {
			if (active) return "raw token stream contained nested tool-call delimiters";
			active = true;
			rawCalls += 1;
		} else if (token === TOOL_CALL_END_TOKEN_ID) {
			if (!active) return "raw token stream contained an unmatched tool-call end delimiter";
			active = false;
		}
	}
	if (active) return "raw token stream ended inside an incomplete structured tool call";

	const parsedCalls = Array.isArray(message?.content)
		? message.content.filter((block) => block?.type === "toolCall").length
		: 0;
	if (rawCalls !== parsedCalls) {
		return "raw structured tool-call count differed from Pi's parsed tool calls";
	}
	if (rawCalls > 0 && message.stopReason !== "toolUse") {
		return "raw structured tool calls did not end with toolUse";
	}
	return undefined;
}

function validateToolCalls(message, records, streamFault) {
	if (streamFault !== undefined) return streamFault;
	if (message.stopReason !== "toolUse") return "tool calls did not end with toolUse";

	const seenIds = new Set();
	let toolCount = 0;
	for (const [contentIndex, block] of message.content.entries()) {
		if (block?.type !== "toolCall") continue;
		toolCount += 1;
		if (typeof block.id !== "string" || block.id.length === 0 || seenIds.has(block.id)) {
			return "tool call IDs were empty or duplicated";
		}
		seenIds.add(block.id);
		const record = records.get(contentIndex);
		if (record === undefined || !record.started || !record.ended) {
			return "tool call did not have one complete streamed start/end sequence";
		}
		if (record.id !== block.id || record.name !== block.name) {
			return "streamed tool identity differed from the finalized tool call";
		}
		const parsed = strictArguments(record.raw);
		if (parsed.error !== undefined) return parsed.error;
		if (!isDeepStrictEqual(parsed.value, block.arguments)) {
			return "Pi's repaired tool arguments differed from the exact emitted JSON";
		}
		if (block.name === "bash") {
			const syntaxError = validateBashSyntax(parsed.value.command);
			if (syntaxError !== undefined) return syntaxError;
		}
	}
	if (toolCount === 0) return undefined;
	if (records.size !== toolCount) return "stream contained an unmatched or duplicate tool-call record";
	return undefined;
}

export default function qwenToolCallIntegrity(pi) {
	let requestActive = false;
	let recoveryState = "none";
	let records = new Map();
	let streamFault;

	function resetStream() {
		records = new Map();
		streamFault = undefined;
	}

	pi.on("before_provider_request", (event, ctx) => {
		if (!isTargetContext(ctx) || !isTargetPayload(event?.payload)) {
			requestActive = false;
			resetStream();
			return undefined;
		}
		requestActive = true;
		resetStream();
		const observablePayload = { ...event.payload, return_token_ids: true };
		if (recoveryState !== "pending" && recoveryState !== "active") return observablePayload;

		recoveryState = "active";
		if (ctx.mode === "tui") {
			ctx.ui.setStatus?.(STATUS_KEY, "Qwen incomplete tool call blocked; corrective recovery in progress");
		}
		return {
			...observablePayload,
			...STOCHASTIC_RECOVERY,
			[EPHEMERAL_RECOVERY_REQUEST]: true,
			messages: Array.isArray(event.payload.messages)
				? [...event.payload.messages, { role: "user", content: RECOVERY_INSTRUCTION }]
				: event.payload.messages,
		};
	});

	pi.on("message_start", (event) => {
		if (!requestActive || event?.message?.role !== "assistant") return;
		resetStream();
	});

	pi.on("message_update", (event) => {
		if (!requestActive) return;
		const update = event?.assistantMessageEvent;
		if (!isRecord(update) || !Number.isSafeInteger(update.contentIndex)) return;
		const index = update.contentIndex;
		if (update.type === "toolcall_start") {
			if (records.has(index)) {
				streamFault = "stream emitted a duplicate tool-call start";
				return;
			}
			const block = toolBlockAt(update.partial, index);
			records.set(index, {
				started: true,
				ended: false,
				raw: "",
				id: block?.id,
				name: block?.name,
			});
			return;
		}
		if (update.type === "toolcall_delta") {
			const record = records.get(index);
			if (record === undefined || record.ended || typeof update.delta !== "string") {
				streamFault = "stream emitted a tool-call delta outside its start/end sequence";
				return;
			}
			record.raw += update.delta;
			const block = toolBlockAt(update.partial, index);
			if (block?.id) record.id = block.id;
			if (block?.name) record.name = block.name;
			return;
		}
		if (update.type === "toolcall_end") {
			const record = records.get(index);
			if (record === undefined || record.ended || update.toolCall?.type !== "toolCall") {
				streamFault = "stream emitted an unmatched or duplicate tool-call end";
				return;
			}
			record.ended = true;
			record.id = update.toolCall.id;
			record.name = update.toolCall.name;
			const parsed = strictArguments(record.raw);
			if (parsed.error !== undefined || !isDeepStrictEqual(parsed.value, update.toolCall.arguments)) {
				streamFault = parsed.error ?? "tool-call end differed from the exact emitted JSON";
			}
		}
	});

	pi.on("message_end", (event, ctx) => {
		const message = event?.message;
		if (!isTargetMessage(message)) return undefined;
		requestActive = false;

		if (message.stopReason === "error" || message.stopReason === "aborted") {
			if (
				message.stopReason === "error" &&
				typeof message.errorMessage === "string" &&
				message.errorMessage.includes("Provider tool-call integrity server error")
			) {
				const retryable = recoveryState === "none";
				recoveryState = retryable ? "pending" : "none";
				resetStream();
				if (ctx.mode === "tui") {
					ctx.ui.setStatus?.(
						STATUS_KEY,
						retryable
							? "Pi rejected incomplete JSON; retrying once with corrective steering"
							: "Pi rejected incomplete JSON again; stopped safely",
					);
				}
				return {
					message: rejectedMessage(
						scrubToolCalls(message),
						retryable,
						"provider stream did not contain one complete JSON argument object",
					),
				};
			}
			resetStream();
			return { message: scrubToolCalls(message) };
		}
		const hasToolCalls = Array.isArray(message.content) && message.content.some((block) => block?.type === "toolCall");
		let violation;
		try {
			violation = validateRawToolProtocol(message);
		} catch {
			violation = "raw tool-call protocol validation failed internally";
		}
		if (!hasToolCalls && violation === undefined) {
			resetStream();
			if (recoveryState === "active") {
				recoveryState = "none";
				if (ctx.mode === "tui") {
					ctx.ui.setStatus?.(STATUS_KEY, "Qwen tool-call recovery omitted the tool; stopped safely");
				}
				return {
					message: rejectedMessage(
						message,
						false,
						"corrective recovery ended without a complete tool call",
					),
				};
			}
			return undefined;
		}

		if (violation === undefined) {
			try {
				violation = validateToolCalls(message, records, streamFault);
			} catch {
				violation = "tool-call integrity validation failed internally";
			}
		}
		resetStream();
		if (violation === undefined) {
			if (recoveryState === "active") {
				recoveryState = "none";
				if (ctx.mode === "tui") ctx.ui.setStatus?.(STATUS_KEY, "Qwen tool-call recovery succeeded");
			}
			return undefined;
		}

		const retryable = recoveryState === "none";
		recoveryState = retryable ? "pending" : "none";
		if (ctx.mode === "tui") {
			ctx.ui.setStatus?.(
				STATUS_KEY,
				retryable
					? "Qwen incomplete tool call blocked; retrying once with corrective steering"
					: "Qwen incomplete tool call recurred; stopped safely",
			);
		}
		return { message: rejectedMessage(message, retryable, violation) };
	});

	pi.on("agent_settled", () => {
		requestActive = false;
		recoveryState = "none";
		resetStream();
	});
}
