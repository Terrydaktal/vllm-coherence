import { createHash } from "node:crypto";
import {
	closeSync,
	constants,
	fstatSync,
	fsyncSync,
	lstatSync,
	openSync,
	readFileSync,
	realpathSync,
	writeFileSync,
} from "node:fs";
import { dirname } from "node:path";

const TARGET_API = "openai-completions";
const TARGET_MODEL = "qwen3.8-27b-frozenlock";
const TARGET_PROVIDER = "qwen-r9700";
const CAPTURE_PATH_ENV = "QWEN_PI_STRUCTURED_OUTCOME_CAPTURE_PATH";

export const FINAL_ANSWER_TOOL_NAME = "qwen_final_answer";
export const STRUCTURED_OUTCOME_ERROR_PREFIX = "Qwen structured-outcome protocol violation";

export const FINAL_ANSWER_PARAMETERS = Object.freeze({
	type: "object",
	additionalProperties: false,
	required: Object.freeze(["answer"]),
	properties: Object.freeze({
		answer: Object.freeze({ type: "string", minLength: 1 }),
	}),
});

const FINAL_ANSWER_DESCRIPTION =
	"Use this reserved tool only when the response is complete and no further external tool action is needed. Put the entire user-facing final response in answer. Never call it together with another tool.";

function isRecord(value) {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isTargetContext(ctx) {
	return ctx?.model?.provider === TARGET_PROVIDER && ctx.model.api === TARGET_API;
}

function isTargetPayload(payload) {
	return isRecord(payload) && payload.model === TARGET_MODEL && payload.stream === true;
}

function isTargetMessage(message) {
	return (
		message?.role === "assistant" &&
		message.provider === TARGET_PROVIDER &&
		message.api === TARGET_API &&
		message.model === TARGET_MODEL
	);
}

function finalAnswerParameters() {
	return {
		type: "object",
		additionalProperties: false,
		required: ["answer"],
		properties: {
			answer: { type: "string", minLength: 1 },
		},
	};
}

function rawFinalAnswerTool() {
	return {
		type: "function",
		function: {
			name: FINAL_ANSWER_TOOL_NAME,
			description: FINAL_ANSWER_DESCRIPTION,
			parameters: finalAnswerParameters(),
		},
	};
}

function stableProtectedFile(path, expected) {
	const before = lstatSync(path);
	if (
		!before.isFile() ||
		before.isSymbolicLink() ||
		before.uid !== process.getuid() ||
		(before.mode & 0o777) !== 0o600 ||
		before.nlink !== 1 ||
		realpathSync(path) !== path
	) {
		throw new Error(`structured-outcome capture has an unsafe identity: ${path}`);
	}
	const observed = readFileSync(path);
	const after = lstatSync(path);
	if (
		before.dev !== after.dev ||
		before.ino !== after.ino ||
		before.size !== after.size ||
		before.mtimeMs !== after.mtimeMs ||
		before.ctimeMs !== after.ctimeMs ||
		!observed.equals(expected)
	) {
		throw new Error(`structured-outcome capture changed or differs: ${path}`);
	}
}

function publishCapture(path, payload) {
	if (typeof path !== "string" || !path.startsWith("/") || path.includes("\0")) {
		throw new Error(`${CAPTURE_PATH_ENV} is not an absolute safe path`);
	}
	const parent = dirname(path);
	const parentInfo = lstatSync(parent);
	if (
		!parentInfo.isDirectory() ||
		parentInfo.isSymbolicLink() ||
		parentInfo.uid !== process.getuid() ||
		(parentInfo.mode & 0o777) !== 0o700 ||
		realpathSync(parent) !== parent
	) {
		throw new Error(`structured-outcome capture parent has an unsafe identity: ${parent}`);
	}
	const encoded = Buffer.from(`${JSON.stringify(payload, null, 2)}\n`, "utf8");
	try {
		const descriptor = openSync(
			path,
			constants.O_WRONLY |
				constants.O_CREAT |
				constants.O_EXCL |
				constants.O_CLOEXEC |
				(constants.O_NOFOLLOW ?? 0),
			0o600,
		);
		try {
			writeFileSync(descriptor, encoded);
			fsyncSync(descriptor);
			const info = fstatSync(descriptor);
			if (info.uid !== process.getuid() || (info.mode & 0o777) !== 0o600 || info.nlink !== 1) {
				throw new Error("structured-outcome capture publication identity changed");
			}
		} finally {
			closeSync(descriptor);
		}
		const directory = openSync(parent, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_CLOEXEC);
		try {
			fsyncSync(directory);
		} finally {
			closeSync(directory);
		}
	} catch (error) {
		if (error?.code !== "EEXIST") throw error;
	}
	stableProtectedFile(path, encoded);
	return createHash("sha256").update(encoded).digest("hex");
}

function rawToolName(tool) {
	return tool?.type === "function" && isRecord(tool.function) ? tool.function.name : undefined;
}

function canonicalTools(tools) {
	const source = Array.isArray(tools) ? tools : [];
	const result = [];
	let insertionIndex;

	for (const tool of source) {
		if (rawToolName(tool) === FINAL_ANSWER_TOOL_NAME) {
			insertionIndex ??= result.length;
			continue;
		}
		result.push(tool);
	}

	result.splice(insertionIndex ?? result.length, 0, rawFinalAnswerTool());
	return result;
}

function isThinkingBlock(block) {
	return isRecord(block) && block.type === "thinking" && typeof block.thinking === "string";
}

function isTextBlock(block) {
	return isRecord(block) && block.type === "text" && typeof block.text === "string";
}

function isToolCall(block) {
	return isRecord(block) && block.type === "toolCall";
}

function isStructurallyValidToolCall(block) {
	return (
		isToolCall(block) &&
		typeof block.id === "string" &&
		block.id.length > 0 &&
		typeof block.name === "string" &&
		block.name.length > 0 &&
		isRecord(block.arguments)
	);
}

function finalAnswerFromArguments(args) {
	if (!isRecord(args)) return undefined;
	const keys = Object.keys(args);
	if (keys.length !== 1 || keys[0] !== "answer") return undefined;
	if (typeof args.answer !== "string" || args.answer.trim().length === 0) return undefined;
	return args.answer;
}

function failedMessage(message, reason) {
	return {
		...message,
		content: [],
		stopReason: "error",
		rawStopReason: "structured_outcome_violation",
		errorMessage: `${STRUCTURED_OUTCOME_ERROR_PREFIX}: ${reason}`,
	};
}

function abortRequest(ctx, payload, reason) {
	try {
		ctx?.abort?.();
		ctx?.shutdown?.();
	} catch {
		// The returned request is independently inert. Pi currently treats extension
		// exceptions as non-fatal, so this handler must never rely on throwing.
	}
	return {
		model: payload.model,
		messages: [{ role: "user", content: "." }],
		max_tokens: 1,
		stream: false,
		temperature: 0,
		top_p: 1,
		_qwen_structured_outcome_error: reason,
	};
}

function scrubIncompleteMessage(message) {
	return {
		...message,
		content: [],
		errorMessage:
			message.errorMessage ??
			(message.stopReason === "aborted"
				? "Operation aborted before a complete structured outcome"
				: "Provider failed before a complete structured outcome"),
	};
}

function validateSuccessfulOutcome(message, allowedRealTools) {
	if (!Array.isArray(message.content)) {
		return { error: "assistant content is not an array" };
	}

	const calls = [];
	const callIds = new Set();
	const canonicalContent = [];
	let visibleText = "";
	for (const block of message.content) {
		if (isThinkingBlock(block)) {
			canonicalContent.push(block);
			continue;
		}
		// Qwen's native tool syntax permits a plain-text preamble before the
		// <tool_call> tag, and vLLM deliberately returns that text alongside
		// tool_calls. Preserve it provisionally: a text-only stop is a valid final
		// answer, while text accompanying a structured call is discarded below so
		// the validated call remains authoritative.
		if (isTextBlock(block)) {
			visibleText += block.text;
			canonicalContent.push(block);
			continue;
		}
		if (!isToolCall(block)) {
			return { error: "assistant emitted a malformed or unsupported content block" };
		}
		if (!isStructurallyValidToolCall(block)) {
			return { error: "assistant emitted a malformed structured tool call" };
		}
		if (callIds.has(block.id)) {
			return { error: "assistant emitted duplicate structured tool-call identifiers" };
		}
		callIds.add(block.id);
		calls.push(block);
		canonicalContent.push(block);
	}

	if (calls.length === 0) {
		// The live Qwen/vLLM path has demonstrated that tool_choice="required" is
		// not a universal generation guarantee: it can return a complete ordinary
		// answer with finish_reason=stop. Losing that already-rendered answer is
		// worse than accepting the provider's normal final-answer representation.
		// The earlier repetition/premature-action guard still rejects text that
		// merely announces an unperformed action before this handler runs.
		if (message.stopReason === "stop" && visibleText.trim().length > 0) {
			return { ordinaryFinal: true };
		}
		return { error: "assistant ended without a usable final answer or structured tool outcome" };
	}

	// Visible text next to a tool call is only a preamble. Do not persist it as
	// though the announced work had already completed.
	const structuredContent = canonicalContent.filter((block) => !isTextBlock(block));

	const finalCalls = calls.filter((call) => call.name === FINAL_ANSWER_TOOL_NAME);
	if (finalCalls.length === 0) {
		if (message.stopReason !== "toolUse") {
			return { error: "real tool calls did not end with toolUse" };
		}
		const unknown = calls.find((call) => !allowedRealTools.has(call.name));
		if (unknown !== undefined) {
			return { error: `assistant emitted an unadvertised tool call ${unknown.name}` };
		}
		return { realTools: true, content: structuredContent };
	}
	if (finalCalls.length !== 1) {
		return { error: `${FINAL_ANSWER_TOOL_NAME} must appear exactly once` };
	}
	if (calls.length !== 1) {
		return { error: `${FINAL_ANSWER_TOOL_NAME} cannot be combined with another tool call` };
	}
	if (message.stopReason !== "toolUse") {
		return { error: `${FINAL_ANSWER_TOOL_NAME} did not end with toolUse` };
	}

	const answer = finalAnswerFromArguments(finalCalls[0].arguments);
	if (answer === undefined) {
		return { error: `${FINAL_ANSWER_TOOL_NAME} arguments do not match the required schema` };
	}
	return { answer };
}

export default function qwenStructuredOutcome(pi) {
	let allowedRealTools = new Set();

	pi.registerTool({
		name: FINAL_ANSWER_TOOL_NAME,
		label: "Final answer",
		description: FINAL_ANSWER_DESCRIPTION,
		parameters: finalAnswerParameters(),
		async execute() {
			throw new Error(
				`${FINAL_ANSWER_TOOL_NAME} reached tool execution; message_end interception invariant failed`,
			);
		},
	});

	pi.on("before_provider_request", (event, ctx) => {
		if (!isTargetContext(ctx) || !isTargetPayload(event?.payload)) return undefined;
		const payload = event.payload;
		if (
			(payload.tool_choice !== undefined &&
				payload.tool_choice !== "auto" &&
				payload.tool_choice !== "required") ||
			payload.response_format !== undefined ||
			payload.structured_outputs !== undefined ||
			payload.structured_output !== undefined
		) {
			allowedRealTools = new Set();
			return abortRequest(ctx, payload, "conflicting structured-output request controls");
		}
		allowedRealTools = new Set(
			(Array.isArray(payload.tools) ? payload.tools : [])
				.map(rawToolName)
				.filter((name) => typeof name === "string" && name !== FINAL_ANSWER_TOOL_NAME),
		);
		const transformed = {
			...payload,
			tools: canonicalTools(payload.tools),
			tool_choice: "required",
		};
		const capturePath = process.env[CAPTURE_PATH_ENV];
		if (capturePath !== undefined) {
			try {
				publishCapture(capturePath, transformed);
			} catch (error) {
				allowedRealTools = new Set();
				return abortRequest(
					ctx,
					payload,
					`cannot durably capture the qualified provider payload: ${error instanceof Error ? error.message : String(error)}`,
				);
			}
		}
		return transformed;
	});

	pi.on("message_end", (event) => {
		const message = event?.message;
		if (!isTargetMessage(message)) return undefined;
		const requestAllowedRealTools = allowedRealTools;
		allowedRealTools = new Set();

		if (message.stopReason === "error" || message.stopReason === "aborted") {
			return { message: scrubIncompleteMessage(message) };
		}
		if (message.stopReason !== "stop" && message.stopReason !== "toolUse") {
			return {
				message: failedMessage(
					message,
					`completion ended with disallowed stop reason ${String(message.stopReason)}`,
				),
			};
		}

		const outcome = validateSuccessfulOutcome(message, requestAllowedRealTools);
		if (outcome.error !== undefined) {
			return { message: failedMessage(message, outcome.error) };
		}
		if (outcome.realTools) {
			return {
				message: {
					...message,
					content: outcome.content,
				},
			};
		}
		if (outcome.ordinaryFinal) {
			// Return no replacement so Pi persists the exact already-rendered answer,
			// including formatting, usage, reasoning and finish metadata.
			return undefined;
		}

		return {
			message: {
				...message,
				content: [
					...message.content.filter((block) => block.type === "thinking"),
					{ type: "text", text: outcome.answer },
				],
				stopReason: "stop",
				rawStopReason: "stop",
				endTurn: true,
				errorMessage: undefined,
				deferred: undefined,
			},
		};
	});
}
