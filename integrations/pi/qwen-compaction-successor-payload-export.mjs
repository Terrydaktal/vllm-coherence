import { createHash, randomBytes } from "node:crypto";
import {
	closeSync,
	constants,
	fsyncSync,
	linkSync,
	lstatSync,
	openSync,
	realpathSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { dirname } from "node:path";

const OUTPUT_ENV = "QWEN_PI_COMPACTION_SUCCESSOR_PAYLOAD_EXPORT";
const PLACEHOLDER_ENV = "QWEN_PI_COMPACTION_SUCCESSOR_PLACEHOLDER";
const TARGET_MODEL = "qwen3.8-27b-frozenlock";
const CONTROL_TYPE = "qwen-compaction-successor-payload-export-v1";
const SCHEMA = "urn:qwen-r9700:compaction-successor-payload-template:v1";

function protectedDirectory(path) {
	const metadata = lstatSync(path);
	if (
		!metadata.isDirectory() ||
		metadata.isSymbolicLink() ||
		metadata.uid !== process.getuid() ||
		(metadata.mode & 0o777) !== 0o700 ||
		realpathSync(path) !== path
	) {
		throw new Error(`successor payload-template directory has an unsafe identity: ${path}`);
	}
}

function fsyncDirectory(path) {
	const descriptor = openSync(path, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW);
	try {
		fsyncSync(descriptor);
	} finally {
		closeSync(descriptor);
	}
}

function publishCreateOnly(path, payload) {
	const directory = dirname(path);
	protectedDirectory(directory);
	const temporary = `${path}.publish-${process.pid}-${randomBytes(16).toString("hex")}.tmp`;
	let descriptor;
	try {
		descriptor = openSync(
			temporary,
			constants.O_WRONLY |
				constants.O_CREAT |
				constants.O_EXCL |
				constants.O_CLOEXEC |
				constants.O_NOFOLLOW,
			0o400,
		);
		writeFileSync(descriptor, payload);
		fsyncSync(descriptor);
		closeSync(descriptor);
		descriptor = undefined;
		linkSync(temporary, path);
		fsyncDirectory(directory);
	} finally {
		if (descriptor !== undefined) closeSync(descriptor);
		try {
			unlinkSync(temporary);
			fsyncDirectory(directory);
		} catch (error) {
			if (error?.code !== "ENOENT") throw error;
		}
	}
}

function countPlaceholder(value, placeholder) {
	if (typeof value === "string") return value.split(placeholder).length - 1;
	if (Array.isArray(value)) {
		return value.reduce((total, child) => total + countPlaceholder(child, placeholder), 0);
	}
	if (value !== null && typeof value === "object") {
		return Object.values(value).reduce(
			(total, child) => total + countPlaceholder(child, placeholder),
			0,
		);
	}
	return 0;
}

function tokenizePayload(payload) {
	const result = { model: payload.model, messages: payload.messages };
	for (const key of [
		"add_generation_prompt",
		"add_special_tokens",
		"chat_template",
		"chat_template_kwargs",
		"continue_final_message",
		"tools",
	]) {
		if (payload[key] !== undefined) result[key] = payload[key];
	}
	return result;
}

function scrubbedAbortPayload(payload) {
	return {
		max_tokens: 1,
		messages: [{ content: ".", role: "user" }],
		model: payload.model,
		stream: false,
		temperature: 0,
		top_p: 1,
	};
}

export default function qwenCompactionSuccessorPayloadExport(pi) {
	const output = process.env[OUTPUT_ENV];
	const placeholder = process.env[PLACEHOLDER_ENV];
	if (typeof output !== "string" || !output.startsWith("/") || output.includes("\0")) {
		throw new Error(`${OUTPUT_ENV} is missing or invalid`);
	}
	if (
		typeof placeholder !== "string" ||
		placeholder.length < 64 ||
		placeholder.length > 131072 ||
		placeholder.includes("\0")
	) {
		throw new Error(`${PLACEHOLDER_ENV} is missing or invalid`);
	}
	let triggered = false;
	let exported = false;
	pi.on("context", (event) => {
		const messages = Array.isArray(event?.messages) ? event.messages : [];
		const matching = messages.filter((message) => message?.customType === CONTROL_TYPE);
		if (triggered && matching.length !== 1) {
			throw new Error("successor payload-export control-message cardinality differs");
		}
		return { messages: messages.filter((message) => message?.customType !== CONTROL_TYPE) };
	});
	pi.on("session_start", () => {
		if (triggered) throw new Error("successor payload export autostart fired more than once");
		triggered = true;
		pi.sendMessage(
			{
				customType: CONTROL_TYPE,
				content: "successor payload export transport control",
				display: false,
			},
			{ triggerTurn: true },
		);
	});
	pi.on("before_provider_request", (event, ctx) => {
		const payload = event?.payload;
		if (payload?.model !== TARGET_MODEL) return undefined;
		if (exported) throw new Error("successor payload was exported more than once");
		const tokenizationPayload = tokenizePayload(payload);
		if (countPlaceholder(tokenizationPayload, placeholder) !== 1) {
			throw new Error("successor payload does not contain exactly one sealed placeholder");
		}
		const payloadBytes = Buffer.from(JSON.stringify(tokenizationPayload), "utf8");
		const document = {
			payload: tokenizationPayload,
			payload_bytes: payloadBytes.length,
			payload_sha256: createHash("sha256").update(payloadBytes).digest("hex"),
			placeholder_sha256: createHash("sha256").update(placeholder).digest("hex"),
			schema: SCHEMA,
		};
		publishCreateOnly(output, Buffer.from(`${JSON.stringify(document)}\n`, "utf8"));
		exported = true;
		ctx.abort();
		ctx.shutdown();
		return scrubbedAbortPayload(payload);
	});
}
