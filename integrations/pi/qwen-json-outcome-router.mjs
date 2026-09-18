import { createHash } from "node:crypto";

const TARGET_API = "openai-completions";
const TARGET_PROVIDER = "qwen-r9700";
const TARGET_MODELS = new Set([
	"qwen3.8-27b-frozenlock",
	"qwen3.8-27b-hauhau-aggressive",
	"qwen3.8-27b-philbert-aggressive",
	"qwen3.8-27b-hauhau-delta-aggressive",
]);

export const OUTCOME_FINAL_STOP_REASON = "qwen_json_outcome_final";
export const OUTCOME_TOOL_STOP_REASON = "qwen_json_outcome_tool";
export const OUTCOME_ERROR_PREFIX = "Qwen JSON-outcome protocol violation";

function isRecord(value) {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isTargetContext(ctx) {
	return ctx?.model?.provider === TARGET_PROVIDER && ctx.model.api === TARGET_API;
}

function isTargetPayload(payload) {
	return isRecord(payload) && TARGET_MODELS.has(payload.model) && payload.stream === true;
}

function isTargetMessage(message, model) {
	return (
		message?.role === "assistant" &&
		message.provider === TARGET_PROVIDER &&
		message.api === TARGET_API &&
		message.model === model
	);
}

function rawToolName(tool) {
	return tool?.type === "function" && isRecord(tool.function) ? tool.function.name : undefined;
}

function strictSchema(value, path = "schema") {
	if (typeof value === "boolean") return value;
	if (!isRecord(value)) throw new Error(`${path} is not a JSON-schema object`);
	const result = {};
	for (const [key, item] of Object.entries(value)) {
		if (key === "additionalProperties") continue;
		if (key === "properties" || key === "$defs" || key === "definitions") {
			if (!isRecord(item)) throw new Error(`${path}.${key} is not an object`);
			result[key] = Object.fromEntries(
				Object.entries(item).map(([name, child]) => [name, strictSchema(child, `${path}.${key}.${name}`)]),
			);
			continue;
		}
		if (key === "items" || key === "contains" || key === "not" || key === "if" || key === "then" || key === "else") {
			result[key] = strictSchema(item, `${path}.${key}`);
			continue;
		}
		if (key === "allOf" || key === "anyOf" || key === "oneOf" || key === "prefixItems") {
			if (!Array.isArray(item)) throw new Error(`${path}.${key} is not an array`);
			result[key] = item.map((child, index) => strictSchema(child, `${path}.${key}[${index}]`));
			continue;
		}
		result[key] = structuredClone(item);
	}
	if (value.type === "object" || isRecord(value.properties)) {
		if (value.additionalProperties !== undefined && value.additionalProperties !== false) {
			throw new Error(`${path} permits unbounded additional object properties`);
		}
		result.additionalProperties = false;
	}
	return result;
}

function toolBranches(tools) {
	if (!Array.isArray(tools)) return { branches: [], schemas: new Map() };
	const names = new Set();
	const branches = [];
	const schemas = new Map();
	for (const [index, tool] of tools.entries()) {
		const name = rawToolName(tool);
		if (typeof name !== "string" || name.length === 0) {
			throw new Error(`tools[${index}] is not a named function tool`);
		}
		if (names.has(name)) throw new Error(`tool name ${name} is duplicated`);
		names.add(name);
		const parameters = strictSchema(tool.function.parameters ?? { type: "object", properties: {} }, `tools[${index}].parameters`);
		schemas.set(name, parameters);
		branches.push({
			type: "object",
			properties: {
				kind: { type: "string", const: "tool" },
				name: { type: "string", const: name },
				arguments: parameters,
			},
			required: ["kind", "name", "arguments"],
			additionalProperties: false,
		});
	}
	return { branches, schemas };
}

function outcomeResponseFormat(tools) {
	const { branches, schemas } = toolBranches(tools);
	branches.push({
		type: "object",
		properties: {
			kind: { type: "string", const: "final" },
			answer: { type: "string", minLength: 1 },
		},
		required: ["kind", "answer"],
		additionalProperties: false,
	});
	return {
		responseFormat: {
			type: "json_schema",
			json_schema: {
				name: "qwen_pi_outcome",
				strict: true,
				schema: { oneOf: branches },
			},
		},
		schemas,
	};
}

function resolveRef(root, reference) {
	if (typeof reference !== "string" || !reference.startsWith("#/")) return undefined;
	let current = root;
	for (const encoded of reference.slice(2).split("/")) {
		const key = encoded.replaceAll("~1", "/").replaceAll("~0", "~");
		if (!isRecord(current) || !(key in current)) return undefined;
		current = current[key];
	}
	return current;
}

function matchesType(value, type) {
	if (type === "null") return value === null;
	if (type === "object") return isRecord(value);
	if (type === "array") return Array.isArray(value);
	if (type === "integer") return Number.isSafeInteger(value);
	if (type === "number") return typeof value === "number" && Number.isFinite(value);
	return typeof value === type;
}

function schemaError(value, schema, root = schema, path = "arguments", depth = 0) {
	if (depth > 64) return `${path} exceeds the validation depth limit`;
	if (schema === true) return undefined;
	if (schema === false || !isRecord(schema)) return `${path} is forbidden by its schema`;
	if (schema.$ref !== undefined) {
		const resolved = resolveRef(root, schema.$ref);
		return resolved === undefined
			? `${path} contains an unsupported schema reference`
			: schemaError(value, resolved, root, path, depth + 1);
	}
	if (schema.const !== undefined && JSON.stringify(value) !== JSON.stringify(schema.const)) {
		return `${path} differs from its required constant`;
	}
	if (Array.isArray(schema.enum) && !schema.enum.some((item) => JSON.stringify(item) === JSON.stringify(value))) {
		return `${path} is outside its allowed enumeration`;
	}
	for (const key of ["allOf"]) {
		if (Array.isArray(schema[key])) {
			for (const child of schema[key]) {
				const error = schemaError(value, child, root, path, depth + 1);
				if (error !== undefined) return error;
			}
		}
	}
	for (const key of ["anyOf", "oneOf"]) {
		if (Array.isArray(schema[key])) {
			const matches = schema[key].filter(
				(child) => schemaError(value, child, root, path, depth + 1) === undefined,
			).length;
			if ((key === "anyOf" && matches === 0) || (key === "oneOf" && matches !== 1)) {
				return `${path} does not match ${key}`;
			}
		}
	}
	if (schema.type !== undefined) {
		const types = Array.isArray(schema.type) ? schema.type : [schema.type];
		if (!types.some((type) => matchesType(value, type))) return `${path} has the wrong type`;
	}
	if (isRecord(value)) {
		const properties = isRecord(schema.properties) ? schema.properties : {};
		for (const name of Array.isArray(schema.required) ? schema.required : []) {
			if (!(name in value)) return `${path}.${name} is required`;
		}
		if (schema.additionalProperties === false) {
			const unknown = Object.keys(value).find((name) => !(name in properties));
			if (unknown !== undefined) return `${path}.${unknown} is not allowed`;
		}
		for (const [name, child] of Object.entries(properties)) {
			if (name in value) {
				const error = schemaError(value[name], child, root, `${path}.${name}`, depth + 1);
				if (error !== undefined) return error;
			}
		}
	}
	if (Array.isArray(value)) {
		if (Number.isSafeInteger(schema.minItems) && value.length < schema.minItems) return `${path} is too short`;
		if (Number.isSafeInteger(schema.maxItems) && value.length > schema.maxItems) return `${path} is too long`;
		if (schema.items !== undefined) {
			for (const [index, item] of value.entries()) {
				const error = schemaError(item, schema.items, root, `${path}[${index}]`, depth + 1);
				if (error !== undefined) return error;
			}
		}
	}
	if (typeof value === "string") {
		if (Number.isSafeInteger(schema.minLength) && value.length < schema.minLength) return `${path} is too short`;
		if (Number.isSafeInteger(schema.maxLength) && value.length > schema.maxLength) return `${path} is too long`;
		if (typeof schema.pattern === "string") {
			try {
				if (!new RegExp(schema.pattern, "u").test(value)) return `${path} does not match its required pattern`;
			} catch {
				return `${path} contains an invalid schema pattern`;
			}
		}
	}
	if (typeof value === "number") {
		if (typeof schema.minimum === "number" && value < schema.minimum) return `${path} is below its minimum`;
		if (typeof schema.maximum === "number" && value > schema.maximum) return `${path} is above its maximum`;
		if (typeof schema.exclusiveMinimum === "number" && value <= schema.exclusiveMinimum) return `${path} is below its exclusive minimum`;
		if (typeof schema.exclusiveMaximum === "number" && value >= schema.exclusiveMaximum) return `${path} is above its exclusive maximum`;
	}
	return undefined;
}

function visibleJson(message) {
	if (!Array.isArray(message?.content)) return undefined;
	if (message.content.some((block) => block?.type === "toolCall")) return undefined;
	const blocks = message.content.filter((block) => block?.type === "text");
	if (blocks.length === 0 || blocks.some((block) => typeof block.text !== "string")) return undefined;
	return blocks.map((block) => block.text).join("");
}

function thinkingBlocks(message) {
	return Array.isArray(message?.content)
		? message.content.filter((block) => block?.type === "thinking" && typeof block.thinking === "string")
		: [];
}

function failedMessage(message, reason) {
	return {
		...message,
		content: [],
		stopReason: "error",
		rawStopReason: "json_outcome_protocol_violation",
		errorMessage: `${OUTCOME_ERROR_PREFIX}: ${reason}`,
	};
}

function inertRequest(payload, ctx, reason) {
	try {
		ctx?.abort?.();
		ctx?.shutdown?.();
	} catch {
		// Pi currently swallows extension hook exceptions. Return an independently
		// inert request as well, so malformed protocol setup cannot reach the model.
	}
	return {
		model: payload.model,
		messages: [{ role: "user", content: "." }],
		max_tokens: 1,
		stream: false,
		temperature: 0,
		top_p: 1,
		_qwen_json_outcome_error: reason,
	};
}

function callId(message, raw) {
	return `qwen_outcome_${createHash("sha256")
		.update(String(message.responseId ?? ""))
		.update("\0")
		.update(raw)
		.digest("hex")
		.slice(0, 24)}`;
}

export default function qwenJsonOutcomeRouter(pi) {
	let active;

	pi.on("before_provider_request", (event, ctx) => {
		if (!isTargetContext(ctx) || !isTargetPayload(event?.payload)) {
			active = undefined;
			return undefined;
		}
		const payload = event.payload;
		if (
			payload.response_format !== undefined ||
			payload.structured_outputs !== undefined ||
			payload.structured_output !== undefined ||
			(payload.tool_choice !== undefined && payload.tool_choice !== "auto" && payload.tool_choice !== "none")
		) {
			active = undefined;
			return inertRequest(payload, ctx, "provider request already contains conflicting outcome controls");
		}
		try {
			const { responseFormat, schemas } = outcomeResponseFormat(payload.tools);
			active = { model: payload.model, schemas };
			return {
				...payload,
				response_format: responseFormat,
				tool_choice: "none",
			};
		} catch (error) {
			active = undefined;
			return inertRequest(
				payload,
				ctx,
				`cannot construct the strict outcome schema: ${error instanceof Error ? error.message : String(error)}`,
			);
		}
	});

	pi.on("message_end", (event) => {
		const message = event?.message;
		const request = active;
		active = undefined;
		if (request === undefined || !isTargetMessage(message, request.model)) return undefined;
		if (message.stopReason === "error" || message.stopReason === "aborted") {
			return { message: { ...message, content: [] } };
		}
		if (message.stopReason !== "stop") {
			return { message: failedMessage(message, `provider ended with ${String(message.stopReason)}`) };
		}
		const raw = visibleJson(message);
		if (raw === undefined) return { message: failedMessage(message, "provider did not emit one JSON outcome") };
		let outcome;
		try {
			outcome = JSON.parse(raw);
		} catch {
			return { message: failedMessage(message, "provider emitted incomplete JSON") };
		}
		if (!isRecord(outcome)) return { message: failedMessage(message, "outcome is not an object") };
		const keys = Object.keys(outcome).sort();
		if (outcome.kind === "final") {
			if (
				JSON.stringify(keys) !== JSON.stringify(["answer", "kind"]) ||
				typeof outcome.answer !== "string" ||
				outcome.answer.trim().length === 0
			) {
				return { message: failedMessage(message, "final outcome does not match its schema") };
			}
			return {
				message: {
					...message,
					content: [...thinkingBlocks(message), { type: "text", text: outcome.answer }],
					stopReason: "stop",
					rawStopReason: OUTCOME_FINAL_STOP_REASON,
					endTurn: true,
					errorMessage: undefined,
					deferred: undefined,
				},
			};
		}
		if (
			outcome.kind !== "tool" ||
			JSON.stringify(keys) !== JSON.stringify(["arguments", "kind", "name"]) ||
			typeof outcome.name !== "string" ||
			!isRecord(outcome.arguments)
		) {
			return { message: failedMessage(message, "tool outcome does not match its envelope schema") };
		}
		const schema = request.schemas.get(outcome.name);
		if (schema === undefined) return { message: failedMessage(message, `tool ${outcome.name} was not advertised`) };
		const error = schemaError(outcome.arguments, schema);
		if (error !== undefined) return { message: failedMessage(message, error) };
		return {
			message: {
				...message,
				content: [
					...thinkingBlocks(message),
					{
						type: "toolCall",
						id: callId(message, raw),
						name: outcome.name,
						arguments: outcome.arguments,
					},
				],
				stopReason: "toolUse",
				rawStopReason: OUTCOME_TOOL_STOP_REASON,
				endTurn: false,
				errorMessage: undefined,
				deferred: undefined,
			},
		};
	});

	for (const event of ["agent_settled", "session_shutdown"]) {
		pi.on(event, () => {
			active = undefined;
		});
	}
}
