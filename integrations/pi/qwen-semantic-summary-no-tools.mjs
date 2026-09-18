const TARGET_PROVIDER = "qwen-r9700";
const TARGET_API = "openai-completions";
const TARGET_MODEL = "qwen3.8-27b-frozenlock";
const MAX_TOKENS_ENV = "QWEN_PI_SEMANTIC_SUMMARY_MAX_TOKENS";
const THINKING_BUDGET_ENV = "QWEN_PI_SEMANTIC_SUMMARY_THINKING_TOKEN_BUDGET";
const PROMPT_THINKING_ENV = "QWEN_PI_SEMANTIC_SUMMARY_PROMPT_THINKING";
const TOOL_NAMES_ENV = "QWEN_PI_SEMANTIC_SUMMARY_TOOL_NAMES";

function loadMaxTokens() {
	const value = Number(process.env[MAX_TOKENS_ENV]);
	if (!Number.isSafeInteger(value) || value < 500 || value > 8000) {
		throw new Error(`${MAX_TOKENS_ENV} is missing or outside 500..8000`);
	}
	return value;
}

function loadThinkingContract(maxTokens) {
	const budget = Number(process.env[THINKING_BUDGET_ENV]);
	if (!Number.isSafeInteger(budget) || budget < 0 || budget > maxTokens - 1) {
		throw new Error(`${THINKING_BUDGET_ENV} is missing or outside 0..max_tokens-1`);
	}
	const promptThinking = process.env[PROMPT_THINKING_ENV];
	if (!new Set(["off", "minimal", "low", "medium", "high", "xhigh", "max"]).has(promptThinking)) {
		throw new Error(`${PROMPT_THINKING_ENV} is missing or invalid`);
	}
	return { budget, promptThinking };
}

function loadToolNames() {
	let names;
	try {
		names = JSON.parse(process.env[TOOL_NAMES_ENV]);
	} catch {
		throw new Error(`${TOOL_NAMES_ENV} is missing or invalid JSON`);
	}
	if (
		!Array.isArray(names) ||
		names.length === 0 ||
		names.some((name) => typeof name !== "string" || name.length === 0) ||
		new Set(names).size !== names.length
	) {
		throw new Error(`${TOOL_NAMES_ENV} is not a unique non-empty tool-name list`);
	}
	return names;
}

function isRecord(value) {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isTarget(ctx, payload) {
	return (
		ctx?.model?.provider === TARGET_PROVIDER &&
		ctx.model.api === TARGET_API &&
		isRecord(payload) &&
		payload.model === TARGET_MODEL
	);
}

export default function qwenSemanticSummaryNoTools(pi) {
	const maxTokens = loadMaxTokens();
	const { budget, promptThinking } = loadThinkingContract(maxTokens);
	const expectedToolNames = loadToolNames();
	pi.on("before_provider_request", (event, ctx) => {
		const payload = event?.payload;
		if (!isTarget(ctx, payload)) return undefined;
		const kwargs = payload.chat_template_kwargs;
		if (
			!isRecord(kwargs) ||
			kwargs.enable_thinking !== (promptThinking !== "off") ||
			kwargs.preserve_thinking !== true ||
			kwargs.reasoning_effort !== promptThinking ||
			Object.keys(kwargs).length !== 3
		) {
			throw new Error("semantic-summary request changed its tokenized reasoning prompt ABI");
		}
		if (!Array.isArray(payload.tools)) {
			throw new Error("semantic-summary request omitted its snapshot-bound tool schema");
		}
		const observedToolNames = payload.tools.map((tool) => tool?.function?.name);
		if (JSON.stringify(observedToolNames) !== JSON.stringify(expectedToolNames)) {
			throw new Error("semantic-summary request changed its snapshot-bound tool names or order");
		}
		if (payload.tool_choice !== undefined && payload.tool_choice !== "none") {
			throw new Error("semantic-summary request already has a conflicting tool_choice");
		}
		if (payload.max_tokens !== undefined && !Number.isSafeInteger(payload.max_tokens)) {
			throw new Error("semantic-summary request has an invalid max_tokens value");
		}
		if (payload.stop !== undefined) {
			throw new Error("semantic-summary request already has an unexpected stop sequence");
		}
		if (payload.return_token_ids !== undefined && payload.return_token_ids !== true) {
			throw new Error("semantic-summary request disables its exact token-ID stream");
		}
		if (payload.include_reasoning !== undefined && payload.include_reasoning !== true) {
			throw new Error("semantic-summary request hides reasoning token IDs");
		}
		return {
			...payload,
			include_reasoning: true,
			max_tokens: maxTokens,
			return_token_ids: true,
			thinking_token_budget: budget,
			tool_choice: "none",
		};
	});
}
