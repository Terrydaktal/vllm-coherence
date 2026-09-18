const STATUS_KEY = "qwen-repetition-guard";
const TARGET_API = "openai-completions";
const TARGET_MODELS = new Set([
	"qwen3.8-27b-frozenlock",
	"qwen3.8-27b-hauhau-aggressive",
	"qwen3.8-27b-philbert-aggressive",
	"qwen3.8-27b-hauhau-delta-aggressive",
]);
const TARGET_PROVIDER = "qwen-r9700";
const EPHEMERAL_RECOVERY_REQUEST = Symbol.for("qwen-r9700:ephemeral-recovery-request:v1");
const INTERNAL_GUARD_RETRY = Symbol.for("qwen-r9700:internal-guard-retry:v1");

export const REPETITION_DETECTION = Object.freeze({
	max_pattern_size: 8,
	min_pattern_size: 1,
	min_count: 7,
});

const STOCHASTIC_FALLBACK = Object.freeze({
	temperature: 1,
	top_p: 0.95,
	top_k: 20,
});

const RETRYABLE_REPETITION_ERROR =
	"Qwen repetition guard server error: a periodic suffix was blocked before release; retrying once with stock stochastic sampling";
const TERMINAL_REPETITION_ERROR =
	"Qwen repetition guard stopped the response because repetition recurred during stochastic recovery";
const RETRYABLE_PREMATURE_ACTION_ERROR =
	"Qwen premature-action guard server error: the response stopped after announcing a tool action without emitting it; retrying once with corrective stochastic steering";
const TERMINAL_PREMATURE_ACTION_ERROR =
	"Qwen premature-action guard stopped the response because an announced tool action was omitted again during stochastic recovery";
const PREMATURE_ACTION_RECOVERY_INSTRUCTION =
	"The preceding assistant attempt was rejected because it ended after announcing an action without emitting the structured tool call. Continue the original task now. Do not narrate, promise, or announce the next action again. If an external action is required, emit the actual structured tool call in this response before stopping. If no external action is required, provide the completed result directly.";
const ANNOUNCED_ACTION =
	/(?:^|[.!?]\s+|\n)\s*(?:(?:(?:now|next)[,:]?\s+)(?:(?:i(?:['’]ll| will| am going to)|let me|let(?:['’]s| us))\s+(?:(?:first|now|also|then)\s+)?)?|(?:i(?:['’]ll| will| am going to)|let me|let(?:['’]s| us))\s+(?:(?:first|now|also|then)\s+)?)(?:creat(?:e|ing)|writ(?:e|ing)|vendor(?:ing)?|edit(?:ing)?|modif(?:y|ying)|patch(?:ing)?|implement(?:ing)?|appl(?:y|ying)|run(?:ning)?|execut(?:e|ing)|(?:re-?)?test(?:ing)?|check(?:ing)?|inspect(?:ing)?|read(?:ing)?|open(?:ing)?|search(?:ing)?|look(?:ing)?\s+up|find(?:ing)?|fix(?:ing)?|updat(?:e|ing)|add(?:ing)?|remov(?:e|ing)|replac(?:e|ing)|try(?:ing)?|construct(?:ing)?|build(?:ing)?|generat(?:e|ing)|compil(?:e|ing)|validat(?:e|ing)|verif(?:y|ying)|install(?:ing)?|load(?:ing)?|fetch(?:ing)?|quer(?:y|ying)|measur(?:e|ing)|benchmark(?:ing)?|debug(?:ging)?|diagnos(?:e|ing)|investigat(?:e|ing)|review(?:ing)?|confirm(?:ing)?|prob(?:e|ing)|pull(?:ing)?)\b[^\n]{0,600}[.:]\s*$/i;

function isRecord(value) {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isTargetContext(ctx) {
	return ctx?.model?.provider === TARGET_PROVIDER && ctx.model.api === TARGET_API;
}

function isTargetPayload(payload) {
	return isRecord(payload) && TARGET_MODELS.has(payload.model) && payload.stream === true;
}

function isTargetMessage(message, requestModel) {
	return (
		message?.role === "assistant" &&
		message.provider === TARGET_PROVIDER &&
		message.api === TARGET_API &&
		TARGET_MODELS.has(message.model) &&
		message.model === requestModel
	);
}

function isGreedy(payload) {
	return payload.temperature === 0 && payload.top_p === 1 && payload.top_k === 1;
}

function isStochastic(payload) {
	return (
		payload.temperature === STOCHASTIC_FALLBACK.temperature &&
		payload.top_p === STOCHASTIC_FALLBACK.top_p &&
		payload.top_k === STOCHASTIC_FALLBACK.top_k
	);
}

function isRepetitionStop(message) {
	return (
		message?.role === "assistant" &&
		message.stopReason === "error" &&
		(message.rawStopReason === "repetition" ||
			message.errorMessage === "Provider finish_reason: repetition")
	);
}

function isPrematureActionStop(message) {
	if (message?.rawStopReason === "qwen_json_outcome_final") return false;
	// Keep this guard active for every prompt ABI. The outcome ABI prefers a
	// reserved structured final-answer tool, but the live serving path can still
	// return an ordinary text stop. This handler runs before the outcome handler,
	// so an announced-but-unperformed action is rejected while a substantive
	// completed text answer remains valid and visible.
	if (message?.role !== "assistant" || message.stopReason !== "stop" || !Array.isArray(message.content)) {
		return false;
	}
	if (message.content.some((item) => item?.type === "toolCall")) return false;
	const text = message.content
		.flatMap((item) => {
			if (item?.type === "text" && typeof item.text === "string") return [item.text];
			if (item?.type === "thinking" && typeof item.thinking === "string") return [item.thinking];
			return [];
		})
		.join("\n")
		.trim();
	return text.length > 0 && ANNOUNCED_ACTION.test(text.slice(-800));
}

function visibleAssistantText(message) {
	if (!Array.isArray(message?.content)) return "";
	return message.content
		.filter((item) => item?.type === "text" && typeof item.text === "string")
		.map((item) => item.text)
		.join("\n")
		.trim()
		.slice(-800);
}

function guardedPayload(payload, useFallback, recoveryKind, recoveryActionText) {
	const messages =
		useFallback && recoveryKind === "premature-action" && Array.isArray(payload.messages)
			? [
					...payload.messages,
					...(recoveryActionText.length > 0
						? [{ role: "assistant", content: recoveryActionText }]
						: []),
					{ role: "user", content: PREMATURE_ACTION_RECOVERY_INSTRUCTION },
				]
			: payload.messages;
	return {
		...payload,
		...(useFallback ? STOCHASTIC_FALLBACK : {}),
		...(messages === undefined ? {} : { messages }),
		repetition_detection: { ...REPETITION_DETECTION },
		...(useFallback ? { [EPHEMERAL_RECOVERY_REQUEST]: true } : {}),
	};
}

function rejectedMessage(message, retryable, kind) {
	const repetition = kind === "repetition";
	return {
		...message,
		content: [],
		stopReason: "error",
		rawStopReason: repetition ? "repetition" : "premature_action_stop",
		...(retryable ? { [INTERNAL_GUARD_RETRY]: true } : {}),
		errorMessage: retryable
			? repetition
				? RETRYABLE_REPETITION_ERROR
				: RETRYABLE_PREMATURE_ACTION_ERROR
			: repetition
				? TERMINAL_REPETITION_ERROR
				: TERMINAL_PREMATURE_ACTION_ERROR,
	};
}

export default function qwenRepetitionGuard(pi) {
	let recoveryState = "none";
	let recoveryKind = "none";
	let recoveryActionText = "";
	let requestMode = "none";
	let requestModel;

	pi.on("before_provider_request", (event, ctx) => {
		if (!isTargetContext(ctx) || !isTargetPayload(event?.payload)) return undefined;
		requestModel = event.payload.model;

		const useFallback = recoveryState === "pending" || recoveryState === "active";
		if (useFallback) {
			recoveryState = "active";
			requestMode = "fallback";
			if (ctx.mode === "tui") {
				ctx.ui.setStatus?.(
					STATUS_KEY,
						recoveryKind === "premature-action"
							? "Qwen omitted an announced tool action; corrective recovery in progress"
						: "Qwen repetition blocked; stochastic recovery in progress",
				);
			}
		} else if (isGreedy(event.payload)) {
			requestMode = "greedy";
		} else if (isStochastic(event.payload)) {
			requestMode = "stochastic";
		} else {
			requestMode = "other";
		}

		return guardedPayload(event.payload, useFallback, recoveryKind, recoveryActionText);
	});

	pi.on("message_end", (event, ctx) => {
		const message = event?.message;
		// A response without a matching tracked request must be left untouched. This
		// prevents a newly added model from being classified as a failed *second*
		// attempt merely because its request model was absent from TARGET_MODELS.
		if (requestMode === "none" || !isTargetMessage(message, requestModel)) {
			return undefined;
		}

		const failureKind = isRepetitionStop(message)
			? "repetition"
			: isPrematureActionStop(message)
				? "premature-action"
				: undefined;
		if (failureKind !== undefined) {
			const retryable =
				(requestMode === "greedy" || requestMode === "stochastic") &&
				recoveryState === "none";
			recoveryState = retryable ? "pending" : "none";
			recoveryKind = retryable ? failureKind : "none";
			recoveryActionText =
				retryable && failureKind === "premature-action" ? visibleAssistantText(message) : "";
			requestMode = "none";
			if (ctx.mode === "tui") {
				ctx.ui.setStatus?.(
					STATUS_KEY,
						failureKind === "premature-action"
							? retryable
								? "Qwen omitted an announced tool action; retrying once with corrective steering"
							: "Qwen omitted an announced tool action again; stopped safely"
						: retryable
							? "Qwen repetition blocked; retrying once with stochastic sampling"
							: "Qwen repetition blocked again; stopped safely",
				);
			}
			return { message: rejectedMessage(message, retryable, failureKind) };
		}

		if (recoveryState === "active" && message.stopReason !== "error" && message.stopReason !== "aborted") {
			const completedKind = recoveryKind;
			recoveryState = "none";
			recoveryKind = "none";
			recoveryActionText = "";
			requestMode = "none";
			if (ctx.mode === "tui") {
				ctx.ui.setStatus?.(
					STATUS_KEY,
					completedKind === "premature-action"
						? "Qwen announced-action recovery succeeded"
						: "Qwen repetition recovery succeeded",
				);
			}
		}
		if (recoveryState !== "active") {
			requestMode = "none";
			requestModel = undefined;
		}
		return undefined;
	});

	pi.on("agent_settled", () => {
		// A pending recovery can remain only when Pi auto-retry was disabled or a
		// non-retryable transport failure ended the run. Never carry it into the
		// next user turn.
		recoveryState = "none";
		recoveryKind = "none";
		recoveryActionText = "";
		requestMode = "none";
		requestModel = undefined;
	});
}
