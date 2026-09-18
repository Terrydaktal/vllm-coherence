import { createSchedulerTelemetry, toolGraceDescription, requestPhaseStatus, REQUEST_PHASE_LABELS } from "./qwen-radiance-scheduler-telemetry.mjs";

const TICK_MS = 1000;
const MIN_LIVE_RATE_SECONDS = 0.5;
const OUTPUT_IDLE_MS = 2000;
const ROLLING_RATE_MS = 3000;
const STATUS_KEY = "qwen-output-rate";
const TARGET_API = "openai-completions";
const TARGET_PROVIDER = "qwen-r9700";

function formatContextTokens(tokens) {
	if (!Number.isFinite(tokens) || tokens < 0) return undefined;
	if (tokens < 1000) return `${Math.round(tokens)}`;
	if (tokens < 1_000_000) return `${(tokens / 1000).toFixed(1)}K`;
	return `${(tokens / 1_000_000).toFixed(1)}M`;
}

function formatFirstData(milliseconds) {
	const seconds = milliseconds / 1000;
	return seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`;
}

function monotonicNow() {
	return typeof globalThis.performance?.now === "function" ? globalThis.performance.now() : Date.now();
}

function enableContinuousUsage(event, ctx) {
	if (ctx?.model?.provider !== TARGET_PROVIDER || ctx.model.api !== TARGET_API) return undefined;

	const payload = event?.payload;
	if (payload === null || typeof payload !== "object" || Array.isArray(payload)) return undefined;
	if (payload.stream !== true) return undefined;

	const current = payload.stream_options;
	if (current !== undefined && (current === null || typeof current !== "object" || Array.isArray(current))) {
		return undefined;
	}

	return {
		...payload,
		stream_options: {
			...(current ?? {}),
			include_usage: true,
			continuous_usage_stats: true,
		},
	};
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

function reportedReasoningTokens(update) {
	const message =
		update?.type === "done"
			? update.message
			: update?.type === "error"
				? update.error
				: update?.partial;
	const reasoning = message?.usage?.reasoning;
	return Number.isSafeInteger(reasoning) && reasoning >= 0 ? reasoning : undefined;
}

export default function qwenProgress(pi, { scheduler = createSchedulerTelemetry() } = {}) {
	let timer;
	let startedAt = 0;
	let firstDataAt = 0;
	let contextTokens;
	let phase = "preparing request";
	let exactOutputTokens;
	let exactReasoningTokens;
	let firstDataOutputTokens;
	let finishedAt = 0;
	let lastOutputAt = 0;
	let toolsFinishedAt = 0;
	let argumentsStartedAt = 0;
	let generatingToolName;
	const runningTools = new Map();
	const rateSamples = [];
	let activeMode;
	let activeUi;
	let providerHeadersReceived = false;
	let schedulerSignature;
	let schedulerStateSince = 0;
	let schedulerComputedTokens;
	let schedulerProgressAt = 0;
	let schedulerProgressObserved = false;
	let schedulerRatePauseAt = 0;
	let excludedSchedulerWaitMs = 0;
	let firstDataRateAt = 0;
	let lastOutputRateAt = 0;
	let requestTiming;
	let lastTiming;
	let nextRequestFromToolsAt;
	const unsubscribe = scheduler.subscribe?.(() => render());

	pi.registerCommand?.("qwen-timing", {
		description: "Show numeric phase timings for the current or last model response",
		handler: async (_args, ctx) => {
			const report = requestTiming ?? lastTiming;
			if (!report) { ctx.ui.notify("No response timing has been observed in this Pi window yet.", "info"); return; }
			const ms = (value) => value === undefined ? "not observed" : `${value.toFixed(1)} ms`;
			const lines = [
				`Tool completion → request hook: ${ms(report.toolGapMs)}`,
				`Request hook → HTTP response headers: ${ms(report.headersMs)}`,
				`Request hook → first output: ${ms(report.firstDataMs)}`,
			];
			if (report.backend) {
				for (const [phase, elapsed] of Object.entries(report.backend.timings_ms)) {
					if (phase !== "complete") lines.push(`${REQUEST_PHASE_LABELS[phase]}: ${ms(elapsed)}`);
				}
				if (report.backend.first_token_ms !== null) {
					lines.push(`Backend admission → first token: ${ms(report.backend.first_token_ms)}`);
					if (report.firstDataMs !== undefined) lines.push(`Preparation, network and stream delivery combined: ${ms(Math.max(0, report.firstDataMs - report.backend.first_token_ms))}`);
				}
			} else lines.push("Detailed backend timings: not observed");
			lines.push("These are elapsed times, not GPU kernel timings. Escape closes this view.");
			if (ctx.ui.select) await ctx.ui.select("Qwen response timing", lines);
			else ctx.ui.notify(lines.join("\n"), "info");
		},
	});

	function resetSchedulerObservation() {
		schedulerSignature = undefined;
		schedulerStateSince = 0;
		schedulerComputedTokens = undefined;
		schedulerProgressAt = 0;
		schedulerProgressObserved = false;
	}

	function schedulerObservation(now) {
		let observation;
		try {
			observation = scheduler.read?.(Date.now()) ?? { available: false };
		} catch {
			observation = { available: false };
		}
			if (!observation.available) return observation;
			if (requestTiming && observation.requestPhase?.request_id === requestTiming.ignoreRequestId) {
				observation = { ...observation, requestPhase: undefined };
			}
			const timing = observation.requestPhase ?? observation.lastRequestTiming;
			if (requestTiming && timing && timing.request_id !== requestTiming.ignoreRequestId) requestTiming.backend = timing;
		const request = observation.request;
		const signature = request
			? `${request.state}:${observation.otherRunningChatId ?? ""}`
			: `absent:${observation.otherRunningChatId ?? ""}`;
		if (signature !== schedulerSignature) {
			schedulerSignature = signature;
			schedulerStateSince = now;
			schedulerComputedTokens = undefined;
			schedulerProgressAt = now;
			schedulerProgressObserved = false;
		}
		if (request !== undefined) {
			if (
				schedulerComputedTokens !== undefined &&
				request.computed_tokens > schedulerComputedTokens
			) {
				schedulerProgressAt = now;
				schedulerProgressObserved = true;
			}
			if (
				schedulerComputedTokens === undefined ||
				request.computed_tokens >= schedulerComputedTokens
			) {
				schedulerComputedTokens = request.computed_tokens;
			}
		}
		return {
			...observation,
			stateSeconds: Math.max(0, Math.floor((now - schedulerStateSince) / 1000)),
			progressSeconds: Math.max(0, Math.floor((now - schedulerProgressAt) / 1000)),
			progressObserved: schedulerProgressObserved,
		};
	}

	function otherChat(observation) {
		return observation.otherRunningChatId
			? `another chat ${observation.otherRunningChatId.slice(0, 12)}`
			: "another chat";
	}

	function rateClock(now) {
		const currentWait = schedulerRatePauseAt === 0 ? 0 : Math.max(0, now - schedulerRatePauseAt);
		return now - excludedSchedulerWaitMs - currentWait;
	}

	function resumeRateClock(now) {
		if (schedulerRatePauseAt === 0) return;
		excludedSchedulerWaitMs += Math.max(0, now - schedulerRatePauseAt);
		schedulerRatePauseAt = 0;
	}

	function syncRateClock(observation, now) {
		if (firstDataAt === 0 || !observation.available) return;
		const request = observation.request;
		const physicalOtherChat = observation.workerAvailable === true &&
			observation.activeChat != null && observation.cacheResidency !== "gpu";
		const waitingForOtherChat = physicalOtherChat || observation.priorityHold !== undefined || (
			observation.otherRunningChatId !== undefined && request?.state !== "running"
		);
		if (waitingForOtherChat) {
			schedulerRatePauseAt ||= now;
			return;
		}
		resumeRateClock(now);
	}

	function promptProgress(observation) {
		const request = observation.request;
		if (request.input_tokens <= 0 || request.computed_tokens >= request.input_tokens) return "";
		if (request.computed_tokens === 0) {
			if (observation.cacheResidency === "ram") {
				return " · KV cache parked in RAM; restore follows GPU handover";
			}
			return " · cache lookup/reuse pending after GPU admission";
		}
		const position = `${request.computed_tokens.toLocaleString("en-US")} / ` +
			`${request.input_tokens.toLocaleString("en-US")} tok`;
		return observation.cacheResidency === "ram"
			? ` · KV state through ${position} parked in RAM`
			: ` · prompt/KV state through ${position} preserved`;
	}

	function blockedStatus(observation, beforeFirstToken) {
				const request = observation.request;
				if (observation.requestPhase) {
						return ["gpu_queue", "priority_wait", "priority_preempt", "tool_grace", "handover", "ram_allocation"].includes(observation.requestPhase.phase)
						? requestPhaseStatus(observation) : undefined;
				}
				if (observation.priorityHold && request?.state !== "running") {
					return { phase: "waiting for higher-priority chat", detail: `chat ${observation.priorityHold.chat_id.slice(0, 12)} keeps the GPU through tools until its answer finishes` };
				}
				if (observation.toolGrace && request?.state !== "running") {
				return { phase: "queued for GPU", detail: toolGraceDescription(observation.toolGrace) };
			}
		if (request?.state === "paused") {
				return {
					phase: beforeFirstToken ? "request paused" : "generation paused",
					detail: `${otherChat(observation)} has the GPU · paused ${observation.stateSeconds}s` +
						promptProgress(observation),
			};
		}
			if (request?.state === "queued") {
			const boundary = observation.policy === "response_boundary"
				? "; waiting for its response to finish"
				: "";
				return observation.otherRunningChatId ? {
					phase: "queued for GPU",
					detail: `${otherChat(observation)} has the GPU${boundary} · queued ${observation.stateSeconds}s` + promptProgress(observation),
				} : { phase: "preparing next response", detail: `cache preparation stage not reported by this backend · ${observation.stateSeconds}s` };
		}
		return undefined;
	}

	function firstTokenStatus(observation) {
			const exact = requestPhaseStatus(observation);
			if (exact) return `${exact.phase} · ${exact.detail}`;
		const blocked = blockedStatus(observation, true);
		if (blocked) return `${blocked.phase} · ${blocked.detail}`;
		if (!observation.available) {
			return providerHeadersReceived
				? observation.configured === false
					? "backend response open · waiting for provider stream"
					: "scheduler telemetry unavailable · backend wait unknown"
				: undefined;
		}
		const request = observation.request;
		if (request?.state === "running") {
			if (request.computed_tokens < request.input_tokens) {
				if (request.computed_tokens === 0) {
					return "GPU active · checking/restoring reusable KV cache; prompt reuse pending";
				}
				return `GPU active · prompt/KV state (restored or prefilled) ` +
					`${request.computed_tokens.toLocaleString("en-US")} / ` +
					`${request.input_tokens.toLocaleString("en-US")} tok`;
			}
			return "GPU active · prompt ready; generating first token";
		}
		if (observation.otherRunningChatId) {
			return `scheduler admission pending · ${otherChat(observation)} has the GPU` +
				(observation.policy === "response_boundary" ? "; waiting for its response to finish" : "");
		}
		return providerHeadersReceived ? "backend response open · scheduler admission pending" : undefined;
	}

	function idleStatus(observation, idleSeconds) {
			const exact = requestPhaseStatus(observation);
			if (exact && observation.requestPhase.phase !== "generate") return exact;
		const blocked = blockedStatus(observation, false);
		if (blocked) return blocked;
		const subject = argumentsStartedAt !== 0
			? `${generatingToolName ?? "tool"} arguments`
			: "model output";
		if (!observation.available) {
			if (observation.configured !== false) {
				return {
					phase: "scheduler telemetry unavailable",
					detail: `backend wait unknown while awaiting ${subject} · ` +
						`client stream unchanged ${idleSeconds}s`,
				};
			}
			return {
				phase: `waiting for ${subject}`,
				detail: `provider stream unchanged ${idleSeconds}s · scheduler telemetry not configured`,
			};
		}
		const request = observation.request;
		if (request?.state === "running") {
			if (request.computed_tokens < request.input_tokens) {
				return {
					phase: "GPU active",
					detail: request.computed_tokens === 0
						? "checking/restoring reusable KV cache; prompt reuse pending"
						: `prompt/KV state (restored or prefilled) ` +
							`${request.computed_tokens.toLocaleString("en-US")} / ` +
							`${request.input_tokens.toLocaleString("en-US")} tok`,
				};
			}
			if (observation.progressObserved && observation.progressSeconds <= 1) {
				return {
					phase: "GPU active",
					detail: `backend token count advancing · next ${subject} stream update pending ` +
						`(${idleSeconds}s since client update)`,
				};
			}
			return {
				phase: "GPU active",
				detail: `backend token count observed unchanged ${observation.progressSeconds}s · ` +
					`client stream unchanged ${idleSeconds}s`,
			};
		}
		return {
			phase: `waiting for ${subject}`,
			detail: observation.otherRunningChatId
				? `this request is no longer scheduled · ${otherChat(observation)} has the GPU · ` +
					`client completion pending ${idleSeconds}s`
				: `this request is no longer scheduled · client completion pending ${idleSeconds}s`,
		};
	}

	function postFirstRate(minimumSeconds = MIN_LIVE_RATE_SECONDS) {
		if (
			firstDataAt === 0 ||
			exactOutputTokens === undefined ||
			firstDataOutputTokens === undefined ||
			exactOutputTokens <= firstDataOutputTokens ||
			lastOutputRateAt === 0
		) {
			return undefined;
		}
		const elapsedSeconds = (lastOutputRateAt - firstDataRateAt) / 1000;
		if (elapsedSeconds <= 0 || elapsedSeconds < minimumSeconds) return undefined;
		return (exactOutputTokens - firstDataOutputTokens) / elapsedSeconds;
	}

	function trimRateSamples(cutoff) {
		// Retain one count at/before the window boundary for interpolation.
		let discard = 0;
		while (discard + 1 < rateSamples.length && rateSamples[discard + 1].at <= cutoff) discard++;
		if (discard) rateSamples.splice(0, discard);
	}

	function recordRateSample(at, tokens) {
		const previous = rateSamples.at(-1);
		if (previous?.at === at) previous.tokens = tokens;
		else rateSamples.push({ at, tokens });
		trimRateSamples(at - ROLLING_RATE_MS);
	}

	function rollingRate(now) {
		if (firstDataAt === 0 || exactOutputTokens === undefined || rateSamples.length === 0) return undefined;
		const activeNow = rateClock(now);
		const start = Math.max(firstDataRateAt, activeNow - ROLLING_RATE_MS);
		const seconds = (activeNow - start) / 1000;
		if (seconds < MIN_LIVE_RATE_SECONDS) return undefined;
		trimRateSamples(start);
		const [before, after] = rateSamples;
		let tokensAtStart = before.tokens;
		if (after && before.at < start) {
			// Sparse counters only locate output between two reports. Interpolate
			// the boundary instead of assigning a whole delayed batch to this window.
			tokensAtStart += (after.tokens - before.tokens) * (start - before.at) / (after.at - before.at);
		}
		return Math.max(0, exactOutputTokens - tokensAtStart) / seconds;
	}

	function observeOutput(outputTokens, reasoningTokens, hasDelta = false, now = monotonicNow()) {
		const advanced = outputTokens !== undefined && outputTokens > (exactOutputTokens ?? 0);
		const reasoningAdvanced = reasoningTokens !== undefined && reasoningTokens > (exactReasoningTokens ?? 0);
		if (outputTokens !== undefined && (exactOutputTokens === undefined || outputTokens >= exactOutputTokens)) {
			exactOutputTokens = outputTokens;
		}
		if (reasoningTokens !== undefined && (exactReasoningTokens === undefined || reasoningTokens >= exactReasoningTokens)) {
			exactReasoningTokens = reasoningTokens;
		}
		// A late reasoning breakdown of an unchanged total is accounting, not
		// evidence of more output. Use it only while total usage is unavailable.
		const active = advanced || hasDelta || (exactOutputTokens === undefined && reasoningAdvanced);
		if (active) {
			// Streamed output proves that this chat is running even if the shared
			// scheduler sample has not yet published the hand-back transition.
			resumeRateClock(now);
			lastOutputAt = now;
			if (firstDataAt !== 0) lastOutputRateAt = rateClock(now);
		}
		if (advanced && firstDataAt !== 0) recordRateSample(rateClock(now), exactOutputTokens);
		return active;
	}

	function stopTimer() {
		if (timer !== undefined) {
			clearInterval(timer);
			timer = undefined;
		}
	}

	function render() {
		if (activeMode !== "tui" || startedAt === 0) return;

		const now = monotonicNow();
		const elapsedSeconds = Math.max(0, Math.floor((now - startedAt) / 1000));
		if (runningTools.size > 0) {
			const tools = [...runningTools.values()];
			const descriptions = tools.slice(0, 3).map((tool) =>
				`${tool.name === "edit" ? "applying edit" : `running ${tool.name}`} ` +
				`${Math.max(0, Math.floor((now - tool.startedAt) / 1000))}s`);
			if (tools.length > 3) descriptions.push(`+${tools.length - 3} more`);
			activeUi.setWorkingMessage(`Qwen ${descriptions.join(", ")} \u2022 model output ended`);
			return;
		}
		if (finishedAt !== 0) {
			const waitingSince = toolsFinishedAt || finishedAt;
			activeUi.setWorkingMessage(`Qwen ${toolsFinishedAt ? "tools finished" : "model response ended"} ` +
				`\u2022 waiting for next step ${Math.max(0, Math.floor((now - waitingSince) / 1000))}s`);
			return;
		}
		const schedulerState = schedulerObservation(now);
		syncRateClock(schedulerState, now);
		if (firstDataAt !== 0) {
			const firstData = formatFirstData(firstDataAt - startedAt);
			const idle = now - lastOutputAt >= OUTPUT_IDLE_MS;
			const rate = postFirstRate();
			const recentRate = rollingRate(now);
			const reasoning =
				exactReasoningTokens > 0
					? ` (${exactReasoningTokens.toLocaleString("en-US")} reasoning)`
					: "";
			const output =
				exactOutputTokens === undefined
					? "exact tokens pending"
					: `${exactOutputTokens.toLocaleString("en-US")} tok${reasoning}`;
			const blocked = idle || schedulerStateSince > lastOutputAt ? blockedStatus(schedulerState, false) : undefined;
			const idleSeconds = Math.floor((now - lastOutputAt) / 1000);
			const waiting = blocked ?? (idle ? idleStatus(schedulerState, idleSeconds) : undefined);
			const rateText = waiting?.detail ??
				(rate === undefined || recentRate === undefined
					? "measuring t/s"
					: `${recentRate.toFixed(1)} t/s, ${rate.toFixed(1)} t/s avg`);
			activeUi.setWorkingMessage(
				`Qwen ${waiting?.phase ?? phase}: ${output} \u2022 ${rateText} ` +
					`\u2022 first data ${firstData} \u2022 ${elapsedSeconds}s`,
			);
			return;
		}

		const formattedContext = formatContextTokens(contextTokens);
		const context = formattedContext === undefined ? "" : ` (~${formattedContext} ctx)`;
		const schedulerPhase = firstTokenStatus(schedulerState);
		activeUi.setWorkingMessage(`Qwen ${schedulerPhase ?? phase}${context} \u2022 ${elapsedSeconds}s`);
	}

	function begin(ctx) {
		stopTimer();
		runningTools.clear();
		rateSamples.length = 0;
		schedulerRatePauseAt = 0;
		excludedSchedulerWaitMs = 0;
		firstDataRateAt = 0;
		lastOutputRateAt = 0;
		const mode = ctx.mode;
		if (mode !== "tui") {
			startedAt = 0;
			activeMode = mode;
			activeUi = undefined;
			return;
		}
		// Event contexts become deliberately unusable after a Pi session replacement
		// or shutdown. Retain only the already-resolved UI object and scalar mode; a
		// timer or late lifecycle event must never dereference a stale ctx proxy.
		activeMode = mode;
		activeUi = ctx.ui;
		scheduler.bind?.(ctx);
		startedAt = monotonicNow();
		firstDataAt = 0;
		exactOutputTokens = undefined;
		exactReasoningTokens = undefined;
		firstDataOutputTokens = undefined;
		finishedAt = 0;
		lastOutputAt = 0;
		providerHeadersReceived = false;
		resetSchedulerObservation();
		toolsFinishedAt = 0;
		argumentsStartedAt = 0;
		generatingToolName = undefined;
		phase = "preparing request";
		contextTokens = ctx.getContextUsage()?.tokens ?? undefined;
		activeUi.setStatus?.(STATUS_KEY, undefined);
		render();
		timer = setInterval(render, TICK_MS);
		timer.unref?.();
	}

	function finish() {
		stopTimer();
		runningTools.clear();
		rateSamples.length = 0;
		schedulerRatePauseAt = 0;
		excludedSchedulerWaitMs = 0;
		firstDataRateAt = 0;
		lastOutputRateAt = 0;
		scheduler.clear?.();
			resetSchedulerObservation();
			if (requestTiming) lastTiming = requestTiming;
			requestTiming = undefined;
		if (activeMode === "tui") activeUi?.setWorkingMessage();
		startedAt = 0;
		activeMode = undefined;
		activeUi = undefined;
	}

	// Clear the last-response footer left by an earlier extension version and
	// register this Pi window with the singleton scheduler telemetry stream.
	pi.on("session_start", (_event, ctx) => {
		ctx.ui?.setStatus?.(STATUS_KEY, undefined);
		scheduler.start?.(ctx);
	});
	pi.on("session_switch", (_event, ctx) => {
		ctx.ui?.setStatus?.(STATUS_KEY, undefined);
		scheduler.start?.(ctx);
	});
	pi.on("agent_start", (_event, ctx) => begin(ctx));

	pi.on("before_provider_request", (event, ctx) => {
		const replacementPayload = enableContinuousUsage(event, ctx);
		// One agent turn may contain several independent model requests separated by
		// tool execution.  Restart the rate/TTFT clock for every provider request so
		// tool runtime and the next request's KV restore/prefill are never reported as
		// decode time for the previous response.
			const ignoreRequestId = (requestTiming ?? lastTiming)?.backend?.request_id;
			if (requestTiming) lastTiming = requestTiming;
			requestTiming = undefined;
			begin(ctx);
			if (startedAt === 0) return replacementPayload;
			requestTiming = { ignoreRequestId, toolGapMs: nextRequestFromToolsAt === undefined ? undefined : startedAt - nextRequestFromToolsAt };
			nextRequestFromToolsAt = undefined;
		contextTokens = ctx.getContextUsage()?.tokens ?? contextTokens;
		phase = "KV lookup/prefill";
		render();
		return replacementPayload;
	});

	pi.on("after_provider_response", () => {
		if (startedAt === 0) return;
			providerHeadersReceived = true;
			if (requestTiming) requestTiming.headersMs = monotonicNow() - startedAt;
		phase = "backend response open; locating scheduler state";
		render();
	});

	pi.on("message_update", (event) => {
		if (startedAt === 0 || finishedAt !== 0) return;
		const update = event.assistantMessageEvent;
		const now = monotonicNow();
		syncRateClock(schedulerObservation(now), now);
		const outputTokens = reportedOutputTokens(update);
		const reasoningTokens = reportedReasoningTokens(update);
		const terminal = update.type === "done" || update.type === "error";
		const hasDelta = ["thinking_delta", "text_delta", "toolcall_delta"].includes(update.type) &&
			typeof update.delta === "string" && update.delta.length > 0;
		const advanced = observeOutput(outputTokens, reasoningTokens, hasDelta, now);
		// Usage can advance before the adapter releases buffered tool-call text.
		// Duplicate counters, empty deltas and lifecycle events are not new output.
			if (firstDataAt === 0 && advanced && !terminal) {
				firstDataAt = lastOutputAt;
				if (requestTiming) requestTiming.firstDataMs = firstDataAt - startedAt;
			firstDataOutputTokens = exactOutputTokens ?? 0;
			firstDataRateAt = rateClock(firstDataAt);
			lastOutputRateAt = firstDataRateAt;
			recordRateSample(firstDataRateAt, firstDataOutputTokens);
			if (argumentsStartedAt === 0) phase = "generating (buffered output)";
		}
		if (update.type === "toolcall_start" || update.type === "toolcall_delta") {
			// Read only the tool name at the event's index, never its arguments.
			if (update.type === "toolcall_start") {
				argumentsStartedAt = monotonicNow();
				generatingToolName = undefined;
			}
			const block = update.partial?.content?.[update.contentIndex];
			const name = block?.type === "toolCall" ? block.name : undefined;
			if (typeof name === "string" && /^[\w.-]{1,80}$/.test(name)) generatingToolName = name;
			argumentsStartedAt ||= monotonicNow();
			phase = `generating ${generatingToolName ?? "tool"} arguments`;
			render();
			return;
		}
		// Pi's patched OpenAI adapter forwards continuous usage chunks even when
		// the provider is buffering an incomplete structured tool call. They carry
		// no content delta, so retain the current phase and refresh only the exact
		// counter/rate.
		if (update.type === "usage_update") {
			render();
			return;
		}
		if (update.type === "done" || update.type === "error") {
			finishedAt = monotonicNow();
			render();
			return;
		}
		if (update.type !== "thinking_delta" && update.type !== "text_delta") return;
		if (hasDelta) {
			argumentsStartedAt = 0;
			generatingToolName = undefined;
			phase = update.type === "thinking_delta" ? "reasoning" : "answer";
		}
		render();
	});

	pi.on("message_end", (event) => {
		if (startedAt === 0 || event.message?.role !== "assistant") return;
		const outputTokens = event.message?.usage?.output;
		const reasoningTokens = event.message?.usage?.reasoning;
		observeOutput(Number.isSafeInteger(outputTokens) && outputTokens >= 0 ? outputTokens : undefined,
			Number.isSafeInteger(reasoningTokens) && reasoningTokens >= 0 ? reasoningTokens : undefined);
		finishedAt ||= monotonicNow();
		render();
	});

	pi.on("tool_execution_start", (event) => {
		if (startedAt === 0) return;
		// Track only tool identity and duration, never its arguments or output.
		const name = typeof event.toolName === "string" && /^[\w.-]{1,80}$/.test(event.toolName) ? event.toolName : "tool";
		runningTools.set(event.toolCallId, { name, startedAt: monotonicNow() });
		toolsFinishedAt = 0;
		render();
	});

	pi.on("tool_execution_end", (event) => {
		if (startedAt === 0 || !runningTools.delete(event.toolCallId)) return;
			if (runningTools.size === 0) nextRequestFromToolsAt = toolsFinishedAt = monotonicNow();
		render();
	});

	pi.on("turn_end", () => finish());
	pi.on("agent_end", () => finish());
	pi.on("agent_settled", () => finish());
	pi.on("session_shutdown", () => {
			finish();
			unsubscribe?.();
		scheduler.stop?.();
	});
}
