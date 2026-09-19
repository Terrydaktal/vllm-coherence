import assert from "node:assert/strict";
import test from "node:test";
import install from "../integrations/pi/qwen-progress.mjs";

function fixture(t, schedulerOverride) {
  let now = 1000, tick, intervalMs, stopped = false;
  t.mock.method(globalThis.performance, "now", () => now);
  t.mock.method(globalThis, "setInterval", (callback, milliseconds) => {
    tick = callback; intervalMs = milliseconds; stopped = false; return 1;
  });
  t.mock.method(globalThis, "clearInterval", () => { stopped = true; });
  const handlers = new Map(), commands = new Map(), working = [], statuses = [];
  const ctx = { mode: "tui", model: { provider: "qwen-r9700", api: "openai-completions" },
    getContextUsage: () => ({ tokens: 60000 }), ui: {
      setWorkingMessage: (value) => working.push(value), setStatus: (_key, value) => statuses.push(value),
    } };
  const scheduler = schedulerOverride ?? {
    start() {}, bind() {}, clear() {}, stop() {},
    read() { return { available: false }; },
  };
  install({ on: (name, handler) => handlers.set(name, handler), registerCommand: (name, value) => commands.set(name, value) }, { scheduler });
  const emit = (name, event = {}) => handlers.get(name)?.(event, ctx);
  emit("before_provider_request", { payload: { stream: true } });
  t.after(() => {
    emit("session_shutdown");
    assert.ok(statuses.every((value) => value === undefined), "progress never pins a footer message");
  });
  return { working, statuses, emit, commands, ctx,
    intervalMs() { return intervalMs; },
    advance(ms) { now += ms; if (!stopped) tick(); },
    update(type, output, delta = "", reasoning = 0) { emit("message_update", { assistantMessageEvent: {
      type, delta, partial: { usage: { output, reasoning } },
    } }); },
    end(output) { emit("message_end", { message: { role: "assistant", usage: { output, reasoning: 0 } } }); },
  };
}

test("progress redraws telemetry at 100 ms", (t) => {
  const f = fixture(t);
  f.advance(1000);
  assert.equal(f.intervalMs(), 100);
});

test("a priority reservation names its owner while that owner runs tools", (t) => {
  const observation = { available: true, priorityHold: { chat_id: "b".repeat(64), priority: 2 },
    request: { state: "paused", computed_tokens: 60050, input_tokens: 60000 } };
  const f = fixture(t, { start() {}, bind() {}, clear() {}, stop() {}, read: () => observation });
  f.advance(4000);
  assert.match(f.working.at(-1), /higher-priority chat.*b{12} keeps the GPU through tools until its answer finishes/);
  assert.doesNotMatch(f.working.at(-1), /cold|first token|prefill/);
});

test("a completed 717-token tool call becomes a tool timer without a pinned response rate", (t) => {
  const f = fixture(t);
  f.advance(1800);
  f.update("toolcall_delta", 1, "{");
  f.advance(10000);
  f.update("toolcall_delta", 717, "}");
  assert.match(f.working.at(-1), /generating tool arguments: 717 tok.*71.6 t\/s, 71.6 t\/s avg/);
  f.end(717);
  assert.equal(f.statuses.at(-1), undefined);
  assert.doesNotMatch(f.working.at(-1), /tok\/s|t\/s/);
  f.emit("tool_execution_start", { toolCallId: "one", toolName: "bash",
    get args() { assert.fail("progress must not inspect tool arguments"); } });
  f.advance(56000);
  assert.equal(f.working.at(-1), "Qwen running bash 56s • model output ended");
  assert.equal(f.statuses.at(-1), undefined);
  f.emit("tool_execution_end", { toolCallId: "one",
    get result() { assert.fail("progress must not inspect tool results"); } });
  f.advance(2000);
  assert.match(f.working.at(-1), /tools finished.*waiting for next step 2s/);
  assert.equal(f.statuses.at(-1), undefined);
  f.emit("turn_end");
  assert.equal(f.working.at(-1), undefined);
});

test("silence hides live throughput; duplicate counters and empty deltas do not revive it", (t) => {
  const f = fixture(t);
  f.advance(1000);
  f.update("text_delta", 1, "x");
  f.advance(10000);
  f.update("text_delta", 601, "x");
  assert.match(f.working.at(-1), /60.0 t\/s, 60.0 t\/s avg/);
  f.advance(1000);
  assert.match(f.working.at(-1), /40.0 t\/s, 60.0 t\/s avg/, "the recent rate ages while the existing overall average stays fixed");
  f.advance(1000);
  assert.match(f.working.at(-1), /scheduler telemetry unavailable.*backend wait unknown while awaiting model output.*client stream unchanged 2s/);
  assert.doesNotMatch(f.working.at(-1), /tok\/s|t\/s/);
  f.update("usage_update", 601);
  f.update("text_delta", 601);
  f.emit("message_update", { assistantMessageEvent: { type: "usage_update", partial: { usage: { output: 601, reasoning: 50 } } } });
  assert.match(f.working.at(-1), /client stream unchanged 2s/, "a late reasoning breakdown does not imply new tokens");
  f.advance(8000);
  assert.match(f.working.at(-1), /client stream unchanged 10s/);
  f.update("usage_update", 661);
  assert.match(f.working.at(-1), /Qwen answer: 661 tok.*6.0 t\/s, 33.0 t\/s avg/);
  f.advance(20000);
  f.end(661);
  assert.equal(f.statuses.at(-1), undefined, "terminal housekeeping must not create a footer message");
  assert.doesNotMatch(f.working.at(-1), /tok\/s|t\/s/);
});

test("usage-only buffered output establishes activity, and the next request starts fresh", (t) => {
  const f = fixture(t);
  f.advance(1000);
  f.update("usage_update", 10);
  assert.match(f.working.at(-1), /generating \(buffered output\): 10 tok/);
  f.advance(1000);
  f.update("usage_update", 70);
  assert.match(f.working.at(-1), /60.0 t\/s, 60.0 t\/s avg/);
  f.end(70);
  f.advance(20000);
  f.emit("before_provider_request", { payload: { stream: true } });
  assert.match(f.working.at(-1), /KV lookup\/prefill.*0s/);
  assert.equal(f.statuses.at(-1), undefined);
  f.emit("after_provider_response");
  assert.match(f.working.at(-1), /scheduler telemetry unavailable.*backend wait unknown/);
  assert.doesNotMatch(f.working.at(-1), /accepted|cache hit|cached/);
  f.advance(3000);
  f.update("text_delta", 1, "x");
  f.advance(1000);
  f.update("text_delta", 41, "x");
	assert.match(f.working.at(-1), /40.0 t\/s, 40.0 t\/s avg.*first data 3.0s/);
});

test("generation telemetry adds the current round and three-second acceptance without polling the GPU", (t) => {
	const observation = {
		available: true,
		request: {
			state: "running", computed_tokens: 60_001, input_tokens: 60_000,
		},
		cacheResidency: "gpu",
		workerAvailable: true,
		activeChat: { chat_id: "a".repeat(64), generation: "b".repeat(64) },
		requestPhase: {
			request_id: "c".repeat(64), phase: "generate", last_round_ms: null, acceptance_rate: null,
				last_acceptance_rate: null, acceptance_rate_3s: null,
		},
	};
	const f = fixture(t, { start() {}, bind() {}, clear() {}, stop() {}, read: () => observation });
	f.advance(1000);
	f.update("text_delta", 1, "x");
	observation.requestPhase.last_round_ms = 17.4;
	observation.requestPhase.acceptance_rate = 102 / 189;
		observation.requestPhase.acceptance_rate_3s = 5 / 7;
		f.advance(500);
		assert.match(f.working.at(-1), /round 17\.4 ms.*acceptance 71\.4%/);
	assert.doesNotMatch(f.working.at(-1), /54\.0%/);
});

test("scheduler telemetry identifies another chat as the reason generation is paused", (t) => {
  const thisChat = "a".repeat(64);
  const otherChat = "b".repeat(64);
  const generation = "c".repeat(64);
  let observation = {
    available: true,
    request: {
      chat_id: thisChat,
		generation,
		state: "paused",
		computed_tokens: 0,
		input_tokens: 60_000,
	},
	otherRunningChatId: otherChat,
	cacheResidency: "ram",
	quantumSeconds: 30,
  };
  const scheduler = {
    start() {}, bind() {}, clear() {}, stop() {},
    read() { return observation; },
  };
  const f = fixture(t, scheduler);
  f.emit("after_provider_response");
  f.advance(3000);
	assert.match(
		f.working.at(-1),
		/Qwen request paused.*another chat b{12} has the GPU.*paused 3s.*KV cache parked in RAM; restore follows GPU handover/,
	);
	assert.doesNotMatch(f.working.at(-1), /0 \/ 60,000|prompt\/KV preparation/);
	assert.doesNotMatch(f.working.at(-1), /waiting for first token|no stream output|no stream update/);

	observation = {
		...observation,
		request: { ...observation.request, computed_tokens: 20_000 },
	};
	f.advance(1000);
	assert.match(f.working.at(-1), /KV state through 20,000 \/ 60,000 tok parked in RAM/);

	observation = {
    ...observation,
    request: { ...observation.request, state: "running", computed_tokens: 45_000 },
    otherRunningChatId: undefined,
  };
	f.advance(1000);
	assert.match(f.working.at(-1), /GPU active.*prompt\/KV state \(restored or prefilled\) 45,000 \/ 60,000 tok/);

  observation = {
    ...observation,
    request: { ...observation.request, computed_tokens: 60_001 },
  };
  f.update("text_delta", 10, "x");
  observation = {
    ...observation,
    request: { ...observation.request, state: "paused" },
    otherRunningChatId: otherChat,
  };
  f.advance(1000);
  assert.match(f.working.at(-1), /Qwen generation paused: 10 tok.*another chat b{12} has the GPU/);
  assert.doesNotMatch(f.working.at(-1), /t\/s|stream unchanged/);

  observation = {
    ...observation,
    request: { ...observation.request, state: "running" },
    otherRunningChatId: undefined,
  };
  f.advance(1000);
  f.advance(3000);
  assert.match(f.working.at(-1), /GPU active.*backend token count observed unchanged 3s.*client stream unchanged 5s/);
  observation = {
    ...observation,
    request: { ...observation.request, computed_tokens: 60_002 },
  };
  f.advance(1000);
  assert.match(f.working.at(-1), /GPU active.*backend token count advancing.*next model output stream update pending/);
});

test("another chat's GPU slice is excluded from both throughput averages", (t) => {
  const thisChat = "a".repeat(64);
  const otherChat = "b".repeat(64);
  let observation = {
    available: true,
    request: {
      chat_id: thisChat,
      generation: "c".repeat(64),
      state: "running",
      computed_tokens: 60_001,
      input_tokens: 60_000,
    },
    otherRunningChatId: undefined,
    cacheResidency: "gpu",
    workerAvailable: true,
    activeChat: { chat_id: thisChat, generation: "c".repeat(64) },
  };
  const scheduler = {
    start() {}, bind() {}, clear() {}, stop() {},
    read() { return observation; },
  };
  const f = fixture(t, scheduler);
  f.advance(1000);
  f.update("usage_update", 1);
  f.advance(1000);
  f.update("usage_update", 101);
  assert.match(f.working.at(-1), /100.0 t\/s, 100.0 t\/s avg/);

  observation = {
    ...observation,
    request: { ...observation.request, state: "paused" },
    otherRunningChatId: otherChat,
    cacheResidency: "ram",
    activeChat: { chat_id: otherChat, generation: "d".repeat(64) },
  };
  f.advance(0);
  f.advance(10_000);
  assert.match(f.working.at(-1), /generation paused.*another chat b{12} has the GPU/);

  observation = {
    ...observation,
    request: { ...observation.request, state: "running" },
    otherRunningChatId: undefined,
    cacheResidency: "gpu",
    activeChat: { chat_id: thisChat, generation: "c".repeat(64) },
  };
  f.advance(0);
  f.advance(1000);
  f.update("usage_update", 201);
  assert.match(
    f.working.at(-1),
    /100.0 t\/s, 100.0 t\/s avg/,
    "a ten-second GPU handover must not dilute this chat's decode rates",
  );
});

test("queued responses explain that the current response must finish before GPU handover", (t) => {
  const observation = {
    available: true, policy: "response_boundary",
    request: { state: "queued", computed_tokens: 0, input_tokens: 60_000 },
    otherRunningChatId: "b".repeat(64), cacheResidency: "ram",
  };
  const f = fixture(t, { read: () => observation });
  f.advance(20_000);
  assert.match(f.working.at(-1), /queued for GPU.*another chat b{12} has the GPU; waiting for its response to finish/);
  assert.doesNotMatch(f.working.at(-1), /0 \/ 60,000|waiting for first token|no stream output/);
});

test("queued chat shows the tool grace countdown instead of cache preparation", (t) => {
	const observation = { available: true,
		request: { state: "queued", computed_tokens: 0, input_tokens: 60_000 },
		toolGrace: { chat_id: "b".repeat(64), phase: "tool_grace", remaining_seconds: 1.5 },
	};
	const f = fixture(t, { read: () => observation });
	f.advance(1000);
	assert.match(f.working.at(-1), /queued for GPU.*chat b{12}'s tool.*1.5s grace remaining/);
	assert.doesNotMatch(f.working.at(-1), /cache admission|prompt\/KV|waiting for first token/);
	observation.toolGrace.phase = "response_outcome";
	f.advance(1000);
	assert.match(f.working.at(-1), /confirming whether chat b{12} finished with a tool call/);
});

test("parallel tools retain their own timers until each finishes", (t) => {
  const f = fixture(t);
  f.end(0);
  f.emit("tool_execution_start", { toolCallId: "one", toolName: "bash" });
  f.advance(3000);
  f.emit("tool_execution_start", { toolCallId: "two", toolName: "search" });
  f.advance(4000);
  assert.match(f.working.at(-1), /running bash 7s, running search 4s/);
  f.emit("tool_execution_end", { toolCallId: "one" });
  f.advance(1000);
  assert.match(f.working.at(-1), /running search 5s/);
  assert.doesNotMatch(f.working.at(-1), /bash|tok\/s|t\/s/);
  f.emit("session_shutdown");
  f.advance(60000);
  assert.equal(f.working.at(-1), undefined);
});

test("three-second rolling throughput follows changing speed and interpolates its boundary", (t) => {
  const f = fixture(t);
  f.advance(1000);
  f.update("usage_update", 10);
  for (const tokens of [30, 70, 130, 210]) {
    f.advance(1000);
    f.update("usage_update", tokens);
  }
  assert.match(f.working.at(-1), /60.0 t\/s, 50.0 t\/s avg/);
  f.advance(500);
  assert.match(f.working.at(-1), /53.3 t\/s, 50.0 t\/s avg/);
  f.update("usage_update", 210);
  f.update("usage_update", 200); // A regressing counter must not add output.
  assert.match(f.working.at(-1), /53.3 t\/s, 50.0 t\/s avg/);
  f.advance(500);
  f.update("usage_update", 310);
  assert.match(f.working.at(-1), /80.0 t\/s, 60.0 t\/s avg/);
  f.advance(3000);
  assert.match(f.working.at(-1), /client stream unchanged 3s/);
  f.update("usage_update", 340);
  assert.match(f.working.at(-1), /10.0 t\/s, 41.3 t\/s avg/);
});

test("session reload and switching clear an older last-response footer", (t) => {
  const f = fixture(t);
  for (const event of ["session_start", "session_switch"]) {
    const count = f.statuses.length;
    f.emit(event);
    assert.equal(f.statuses.length, count + 1);
    assert.equal(f.statuses.at(-1), undefined);
  }
});

test("thinking with an unreported token breakdown omits the reasoning segment", (t) => {
  const f = fixture(t);
  f.advance(1000);
  f.update("thinking_delta", 10, "synthetic thinking");
  assert.match(f.working.at(-1), /10 tok/);
  assert.doesNotMatch(f.working.at(-1), /reasoning count unavailable|\([\d,]+ reasoning\)/);
  f.advance(1000);
  f.update("thinking_delta", 70, "synthetic thinking");
  assert.match(f.working.at(-1), /60.0 t\/s, 60.0 t\/s avg/, "thinking remains included in total output speed");
  f.end(70);
  assert.equal(f.statuses.at(-1), undefined);

  f.emit("before_provider_request", { payload: { stream: true } });
  f.advance(1000);
  f.update("text_delta", 10, "synthetic answer");
  assert.doesNotMatch(f.working.at(-1), /reasoning/, "the next response must not inherit the previous thinking state");
  f.end(10);
  assert.equal(f.statuses.at(-1), undefined);
});

test("a positive provider reasoning count appears once it is reported", (t) => {
  const f = fixture(t);
  f.advance(1000);
  f.update("thinking_delta", 10, "synthetic thinking");
  assert.doesNotMatch(f.working.at(-1), /reasoning count unavailable|\([\d,]+ reasoning\)/);
  f.advance(1000);
  f.update("thinking_delta", 70, "synthetic thinking", 60);
  assert.match(f.working.at(-1), /70 tok \(60 reasoning\)/);
  f.end(70);
  assert.equal(f.statuses.at(-1), undefined);
});

test("buffered edits report argument generation, then use a fresh application timer", (t) => {
  const f = fixture(t);
  const block = { type: "toolCall", name: "edit",
    get arguments() { assert.fail("progress must not inspect edit arguments"); } };
  f.advance(2300);
  f.emit("message_update", { assistantMessageEvent: { type: "toolcall_start", contentIndex: 0,
    partial: { content: [block], usage: { output: 111 } } } });
  for (let second = 1; second <= 34; second++) {
    f.advance(1000);
    f.update("usage_update", 111 + 50 * second);
    assert.match(f.working.at(-1), /generating edit arguments/);
    assert.doesNotMatch(f.working.at(-1), /waiting|0 reasoning/);
  }
  assert.match(f.working.at(-1), /1,811 tok.*50.0 t\/s, 50.0 t\/s avg/);
  assert.doesNotMatch(f.working.at(-1), /arguments \d+s/);
  f.advance(2000);
  assert.match(f.working.at(-1), /scheduler telemetry unavailable.*backend wait unknown while awaiting edit arguments.*client stream unchanged 2s/);
  assert.doesNotMatch(f.working.at(-1), /tok\/s/);
  f.end(1811);
  f.emit("tool_execution_start", { toolCallId: "edit-one", toolName: "edit" });
  assert.equal(f.working.at(-1), "Qwen applying edit 0s • model output ended");
  f.advance(3000);
  assert.equal(f.working.at(-1), "Qwen applying edit 3s • model output ended");
  f.emit("tool_execution_end", { toolCallId: "edit-one" });
  f.emit("before_provider_request", { payload: { stream: true } });
  f.advance(1000);
  f.update("usage_update", 5);
  assert.match(f.working.at(-1), /generating \(buffered output\)/);
  assert.doesNotMatch(f.working.at(-1), /edit|arguments/);
});

test("a tool name received before usage survives the first buffered count", (t) => {
  const f = fixture(t);
  f.emit("message_update", { assistantMessageEvent: { type: "toolcall_start", contentIndex: 0,
    partial: { content: [{ type: "toolCall", name: "edit" }] } } });
  f.advance(1000);
  f.update("usage_update", 100);
  assert.match(f.working.at(-1), /generating edit arguments/);
  assert.doesNotMatch(f.working.at(-1), /arguments \d+s/);
});

test("a queued request without a blocker does not claim GPU contention or a cold cache", (t) => {
  const scheduler = { read: () => ({ available: true, request: {
    state: "queued", input_tokens: 150800, computed_tokens: 0,
  } }) };
  const f = fixture(t, scheduler);
  assert.match(f.working.at(-1), /preparing next response.*stage not reported/);
  assert.doesNotMatch(f.working.at(-1), /queued for GPU|0 \/ 150,800|after GPU admission/);
});

test("authoritative cache phases replace stale queue samples and expose timings on demand", async (t) => {
  const phase = { chat_id: "a".repeat(64), generation: "b".repeat(64), request_id: "c".repeat(64),
    phase: "cache_update", blocker: null, input_tokens: 150800, computed_tokens: 0,
    cached_tokens: 150272, elapsed_ms: 237, phase_elapsed_ms: 237, first_token_ms: null,
    timings_ms: { cache_update: 237 } };
  let notify;
  const observation = { available: true, request: { state: "queued", computed_tokens: 0, input_tokens: 150800 },
    requestPhase: phase, phaseObservedAt: Date.now() };
  const scheduler = { read: () => observation, subscribe: (fn) => { notify = fn; return () => { notify = undefined; }; } };
  const f = fixture(t, scheduler);
  assert.match(f.working.at(-1), /finishing previous cache update/);
  assert.doesNotMatch(f.working.at(-1), /queued for GPU/);
  phase.phase = "cache_lookup";
  phase.timings_ms.cache_lookup = 3;
  notify(); // No timer tick: the shared-file event alone updates the spinner.
  assert.match(f.working.at(-1), /checking reusable context/);
  phase.phase = "prefill";
  phase.computed_tokens = 150500;
  notify();
  assert.match(f.working.at(-1), /processing uncached prompt tokens.*228 \/ 528 uncached tok processed.*150,272 reused/);
  phase.phase = "generate";
  phase.first_token_ms = 540;
  f.advance(700);
  f.update("usage_update", 1);
  assert.doesNotMatch(f.working.at(-1), /queued for GPU/);
  const views = [];
  f.ctx.ui.select = async (title, lines) => { views.push({ title, lines }); };
  await f.commands.get("qwen-timing").handler("", f.ctx);
  assert.match(views[0].lines.join("\n"), /Finish.*previous cache update: 237.0 ms/);
  assert.match(views[0].lines.join("\n"), /Checking reusable context: 3.0 ms/);
  assert.match(views[0].lines.join("\n"), /Backend admission → first token: 540.0 ms/);
  assert.match(views[0].lines.join("\n"), /Request hook → first output: 700.0 ms/);
});

test("new streamed output takes precedence over an unchanged queued sample", (t) => {
  const f = fixture(t, { read: () => ({ available: true, otherRunningChatId: "b".repeat(64),
    request: { state: "queued", input_tokens: 100, computed_tokens: 0 } }) });
  f.advance(1000);
  f.update("text_delta", 1, "x");
  f.update("text_delta", 2, "x");
  assert.doesNotMatch(f.working.at(-1), /queued for GPU/);
});
