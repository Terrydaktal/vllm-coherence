import { getCompactionProgress } from "./qwen-radiance-compaction-progress.mjs";
import { CONTEXT_POLICY_ENTRY, contextPolicy, clonePolicy, contextRows, changeSelection,
  effectiveExclusions, selectionChanges, filterContext, hasThinking, estimateMessageTokens, clearContextFailure } from "./qwen-context-policy.mjs";

const CAPTURE = "QWEN_CONTEXT_PREVIEW_BEFORE_NETWORK";
const number = (n) => n.toLocaleString("en-GB");
// Terminal content is untrusted. Snippets stay in the local picker; no snippet,
// message body, arguments or token vector is written to policy metadata/logs.
const plain = (text) => String(text).replace(/\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))/g, "")
  .replace(/[\x00-\x08\x0b-\x1f\x7f\u202a-\u202e\u2066-\u2069]/g, "");
function snippet(message) {
  if (message.role === "bashExecution") return plain(`!${message.command ?? ""}\n${message.output ?? ""}`);
  if (typeof message.content === "string") return plain(message.content);
  return (message.content ?? []).map((block) => {
    if (block?.type === "text") return plain(block.text ?? "");
    if (block?.type === "thinking") return `[thinking] ${plain(block.thinking ?? "")}`;
    if (block?.type === "toolCall") return `[${plain(block.name)}] ${plain(JSON.stringify(block.arguments ?? {}))}`;
    return block?.type === "image" ? "[image]" : "";
  }).filter(Boolean).join("\n");
}
function rowSnippet(message) {
  if (!Array.isArray(message.content)) return snippet(message);
  const preferred = message.content.find((block) => block?.type === "text" && block.text?.trim()) ??
    message.content.find((block) => block?.type === "toolCall") ?? message.content.find((block) => block?.type === "thinking");
  return preferred ? snippet({ ...message, content: [preferred] }) : snippet(message);
}

export async function previewContextTokens(pi, ctx, policy, dependencies, signal) {
  const { convertToLlm, streamSimpleOpenAICompletions, fetcher = fetch } = dependencies;
  if (ctx.model?.api !== "openai-completions" || ctx.model?.provider !== "qwen-r9700") {
    throw new Error("Exact token preview is available on the Radiance backend; message estimates remain available");
  }
  const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
  if (!auth.ok) throw new Error("Cannot authenticate the tokenizer");
  const model = { ...ctx.model, baseUrl: auth.baseUrl ?? ctx.model.baseUrl };
  const base = model.baseUrl.replace(/\/+$/, "");
  if (!base.endsWith("/v1")) throw new Error("Tokenizer connection is not configured");
  const allTools = new Map(pi.getAllTools().map((tool) => [tool.name, tool]));
  const tools = pi.getActiveTools().map((name) => allTools.get(name));
  if (tools.some((tool) => !tool)) throw new Error("An active tool schema is missing");
  const source = ctx.sessionManager.buildSessionContext().messages;
  const saved = contextPolicy(ctx);
  const headers = { "Content-Type": "application/json", ...(auth.apiKey ? { Authorization: `Bearer ${auth.apiKey}` } : {}),
    ...Object.fromEntries(Object.entries(auth.headers ?? {}).filter(([, value]) => value != null)) };
  const count = async (selected) => {
    let payload;
    const capture = await streamSimpleOpenAICompletions(model, { systemPrompt: ctx.getSystemPrompt(),
      messages: convertToLlm(filterContext(source, ctx, selected)), tools }, { ...auth,
      reasoning: pi.getThinkingLevel(), maxTokens: 1, signal,
      onPayload: (value) => { payload = value; throw new Error(CAPTURE); },
      fetch: () => { throw new Error("Context preview must not request model generation"); },
    }).result();
    if (!payload || capture.errorMessage !== CAPTURE) throw new Error("Could not prepare the tokenizer prompt");
    if (selected.preserveFutureThinking) payload.chat_template_kwargs = { ...payload.chat_template_kwargs,
      preserve_thinking: true,
      ...(Object.hasOwn(payload.chat_template_kwargs ?? {}, "preserve_reasoning") ? { preserve_reasoning: true } : {}),
    };
    const response = await fetcher(`${base.slice(0, -3)}/tokenize`, { method: "POST", headers, signal,
      body: JSON.stringify({ model: payload.model, messages: payload.messages, tools: payload.tools,
        chat_template_kwargs: payload.chat_template_kwargs, add_generation_prompt: true, add_special_tokens: false }) });
    if (!response.ok) throw new Error(`Tokenizer returned HTTP ${response.status}`);
    const body = await response.json();
    if (!Array.isArray(body.tokens) || !body.tokens.every((token) => Number.isSafeInteger(token) && token >= 0)) {
      throw new Error("Tokenizer returned an invalid token count");
    }
    return body.tokens.length;
  };
  // CPU tokenization only, with a bounded request lifetime; never a GPU prefill.
  const before = await count(saved), after = await count(policy);
  return { before, after, capacity: model.contextWindow };
}

export class ContextPicker {
  constructor({ rows, policy, messages, ctx, tui, theme, done, matchesKey, truncateToWidth, preview }) {
    Object.assign(this, { rows, messages, ctx, tui, theme, done, matchesKey, truncateToWidth, preview });
    this.original = clonePolicy(policy);
    this.policy = clonePolicy(policy);
    this.cursor = Math.max(0, rows.length - 1);
    this.selected = new Set();
    this.undo = [];
    this.anchor = undefined;
    this.notice = "Selections affect model context; the saved transcript stays intact.";
    this.previewResult = undefined;
    this.revision = 0;
    this.disposed = false;
    this.counts = new Map(rows.map((row) => [row.id, { all: estimateMessageTokens(row.message),
      noThinking: hasThinking(row.message) ? estimateMessageTokens({ ...row.message,
        content: row.message.content.filter((block) => block?.type !== "thinking") }) : estimateMessageTokens(row.message) }]));
    this.readonlyTokens = Math.max(0, messages.reduce((sum, message) => sum + estimateMessageTokens(message), 0) -
      rows.reduce((sum, row) => sum + this.counts.get(row.id).all, 0));
  }
  invalidate() {}
  dispose() { this.disposed = true; this.previewAbort?.abort(); }
  redraw() { if (!this.disposed) this.tui.requestRender(); }
  ids() { return this.selected.size ? [...this.selected] : [this.rows[this.cursor]?.id].filter(Boolean); }
  apply(part, excluded) {
    const next = changeSelection(this.rows, this.policy, this.ids(), part, excluded);
    if (selectionChanges(this.policy, next).length) {
      this.undo.push(clonePolicy(this.policy));
      this.policy = next;
      this.revision++;
      this.previewResult = undefined;
    }
    this.notice = part === "message" ? (excluded ? "Exclude from context (tool calls and their results stay together)." : "Restore to context.")
      : (excluded ? "Exclude selected thinking; future thinking is retained." : "Restore selected thinking to context.");
  }
  handleInput(data) {
    const key = (name) => this.matchesKey(data, name);
    if (key("escape") || key("ctrl+c")) { this.done(undefined); return; }
    if (key("up")) this.cursor = Math.max(0, this.cursor - 1);
    else if (key("down")) this.cursor = Math.min(this.rows.length - 1, this.cursor + 1);
    else if (key("pageUp")) this.cursor = Math.max(0, this.cursor - this.pageSize);
    else if (key("pageDown")) this.cursor = Math.min(this.rows.length - 1, this.cursor + this.pageSize);
    else if (key("home")) this.cursor = 0;
    else if (key("end")) this.cursor = Math.max(0, this.rows.length - 1);
    else if (key("space")) {
      const id = this.rows[this.cursor]?.id;
      if (id) { if (this.selected.has(id)) this.selected.delete(id); else this.selected.add(id); }
    } else if (key("t")) {
      const turn = this.rows[this.cursor]?.turn;
      const ids = this.rows.filter((row) => row.turn === turn).map((row) => row.id);
      const selected = ids.every((id) => this.selected.has(id));
      for (const id of ids) { if (selected) this.selected.delete(id); else this.selected.add(id); }
    } else if (key("r")) {
      if (this.anchor === undefined) { this.anchor = this.cursor; this.notice = "Range start set; move to the end and press r again."; }
      else {
        for (let i = Math.min(this.anchor, this.cursor); i <= Math.max(this.anchor, this.cursor); i++) this.selected.add(this.rows[i].id);
        this.anchor = undefined;
        this.notice = "Range selected.";
      }
    } else if (key("o")) {
      for (const row of this.rows.slice(0, this.cursor + 1)) this.selected.add(row.id);
      this.notice = "Selected all messages through the cursor (oldest first).";
    } else if (key("a")) {
      if (this.selected.size === this.rows.length) this.selected.clear();
      else for (const row of this.rows) this.selected.add(row.id);
    } else if (key("c")) { this.selected.clear(); this.anchor = undefined; }
    else if (key("e")) this.apply("message", true);
    else if (key("s")) this.apply("message", false);
    else if (key("h")) this.apply("thinking", true);
    else if (key("shift+h") || data === "H") this.apply("thinking", false);
    else if (key("u")) {
      if (this.undo.length) { this.policy = this.undo.pop(); this.revision++; this.previewResult = undefined; this.notice = "Undid the last pending change."; }
    } else if (key("p")) {
      if (this.previewAbort) return;
      const revision = this.revision, policy = clonePolicy(this.policy);
      const controller = new AbortController();
      this.previewAbort = controller;
      this.notice = "Counting the rendered prompt with the CPU tokenizer…";
      const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(15000)]);
      void this.preview(policy, signal).then((result) => {
        if (revision === this.revision && !this.disposed) { this.previewResult = result; this.notice = "Exact prompt count includes the system prompt, tool schemas and template."; }
      }).catch(() => {
        if (!this.disposed) this.notice = "Exact preview unavailable; selections are unchanged. Message counts are estimates.";
      }).finally(() => { this.previewAbort = undefined; this.redraw(); });
    } else if (key("enter")) { this.done(this.policy); return; }
    this.redraw();
  }
  render(width) {
    const fit = (value) => this.truncateToWidth(value, Math.max(1, width));
    const excluded = effectiveExclusions(this.rows, this.policy);
    const estimate = (selected) => {
      const hidden = effectiveExclusions(this.rows, selected);
      return this.readonlyTokens + this.rows.reduce((sum, row) => sum + (hidden.has(row.id) ? 0 :
        this.counts.get(row.id)[selected.thinking.has(row.id) ? "noThinking" : "all"]), 0);
    };
    const before = estimate(this.original), after = estimate(this.policy), result = this.previewResult;
    const header = result
      ? `Exact prompt: ${number(result.before)} → ${number(result.after)} / ${number(result.capacity)} tok · ${number(result.before - result.after)} freed`
      : `Message estimate: ≈${number(before)} → ≈${number(after)} tok (excludes system prompt, tools and template)`;
    const lines = [this.theme.fg("accent", "/context — choose what the model remembers"), header,
      `${this.selected.size} selected · ${selectionChanges(this.original, this.policy).length} pending changes · ${excluded.size} excluded messages`,
      "↑↓ move · Space select · t turn · r range · o older · a all · c clear selection",
      "e exclude message · s restore message · h exclude thinking · H restore thinking",
      "p exact count · u undo change · Enter save · Esc cancel", ""];
    this.pageSize = Math.max(1, Math.min(12, (this.tui.terminal?.rows ?? 30) - 16));
    const start = Math.max(0, Math.min(this.cursor - Math.floor(this.pageSize / 2), this.rows.length - this.pageSize));
    for (const row of this.rows.slice(start, start + this.pageSize)) {
      const state = excluded.has(row.id) ? "excluded" : this.policy.thinking.has(row.id) ? "no thinking" : "included";
      const label = plain(row.message.role === "toolResult" ? `tool ${row.message.toolName ?? "result"}` : row.message.role);
      const line = `${row.index === this.cursor ? "›" : " "} [${this.selected.has(row.id) ? "x" : " "}] ${row.index + 1}. ${label} · ${state} · ` +
        `≈${number(this.counts.get(row.id).all)} tok · ${rowSnippet(row.message).replace(/\s+/g, " ")}`;
      lines.push(row.index === this.cursor ? this.theme.fg("accent", fit(line)) : fit(line));
    }
    if (!this.rows.length) lines.push("No individual messages remain after the latest checkpoint.");
    const focused = this.rows[this.cursor];
    if (focused) {
      lines.push("", `Message ${this.cursor + 1} · turn ${focused.turn}${hasThinking(focused.message) ? " · contains thinking" : ""}`);
      lines.push(...snippet(focused.message).split("\n").filter(Boolean).slice(0, 3));
    }
    lines.push("", this.notice, "Earlier compacted history is represented by the checkpoint, not individual messages.");
    return lines.map(fit);
  }
}

export function installContextPicker(pi, dependencies) {
  const { Text } = dependencies;
  if (Text) pi.registerEntryRenderer(CONTEXT_POLICY_ENTRY, (entry, _options, theme) => {
    const changes = entry.data?.changes ?? [];
    const count = (part, excluded) => changes.filter((change) => change.part === part && change.excluded === excluded).length;
    return new Text(theme.fg("dim", `Context updated: ${count("message", true)} messages excluded, ${count("message", false)} restored; ` +
      `${count("thinking", true)} thinking blocks excluded, ${count("thinking", false)} restored.`), 0, 0);
  });
  pi.registerCommand("context", {
    description: "Choose messages or thinking to exclude from model context without deleting the transcript. /context undo restores the last change.",
    handler: async (args, ctx) => {
      if (!["", "undo", "status"].includes(args.trim())) { ctx.ui.notify("Usage: /context [undo|status]", "warning"); return; }
      if (!ctx.isIdle() || getCompactionProgress(ctx)?.snapshot().state === "running") {
        ctx.ui.notify("Stop generation or compaction with Escape before /context.", "warning"); return;
      }
      if (ctx.contextFilterErrorsPropagate !== true) {
        ctx.ui.notify("Restart Pi to activate the installed context-safety runtime patch before using /context.", "warning"); return;
      }
      const before = contextPolicy(ctx), rows = contextRows(ctx);
      if (args.trim() === "status") {
        const activeIds = new Set(rows.map((row) => row.id));
        const excluded = effectiveExclusions(rows, before);
        ctx.ui.notify(`Active context: ${rows.length} messages; ${rows.filter((row) => excluded.has(row.id)).length} excluded; ` +
          `${[...before.thinking].filter((id) => activeIds.has(id)).length} with thinking excluded. Saved transcript retained.`, "info"); return;
      }
      let after, undoOf;
      const leaf = ctx.sessionManager.getLeafId();
      if (args.trim() === "undo") {
        const changes = ctx.sessionManager.getBranch().filter((entry) => entry.type === "custom" && entry.customType === CONTEXT_POLICY_ENTRY);
        const undone = new Set(changes.map((entry) => entry.data.undoOf).filter(Boolean));
        const last = changes.findLast((entry) => !entry.data.undoOf && !undone.has(entry.id));
        if (!last) { ctx.ui.notify("No context selection to undo on this branch.", "info"); return; }
        // Re-evaluate the branch before that entry, then revert only the parts it
        // changed. Later /purge-thinking decisions for other entries stay intact.
        const branch = ctx.sessionManager.getBranch();
        const earlier = contextPolicy({ ...ctx, sessionManager: { getBranch: () => branch.slice(0, branch.findIndex((entry) => entry.id === last.id)),
          buildContextEntries: () => ctx.sessionManager.buildContextEntries() } });
        after = clonePolicy(before);
        for (const change of last.data.changes) {
          const key = change.part === "message" ? "excluded" : "thinking";
          if (earlier[key].has(change.entryId)) after[key].add(change.entryId); else after[key].delete(change.entryId);
        }
        undoOf = last.id;
      } else {
        if (ctx.mode && ctx.mode !== "tui") { ctx.ui.notify("/context needs the interactive terminal. /context status and undo work in other modes.", "warning"); return; }
        after = await ctx.ui.custom((tui, theme, _keybindings, done) => new ContextPicker({ rows, policy: before,
          messages: ctx.sessionManager.buildSessionContext().messages, ctx, tui, theme, done, ...dependencies,
          preview: (policy, signal) => previewContextTokens(pi, ctx, policy, dependencies, signal) }), { overlay: true,
          overlayOptions: { width: "95%", maxHeight: "95%", anchor: "center" } });
        if (!after) return;
      }
      if (!ctx.isIdle() || ctx.sessionManager.getLeafId() !== leaf) {
        ctx.ui.notify("The chat changed while /context was open; selections were not saved.", "warning"); return;
      }
      const changes = selectionChanges(before, after);
      if (!changes.length && !undoOf) { ctx.ui.notify("Context unchanged.", "info"); return; }
      filterContext(ctx.sessionManager.buildSessionContext().messages, ctx, after); // Validate before persisting.
      pi.appendEntry(CONTEXT_POLICY_ENTRY, { version: 1, changes, preserveFutureThinking: after.preserveFutureThinking,
        ...(undoOf ? { undoOf } : {}) });
      clearContextFailure(ctx);
      ctx.ui.notify(`${undoOf ? "Undid the last context selection" : "Saved context selection"}. Transcript retained; the next request may rebuild the changed cache prefix.`, "info");
    },
  });
}
