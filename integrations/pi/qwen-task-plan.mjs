import { contextPolicy, effectiveExclusions } from "./qwen-context-policy.mjs";
import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalToolName, toolNamesMatch } from "./qwen-tool-names.mjs";

const LEGACY_TOOL_NAME = "qwen_plan";
const TOOL_NAME = canonicalToolName(LEGACY_TOOL_NAME);
const SESSION_SEARCH_TOOL_NAME = canonicalToolName("session_search");
const REHYDRATE_TOOL_NAME = canonicalToolName("qwen_rehydrate_tool_turn");

export const TASK_PLAN_ENTRY = "qwen-task-plan";
export const TASK_PLAN_CONTEXT = "qwen-task-plan-context";
export const TASK_PLAN_VERSION = 1;
export const DEFAULT_PLAN_CONTEXT_CHARS = 3000;
const STATUSES = ["pending", "in_progress", "completed", "blocked"];
const MAX_STEPS = 30;
const ID = /^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}$/;
const emptyState = () => ({ version: TASK_PLAN_VERSION, mode: "execute", plan: null });
const clone = (value) => structuredClone(value);
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
function keys(value, allowed, label) {
  if (!object(value) || Object.keys(value).some((key) => !allowed.includes(key))) throw new Error(`Invalid ${label} fields`);
}
function text(value, limit, label) {
  if (typeof value !== "string" || !value.trim() || value.length > limit || /[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(value)) {
    throw new Error(`${label} must be nonempty text of at most ${limit} characters`);
  }
  return value.trim();
}
function list(value, max, mapper, label) {
  if (!Array.isArray(value) || value.length > max) throw new Error(`${label} must be an array of at most ${max} items`);
  return value.map(mapper);
}
function sources(value, available) {
  const result = list(value, 8, (id) => text(id, 256, "Source entry ID"), "Source entry IDs");
  if (new Set(result).size !== result.length) throw new Error("Duplicate source entry IDs");
  if (available && result.some((id) => !available.has(id))) throw new Error("Source/evidence entry ID is not on the selected session branch");
  return result;
}
function files(value) {
  const result = list(value, 40, (path) => text(path, 512, "Relevant file"), "Relevant files");
  if (new Set(result).size !== result.length) throw new Error("Duplicate relevant files");
  return result;
}
function annotations(value, max, available, label) {
  return list(value, max, (item) => {
    keys(item, ["text", "sourceEntryIds"], label);
    return { text: text(item.text, 400, label), sourceEntryIds: sources(item.sourceEntryIds, available) };
  }, label);
}
function validateStep(step, available) {
  keys(step, ["id", "title", "status", "sourceEntryIds", "relevantFiles", "evidence"], "step");
  if (!ID.test(step.id)) throw new Error("Step ID must be a stable identifier of at most 64 characters");
  if (!STATUSES.includes(step.status)) throw new Error("Invalid step status");
  return { id: step.id, title: text(step.title, 280, "Step title"), status: step.status,
    sourceEntryIds: sources(step.sourceEntryIds, available), relevantFiles: files(step.relevantFiles),
    evidence: list(step.evidence, 8, (item) => {
      keys(item, ["entryId", "note"], "evidence");
      const [entryId] = sources([item.entryId], available);
      return { entryId, note: text(item.note, 280, "Evidence note") };
    }, "Evidence") };
}
export function validateTaskPlanState(state, { availableIds } = {}) {
  keys(state, ["version", "mode", "plan"], "plan state");
  if (state.version !== TASK_PLAN_VERSION || !["execute", "plan"].includes(state.mode)) throw new Error("Unsupported saved plan state/version");
  if (state.plan === null) return { version: TASK_PLAN_VERSION, mode: state.mode, plan: null };
  const plan = state.plan;
  keys(plan, ["goal", "goalSourceEntryIds", "steps", "nextStepId", "relevantFiles", "relevantFileSources", "constraints", "notes"], "plan");
  if (!Number.isSafeInteger(plan.nextStepId) || plan.nextStepId < 1) throw new Error("Invalid next step ID counter");
  const relevantFiles = files(plan.relevantFiles);
  if (!object(plan.relevantFileSources) || Object.keys(plan.relevantFileSources).some((path) => !relevantFiles.includes(path))) {
    throw new Error("Invalid relevant file source mapping");
  }
  const relevantFileSources = Object.fromEntries(relevantFiles.map((path) => [path, sources(plan.relevantFileSources[path] ?? [], availableIds)]));
  const steps = list(plan.steps, MAX_STEPS, (step) => validateStep(step, availableIds), "Steps");
  if (!steps.length || new Set(steps.map((step) => step.id)).size !== steps.length) throw new Error("Plan needs at least one step and unique step IDs");
  if (steps.filter((step) => step.status === "in_progress").length > 1) throw new Error("Only one plan step may be in progress");
  return { version: TASK_PLAN_VERSION, mode: state.mode, plan: {
    goal: text(plan.goal, 800, "Goal"), goalSourceEntryIds: sources(plan.goalSourceEntryIds, availableIds), steps, nextStepId: plan.nextStepId,
    relevantFiles, relevantFileSources, constraints: annotations(plan.constraints, 16, availableIds, "Constraint"),
    notes: annotations(plan.notes, 12, availableIds, "Note"),
  } };
}

// Replay ONLY the selected root-to-leaf branch, never getEntries(): sibling plans
// are not inherited. Compaction retains ancestors in getBranch(), and forks copy
// their selected ancestry. Invalid saved state fails closed instead of enabling
// execution or resurrecting an earlier cleared plan.
export function replayTaskPlan(branchEntries) {
  let state = emptyState();
  for (const entry of branchEntries) {
    if (entry.type === "custom" && entry.customType === TASK_PLAN_ENTRY) state = validateTaskPlanState(entry.data);
  }
  return state;
}

export function filterTaskPlanForContext(state, excludedIds = new Set()) {
  const result = clone(state), plan = result.plan;
  if (!plan || !excludedIds.size) return result;
  const visible = (ids) => !(ids ?? []).some((id) => excludedIds.has(id));
  if (!visible(plan.goalSourceEntryIds)) { plan.goal = "[Goal withheld by /context]"; plan.goalSourceEntryIds = []; }
  plan.steps = plan.steps.filter((step) => visible(step.sourceEntryIds)).map((step) => ({ ...step,
    evidence: step.evidence.filter((item) => !excludedIds.has(item.entryId)),
    relevantFiles: step.relevantFiles.filter((path) => visible(plan.relevantFileSources[path])),
  }));
  plan.constraints = plan.constraints.filter((item) => visible(item.sourceEntryIds));
  plan.notes = plan.notes.filter((item) => visible(item.sourceEntryIds));
  plan.relevantFiles = plan.relevantFiles.filter((path) => visible(plan.relevantFileSources[path]));
  plan.relevantFileSources = Object.fromEntries(plan.relevantFiles.map((path) => [path, plan.relevantFileSources[path]]));
  return result;
}
function excludedForContext(ctx, branch) {
  if (typeof ctx.sessionManager?.getBranch !== "function") return new Set();
  return effectiveExclusions(branch.filter((entry) => entry.type === "message"), contextPolicy(ctx));
}
export function taskPlanState(ctx, { filter = true } = {}) {
  const branch = ctx.sessionManager?.getBranch?.() ?? [];
  const state = replayTaskPlan(branch);
  return filter ? filterTaskPlanForContext(state, excludedForContext(ctx, branch)) : state;
}
function contextBudget(value) {
  const parsed = Number(value ?? DEFAULT_PLAN_CONTEXT_CHARS);
  return Number.isFinite(parsed) ? Math.max(0, Math.min(12000, Math.floor(parsed))) : DEFAULT_PLAN_CONTEXT_CHARS;
}
const safeText = (value) => value.replace(/[<>\u2028\u2029]/gu, (character) => `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`);
const oneLine = (value) => safeText(value.replace(/\s+/g, " "));
const refs = (ids) => ids?.length ? ` [source:${ids.map(safeText).join(",")}]` : "";
function safeSlice(value, length) {
  let end = Math.max(0, length);
  if (end < value.length && /[\uD800-\uDBFF]/u.test(value[end - 1] ?? "")) end--;
  return value.slice(0, end);
}
export function renderTaskPlan(state, { maxChars = DEFAULT_PLAN_CONTEXT_CHARS } = {}) {
  const budget = contextBudget(maxChars);
  const lines = [`Plan mode: ${state.mode === "plan" ? "READ ONLY; /plan execute resumes execution" : "execute"}.`];
  const stepRows = [];
  const addStep = (step) => {
    stepRows.push({ line: lines.length, status: step.status });
    lines.push(`${safeText(step.id)} [${step.status}${step.status === "completed" ? "; model reported" : ""}] ${oneLine(step.title)}${refs(step.sourceEntryIds)}`);
    if (step.relevantFiles.length) lines.push(`  Files: ${step.relevantFiles.map(oneLine).join(", ")}`);
    for (const evidence of step.evidence) lines.push(`  Evidence (model cited): ${safeText(evidence.entryId)} ${oneLine(evidence.note)}`);
  };
  if (!state.plan) lines.push(`No active task plan. Use ${TOOL_NAME} create for substantial work.`);
  else {
    const p = state.plan;
    const goal = `Goal: ${oneLine(p.goal)}${refs(p.goalSourceEntryIds)}`;
    const goalLimit = Math.max(80, Math.floor(budget / 3));
    lines.push(goal.length > goalLimit ? `${safeSlice(goal, goalLimit - 1)}…` : goal);
    for (const status of ["in_progress", "blocked", "pending"]) for (const step of p.steps.filter((item) => item.status === status)) addStep(step);
    if (p.relevantFiles.length) lines.push(`Relevant files: ${p.relevantFiles.map((path) => `${oneLine(path)}${refs(p.relevantFileSources[path])}`).join("; ")}`);
    for (const item of p.constraints) lines.push(`Constraint: ${oneLine(item.text)}${refs(item.sourceEntryIds)}`);
    for (const step of p.steps.filter((item) => item.status === "completed")) addStep(step);
    for (const item of p.notes) lines.push(`Note: ${oneLine(item.text)}${refs(item.sourceEntryIds)}`);
  }
  const rendered = lines.join("\n");
  if (rendered.length <= budget) return rendered;
  const footer = (count) => `\n[Plan excerpt truncated; ${count} step(s) omitted. ${TOOL_NAME} show expands it; ${SESSION_SEARCH_TOOL_NAME} recovers sources.]`;
  const reserve = footer(MAX_STEPS).length;
  if (budget <= reserve) return "[Plan truncated]".slice(0, budget);
  const prefix = safeSlice(rendered, budget - reserve);
  const omitted = stepRows.filter((row) => lines.slice(0, row.line + 1).join("\n").length > prefix.length).length;
  return prefix + footer(omitted);
}
export function taskPlanContext(state, options) {
  const budget = contextBudget(options?.maxChars);
  const preface = "[Persistent task plan; metadata and model-reported progress, not verification. Follow current user instructions.]\n";
  if (budget <= preface.length) return safeSlice(preface, budget);
  return preface + renderTaskPlan(state, { maxChars: budget - preface.length });
}
export function planRelevantPaths(state) {
  if (!state.plan) return [];
  const plan = state.plan, result = new Map();
  for (const path of plan.relevantFiles) result.set(path, new Set(plan.relevantFileSources[path] ?? []));
  for (const step of plan.steps) for (const path of step.relevantFiles) {
    if (!result.has(path)) result.set(path, new Set());
    for (const id of step.sourceEntryIds) result.get(path).add(id);
  }
  return [...result].map(([path, ids]) => ({ path, sourceIds: [...ids] }));
}
function allSourceIds(state) {
  if (!state.plan) return [];
  const p = state.plan;
  return [...new Set([...p.goalSourceEntryIds, ...Object.values(p.relevantFileSources).flat(),
    ...p.constraints.flatMap((item) => item.sourceEntryIds), ...p.notes.flatMap((item) => item.sourceEntryIds),
    ...p.steps.flatMap((step) => [...step.sourceEntryIds, ...step.evidence.map((item) => item.entryId)])])];
}

const stringSchema = (maxLength) => ({ type: "string", minLength: 1, maxLength });
const sourceSchema = { type: "array", maxItems: 8, uniqueItems: true, items: stringSchema(256) };
const fileSchema = { type: "array", maxItems: 40, uniqueItems: true, items: stringSchema(512) };
const annotationSchema = { type: "object", additionalProperties: false, required: ["text"], properties: {
  text: stringSchema(400), sourceEntryIds: sourceSchema,
} };
export const TASK_PLAN_PARAMETERS = { type: "object", additionalProperties: false, required: ["action"], properties: {
  action: { type: "string", enum: ["create", "update", "revise", "show", "clear"] },
  goal: stringSchema(800), goalSourceEntryIds: sourceSchema,
  relevantFiles: fileSchema,
  relevantFileSources: { type: "object", additionalProperties: sourceSchema, maxProperties: 40 },
  constraints: { type: "array", maxItems: 16, items: annotationSchema },
  notes: { type: "array", maxItems: 12, items: annotationSchema },
  steps: { type: "array", maxItems: MAX_STEPS, items: { type: "object", additionalProperties: false,
    properties: { id: { type: "string", pattern: ID.source }, title: stringSchema(280),
      status: { type: "string", enum: STATUSES }, sourceEntryIds: sourceSchema, relevantFiles: fileSchema,
      evidence: { type: "array", maxItems: 8, items: { type: "object", additionalProperties: false,
        required: ["entryId", "note"], properties: { entryId: stringSchema(256), note: stringSchema(280) } } },
    } } },
  maxChars: { type: "integer", minimum: 256, maximum: 12000 },
} };

export function applyTaskPlanAction(state, params, branchEntries) {
  keys(params, Object.keys(TASK_PLAN_PARAMETERS.properties), `${TOOL_NAME} parameters`);
  if (!["create", "update", "revise", "show", "clear"].includes(params.action)) throw new Error("Invalid plan action");
  if (params.maxChars !== undefined && (!Number.isInteger(params.maxChars) || params.maxChars < 256 || params.maxChars > 12000)) throw new Error("maxChars must be an integer from 256 to 12000");
  if (["show", "clear"].includes(params.action)) {
    if (Object.keys(params).some((key) => !["action", "maxChars"].includes(key))) throw new Error("show/clear do not accept plan changes");
    return params.action === "clear" ? { ...clone(state), plan: null } : clone(state);
  }
  if (params.maxChars !== undefined) throw new Error("maxChars is only for show/clear");
  if (params.action === "create" && state.plan) throw new Error("An active plan exists; use revise, or clear it before create");
  if (params.action !== "create" && !state.plan) throw new Error("No active plan; create it first");
  const availableIds = new Set(branchEntries.map((entry) => entry.id));
  const latestUser = branchEntries.findLast((entry) => entry.type === "message" && entry.message?.role === "user");
  const defaults = latestUser ? [latestUser.id] : [];
  const next = clone(state);
  if (params.action === "create") next.plan = { goal: params.goal, goalSourceEntryIds: params.goalSourceEntryIds ?? defaults,
    steps: [], nextStepId: 1, relevantFiles: [], relevantFileSources: {}, constraints: [], notes: [] };
  const p = next.plan;
  if (params.goal !== undefined) { p.goal = params.goal; p.goalSourceEntryIds = params.goalSourceEntryIds ?? defaults; }
  else if (params.goalSourceEntryIds !== undefined) p.goalSourceEntryIds = params.goalSourceEntryIds;
  if (params.relevantFiles !== undefined) {
    p.relevantFiles = files(params.relevantFiles);
    p.relevantFileSources = Object.fromEntries(p.relevantFiles.map((path) => [path,
      params.relevantFileSources?.[path] ?? p.relevantFileSources[path] ?? defaults]));
  }
  if (params.relevantFileSources !== undefined) {
    if (!object(params.relevantFileSources) || Object.keys(params.relevantFileSources).some((path) => !p.relevantFiles.includes(path))) {
      throw new Error("File source mapping must name current relevantFiles");
    }
    p.relevantFileSources = { ...p.relevantFileSources, ...params.relevantFileSources };
  }
  for (const key of ["constraints", "notes"]) if (params[key] !== undefined) {
    p[key] = list(params[key], key === "constraints" ? 16 : 12, (item) => {
      keys(item, ["text", "sourceEntryIds"], key);
      return { ...item, sourceEntryIds: item.sourceEntryIds ?? defaults };
    }, key);
  }
  if (params.steps !== undefined) {
    const requested = list(params.steps, MAX_STEPS, (step) => {
      keys(step, ["id", "title", "status", "sourceEntryIds", "relevantFiles", "evidence"], "step update");
      return step;
    }, "Steps");
    const existing = new Map(p.steps.map((step) => [step.id, step]));
    const used = new Set(existing.keys());
    const suppliedIds = requested.filter((step) => step.id !== undefined).map((step) => step.id);
    if (new Set(suppliedIds).size !== suppliedIds.length) throw new Error("Duplicate step IDs");
    const normalize = (step) => {
      let id = step.id;
      if (params.action === "update" && (!id || !existing.has(id))) throw new Error("update needs an existing stable step ID; use revise to add steps");
      if (!id) { let n = p.nextStepId; while (used.has(`s${n}`) || suppliedIds.includes(`s${n}`)) n++; id = `s${n}`; p.nextStepId = n + 1; }
      used.add(id);
      const previous = existing.get(id);
      return { id, title: previous?.title, status: previous?.status ?? "pending", sourceEntryIds: previous?.sourceEntryIds ?? defaults,
        relevantFiles: previous?.relevantFiles ?? [], evidence: previous?.evidence ?? [], ...step,
        id, ...(step.sourceEntryIds === undefined ? { sourceEntryIds: [...new Set([...(previous?.sourceEntryIds ?? []), ...defaults])] } : {}) };
    };
    const normalized = requested.map(normalize);
    p.steps = params.action === "update" ? p.steps.map((step) => normalized.find((item) => item.id === step.id) ?? step) : normalized;
  }
  return validateTaskPlanState(next, { availableIds });
}

// This is a deliberately small shell language, not a claim that arbitrary bash
// is read only. Only one literal command is parsed, command-specific option lists
// are checked, and execution is rewritten to an absolute binary with quoted argv.
function literalArgv(command) {
  if (typeof command !== "string" || !command.trim() || command.length > 4096 || /[\x00-\x1f\x7f$`\\;|&<>]/.test(command)) {
    throw new Error("Only one literal read command is allowed; shell operators, expansions, escapes and newlines are blocked");
  }
  const args = []; let token = "", started = false, quote = null;
  for (const char of command) {
    if (quote) { if (char === quote) quote = null; else token += char; started = true; continue; }
    if (char === "'" || char === '"') { quote = char; started = true; continue; }
    if (char === " ") { if (started) { args.push(token); token = ""; started = false; } continue; }
    if (!/[a-zA-Z0-9_.,:/@%+=-]/.test(char)) throw new Error("Unquoted shell syntax is blocked; quote literal search text and paths");
    token += char; started = true;
  }
  if (quote) throw new Error("Unclosed quote");
  if (started) args.push(token);
  return args;
}
const quoteArg = (arg) => `'${arg.replaceAll("'", "'\\''")}'`;
const READ_OPTIONS = {
  pwd: { flags: ["-L", "-P"], values: [] },
  cat: { flags: ["-n", "-b", "-s", "-E", "-T", "-v", "-A", "--number", "--number-nonblank", "--squeeze-blank"], values: [] },
  ls: { flags: ["-a", "-A", "-l", "-h", "-d", "-R", "-F", "-p", "-1", "--all", "--almost-all", "--directory", "--human-readable"], values: [] },
  head: { flags: ["-q", "-v"], values: ["-n", "--lines", "-c", "--bytes"] },
  tail: { flags: ["-q", "-v"], values: ["-n", "--lines", "-c", "--bytes"] },
  wc: { flags: ["-l", "-w", "-c", "-m", "-L", "--lines", "--words", "--bytes", "--chars", "--max-line-length"], values: [] },
  stat: { flags: ["-L", "-f", "--dereference", "--file-system"], values: ["-c", "--format"] },
  tree: { flags: ["-a", "-d", "-f", "-i", "--dirsfirst"], values: ["-L"] },
  rg: { flags: ["-n", "-i", "-s", "-F", "-w", "-x", "-l", "-L", "-c", "-q", "-u", "--hidden", "--files", "--no-ignore", "--no-heading", "--line-number", "--fixed-strings", "--ignore-case", "--glob-case-insensitive"],
    values: ["-e", "--regexp", "-g", "--glob", "-t", "--type", "-T", "--type-not", "-A", "-B", "-C", "--after-context", "--before-context", "--context", "-m", "--max-count", "--max-columns"] },
  fd: { flags: ["-H", "-I", "-i", "-s", "-a", "-l", "--hidden", "--no-ignore", "--absolute-path", "--full-path", "--fixed-strings"],
    values: ["-t", "--type", "-e", "--extension", "-E", "--exclude", "-d", "--max-depth", "--min-depth", "--base-directory"] },
};
function checkOptions(args, spec) {
  let operands = false;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (operands || !arg.startsWith("-") || arg === "-") continue;
    if (arg === "--") { operands = true; continue; }
    if (spec.flags.includes(arg)) continue;
    const [flag, inline] = arg.split(/=(.*)/s);
    if (spec.flags.includes(flag) && inline === undefined) continue;
    if (spec.values.includes(flag)) { if (inline === undefined && ++i >= args.length) throw new Error(`Missing value for ${flag}`); continue; }
    // Only bundle explicitly allowed no-value short flags, never interpret an
    // attached value or an unrecognized command option.
    if (/^-[A-Za-z]+$/.test(arg) && [...arg.slice(1)].every((letter) => spec.flags.includes(`-${letter}`))) continue;
    throw new Error(`Option ${arg} is not in the read-only allowlist`);
  }
}
export function validateReadOnlyBash(command) {
  try {
    const [binary, ...args] = literalArgv(command);
    const name = binary.startsWith("/usr/bin/") ? binary.slice(9) : binary;
    if (name.includes("/")) throw new Error("Only named system read commands are allowed");
    let argv;
    if (name === "git") {
      let rest = [...args], directory = [];
      if (rest[0] === "--no-pager") rest.shift();
      if (rest[0] === "-C" && rest[1]) directory = rest.splice(0, 2);
      const subcommand = rest.shift();
      const gitSpecs = {
        "ls-files": { flags: ["--cached", "--stage", "-z"], values: [] },
      };
      // Working-tree status/diff may execute repository clean/process filters;
      // object diffs can also fetch missing objects. Permit index metadata only.
      if (!gitSpecs[subcommand]) throw new Error("Only git ls-files index metadata is allowed in plan mode; status/diff may execute repository helpers");
      checkOptions(rest, gitSpecs[subcommand]);
      argv = ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "LC_ALL=C", "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", "GIT_OPTIONAL_LOCKS=0",
        "/usr/bin/git", "--no-pager", "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "-c", "core.pager=cat", ...directory, subcommand, ...rest];
    } else {
      if (!READ_OPTIONS[name]) throw new Error(`Command ${binary} is not in the read-only allowlist`);
      checkOptions(args, READ_OPTIONS[name]);
      argv = [`/usr/bin/${name}`, ...(name === "rg" ? ["--no-config"] : []), ...args];
    }
    return { allowed: true, command: argv.map(quoteArg).join(" "), argv };
  } catch (error) { return { allowed: false, reason: error.message }; }
}

const READ_TOOLS = new Set(["read", "grep", "find", "ls", "session_search", "qwen_rehydrate_tool_turn", "qwen_plan"].map(canonicalToolName));
const BUILTIN_READ_TOOLS = new Set(["read", "grep", "find", "ls", "bash"]);
const extensionDirectory = dirname(fileURLToPath(import.meta.url));
const TRUSTED_EXTENSIONS = {
  [SESSION_SEARCH_TOOL_NAME]: ["qwen-session-search.mjs"],
  [REHYDRATE_TOOL_NAME]: ["qwen-tool-turn-rehydrate.mjs"],
  [TOOL_NAME]: ["qwen-task-plan.mjs", "qwen-task-plan.ts"],
  ...Object.fromEntries([...BUILTIN_READ_TOOLS].map((name) => [canonicalToolName(name), ["qwen-tool-names.ts"]])),
};
export function isTrustedPlanTool(name, info) {
  if (BUILTIN_READ_TOOLS.has(name)) return info?.sourceInfo?.source === "builtin" && info.sourceInfo.path === `<builtin:${name}>`;
  const canonical = canonicalToolName(name);
  if (!TRUSTED_EXTENSIONS[canonical] || typeof info?.sourceInfo?.path !== "string") return false;
  try {
    const source = realpathSync(info.sourceInfo.path);
    return TRUSTED_EXTENSIONS[canonical].some((file) => source === realpathSync(join(extensionDirectory, file)));
  } catch { return false; }
}
const planDigest = (state, maxChars) => createHash("sha256").update(taskPlanContext(state, { maxChars })).digest("hex");
export function planToolGate(state, event) {
  if (state.mode !== "plan") return undefined;
  if (READ_TOOLS.has(canonicalToolName(event.toolName))) return undefined;
  if (toolNamesMatch(event.toolName, "bash")) {
    const checked = validateReadOnlyBash(event.input?.command);
    if (checked.allowed) { event.input.command = checked.command; return undefined; }
    return { block: true, reason: `Read-only plan mode: ${checked.reason}. Use read_file/search_file_contents/find_files/list_directory, or /plan execute to resume execution.` };
  }
  return { block: true, reason: `Read-only plan mode blocks ${event.toolName}; only known read tools and plan metadata updates are allowed. /plan execute resumes execution.` };
}
export function installTaskPlan(pi, { maxContextChars = process.env.QWEN_PI_PLAN_CONTEXT_MAX_CHARS } = {}) {
  const maxChars = contextBudget(maxContextChars);
  const persist = (state) => pi.appendEntry(TASK_PLAN_ENTRY, validateTaskPlanState(state));
  const updateStatus = (ctx) => {
    const state = taskPlanState(ctx, { filter: false });
    ctx.ui?.setStatus?.("qwen-plan", state.mode === "plan" ? "plan: read only" : state.plan ? "task plan" : undefined);
  };
  const definition = { name: TOOL_NAME, label: "Persistent task plan",
    description: "For substantial multi-step tasks, create or revise a durable task plan before implementation and update it as progress/evidence/constraints change. Normal execution needs no mandatory approval; simple one-step work needs no plan. Create, update, revise, show or clear the current session branch's plan. create requires goal and steps; missing new step IDs are assigned. update patches existing steps by stable ID; revise replaces the ordered step list and may add/remove steps. Other supplied arrays replace that field. Completed status is model reported; cite existing selected-branch evidence entry IDs when available. show lists state and source IDs; maxChars expands its excerpt. clear removes the active plan, keeps mode and preserves history. Plan metadata changes are allowed in read-only plan mode; the tool never changes execution mode.",
    promptSnippet: "Track substantial tasks in a persistent goal and ordered steps with source-linked constraints and evidence",
    promptGuidelines: [
      `For substantial tasks with multiple steps, create or revise ${TOOL_NAME} before implementation, then update stable step IDs as work advances, evidence is obtained, or constraints change. Simple one-step tasks do not require a plan.`,
      "Maintain one in_progress step; preserve relevantFiles, user constraints and useful source entry IDs. Plan statuses and evidence notes are model-reported claims, never proof that checks passed.",
      "Normal execution proceeds without a mandatory approval flow. In explicitly enabled read-only plan mode, inspect and update plan metadata only; do not mutate files, run tests or call unknown tools. Only the user /plan off or /plan execute command resumes execution.",
      `Recover exact source details with ${SESSION_SEARCH_TOOL_NAME} or ${REHYDRATE_TOOL_NAME}; respect /context exclusions and current user instructions over old plan text.`,
    ], parameters: TASK_PLAN_PARAMETERS,
    async execute(_id, params, signal, _onUpdate, ctx) {
      if (signal?.aborted) throw new Error("Plan operation cancelled");
      const branch = ctx.sessionManager.getBranch(), before = replayTaskPlan(branch);
      const next = applyTaskPlanAction(before, params, branch);
      if (params.action !== "show") persist(next);
      updateStatus(ctx);
      const visible = filterTaskPlanForContext(next, excludedForContext(ctx, ctx.sessionManager.getBranch()));
      return { content: [{ type: "text", text: renderTaskPlan(visible, { maxChars: params.maxChars ?? maxChars }) }],
        details: { mode: next.mode, stepIds: visible.plan?.steps.map((step) => step.id) ?? [], modelReported: true,
          sourceEntryIds: allSourceIds(visible), planDigest: planDigest(visible, maxChars) } };
    },
  };
  pi.registerTool(definition);
  pi.registerCommand("plan", { description: "Toggle read-only planning; /plan [show|on|off|execute|clear]. clear keeps the current mode.",
    handler: async (args, ctx) => {
      const action = args.trim().toLowerCase();
      if (!["", "show", "on", "off", "execute", "clear"].includes(action)) { ctx.ui.notify("Usage: /plan [show|on|off|execute|clear]", "warning"); return; }
      const state = taskPlanState(ctx, { filter: false });
      if (action === "show") { ctx.ui.notify(renderTaskPlan(taskPlanState(ctx), { maxChars: 12000 }), "info"); return; }
      if (action === "clear") state.plan = null;
      else state.mode = action === "on" ? "plan" : ["off", "execute"].includes(action) ? "execute" : state.mode === "plan" ? "execute" : "plan";
      persist(state); updateStatus(ctx);
      ctx.ui.notify(action === "clear" ? `Plan cleared; history retained. Mode: ${state.mode}.` :
        state.mode === "plan" ? "Read-only plan mode enabled. /plan execute resumes execution." : "Execution mode enabled.", "info");
    } });
  for (const event of ["session_start", "session_tree", "session_compact"]) pi.on(event, (_event, ctx) => {
    if (event === "session_start") {
      const active = pi.getActiveTools(), tools = active.filter((name) => name !== LEGACY_TOOL_NAME);
      if (!tools.includes(TOOL_NAME)) tools.push(TOOL_NAME);
      if (tools.length !== active.length || tools.some((name, index) => name !== active[index])) pi.setActiveTools(tools);
    }
    updateStatus(ctx);
  });
  pi.on("tool_call", (event, ctx) => {
    try {
      const state = taskPlanState(ctx, { filter: false });
      if (state.mode === "plan" && (READ_TOOLS.has(canonicalToolName(event.toolName)) || toolNamesMatch(event.toolName, "bash")) &&
          !isTrustedPlanTool(event.toolName, pi.getAllTools?.().find((tool) => tool.name === event.toolName))) {
        return { block: true, reason: `Read-only plan mode blocks unverified implementation of ${event.toolName}; a tool name alone does not establish read-only behavior.` };
      }
      return planToolGate(state, event);
    }
    catch { return { block: true, reason: "Saved plan state could not be safely restored; tool execution is blocked." }; }
  });
  pi.on("user_bash", (event, ctx) => {
    let output;
    try {
      if (taskPlanState(ctx, { filter: false }).mode !== "plan") return undefined;
      output = "Read-only plan mode blocks ! shell commands. Use /plan execute to resume execution.";
    } catch { output = "Saved plan state could not be safely restored; shell execution is blocked."; }
    // User shell commands have a different SDK API without input rewriting.
    // Reject them explicitly so ! cannot silently bypass read-only planning.
    return { result: { output, exitCode: 1, cancelled: false, truncated: false } };
  });
  pi.on("before_agent_start", (_event, ctx) => {
    const state = taskPlanState(ctx);
    const active = ctx.sessionManager.buildContextEntries?.() ?? [];
    const prior = active.findLast((entry) => entry.type === "custom_message" && entry.customType === TASK_PLAN_CONTEXT ||
      entry.type === "message" && entry.message?.role === "toolResult" && toolNamesMatch(entry.message.toolName, TOOL_NAME));
    const details = prior?.type === "custom_message" ? prior.details : prior?.message?.details;
    const digest = planDigest(state, maxChars);
    if (details?.planDigest === digest || (!prior && !state.plan && state.mode === "execute")) return undefined;
    return { message: { customType: TASK_PLAN_CONTEXT, display: false, content: taskPlanContext(state, { maxChars }),
      details: { version: TASK_PLAN_VERSION, sourceEntryIds: allSourceIds(state), planDigest: digest } } };
  });
  pi.on("context", (event, ctx) => {
    const branch = ctx.sessionManager.getBranch(), excluded = excludedForContext(ctx, branch);
    if (!excluded.size) return undefined;
    // Only explicit source exclusion changes old plan messages; normal updates
    // append context and tool results without rewriting the cached history.
    let changed = false;
    const messages = event.messages.flatMap((message) => {
      if (!(message.details?.sourceEntryIds ?? []).some((id) => excluded.has(id))) return [message];
      if (message.role === "custom" && message.customType === TASK_PLAN_CONTEXT) { changed = true; return []; }
      if (message.role === "toolResult" && toolNamesMatch(message.toolName, TOOL_NAME)) {
        changed = true;
        return [{ ...message, content: [{ type: "text", text: `[Plan output withheld by /context; ${TOOL_NAME} show displays permitted current state.]` }], details: { withheld: true } }];
      }
      return [message];
    });
    return changed ? { messages } : undefined;
  });
}

export default installTaskPlan;
