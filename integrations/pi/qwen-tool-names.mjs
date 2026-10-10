// One naming contract for new calls and historical Pi tool identities.
export const TOOL_NAME_ALIASES = Object.freeze({
  read: "read_file",
  bash: "run_shell_command",
  edit: "edit_file",
  write: "write_file",
  grep: "search_file_contents",
  find: "find_files",
  ls: "list_directory",
  search: "google_ai_search",
  fetch: "fetch_webpage",
  extract: "extract_webpage_snippets",
  qwen_rehydrate_tool_turn: "rehydrate_tool_result",
  session_search: "pi_session_search",
  qwen_plan: "manage_task_plan",
});

// Previously deployed names are recognized in saved history only; they are not
// registered as executable tools.
export const TOOL_NAME_COMPATIBILITY_ALIASES = Object.freeze({
  read_archived_tool_result: "rehydrate_tool_result",
});

export const BUILTIN_TOOL_NAMES = Object.freeze(["read", "bash", "edit", "write", "grep", "find", "ls"]);

export function canonicalToolName(name) {
  if (Object.hasOwn(TOOL_NAME_ALIASES, name)) return TOOL_NAME_ALIASES[name];
  return Object.hasOwn(TOOL_NAME_COMPATIBILITY_ALIASES, name) ? TOOL_NAME_COMPATIBILITY_ALIASES[name] : name;
}

export function toolNamesMatch(left, right) {
  return canonicalToolName(left) === canonicalToolName(right);
}

export function legacyToolName(name) {
  return Object.entries(TOOL_NAME_ALIASES).find(([, canonical]) => canonical === canonicalToolName(name))?.[0] ?? name;
}

// Change only tool identity metadata in the outgoing context, never stored JSONL
// or archived bytes. Tool IDs, arguments, outputs and message order stay intact.
export function canonicalizeToolMessages(messages) {
  let changed = false;
  const normalized = messages.map((message) => {
    if (message?.role === "toolResult") {
      const name = canonicalToolName(message.toolName);
      if (name !== message.toolName) { changed = true; return { ...message, toolName: name }; }
    }
    if (message?.role === "assistant" && Array.isArray(message.content)) {
      let contentChanged = false;
      const content = message.content.map((block) => {
        if (block?.type !== "toolCall") return block;
        const name = canonicalToolName(block.name);
        if (name === block.name) return block;
        contentChanged = true;
        return { ...block, name };
      });
      if (contentChanged) { changed = true; return { ...message, content }; }
    }
    return message;
  });
  return changed ? normalized : messages;
}

export function activateCanonicalTools(pi, historicalDefaults = []) {
  const registered = new Set(pi.getAllTools().map((tool) => tool.name));
  const active = pi.getActiveTools();
  // Launcher exclusions remove native names before session_start. Recover only
  // the retired native selections from old settings, not inactive current tools.
  const migratedDefaults = historicalDefaults.filter((name) => BUILTIN_TOOL_NAMES.includes(name))
    .map(canonicalToolName).filter((name) => registered.has(name));
  const canonical = [...new Set([...active.map((name) => {
    const renamed = canonicalToolName(name);
    return registered.has(renamed) ? renamed : name;
  }), ...migratedDefaults])];
  if (canonical.length !== active.length || canonical.some((name, index) => name !== active[index])) pi.setActiveTools(canonical);
}
