import {
  createReadToolDefinition, createBashToolDefinition, createEditToolDefinition,
  createWriteToolDefinition, createGrepToolDefinition, createFindToolDefinition,
  createLsToolDefinition, SettingsManager, type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import { activateCanonicalTools, canonicalToolName, canonicalizeToolMessages } from "./qwen-tool-names.mjs";

const namedGuidance = (text: string) => text.replace(/\b(read|bash|edit|write|grep|find|ls)\b/g, canonicalToolName);
const searchGuidelines = [
  "Use search_file_contents for repository text searches. Start with a relevant path, literal: true for exact paths/symbols/errors, context: 0 and limit: 50; request context around the relevant matches afterward.",
  "Use find_files for filename discovery. Narrow the directory and glob before widening a search to a whole repository collection.",
  "For matching filenames only, multiple exclusions, or ignored build files, use run_shell_command with rg -l -F, traversal exclusions using repeated -g flags, and an explicit timeout (20 seconds initially). The native search tool has one glob filter and no files-only or include-ignored option.",
  "Exclude irrelevant directories before scanning. A downstream grep -v filters results after the files were read; head bounds displayed output, not the cost of finding it. --include='*' excludes nothing. Search relevant build files explicitly when needed.",
];
const shellGuidelines = [
  "Use run_shell_command for builds, tests, version control and other shell operations. Prefer search_file_contents, find_files and list_directory for ordinary repository inspection.",
  ...searchGuidelines.slice(2),
];

// Delegate to the installed SDK definitions. This preserves schemas, rendering,
// cancellation, truncation, image support and the SDK's shared file-mutation queue.
export default function namedTools(pi: ExtensionAPI) {
  const factories = [createReadToolDefinition, createBashToolDefinition, createEditToolDefinition,
    createWriteToolDefinition, createGrepToolDefinition, createFindToolDefinition, createLsToolDefinition];
  for (const createDefinition of factories) {
    const definition = createDefinition(process.cwd());
    const guidelines = definition.name === "grep" ? searchGuidelines : definition.name === "bash" ? shellGuidelines : [];
    pi.registerTool({
      ...definition,
      name: canonicalToolName(definition.name),
      description: definition.description + (guidelines.length ? " " + guidelines[0] : ""),
      promptSnippet: definition.name === "bash" ? "Run builds, tests, version-control commands and other shell operations"
        : definition.name === "grep" ? "Search repository text using ripgrep with a narrow path, optional literal pattern and bounded matches"
        : definition.promptSnippet && namedGuidance(definition.promptSnippet),
      promptGuidelines: [...(definition.promptGuidelines?.map(namedGuidance) ?? []), ...guidelines],
      execute: (id, params, signal, onUpdate, ctx) => {
        // Honor persisted Pi shell/image options just as native tools do. Read
        // on invocation so /settings changes are not stranded in this delegate.
        const settings = ["read", "bash"].includes(definition.name)
          ? SettingsManager.create(ctx.cwd, process.env.PI_CODING_AGENT_DIR, { projectTrusted: ctx.isProjectTrusted() }) : undefined;
        const options = definition.name === "read" ? { autoResizeImages: settings!.getImageAutoResize() }
          : definition.name === "bash" ? { commandPrefix: settings!.getShellCommandPrefix(), shellPath: settings!.getShellPath() } : undefined;
        return createDefinition(ctx.cwd, options).execute(id, params, signal, onUpdate, ctx);
      },
    });
  }
  pi.on("session_start", async (_event, ctx) => {
    // Respect explicit CLI tool disabling. Otherwise migrate old native defaults
    // that the launcher has removed from the registry before extension startup.
    const disabledDefaults = process.argv.some((arg) => ["--no-tools", "-nt", "--no-builtin-tools", "-nbt"].includes(arg));
    const defaults = disabledDefaults ? [] : SettingsManager.create(ctx.cwd, process.env.PI_CODING_AGENT_DIR,
      { projectTrusted: ctx.isProjectTrusted() }).getDefaultTools() ?? ["read", "bash", "edit", "write"];
    activateCanonicalTools(pi, defaults);
  });
  pi.on("session_switch", async () => activateCanonicalTools(pi));
  pi.on("context", async (event) => {
    const messages = canonicalizeToolMessages(event.messages);
    return messages === event.messages ? undefined : { messages };
  });
}
