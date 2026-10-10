// Inspect Pi's current tool registry without an inference request or tool changes.
export function installToolListing(pi) {
  pi.registerCommand("tools", {
    description: "List loaded tools, their descriptions and enabled status",
    handler: async (_args, ctx) => {
      let tools, active;
      try {
        tools = pi.getAllTools();
        active = new Set(pi.getActiveTools());
      } catch (error) {
        ctx.ui.notify(`Could not list loaded tools: ${error instanceof Error ? error.message : String(error)}`, "error");
        return;
      }
      if (!tools.length) {
        ctx.ui.notify("No tools are registered in this Pi session.", "info");
        return;
      }
      const ordered = [...tools].sort((a, b) => a.name.localeCompare(b.name));
      const enabled = ordered.filter((tool) => active.has(tool.name)).length;
      const color = (name, text) => ctx.ui.theme?.fg(name, text) ?? text;
      const rows = ordered.map((tool) => {
        const status = active.has(tool.name) ? "enabled" : "disabled";
        const description = tool.description?.replace(/\s+/g, " ").trim() || "No description provided.";
        return color("dim", "[") + color(status === "enabled" ? "success" : "error", status)
          + color("dim", "] ") + color("text", tool.name) + color("dim", ` — ${description}`);
      });
      ctx.ui.notify(`Loaded tools: ${tools.length} (${enabled} enabled, ${tools.length - enabled} disabled)\n\n${rows.join("\n")}`, "info");
    },
  });
}

export default installToolListing;
