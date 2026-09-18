/**
 * Frozen model-visible search tool for the searchtool-toolarchive-v1 prompt ABI.
 *
 * This adapter is used only for a tool-disabled compaction request. It preserves
 * the old prompt/tool schema without importing the mutable searchtool project.
 * A successor prompt ABI may expose newer tools after compaction.
 */

export default function searchtoolToolarchiveV1Compat(pi) {
	pi.on("session_start", () => {
		const active = pi.getActiveTools();
		const missing = ["search", "fetch", "extract"].filter((name) => !active.includes(name));
		if (missing.length > 0) pi.setActiveTools([...active, ...missing]);
	});

	pi.registerTool({
		name: "search",
		label: "Search",
		description:
			"Search Google AI mode (Google Search AI) with a query and return the full AI mode response as Markdown, including cited sources. Runs against the logged-in chromium-chatbot browser profile; the browser is auto-launched if not running.",
		promptSnippet: "Search Google AI mode and return the response",
		promptGuidelines: [
			"Use search when you need current web information, facts, news, or research that Google AI mode can answer.",
			"Keep search queries focused; make multiple search calls for multi-part questions.",
			"The response includes a Sources section with URLs you can pass to fetch or extract to read the original pages.",
		],
		parameters: {
			type: "object",
			required: ["query"],
			properties: {
				query: {
					type: "string",
					description: "The search query to send to Google AI mode",
				},
			},
		},
		async execute() {
			throw new Error("searchtool-toolarchive-v1 compatibility tool is summary-only");
		},
	});

	pi.registerTool({
		name: "fetch",
		label: "Fetch",
		description:
			"Fetch a web page and return its main content as Markdown. Uses the logged-in browser profile and handles JavaScript-rendered pages. Navigation chrome (nav, footers, sidebars) is stripped via Readability.",
		promptSnippet: "Fetch a web page's main content as Markdown",
		promptGuidelines: [
			"Use fetch to read the full content of a specific URL, e.g. a source returned by search.",
			"For long pages, consider extract to pull only the relevant part.",
		],
		parameters: {
			type: "object",
			required: ["url"],
			properties: {
				url: {
					type: "string",
					description: "The URL to fetch",
				},
			},
		},
		async execute() {
			throw new Error("searchtool-toolarchive-v1 compatibility tool is summary-only");
		},
	});

	pi.registerTool({
		name: "extract",
		label: "Extract",
		description:
			"Fetch a web page and return the sections around a keyword or phrase (snippeting). Useful for pulling the relevant part of a long page without reading the whole thing.",
		promptSnippet: "Extract the relevant part of a web page around a keyword",
		promptGuidelines: [
			"Use extract to pull the part of a page that discusses a specific keyword or phrase.",
			"Increase contextChars for more surrounding text; increase maxMatches for more occurrences.",
		],
		parameters: {
			type: "object",
			required: ["url", "query"],
			properties: {
				url: {
					type: "string",
					description: "The URL to fetch",
				},
				query: {
					type: "string",
					description: "The keyword or phrase to find on the page",
				},
				contextChars: {
					type: "number",
					description: "Characters of context before and after each match (default 400)",
				},
				maxMatches: {
					type: "number",
					description: "Maximum number of matches to return (default 5)",
				},
			},
		},
		async execute() {
			throw new Error("searchtool-toolarchive-v1 compatibility tool is summary-only");
		},
	});
}
