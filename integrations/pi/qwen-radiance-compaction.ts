// Static imports let Pi's extension loader resolve its own exact package aliases.
// Keep the core adapter dependency-injected so its failure paths run in node:test.
import { convertToLlm } from "@earendil-works/pi-coding-agent";
import { streamSimpleOpenAICompletions } from "@earendil-works/pi-ai/compat";
import { Text, matchesKey, truncateToWidth } from "@earendil-works/pi-tui";
import { installRadianceCompaction } from "./qwen-radiance-compaction.mjs";
import { installRadianceErrors } from "./qwen-radiance-errors.mjs";
import { installThinkingPurge } from "./qwen-radiance-thinking.mjs";
import { installContextPicker } from "./qwen-context.mjs";
import { installToolListing } from "./qwen-tools.mjs";

export default function radianceCompaction(pi) {
  installToolListing(pi);
  installRadianceErrors(pi, { Text });
  installThinkingPurge(pi, { Text });
  installContextPicker(pi, { Text, matchesKey, truncateToWidth, convertToLlm, streamSimpleOpenAICompletions });
  installRadianceCompaction(pi, { convertToLlm, streamSimpleOpenAICompletions });
}
