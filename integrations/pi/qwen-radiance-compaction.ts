// Static imports let Pi's extension loader resolve its own exact package aliases.
// Keep the core adapter dependency-injected so its failure paths run in node:test.
import { convertToLlm } from "@earendil-works/pi-coding-agent";
import { streamSimpleOpenAICompletions } from "@earendil-works/pi-ai/compat";
import { Text } from "@earendil-works/pi-tui";
import { installRadianceCompaction } from "./qwen-radiance-compaction.mjs";
import { installRadianceErrors } from "./qwen-radiance-errors.mjs";

export default function radianceCompaction(pi) {
  installRadianceErrors(pi, { Text });
  installRadianceCompaction(pi, { convertToLlm, streamSimpleOpenAICompletions });
}
